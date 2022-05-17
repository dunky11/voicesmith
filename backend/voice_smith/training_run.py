from pathlib import Path
import shutil
from typing import Literal
import sys
import torch
import json
import sqlite3
from typing import Union, Literal, Tuple, Dict, Any
import argparse
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.preprocessing.extract_data import extract_data
from voice_smith.preprocessing.ground_truth_alignment import ground_truth_alignment
from voice_smith.acoustic_training import train_acoustic
from voice_smith.vocoder_training import train_vocoder
from voice_smith.config.preprocess_config import preprocess_config
from voice_smith.config.acoustic_fine_tuning_config import acoustic_fine_tuning_config
from voice_smith.config.acoustic_model_config import acoustic_model_config
from voice_smith.config.vocoder_model_config import vocoder_model_config
from voice_smith.config.vocoder_fine_tuning_config import vocoder_fine_tuning_config
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.export import acoustic_to_torchscript, vocoder_to_torchscript
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con
from voice_smith.config.symbols import symbol2id
from voice_smith.docker.api import generate_vocab, align, reload_docker

def step_from_ckpt(ckpt: str):
    ckpt_path = Path(ckpt)
    return int(ckpt_path.stem.split("_")[1])


def recalculate_train_size(batch_size: int, grad_acc_step: int, target_size: int):
    if batch_size <= 1:
        raise Exception("Batch size has to be greater than one")

    batch_size = batch_size - 1
    while batch_size * grad_acc_step < target_size:
        grad_acc_step = grad_acc_step + 1
    return batch_size, grad_acc_step


def get_available_name(model_dir: str, name: str):
    model_path = Path(model_dir)
    i = 1
    while (model_path / name).exists():
        name = name + f" ({i})"
        i += 1
    return name


def get_latest_checkpoint(name: str, ckpt_dir: str):
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None, 0
    ckpts = [str(el) for el in ckpt_path.iterdir()]
    if len(ckpts) == 0:
        return None, 0
    ckpts = list(map(step_from_ckpt, ckpts))
    ckpts.sort()
    last_ckpt = ckpts[-1]
    last_ckpt_path = ckpt_path / f"{name}_{last_ckpt}.pt"
    return last_ckpt_path, last_ckpt


def get_acoustic_configs(
    cur: sqlite3.Cursor, training_run_id: int
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    row = cur.execute(
        """
        SELECT validation_size, acoustic_learning_rate, acoustic_training_iterations, acoustic_batch_size, acoustic_grad_accum_steps, acoustic_validate_every, only_train_speaker_emb_until 
        FROM training_run WHERE ID=?
        """,
        (training_run_id,),
    ).fetchone()
    (
        validation_size,
        acoustic_learning_rate,
        acoustic_training_iterations,
        batch_size,
        grad_acc_step,
        acoustic_validate_every,
        only_train_speaker_until,
    ) = row
    p_config = preprocess_config.copy()
    m_config = acoustic_model_config.copy()
    t_config = acoustic_fine_tuning_config.copy()
    p_config["val_size"] = validation_size / 100.0
    t_config["batch_size"] = batch_size
    t_config["grad_acc_step"] = grad_acc_step
    t_config["step"]["train_steps"] = acoustic_training_iterations
    t_config["optimizer"]["learning_rate"] = acoustic_learning_rate
    t_config["step"]["val_step"] = acoustic_validate_every
    t_config["step"]["only_train_speaker_until"] = only_train_speaker_until
    return p_config, m_config, t_config


def get_vocoder_configs(cur: sqlite3.Cursor, training_run_id):
    row = cur.execute(
        """
        SELECT vocoder_learning_rate, vocoder_training_iterations, vocoder_batch_size, vocoder_grad_accum_steps, vocoder_validate_every 
        FROM training_run WHERE ID=?
        """,
        (training_run_id,),
    ).fetchone()
    (
        vocoder_learning_rate,
        vocoder_training_iterations,
        vocoder_batch_size,
        vocoder_grad_accum_steps,
        vocoder_validate_every,
    ) = row
    p_config = preprocess_config.copy()
    m_config = vocoder_model_config.copy()
    t_config = vocoder_fine_tuning_config.copy()
    t_config["batch_size"] = vocoder_batch_size
    t_config["grad_accum_steps"] = vocoder_grad_accum_steps
    t_config["train_steps"] = vocoder_training_iterations
    t_config["learning_rate"] = vocoder_learning_rate
    t_config["validation_interval"] = vocoder_validate_every
    return p_config, m_config, t_config


def get_device(device: Union[Literal["CPU"], Literal["GPU"]]) -> torch.device:
    if device == "CPU":
        return torch.device("cpu")
    elif device == "GPU":
        if not torch.cuda.is_available():
            raise Exception(
                f"Mode was set to 'GPU' but no available GPU could be found ..."
            )
        return torch.device("cuda")
    else:
        raise Exception(f"Device '{device}' is not a valid device type ...")


def continue_training_run(
    training_run_id: int, 
    training_runs_path: str, 
    assets_path: str, 
    db_path: str, 
    models_path: str, 
    datasets_path: str,
    user_data_path: str
):
    con = get_con(db_path)
    cur = con.cursor()
    data_path = Path(training_runs_path) / str(training_run_id)
    model_path = Path(models_path)
    dataset_path = Path(datasets_path)
    stage = None

    def get_logger():
        con = get_con(db_path)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=training_run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="preprocessing",
        )
        return logger

    while stage != "finished":
        row = cur.execute(
            "SELECT stage, preprocessing_stage FROM training_run WHERE ID=?",
            (training_run_id,),
        ).fetchone()
        stage, preprocessing_stage = row
        if stage == "not_started":
            cur.execute(
                "UPDATE training_run SET stage='preprocessing' WHERE ID=?",
                (training_run_id,),
            )
            con.commit()
            if data_path.exists():
                shutil.rmtree(data_path)
            (data_path / "logs").mkdir(exist_ok=True, parents=True)
            (data_path / "raw_data").mkdir(exist_ok=True)

        elif stage == "preprocessing":
            logger = SQLLogger(
                training_run_id=training_run_id,
                con=con,
                cursor=cur,
                out_dir=str(data_path),
                stage="preprocessing",
            )
            if preprocessing_stage == "not_started":
                set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
                cur.execute(
                    "UPDATE training_run SET preprocessing_stage='copying_files' WHERE ID=?",
                    (training_run_id,),
                )
                con.commit()

            elif preprocessing_stage == "copying_files":
                set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
                txt_paths, texts, audio_paths, names = [], [], [], []
                for (
                    txt_path,
                    text,
                    audio_path,
                    speaker_name,
                    dataset_id,
                    speaker_id,
                ) in cur.execute(
                    """
                    SELECT sample.txt_path, sample.text, sample.audio_path, speaker.name AS speaker_name, dataset.ID AS dataset_id, speaker.ID as speaker_id FROM training_run INNER JOIN dataset ON training_run.dataset_id = dataset.ID 
                    INNER JOIN speaker on speaker.dataset_id = dataset.ID
                    INNER JOIN sample on sample.speaker_id = speaker.ID
                    WHERE training_run.ID=?
                    """,
                    (training_run_id,),
                ).fetchall():
                    full_audio_path = (
                        dataset_path
                        / str(dataset_id)
                        / "speakers"
                        / str(speaker_id)
                        / audio_path
                    )
                    txt_paths.append(txt_path)
                    texts.append(text)
                    audio_paths.append(str(full_audio_path))
                    names.append(speaker_name)
                copy_files(
                    db_id=training_run_id,
                    table_name="training_run",
                    data_path=str(data_path),
                    txt_paths=txt_paths,
                    texts=texts,
                    audio_paths=audio_paths,
                    names=names,
                    get_logger=get_logger,
                )
                cur.execute(
                    "UPDATE training_run SET preprocessing_stage='gen_vocab' WHERE ID=?",
                    (training_run_id,),
                )
                con.commit()

            elif preprocessing_stage == "gen_vocab":
                (data_path / "data").mkdir(exist_ok=True)
                set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
                print(user_data_path)
                container = reload_docker(user_data_path=user_data_path)
                generate_vocab(container, training_run_name=str(training_run_id))
                quit()
                cur.execute(
                    "UPDATE training_run SET preprocessing_stage='gen_alignments', preprocessing_gen_vocab_progress=1.0 WHERE ID=?",
                    (training_run_id,),
                )
                con.commit()

            elif preprocessing_stage == "gen_alignments":
                set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
                container = reload_docker(user_data_path=user_data_path)
                align(container, training_run_name=str(training_run_id))
                cur.execute(
                    "UPDATE training_run SET preprocessing_stage='extract_data', preprocessing_gen_align_progress=1.0 WHERE ID=?",
                    (training_run_id,),
                )
                con.commit()

            elif preprocessing_stage == "extract_data":
                set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
                row = cur.execute(
                    "SELECT validation_size FROM training_run WHERE ID=?",
                    (training_run_id,),
                ).fetchone()
                p_config, _, _ = get_acoustic_configs(
                    cur=cur, training_run_id=training_run_id
                )
                extract_data(
                    db_id=training_run_id,
                    table_name="training_run",
                    training_run_name=str(training_run_id),
                    preprocess_config=p_config,
                    get_logger=get_logger,
                    assets_path=assets_path,
                    training_runs_path=training_runs_path
                )
                cur.execute(
                    "UPDATE training_run SET stage='acoustic_fine_tuning', preprocessing_stage='finished' WHERE ID=?",
                    (training_run_id,),
                )
                con.commit()
                if (data_path / "raw_data").exists():
                    shutil.rmtree(data_path / "raw_data")

            elif preprocessing_stage == "finished":
                pass

            else:
                raise Exception(
                    f"Preprocessing stage '{preprocessing_stage}' is not a valid stage ..."
                )

        elif stage == "acoustic_fine_tuning":
            set_stream_location(str(data_path / "logs" / "acoustic_fine_tuning.txt"))
            print("Fine-Tuning Acoustic Model ...")
            row = cur.execute(
                "SELECT device FROM training_run WHERE ID=?",
                (training_run_id,),
            ).fetchone()

            device = row[0]
            device = get_device(device)

            p_config, m_config, t_config = get_acoustic_configs(
                cur=cur, training_run_id=training_run_id
            )

            logger = SQLLogger(
                training_run_id=training_run_id,
                con=con,
                cursor=cur,
                out_dir=str(data_path),
                stage="acoustic",
            )

            target_batch_size_total = t_config["batch_size"] * t_config["grad_acc_step"]

            while True:
                try:
                    checkpoint_acoustic, step = get_latest_checkpoint(
                        name="acoustic", ckpt_dir=str(data_path / "ckpt" / "acoustic")
                    )
                    checkpoint_style, step = get_latest_checkpoint(
                        name="style", ckpt_dir=str(data_path / "ckpt" / "acoustic")
                    )
                    if checkpoint_acoustic is None or checkpoint_style is None:
                        reset = True
                        checkpoint_acoustic = str(
                            Path(".") / "assets" / "acoustic_pretrained.pt"
                        )
                        checkpoint_style = str(
                            Path(".") / "assets" / "style_pretrained.pt"
                        )
                    else:
                        reset = False

                    cur.execute(
                        "DELETE FROM image_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                        (training_run_id, step),
                    )
                    cur.execute(
                        "DELETE FROM graph_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                        (training_run_id, step),
                    )
                    cur.execute(
                        "DELETE FROM audio_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                        (training_run_id, step),
                    )
                    con.commit()

                    train_acoustic(
                        db_id=training_run_id,
                        training_run_name=str(training_run_id),
                        preprocess_config=p_config,
                        model_config=m_config,
                        train_config=t_config,
                        logger=logger,
                        device=device,
                        reset=reset,
                        checkpoint_acoustic=checkpoint_acoustic,
                        checkpoint_style=checkpoint_style,
                        fine_tuning=True,
                        overwrite_saves=True,
                        assets_path=assets_path,
                        training_runs_path=training_runs_path
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if t_config["batch_size"] > 1:
                            old_batch_size, old_grad_accum_steps = (
                                t_config["batch_size"],
                                t_config["grad_acc_step"],
                            )
                            batch_size, grad_acc_step = recalculate_train_size(
                                batch_size=old_batch_size,
                                grad_acc_step=old_grad_accum_steps,
                                target_size=target_batch_size_total,
                            )
                            print(
                                f"""
                                Ran out of VRAM during acoustic model training, setting batch size from {old_batch_size} 
                                to {batch_size} and gradient accumulation steps from {old_grad_accum_steps} to {grad_acc_step} and trying again...
                                """
                            )
                            t_config["batch_size"] = batch_size
                            t_config["grad_acc_step"] = grad_acc_step
                        else:
                            raise Exception(
                                f"""
                                Ran out of VRAM during acoustic model training, batch size is {t_config["batch_size"]}, so cannot set it lower. 
                                Please restart your PC and try again. If this error continues you may not
                                have enough VRAM to run this software. You could try training on CPU
                                instead of on GPU.
                                """
                            )
                    else:
                        raise e

            cur.execute(
                "UPDATE training_run SET stage='ground_truth_alignment' WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "ground_truth_alignment":
            set_stream_location(str(data_path / "logs" / "ground_truth_alignment.txt"))
            logger = SQLLogger(
                training_run_id=training_run_id,
                con=con,
                cursor=cur,
                out_dir=str(data_path),
                stage="ground_truth_alignment",
            )
            row = cur.execute(
                "SELECT acoustic_batch_size, device FROM training_run WHERE ID=?",
                (training_run_id,),
            ).fetchone()
            batch_size, device = row
            device = get_device(device)
            checkpoint_acoustic, step = get_latest_checkpoint(
                name="acoustic", ckpt_dir=str(data_path / "ckpt" / "acoustic")
            )
            checkpoint_style, step = get_latest_checkpoint(
                name="style", ckpt_dir=str(data_path / "ckpt" / "acoustic")
            )
            while True:
                try:
                    ground_truth_alignment(
                        db_id=training_run_id,
                        table_name="training_run",
                        training_run_name=str(training_run_id),
                        batch_size=3 * batch_size,
                        group_size=3,
                        device=device,
                        checkpoint_acoustic=str(checkpoint_acoustic),
                        checkpoint_style=str(checkpoint_style),
                        logger=logger,
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if batch_size > 1:
                            old_batch_size = batch_size
                            batch_size = batch_size - 1
                            print(
                                f"""
                                Ran out of VRAM during ground truth alignment, setting batch size from {old_batch_size} 
                                to {batch_size} and trying again...
                                """
                            )
                        else:
                            raise Exception(
                                f"""
                                Ran out of VRAM during ground truth alignment, batch size is {batch_size}, so cannot set lower. 
                                Please restart your PC and try again. If this error continues you may not
                                have enough VRAM to run this software. You could try training on CPU
                                instead of on GPU.
                                """
                            )
                    else:
                        raise e

            cur.execute(
                "UPDATE training_run SET stage='vocoder_fine_tuning' WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "vocoder_fine_tuning":
            set_stream_location(str(data_path / "logs" / "vocoder_fine_tuning.txt"))
            print("Fine-Tuning Vocoder ...")

            row = cur.execute(
                "SELECT device FROM training_run WHERE ID=?",
                (training_run_id,),
            ).fetchone()
            device = row[0]
            device = get_device(device)

            p_config, m_config, t_config = get_vocoder_configs(
                cur=cur, training_run_id=training_run_id
            )

            logger = SQLLogger(
                training_run_id=training_run_id,
                con=con,
                cursor=cur,
                out_dir=str(data_path),
                stage="vocoder",
            )

            target_batch_size_total = (
                t_config["batch_size"] * t_config["grad_accum_steps"]
            )

            while True:
                try:
                    checkpoint_path, step = get_latest_checkpoint(
                        name="vocoder", ckpt_dir=str(data_path / "ckpt" / "vocoder")
                    )

                    cur.execute(
                        "DELETE FROM image_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                        (training_run_id, step),
                    )
                    cur.execute(
                        "DELETE FROM graph_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                        (training_run_id, step),
                    )
                    cur.execute(
                        "DELETE FROM audio_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                        (training_run_id, step),
                    )
                    con.commit()

                    if checkpoint_path == None:
                        reset = True
                        checkpoint_path = str(
                            Path(".") / "assets" / "vocoder_pretrained.pt"
                        )
                    else:
                        reset = False

                    train_vocoder(
                        db_id=training_run_id,
                        training_run_name=str(training_run_id),
                        train_config=t_config,
                        logger=logger,
                        device=device,
                        reset=reset,
                        checkpoint_path=checkpoint_path,
                        training_runs_path=training_runs_path,
                        fine_tuning=True,
                        overwrite_saves=True,
                    )
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if t_config["batch_size"] > 1:
                            old_batch_size, old_grad_accum_steps = (
                                t_config["batch_size"],
                                t_config["grad_accum_steps"],
                            )
                            batch_size, grad_acc_step = recalculate_train_size(
                                batch_size=old_batch_size,
                                grad_acc_step=old_grad_accum_steps,
                                target_size=target_batch_size_total,
                            )
                            print(
                                f"""
                                Ran out of VRAM during vocoder model training, setting batch size from {old_batch_size} 
                                to {batch_size} and gradient accumulation steps from {old_grad_accum_steps} to {grad_acc_step} and trying again...
                                """
                            )
                            t_config["batch_size"] = batch_size
                            t_config["grad_accum_steps"] = grad_acc_step
                        else:
                            raise Exception(
                                f"""
                                Ran out of VRAM during vocoder model training, batch size is {t_config["batch_size"]}, so cannot set it lower. 
                                Please restart your PC and try again. If this error continues you may not
                                have enough VRAM to run this software. You could try training on CPU
                                instead of on GPU.
                                """
                            )
                    else:
                        raise e

            cur.execute(
                "UPDATE training_run SET stage='save_model' WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "save_model":
            set_stream_location(str(data_path / "logs" / "save_model.txt"))
            print("Saving Model ...")

            checkpoint_acoustic, acoustic_steps = get_latest_checkpoint(
                name="acoustic",
                ckpt_dir=str(data_path / "ckpt" / "acoustic"),
            )
            checkpoint_style, acoustic_steps = get_latest_checkpoint(
                name="style",
                ckpt_dir=str(data_path / "ckpt" / "acoustic"),
            )
            checkpoint_vocoder, vocoder_steps = get_latest_checkpoint(
                name="vocoder", ckpt_dir=str(data_path / "ckpt" / "vocoder")
            )
            if checkpoint_acoustic is None:
                raise ValueError(
                    "Acoustic path is None in save_model, no model has been saved?"
                )

            if checkpoint_style is None:
                raise ValueError(
                    "Style path is None in save_model, no model has been saved?"
                )

            if checkpoint_vocoder is None:
                raise ValueError(
                    "Vocoder path is None in save_model, no model has been saved?"
                )

            acoustic_model, style_predictor = acoustic_to_torchscript(
                checkpoint_acoustic=str(checkpoint_acoustic),
                checkpoint_style=str(checkpoint_style),
                data_path=str(data_path / "data"),
            )
            vocoder = vocoder_to_torchscript(
                ckpt_path=str(checkpoint_vocoder), data_path=str(data_path / "data")
            )

            p_config, m_config_acoustic, t_config_acoustic = get_acoustic_configs(
                cur=cur, training_run_id=training_run_id
            )

            p_config, m_config_vocoder, t_config_vocoder = get_vocoder_configs(
                cur=cur, training_run_id=training_run_id
            )

            with open(Path(data_path) / "data" / "speakers.json", "r") as f:
                speakers = json.load(f)

            # TODO place in transaction

            model_type = "Delighful_FreGANv1_v0.0"
            config = {
                "acousticSteps": acoustic_steps,
                "vocoderSteps": vocoder_steps,
                "preprocessing": p_config,
                "acoustic_model": m_config_acoustic,
                "vocoder_model": m_config_vocoder,
                "fine_tuning_acoustic": t_config_acoustic,
                "fine_tuning_vocoder": t_config_vocoder,
            }
            description = """
                DelightfulTTS (https://arxiv.org/pdf/2110.12612.pdf) like architecture with DeepVoice 3 
                speaker embeddings. Speaker embeddings are pretrained using ECAPA-TDNN. 
                Attention with TinyBERT is used to improve prosody. Uses a mixed char, phoneme and caps embedding. 
                Duration is unsupervised using the scheme from "One TTS Alignment To Rule Them All" 
                (https://arxiv.org/pdf/2108.10447.pdf).
                Vocoder is 44.1khz FreGANv1 with HiFiGAN discriminators, both generator and discriminator are
                speaker conditional.
            """

            row = cur.execute(
                "SELECT name FROM training_run WHERE ID=?",
                (training_run_id,),
            ).fetchone()
            con.commit()

            name = get_available_name(model_dir=str(model_path), name=row[0])
            (model_path / name).mkdir(exist_ok=True, parents=True)
            models_dir = model_path / name / "torchscript"
            models_dir.mkdir(exist_ok=True)
            acoustic_model.save(str(models_dir / "acoustic_model.pt"))
            style_predictor.save(models_dir / "style_predictor.pt")
            vocoder.save(models_dir / "vocoder.pt")
            with open(model_path / name / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f)

            lexicon = {}
            with open(
                Path(".")
                / "training_runs"
                / str(training_run_id)
                / "data"
                / "lexicon_post.txt",
                "r",
                encoding="utf-8",
            ) as f:
                for line in f:
                    split = line.strip().split(" ")
                    lexicon[split[0]] = split[1:]

            cur.execute(
                "INSERT INTO model(name, type, description) VALUES (?, ?, ?)",
                (name, model_type, description),
            )
            model_id = cur.lastrowid
            for speaker_name in speakers.keys():
                speaker_id = speakers[speaker_name]
                cur.execute(
                    "INSERT INTO model_speaker (name, speaker_id, model_id) VALUES (?, ?, ?)",
                    [speaker_name, speaker_id, model_id],
                )
            for symbol in symbol2id.keys():
                symbol_id = symbol2id[symbol]
                cur.execute(
                    "INSERT INTO symbol (symbol, symbol_id, model_id) VALUES (?, ?, ?)",
                    [symbol, symbol_id, model_id],
                )
            for word in lexicon.keys():
                phonemes = " ".join(lexicon[word])
                cur.execute(
                    "INSERT INTO lexicon_word (word, phonemes, model_id) VALUES (?, ?, ?)",
                    [word, phonemes, model_id],
                )
            con.commit()

            cur.execute(
                "UPDATE training_run SET stage='finished' WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "finished":
            break

        else:
            raise Exception(f"Stage '{stage}' is not a valid stage ...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_run_id", type=int, required=True)
    parser.add_argument("--training_runs_path", type=str, required=True)
    parser.add_argument("--assets_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--user_data_path", type=str, required=True)
    args = parser.parse_args()

    continue_training_run(
        training_run_id=args.training_run_id,
        training_runs_path=args.training_runs_path,
        assets_path=args.assets_path,
        db_path=args.db_path,
        models_path=args.models_path,
        datasets_path=args.datasets_path,
        user_data_path=args.user_data_path
    )