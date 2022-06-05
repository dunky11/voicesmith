from pathlib import Path
import shutil
from typing import Literal
import sys
import torch
import json
import sqlite3
from typing import Union, Literal, Tuple, Dict, Any, Callable
import dataclasses
import argparse
import multiprocessing as mp
from voice_smith.acoustic_training import train_acoustic
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.preprocessing.extract_data import extract_data
from voice_smith.preprocessing.ground_truth_alignment import ground_truth_alignment
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticFinetuningConfig,
    AcousticModelConfig,
    VocoderFinetuningConfig,
    VocoderModelConfig,
)
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.export import acoustic_to_torchscript, vocoder_to_torchscript
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.config.symbols import symbol2id
from voice_smith.utils.tools import warnings_to_stdout, get_workers
from voice_smith.preprocessing.generate_vocab import generate_vocab
from voice_smith.preprocessing.merge_lexika import merge_lexica
from voice_smith.preprocessing.align import align
from voice_smith.utils.punctuation import get_punct
from voice_smith.vocoder_training import train_vocoder
from voice_smith.utils.runs import StageRunner

warnings_to_stdout()


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
    cur: sqlite3.Cursor, run_id: int
) -> Tuple[PreprocessingConfig, AcousticModelConfig, AcousticFinetuningConfig]:
    row = cur.execute(
        """
        SELECT min_seconds, max_seconds, maximum_workers, use_audio_normalization, validation_size, acoustic_learning_rate, acoustic_training_iterations, acoustic_batch_size, acoustic_grad_accum_steps, acoustic_validate_every, only_train_speaker_emb_until 
        FROM training_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (
        min_seconds,
        max_seconds,
        maximum_workers,
        use_audio_normalization,
        validation_size,
        acoustic_learning_rate,
        acoustic_training_iterations,
        batch_size,
        grad_acc_step,
        acoustic_validate_every,
        only_train_speaker_until,
    ) = row
    p_config: PreprocessingConfig = PreprocessingConfig()
    m_config: AcousticModelConfig = AcousticModelConfig()
    t_config: AcousticFinetuningConfig = AcousticFinetuningConfig()
    p_config.val_size = validation_size / 100.0
    p_config.min_seconds = min_seconds
    p_config.max_seconds = max_seconds
    p_config.use_audio_normalization = use_audio_normalization == 1
    p_config.workers = get_workers(maximum_workers)
    t_config.batch_size = batch_size
    t_config.grad_acc_step = grad_acc_step
    t_config.train_steps = acoustic_training_iterations
    t_config.optimizer_config.learning_rate = acoustic_learning_rate
    t_config.val_step = acoustic_validate_every
    t_config.only_train_speaker_until = only_train_speaker_until
    return p_config, m_config, t_config


def get_vocoder_configs(
    cur: sqlite3.Cursor, run_id
) -> Tuple[PreprocessingConfig, VocoderModelConfig, VocoderFinetuningConfig]:
    row = cur.execute(
        """
        SELECT vocoder_learning_rate, vocoder_training_iterations, vocoder_batch_size, vocoder_grad_accum_steps, vocoder_validate_every 
        FROM training_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (
        vocoder_learning_rate,
        vocoder_training_iterations,
        vocoder_batch_size,
        vocoder_grad_accum_steps,
        vocoder_validate_every,
    ) = row
    p_config = PreprocessingConfig()
    m_config = VocoderModelConfig()
    t_config = VocoderFinetuningConfig()
    t_config.batch_size = vocoder_batch_size
    t_config.grad_accum_steps = vocoder_grad_accum_steps
    t_config.train_steps = vocoder_training_iterations
    t_config.learning_rate = vocoder_learning_rate
    t_config.validation_interval = vocoder_validate_every
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


def not_started_stage(
    cur: sqlite3.Cursor, con: sqlite3.Connection, run_id: int, data_path: str, **kwargs
) -> bool:
    data_path = Path(data_path)
    if data_path.exists():
        shutil.rmtree(data_path)
    (data_path / "logs").mkdir(exist_ok=True, parents=True)
    (data_path / "raw_data").mkdir(exist_ok=True, parents=True)
    cur.execute(
        "UPDATE training_run SET stage='preprocessing' WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def preprocessing_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    dataset_path: str,
    get_logger: Callable[[], SQLLogger],
    assets_path: str,
    environment_name: str,
    training_runs_path: str,
    **kwargs,
) -> bool:
    preprocessing_stage = None
    while preprocessing_stage != "finished":
        preprocessing_stage = cur.execute(
            "SELECT preprocessing_stage FROM training_run WHERE ID=?",
            (run_id,),
        ).fetchone()[0]
        if preprocessing_stage == "not_started":
            set_stream_location(str(Path(data_path) / "logs" / "preprocessing.txt"))
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='copying_files' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "copying_files":
            set_stream_location(str(Path(data_path) / "logs" / "preprocessing.txt"))
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
                (run_id,),
            ).fetchall():
                full_audio_path = (
                    Path(dataset_path)
                    / str(dataset_id)
                    / "speakers"
                    / str(speaker_id)
                    / audio_path
                )
                txt_paths.append(txt_path)
                texts.append(text)
                audio_paths.append(str(full_audio_path))
                names.append(speaker_name)
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)

            def progress_cb(progress: float):
                logger = get_logger()
                logger.query(
                    "UPDATE training_run SET preprocessing_copying_files_progress=? WHERE id=?",
                    (progress, run_id),
                )

            copy_files(
                data_path=data_path,
                txt_paths=txt_paths,
                texts=texts,
                audio_paths=audio_paths,
                names=names,
                workers=p_config.workers,
                progress_cb=progress_cb,
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='gen_vocab' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "gen_vocab":
            set_stream_location(str(Path(data_path) / "logs" / "preprocessing.txt"))
            (Path(data_path) / "data").mkdir(exist_ok=True, parents=True)
            row = cur.execute(
                "SELECT device FROM training_run WHERE ID=?",
                (run_id,),
            ).fetchone()
            texts = []
            for (text,) in cur.execute(
                """
                    SELECT sample.text AS text FROM training_run INNER JOIN dataset ON training_run.dataset_id = dataset.ID 
                    INNER JOIN speaker on speaker.dataset_id = dataset.ID
                    INNER JOIN sample on sample.speaker_id = speaker.ID
                    WHERE training_run.ID=?
                    """,
                (run_id,),
            ).fetchall():
                texts.append(text)

            device = row[0]
            device = get_device(device)

            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)
            base_lexica_path = str(Path(data_path) / "data" / "lexicon_pre.txt")
            predicted_phones = generate_vocab(
                texts=texts, lang="en", assets_path=assets_path, device=device
            )
            punct_set = get_punct(lang="en")
            with open(base_lexica_path, "w", encoding="utf-8") as f:
                for word, phones in predicted_phones.items():
                    word = word.lower().strip()
                    phones = " ".join(phones).strip()
                    if len(word) == 0 or len(phones) == 0:
                        continue
                    if word in punct_set:
                        continue
                    f.write(f"{word.lower()} {phones}\n")

            merge_lexica(
                base_lexica_path=base_lexica_path,
                lang="en",
                assets_path=assets_path,
                out_path=str(Path(data_path) / "data" / "lexicon_post.txt"),
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='gen_alignments', preprocessing_gen_vocab_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "gen_alignments":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)
            align(
                environment_name=environment_name,
                in_path=str(Path(data_path) / "raw_data"),
                lexicon_path=str(Path(data_path / "data" / "lexicon_post.txt")),
                out_path=(Path(data_path) / "data" / "textgrid"),
                n_workers=p_config.workers,
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='extract_data', preprocessing_gen_align_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif preprocessing_stage == "extract_data":
            set_stream_location(str(Path(data_path) / "logs" / "preprocessing.txt"))
            row = cur.execute(
                "SELECT validation_size FROM training_run WHERE ID=?",
                (run_id,),
            ).fetchone()
            p_config, _, _ = get_acoustic_configs(cur=cur, run_id=run_id)
            extract_data(
                db_id=run_id,
                table_name="training_run",
                training_run_name=str(run_id),
                preprocess_config=p_config,
                get_logger=get_logger,
                assets_path=assets_path,
                training_runs_path=training_runs_path,
            )
            cur.execute(
                "UPDATE training_run SET stage='acoustic_fine_tuning', preprocessing_stage='finished' WHERE ID=?",
                (run_id,),
            )
            con.commit()
            if (Path(data_path) / "raw_data").exists():
                shutil.rmtree(Path(data_path) / "raw_data")
    return False


def before_run(data_path: str, **kwargs):
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)


def acoustic_fine_tuning_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    training_runs_path: str,
    **kwargs,
) -> bool:
    row = cur.execute(
        "SELECT device FROM training_run WHERE ID=?",
        (run_id,),
    ).fetchone()

    device = row[0]
    device = get_device(device)

    p_config, m_config, t_config = get_acoustic_configs(cur=cur, run_id=run_id)

    logger = SQLLogger(
        training_run_id=run_id,
        con=con,
        cursor=cur,
        out_dir=data_path,
        stage="acoustic",
    )

    target_batch_size_total = t_config.batch_size * t_config.grad_acc_step

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
                checkpoint_acoustic = str(Path(assets_path) / "acoustic_pretrained.pt")
                checkpoint_style = str(Path(assets_path) / "style_pretrained.pt")
            else:
                reset = False

            cur.execute(
                "DELETE FROM image_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                (run_id, step),
            )
            cur.execute(
                "DELETE FROM graph_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                (run_id, step),
            )
            cur.execute(
                "DELETE FROM audio_statistic WHERE training_run_id=? AND step>=? AND stage='acoustic'",
                (run_id, step),
            )
            con.commit()

            train_acoustic(
                db_id=run_id,
                training_run_name=str(run_id),
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
                training_runs_path=training_runs_path,
            )
            break
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                if t_config.batch_size > 1:
                    old_batch_size, old_grad_accum_steps = (
                        t_config.batch_size,
                        t_config.grad_acc_step,
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
                    t_config.batch_size = batch_size
                    t_config.grad_acc_step = grad_acc_step
                else:
                    raise Exception(
                        f"""
                            Ran out of VRAM during acoustic model training, batch size is {t_config.batch_size}, so cannot set it lower. 
                            Please restart your PC and try again. If this error continues you may not
                            have enough VRAM to run this software. You could try training on CPU
                            instead of on GPU.
                            """
                    )
            else:
                raise e

    cur.execute(
        "UPDATE training_run SET stage='ground_truth_alignment' WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def ground_truth_alignment_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    training_runs_path: str,
    **kwargs,
) -> bool:
    logger = SQLLogger(
        training_run_id=run_id,
        con=con,
        cursor=cur,
        out_dir=data_path,
        stage="ground_truth_alignment",
    )
    row = cur.execute(
        "SELECT acoustic_batch_size, device FROM training_run WHERE ID=?",
        (run_id,),
    ).fetchone()
    batch_size, device = row
    device = get_device(device)
    checkpoint_acoustic, step = get_latest_checkpoint(
        name="acoustic", ckpt_dir=str(Path(data_path) / "ckpt" / "acoustic")
    )
    checkpoint_style, step = get_latest_checkpoint(
        name="style", ckpt_dir=str(Path(data_path) / "ckpt" / "acoustic")
    )
    while True:
        try:
            ground_truth_alignment(
                db_id=run_id,
                table_name="training_run",
                training_run_name=str(run_id),
                batch_size=3 * batch_size,
                group_size=3,
                device=device,
                checkpoint_acoustic=str(checkpoint_acoustic),
                checkpoint_style=str(checkpoint_style),
                logger=logger,
                assets_path=assets_path,
                training_runs_path=training_runs_path,
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
        (run_id,),
    )
    con.commit()
    return False


def vocoder_fine_tuning_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    training_runs_path: str,
    **kwargs,
) -> bool:
    row = cur.execute(
        "SELECT device FROM training_run WHERE ID=?",
        (run_id,),
    ).fetchone()
    device = row[0]
    device = get_device(device)

    p_config, m_config, t_config = get_vocoder_configs(cur=cur, run_id=run_id)

    logger = SQLLogger(
        training_run_id=run_id,
        con=con,
        cursor=cur,
        out_dir=data_path,
        stage="vocoder",
    )

    target_batch_size_total = t_config.batch_size * t_config.grad_accum_steps

    while True:
        try:
            checkpoint_path, step = get_latest_checkpoint(
                name="vocoder", ckpt_dir=str(data_path / "ckpt" / "vocoder")
            )

            cur.execute(
                "DELETE FROM image_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                (run_id, step),
            )
            cur.execute(
                "DELETE FROM graph_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                (run_id, step),
            )
            cur.execute(
                "DELETE FROM audio_statistic WHERE training_run_id=? AND step>=? AND stage='vocoder'",
                (run_id, step),
            )
            con.commit()

            if checkpoint_path is None:
                reset = True
                checkpoint_path = str(Path(assets_path) / "vocoder_pretrained.pt")
            else:
                reset = False

            train_vocoder(
                db_id=run_id,
                training_run_name=str(run_id),
                train_config=t_config,
                model_config=VocoderModelConfig(),
                preprocess_config=p_config,
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
                if t_config.batch_size > 1:
                    old_batch_size, old_grad_accum_steps = (
                        t_config.batch_size,
                        t_config.grad_accum_steps,
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
                    t_config.batch_size = batch_size
                    t_config.grad_accum_steps = grad_acc_step
                else:
                    raise Exception(
                        f"""
                            Ran out of VRAM during vocoder model training, batch size is {t_config.batch_size}, so cannot set it lower. 
                            Please restart your PC and try again. If this error continues you may not
                            have enough VRAM to run this software. You could try training on CPU
                            instead of on GPU.
                            """
                    )
            else:
                raise e

    cur.execute(
        "UPDATE training_run SET stage='save_model' WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def save_model_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    models_path: str,
    **kwargs,
) -> bool:
    data_path = Path(data_path)
    models_path = Path(models_path)
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
        raise ValueError("Style path is None in save_model, no model has been saved?")

    if checkpoint_vocoder is None:
        raise ValueError("Vocoder path is None in save_model, no model has been saved?")

    p_config, m_config_acoustic, t_config_acoustic = get_acoustic_configs(
        cur=cur, run_id=run_id
    )

    p_config, m_config_vocoder, t_config_vocoder = get_vocoder_configs(
        cur=cur, run_id=run_id
    )

    acoustic_model, style_predictor = acoustic_to_torchscript(
        checkpoint_acoustic=str(checkpoint_acoustic),
        preprocess_config=p_config,
        model_config=m_config_acoustic,
        train_config=t_config_acoustic,
        assets_path=assets_path,
        checkpoint_style=str(checkpoint_style),
        data_path=str(data_path / "data"),
    )
    vocoder = vocoder_to_torchscript(
        ckpt_path=str(checkpoint_vocoder),
        data_path=str(data_path / "data"),
        preprocess_config=p_config,
        model_config=m_config_vocoder,
        train_config=t_config_vocoder,
    )

    with open(Path(data_path) / "data" / "speakers.json", "r", encoding="utf-8") as f:
        speakers = json.load(f)

    # TODO place in transaction

    model_type = "Delighful_FreGANv1_v0.0"
    config = {
        "acousticSteps": acoustic_steps,
        "vocoderSteps": vocoder_steps,
        "preprocessing": dataclasses.asdict(p_config),
        "acoustic_model": dataclasses.asdict(m_config_acoustic),
        "vocoder_model": dataclasses.asdict(m_config_vocoder),
        "fine_tuning_acoustic": dataclasses.asdict(t_config_acoustic),
        "fine_tuning_vocoder": dataclasses.asdict(t_config_vocoder),
    }
    description = """
            TODO
        """

    row = cur.execute(
        "SELECT name FROM training_run WHERE ID=?",
        (run_id,),
    ).fetchone()
    con.commit()

    name = get_available_name(model_dir=str(models_path), name=row[0])
    (models_path / name).mkdir(exist_ok=True, parents=True)
    models_dir = models_path / name / "torchscript"
    models_dir.mkdir(exist_ok=True)
    acoustic_model.save(str(models_dir / "acoustic_model.pt"))
    style_predictor.save(models_dir / "style_predictor.pt")
    vocoder.save(models_dir / "vocoder.pt")
    with open(models_path / name / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    lexicon = {}
    with open(
        data_path / "data" / "lexicon_post.txt",
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
        (run_id,),
    )
    con.commit()

    return True


def before_stage(
    data_path: str,
    stage_name: str,
    **kwargs,
):
    # set_stream_location(str(Path(data_path) / "logs" / "{stage_name}.txt"))
    pass


def get_stage_name(cur: sqlite3.Cursor, run_id: int, **kwargs):
    row = cur.execute(
        "SELECT stage FROM training_run WHERE ID=?",
        (run_id,),
    ).fetchone()
    return row[0]


def continue_training_run(
    run_id: int,
    training_runs_path: str,
    assets_path: str,
    db_path: str,
    models_path: str,
    datasets_path: str,
    user_data_path: str,
    environment_name: str,
):
    con = get_con(db_path)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = Path(training_runs_path) / str(run_id)

    def get_logger():
        con = get_con(db_path)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=run_id,
            con=con,
            cursor=cur,
            out_dir=data_path,
            stage="preprocessing",
        )
        return logger

    runner = StageRunner(
        cur=cur,
        con=con,
        before_run=before_run,
        before_stage=before_stage,
        get_stage_name=get_stage_name,
        stages=[
            ("not_started", not_started_stage),
            ("preprocessing", preprocessing_stage),
            ("acoustic_fine_tuning", acoustic_fine_tuning_stage),
            ("ground_truth_alignment", ground_truth_alignment_stage),
            ("vocoder_fine_tuning", vocoder_fine_tuning_stage),
            ("save_model", save_model_stage),
        ],
    )
    runner.run(
        cur=cur,
        con=con,
        run_id=run_id,
        data_path=data_path,
        dataset_path=datasets_path,
        get_logger=get_logger,
        assets_path=assets_path,
        environment_name=environment_name,
        training_runs_path=training_runs_path,
        models_path=models_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--training_runs_path", type=str, required=True)
    parser.add_argument("--assets_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--user_data_path", type=str, required=True)
    parser.add_argument("--environment_name", type=str, required=True)
    args = parser.parse_args()

    continue_training_run(
        run_id=args.run_id,
        training_runs_path=args.training_runs_path,
        assets_path=args.assets_path,
        db_path=args.db_path,
        models_path=args.models_path,
        datasets_path=args.datasets_path,
        user_data_path=args.user_data_path,
        environment_name=args.environment_name,
    )
