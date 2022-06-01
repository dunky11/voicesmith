from pathlib import Path
import shutil
from typing import Literal
import sys
import torch
import json
import sqlite3
from typing import Union, Literal, Tuple, Dict, Any
import argparse
import multiprocessing as mp
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
from voice_smith.sql import get_con, save_current_pid
from voice_smith.config.symbols import symbol2id
from voice_smith.utils.tools import warnings_to_stdout
from voice_smith.preprocessing.generate_vocab import generate_vocab
from voice_smith.preprocessing.merge_lexika import merge_lexica
from voice_smith.preprocessing.align import align
from voice_smith.utils.punctuation import get_punct

warnings_to_stdout()


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


def continue_sample_splitting_run(
    run_id: int,
    preprocessing_runs_dir: str,
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
    data_path = Path(preprocessing_runs_dir) / "sample_splitting_runs" / str(run_id)
    model_path = Path(models_path)
    dataset_path = Path(datasets_path)
    stage = None

    def get_logger():
        con = get_con(db_path)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="sample_splitting_run",
        )
        return logger

    while stage != "finished":
        row = cur.execute(
            "SELECT stage FROM sample_splitting_run WHERE ID=?",
            (run_id,),
        ).fetchone()
        stage = row
        if stage == "not_started":
            if data_path.exists():
                shutil.rmtree(data_path)
            (data_path / "logs").mkdir(exist_ok=True, parents=True)
            (data_path / "raw_data").mkdir(exist_ok=True)
            cur.execute(
                "UPDATE sample_splitting_run SET stage='copying_files' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "copying_files":
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
                SELECT sample.txt_path, sample.text, sample.audio_path, speaker.name AS speaker_name, dataset.ID AS dataset_id, speaker.ID as speaker_id FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
                INNER JOIN speaker on speaker.dataset_id = dataset.ID
                INNER JOIN sample on sample.speaker_id = speaker.ID
                WHERE training_run.ID=?
                """,
                (run_id,),
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
            p_config, _, _ = get_acoustic_configs(
                cur=cur, training_run_id=run_id
            )
            copy_files(
                db_id=run_id,
                table_name="training_run",
                data_path=str(data_path),
                txt_paths=txt_paths,
                texts=texts,
                audio_paths=audio_paths,
                names=names,
                get_logger=get_logger,
                workers=p_config["workers"],
            )
            cur.execute(
                "UPDATE sample_splitting_run SET stage='gen_vocab' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "gen_vocab":
            (data_path / "data").mkdir(exist_ok=True, parents=True)
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            row = cur.execute(
                "SELECT device FROM sample_splitting_run WHERE ID=?",
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

            p_config, _, _ = get_acoustic_configs(
                cur=cur, training_run_id=training_run_id
            )
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
                out_path=str(Path(data_path / "data" / "lexicon_post.txt")),
            )
            cur.execute(
                "UPDATE sample_splitting_run SET stage='gen_alignments', preprocessing_gen_vocab_progress=1.0 WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "gen_alignments":
            set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            p_config, _, _ = get_acoustic_configs(
                cur=cur, training_run_id=training_run_id
            )
            align(
                environment_name=environment_name,
                in_path=str(Path(data_path) / "raw_data"),
                lexicon_path=str(Path(data_path / "data" / "lexicon_post.txt")),
                out_path=(Path(data_path) / "data" / "textgrid"),
                n_workers=p_config["workers"],
            )
            cur.execute(
                "UPDATE training_run SET preprocessing_stage='extract_data', preprocessing_gen_align_progress=1.0 WHERE ID=?",
                (training_run_id,),
            )
            con.commit()

        elif stage == "extract_data":
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
                training_runs_path=training_runs_path,
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
    parser.add_argument("--environment_name", type=str, required=True)
    args = parser.parse_args()

    continue_sample_splitting_run(
        training_run_id=args.training_run_id,
        training_runs_path=args.training_runs_path,
        assets_path=args.assets_path,
        db_path=args.db_path,
        models_path=args.models_path,
        datasets_path=args.datasets_path,
        user_data_path=args.user_data_path,
        environment_name=args.environment_name,
    )
