from pathlib import Path
import shutil
from typing import Literal
import sys
import torch
import json
import sqlite3
import argparse
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.config.configs import SampleSplittingRunConfig
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.utils.tools import warnings_to_stdout, get_device, get_workers
from voice_smith.preprocessing.generate_vocab import generate_vocab
from voice_smith.preprocessing.merge_lexika import merge_lexica
from voice_smith.preprocessing.align import align
from voice_smith.preprocessing.sample_splitting import sample_splitting, split_sample
from voice_smith.utils.punctuation import get_punct
from joblib import Parallel, delayed

warnings_to_stdout()


def get_config(cur: sqlite3.Cursor, run_id: int) -> SampleSplittingRunConfig:
    row = cur.execute(
        """
        SELECT device, maximum_workers FROM sample_splitting_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (device, maximum_workers) = row
    device = get_device(device)
    workers = get_workers(maximum_workers)
    return SampleSplittingRunConfig(workers=workers, device=device)


def continue_sample_splitting_run(
    run_id: int,
    preprocessing_runs_dir: str,
    assets_path: str,
    db_path: str,
    datasets_path: str,
    environment_name: str,
):
    con = get_con(db_path)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = Path(preprocessing_runs_dir) / str(run_id)
    splits_path = data_path / "splits"
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
        stage = row[0]
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
            # set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
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
                WHERE sample_splitting_run.ID=?
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

            config = get_config(cur=cur, run_id=run_id)

            def progress_cb(progress: float):
                logger = get_logger()
                logger.query(
                    "UPDATE sample_splitting_run SET copying_files_progress=? WHERE id=?",
                    (progress, run_id),
                )

            copy_files(
                data_path=str(data_path),
                txt_paths=txt_paths,
                texts=texts,
                audio_paths=audio_paths,
                names=names,
                workers=config.workers,
                progress_cb=progress_cb,
            )
            cur.execute(
                "UPDATE sample_splitting_run SET stage='gen_vocab' WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "gen_vocab":
            # set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            (data_path / "data").mkdir(exist_ok=True, parents=True)
            texts = []
            for (text,) in cur.execute(
                """
                SELECT sample.text AS text FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
                INNER JOIN speaker on speaker.dataset_id = dataset.ID
                INNER JOIN sample on sample.speaker_id = speaker.ID
                WHERE sample_splitting_run.ID=?
                """,
                (run_id,),
            ).fetchall():
                texts.append(text)

            row = cur.execute(
                "SELECT device FROM sample_splitting_run WHERE ID=?",
                (run_id,),
            ).fetchone()
            device = row[0]
            device = get_device(device)

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
                "UPDATE sample_splitting_run SET stage='gen_alignments', gen_vocab_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "gen_alignments":
            # set_stream_location(str(data_path / "logs" / "preprocessing.txt"))
            p_config = get_config(cur, run_id)
            align(
                environment_name=environment_name,
                in_path=str(Path(data_path) / "raw_data"),
                lexicon_path=str(Path(data_path / "data" / "lexicon_post.txt")),
                out_path=(Path(data_path) / "data" / "textgrid"),
                n_workers=p_config.workers,
            )
            cur.execute(
                "UPDATE sample_splitting_run SET stage='creating_splits', gen_align_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "creating_splits":
            sample_ids, texts, textgrid_paths, langs = [], [], [], []
            if splits_path.exists():
                shutil.rmtree(splits_path)
            (data_path / "splits").mkdir(parents=True)
            p_config = get_config(cur, run_id)

            # TODO delete other table
            cur.execute(
                """
                DELETE FROM sample_splitting_run_sample WHERE sample_splitting_run_id=?
                """,
                (run_id,),
            )

            con.commit()
            for (sample_id, text, audio_path, speaker_name,) in cur.execute(
                """
                SELECT sample.ID, sample.text, sample.audio_path, speaker.name AS speaker_name FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
                INNER JOIN speaker on speaker.dataset_id = dataset.ID
                INNER JOIN sample on sample.speaker_id = speaker.ID
                WHERE sample_splitting_run.ID=?
                """,
                (run_id,),
            ).fetchall():
                sample_ids.append(sample_id)
                texts.append(text)
                textgrid_paths.append(
                    (Path(data_path) / "data" / "textgrid")
                    / speaker_name
                    / f"{Path(audio_path).stem}.TextGrid"
                )
                langs.append("en")

            splits = sample_splitting(
                ids=sample_ids,
                texts=texts,
                textgrid_paths=textgrid_paths,
                languages=langs,
            )
            run_sample_id_to_split = {}
            for split in splits:
                cur.execute(
                    """
                    INSERT INTO sample_splitting_run_sample (text, sample_splitting_run_id, sample_id)
                    VALUES (?, ?, ?)
                    """,
                    (split.text, run_id, split.sample_id),
                )
                run_sample_id_to_split[cur.lastrowid] = split

            con.commit()

            run_sample_infos = []
            for (
                sample_splitting_run_id,
                txt_path,
                text,
                audio_path,
                dataset_id,
                speaker_id,
            ) in cur.execute(
                """
                SELECT sample_splitting_run_sample.ID, sample.txt_path, sample.text, sample.audio_path, dataset.ID AS dataset_id, speaker.ID as speaker_id FROM sample_splitting_run_sample 
                INNER JOIN sample on sample_splitting_run_sample.sample_id = sample.ID
                INNER JOIN speaker on sample.speaker_id = speaker.ID
                INNER JOIN dataset ON speaker.dataset_id = dataset.ID 
                WHERE sample_splitting_run_sample.sample_splitting_run_id=?
                """,
                (run_id,),
            ).fetchall():
                audio_path = (
                    dataset_path
                    / str(dataset_id)
                    / "speakers"
                    / str(speaker_id)
                    / audio_path
                )
                run_sample_infos.append((sample_splitting_run_id, audio_path))

            Parallel(n_jobs=p_config.workers)(
                delayed(split_sample)(
                    run_sample_id_to_split[sample_splitting_run_id],
                    audio_path,
                    str(splits_path),
                    sample_splitting_run_id,
                )
                for (sample_splitting_run_id, audio_path) in run_sample_infos
            )

            for sample_splitting_run_id, audio_path in run_sample_infos:
                for split_idx, split in enumerate(
                    run_sample_id_to_split[sample_splitting_run_id].splits
                ):
                    cur.execute(
                        """
                        INSERT INTO sample_splitting_run_split (text, split_idx, sample_splitting_run_sample_id)
                        VALUES (?, ?, ?)
                        """,
                        (split.text, split_idx, sample_splitting_run_id),
                    )
            cur.execute(
                "UPDATE sample_splitting_run SET stage='choose_samples', creating_splits_progress=1.0 WHERE ID=?",
                (run_id,),
            )
            con.commit()

        elif stage == "choose_samples":
            break

        elif stage == "finished":
            break

        else:
            raise Exception(f"Stage '{stage}' is not a valid stage ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--preprocessing_runs_dir", type=str, required=True)
    parser.add_argument("--assets_path", type=str, required=True)
    parser.add_argument("--db_path", type=str, required=True)
    parser.add_argument("--datasets_path", type=str, required=True)
    parser.add_argument("--environment_name", type=str, required=True)
    args = parser.parse_args()

    continue_sample_splitting_run(
        run_id=args.run_id,
        preprocessing_runs_dir=args.preprocessing_runs_dir,
        assets_path=args.assets_path,
        db_path=args.db_path,
        datasets_path=args.datasets_path,
        environment_name=args.environment_name,
    )
