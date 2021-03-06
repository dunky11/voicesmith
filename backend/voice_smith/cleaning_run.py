from pathlib import Path
import shutil
import argparse
from typing import Callable
import sqlite3
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.runs import StageRunner
from voice_smith.utils.tools import get_device, get_workers
from voice_smith.preprocessing.gen_speaker_embeddings import (
    gen_file_emeddings,
    gen_speaker_embeddings,
    get_quality_scores,
)
from voice_smith.config.globals import (
    DB_PATH,
    CLEANING_RUNS_PATH,
    DATASETS_PATH,
    ASSETS_PATH,
)
from voice_smith.config.configs import CleaningRunConfig


def get_config(cur: sqlite3.Cursor, run_id: int) -> CleaningRunConfig:
    row = cur.execute(
        """
        SELECT device, maximum_workers, skip_on_error FROM cleaning_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (device, maximum_workers, skip_on_error) = row
    device = get_device(device)
    workers = get_workers(maximum_workers)
    skip_on_error = bool(skip_on_error)
    return CleaningRunConfig(
        workers=workers, device=device, skip_on_error=skip_on_error
    )


def before_run(data_path: str, **kwargs):
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)


def get_log_file_name(stage_name: str) -> str:
    if stage_name in [
        "not_started",
        "copying_files",
        "transcribe",
        "choose_samples",
    ]:
        return "preprocessing.txt"
    elif stage_name == "apply_changes":
        return "apply_changes.txt"
    else:
        raise Exception(
            f"No branch selected in switch-statement, {stage_name} is not a valid case ..."
        )


def before_stage(
    data_path: str, stage_name: str, log_console: bool, **kwargs,
):
    set_stream_location(
        str(Path(data_path) / "logs" / get_log_file_name(stage_name=stage_name)),
        log_console=log_console,
    )


def get_stage_name(cur: sqlite3.Cursor, run_id: int, **kwargs):
    row = cur.execute(
        "SELECT stage FROM cleaning_run WHERE ID=?", (run_id,),
    ).fetchone()
    stage = row[0]
    return stage


def not_started_stage(
    cur: sqlite3.Cursor, con: sqlite3.Connection, run_id: int, data_path: str, **kwargs
) -> bool:
    data_path = Path(data_path)
    if data_path.exists():
        shutil.rmtree(data_path)
    (data_path / "logs").mkdir(exist_ok=True, parents=True)
    (data_path / "raw_data").mkdir(exist_ok=True)
    cur.execute(
        "UPDATE cleaning_run SET stage='copying_files' WHERE ID=?", (run_id,),
    )
    con.commit()
    return False


def copying_files_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    datasets_path: str,
    get_logger: Callable[[], SQLLogger],
    **kwargs,
) -> bool:
    sample_ids, texts, audio_paths, names, langs = [], [], [], [], []
    for (
        sample_id,
        text,
        audio_path,
        speaker_name,
        dataset_id,
        speaker_id,
        lang,
    ) in cur.execute(
        """
        SELECT sample.ID, sample.text, sample.audio_path, 
        speaker.name, dataset.ID, speaker.ID, speaker.language
        FROM cleaning_run INNER JOIN dataset ON cleaning_run.dataset_id = dataset.ID 
        INNER JOIN speaker on speaker.dataset_id = dataset.ID
        INNER JOIN sample on sample.speaker_id = speaker.ID
        WHERE cleaning_run.ID=?
        """,
        (run_id,),
    ).fetchall():
        full_audio_path = (
            Path(datasets_path)
            / str(dataset_id)
            / "speakers"
            / str(speaker_id)
            / audio_path
        )
        sample_ids.append(sample_id)
        texts.append(text)
        audio_paths.append(str(full_audio_path))
        names.append(speaker_name)
        langs.append(lang)

    config = get_config(cur=cur, run_id=run_id)

    def progress_cb(progress: float):
        logger = get_logger()
        logger.query(
            "UPDATE cleaning_run SET copying_files_progress=? WHERE id=?",
            (progress, run_id),
        )

    copy_files(
        data_path=data_path,
        sample_ids=sample_ids,
        texts=texts,
        audio_paths=audio_paths,
        names=names,
        workers=config.workers,
        progress_cb=progress_cb,
        langs=langs,
        skip_on_error=config.skip_on_error,
    )
    cur.execute(
        "UPDATE cleaning_run SET stage='transcribe' WHERE ID=?", (run_id,),
    )
    con.commit()
    return False


def transcribe_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    get_logger: Callable[[], SQLLogger],
    **kwargs,
) -> bool:
    config = get_config(cur=cur, run_id=run_id)
    langs = [el.name for el in (Path(data_path) / "raw_data").iterdir()]
    for i, lang in enumerate(langs):
        audio_paths = []
        for (sample_id, speaker_name) in cur.execute(
            """
            SELECT sample.ID, speaker.name
            FROM cleaning_run INNER JOIN dataset ON cleaning_run.dataset_id = dataset.ID 
            INNER JOIN speaker on speaker.dataset_id = dataset.ID
            INNER JOIN sample on sample.speaker_id = speaker.ID
            WHERE cleaning_run.ID=? AND speaker.language=?
            """,
            (run_id, lang),
        ).fetchall():
            audio_paths.append(
                str(
                    Path(data_path)
                    / "raw_data"
                    / lang
                    / speaker_name
                    / f"{sample_id}.flac"
                )
            )

        def gen_file_emb_progress_cb(progress: float):
            logger = get_logger()
            logger.query(
                "UPDATE cleaning_run SET transcription_progress=? WHERE id=?",
                (0.75 * i * (1 / len(langs)) + progress * (1 / len(langs)), run_id),
            )
            con.commit()

        gen_file_emeddings(
            in_paths=audio_paths,
            out_dir=str(Path(data_path) / "file_embeddings" / lang),
            callback=gen_file_emb_progress_cb,
            device=config.device,
            workers=config.workers,
        )

    for i, lang in enumerate(langs):

        def gen_speaker_emb_progress_cb(progress: float):
            logger = get_logger()
            logger.query(
                "UPDATE cleaning_run SET transcription_progress=? WHERE id=?",
                (
                    0.75 + 0.25 * i * (1 / len(langs)) + progress * (1 / len(langs)),
                    run_id,
                ),
            )
            con.commit()

        speaker_paths = [
            str(el) for el in (Path(data_path) / "file_embeddings" / lang).iterdir()
        ]

        gen_speaker_embeddings(
            speaker_paths=speaker_paths,
            out_dir=str(Path(data_path) / "speaker_embeddings" / lang),
            callback=gen_speaker_emb_progress_cb,
            device=config.device,
            workers=config.workers,
        )

        sample_ids, scores = get_quality_scores(
            file_embeddings_dir=str(Path(data_path) / "file_embeddings" / lang),
            speaker_embeddings_dir=str(Path(data_path) / "speaker_embeddings" / lang),
            workers=config.workers,
            device=config.device,
        )

        for sample_id, score in zip(sample_ids, scores):
            cur.execute(
                """
                INSERT INTO cleaning_run_sample (quality_score, sample_id, transcription, cleaning_run_id)
                VALUES (?, ?, ?, ?)
                """,
                (score, sample_id, "", run_id),
            )

    cur.execute(
        "UPDATE cleaning_run SET stage='choose_samples', transcription_progress=1.0 WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return True


def apply_changes_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    get_logger: Callable[[], SQLLogger],
    **kwargs,
) -> bool:
    return True


def continue_cleaning_run(run_id: int, log_console: bool):
    con = get_con(DB_PATH)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = str(Path(CLEANING_RUNS_PATH) / str(run_id))

    def get_logger():
        con = get_con(DB_PATH)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="cleaning_run",
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
            ("copying_files", copying_files_stage),
            ("transcribe", transcribe_stage),
            ("appy_changes", apply_changes_stage),
        ],
    )
    runner.run(
        cur=cur,
        con=con,
        run_id=run_id,
        data_path=data_path,
        assets_path=ASSETS_PATH,
        get_logger=get_logger,
        datasets_path=DATASETS_PATH,
        log_console=log_console,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--log_console", action="store_true")
    args = parser.parse_args()
    continue_cleaning_run(run_id=args.run_id, log_console=args.log_console)

