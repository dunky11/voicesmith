from voice_smith.utils.tools import warnings_to_stdout

warnings_to_stdout()
from pathlib import Path
import sqlite3
import shutil
import argparse
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.preprocessing.text_normalization import text_normalize
from voice_smith.utils.runs import StageRunner
from voice_smith.config.globals import (
    DB_PATH,
    TEXT_NORMALIZATION_RUNS_PATH,
    ASSETS_PATH,
)


def before_run(data_path: str, **kwargs):
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)


def before_stage(
    log_console: bool, data_path: str, **kwargs,
):
    set_stream_location(
        str(Path(data_path) / "logs" / "preprocessing.txt"), log_console=log_console
    )


def get_stage_name(cur: sqlite3.Cursor, run_id: int, **kwargs):
    row = cur.execute(
        "SELECT stage FROM text_normalization_run WHERE ID=?", (run_id,),
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
    cur.execute(
        "UPDATE text_normalization_run SET stage='text_normalization' WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def text_normalization_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    assets_path: str,
    **kwargs,
) -> bool:
    cur.execute(
        "DELETE FROM text_normalization_sample WHERE text_normalization_run_id = ?",
        (run_id,),
    )
    con.commit()
    sample_ids = []
    texts = []
    langs = []
    for (sample_id, text, lang) in cur.execute(
        """
        SELECT sample.ID, sample.text, speaker.language FROM sample
        INNER JOIN speaker ON sample.speaker_id = speaker.ID
        INNER JOIN dataset on speaker.dataset_id = dataset.ID
        INNER JOIN text_normalization_run ON text_normalization_run.dataset_id = dataset.ID
        WHERE text_normalization_run.ID = ?
        """,
        (run_id,),
    ).fetchall():
        sample_ids.append(sample_id)
        texts.append(text)
        langs.append(lang)

    def callback(progress: float):
        progress = progress * 0.9
        cur.execute(
            "UPDATE text_normalization_run SET text_normalization_progress=? WHERE ID=?",
            (progress, run_id),
        )
        con.commit()

    normalizations = text_normalize(
        sample_ids=sample_ids,
        texts=texts,
        langs=langs,
        assets_path=assets_path,
        progress_cb=callback,
    )

    for (sample_id, text_in, text_out, reason) in normalizations:
        cur.execute(
            """
            INSERT INTO text_normalization_sample (old_text, new_text, reason, sample_id, text_normalization_run_id) 
            VALUES (?, ?, ?, ?, ?)
            """,
            (text_in, text_out, reason, sample_id, run_id),
        )

    cur.execute(
        "UPDATE text_normalization_run SET stage='choose_samples', text_normalization_progress=1.0 WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return True


def continue_text_normalization_run(run_id: int, log_console: bool):
    con = get_con(DB_PATH)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = str(Path(TEXT_NORMALIZATION_RUNS_PATH) / str(run_id))

    runner = StageRunner(
        cur=cur,
        con=con,
        before_run=before_run,
        before_stage=before_stage,
        get_stage_name=get_stage_name,
        stages=[
            ("not_started", not_started_stage),
            ("text_normalization", text_normalization_stage),
        ],
    )
    runner.run(
        cur=cur,
        con=con,
        run_id=run_id,
        data_path=data_path,
        assets_path=ASSETS_PATH,
        log_console=log_console,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--log_console", action="store_true")
    args = parser.parse_args()
    continue_text_normalization_run(run_id=args.run_id, log_console=args.log_console)
