import sqlite3
from typing import List, Tuple
from pathlib import Path
import shutil
from joblib import delayed, Parallel
from voice_smith.utils.shell import run_conda_in_shell
from voice_smith.utils.mfa import lang_to_mfa_acoustic
from voice_smith.utils.tools import iter_logger


def get_batch(
    cur: sqlite3.Cursor,
    table_name: str,
    lang: str,
    foreign_key_name: str,
    run_id: int,
    batch_size: int,
    data_path: str,
) -> Tuple[List[int], List[str]]:
    sample_ids, base_paths = [], []

    for (sample_id, speaker_name, audio_path, lang) in cur.execute(
        f"""
        SELECT sample.ID, speaker.name,
        sample.audio_path, speaker.language
        FROM {table_name} INNER JOIN dataset ON {table_name}.dataset_id = dataset.ID 
        INNER JOIN speaker on speaker.dataset_id = dataset.ID
        INNER JOIN sample on sample.speaker_id = speaker.ID
        LEFT JOIN sample_to_align ON sample_to_align.sample_id = sample.ID
            AND sample_to_align.{foreign_key_name} = ?
        WHERE {table_name}.ID=?
        AND speaker.language = ?
        AND sample_to_align.ID IS NULL
        LIMIT {batch_size}
        """,
        (run_id, run_id, lang),
    ).fetchall():
        sample_ids.append(sample_id)
        base_paths.append(
            Path(data_path) / "raw_data" / lang / speaker_name / Path(audio_path).stem
        )
    return sample_ids, base_paths


def copy_sample(base_path: str, out_dir: str) -> None:
    base_path = Path(base_path)
    speaker_name = base_path.parent.name
    audio_in_path = f"{str(base_path)}.flac"
    txt_in_path = f"{str(base_path)}.txt"
    out_dir_file = Path(out_dir) / speaker_name
    out_dir_file.mkdir(exist_ok=True, parents=True)
    audio_out_path = out_dir_file / f"{base_path.name}.flac"
    txt_out_path = out_dir_file / f"{base_path.name}.txt"
    shutil.copy2(audio_in_path, audio_out_path)
    shutil.copy2(txt_in_path, txt_out_path)


def copy_batch(base_paths: List[str], out_dir: str, n_workers: int):
    Parallel(n_jobs=n_workers)(
        delayed(copy_sample)(base_path, out_dir)
        for base_path in iter_logger(base_paths)
    )


def finish_batch(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    sample_ids: List[int],
    foreign_key_name: str,
    run_id: int,
):
    for sample_id in sample_ids:
        cur.execute(
            f"""
            INSERT OR IGNORE INTO sample_to_align(sample_id, {foreign_key_name})
            VALUES (?, ?)
            """,
            (sample_id, run_id),
        )
    con.commit()


def align(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    table_name: str,
    foreign_key_name: str,
    run_id: int,
    environment_name: str,
    data_path: str,
    lexicon_path: str,
    out_path: str,
    n_workers: int,
    lang: str,
    batch_size: int,
):
    tmp_dir = Path(data_path) / "tmp"
    while True:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        sample_ids, base_paths = get_batch(
            cur, table_name, lang, foreign_key_name, run_id, batch_size, data_path
        )
        if len(sample_ids) == 0:
            break
        copy_batch(base_paths=base_paths, out_dir=tmp_dir, n_workers=n_workers)
        cmd = f"mfa align --clean -j {n_workers} {tmp_dir} {lexicon_path} {lang_to_mfa_acoustic(lang)} {out_path}"
        success = run_conda_in_shell(cmd, environment_name, stderr_to_stdout=True)
        finish_batch(
            cur=cur,
            con=con,
            sample_ids=sample_ids,
            foreign_key_name=foreign_key_name,
            run_id=run_id,
        )
        # MFA throws an error at end even though it created texgrids, so don't check
        """if not success:
            raise Exception("An error occured in align() ...")"""
