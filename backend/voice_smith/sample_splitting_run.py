from pathlib import Path
import shutil
from typing import Callable, List, Dict
import sys
import torch
import json
import sqlite3
import argparse
from dataclasses import dataclass
from joblib import Parallel, delayed
from voice_smith.preprocessing.copy_files import copy_files
from voice_smith.config.configs import SampleSplittingRunConfig
from voice_smith.utils.sql_logger import SQLLogger
from voice_smith.utils.loggers import set_stream_location
from voice_smith.sql import get_con, save_current_pid
from voice_smith.utils.tools import warnings_to_stdout, get_device, get_workers
from voice_smith.preprocessing.generate_vocab import generate_vocab
from voice_smith.preprocessing.align import align
from voice_smith.preprocessing.sample_splitting import sample_splitting, split_sample
from voice_smith.utils.punctuation import get_punct
from voice_smith.utils.runs import StageRunner
from voice_smith.config.globals import (
    DB_PATH,
    SAMPLE_SPLITTING_RUNS_PATH,
    ENVIRONMENT_NAME,
    DATASETS_PATH,
    ASSETS_PATH,
)

warnings_to_stdout()


def get_config(cur: sqlite3.Cursor, run_id: int) -> SampleSplittingRunConfig:
    row = cur.execute(
        """
        SELECT device, maximum_workers, skip_on_error FROM sample_splitting_run WHERE ID=?
        """,
        (run_id,),
    ).fetchone()
    (device, maximum_workers, skip_on_error) = row
    device = get_device(device)
    workers = get_workers(maximum_workers)
    skip_on_error = bool(skip_on_error)
    return SampleSplittingRunConfig(
        workers=workers, device=device, skip_on_error=skip_on_error
    )


def before_run(data_path: str, **kwargs):
    (Path(data_path) / "logs").mkdir(exist_ok=True, parents=True)


def get_log_file_name(stage_name: str) -> str:
    if stage_name in [
        "not_started",
        "copying_files",
        "gen_vocab",
        "gen_alignments",
        "creating_splits",
    ]:
        return "preprocessing.txt"
    elif stage_name == "apply_changes":
        return "apply_changes.txt"
    else:
        raise Exception(
            f"No branch selected in switch-statement, {stage_name} is not a valid case ..."
        )


def before_stage(
    data_path: str, stage_name: str, **kwargs,
):
    set_stream_location(str(Path(data_path) / "logs" / get_log_file_name(stage_name)))


def get_stage_name(cur: sqlite3.Cursor, run_id: int, **kwargs):
    row = cur.execute(
        "SELECT stage FROM sample_splitting_run WHERE ID=?", (run_id,),
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
        "UPDATE sample_splitting_run SET stage='copying_files' WHERE ID=?", (run_id,),
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
    txt_paths, texts, audio_paths, names, langs = [], [], [], [], []

    for (
        txt_path,
        text,
        audio_path,
        speaker_name,
        dataset_id,
        speaker_id,
        lang,
    ) in cur.execute(
        """
        SELECT sample.txt_path, sample.text, sample.audio_path, 
        speaker.name AS speaker_name, dataset.ID AS dataset_id, 
        speaker.ID as speaker_id, speaker.language
        FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
        INNER JOIN speaker on speaker.dataset_id = dataset.ID
        INNER JOIN sample on sample.speaker_id = speaker.ID
        WHERE sample_splitting_run.ID=?
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
        txt_paths.append(txt_path)
        texts.append(text)
        audio_paths.append(str(full_audio_path))
        names.append(speaker_name)
        langs.append(lang)

    config = get_config(cur=cur, run_id=run_id)

    def progress_cb(progress: float):
        logger = get_logger()
        logger.query(
            "UPDATE sample_splitting_run SET copying_files_progress=? WHERE id=?",
            (progress, run_id),
        )

    copy_files(
        data_path=data_path,
        txt_paths=txt_paths,
        texts=texts,
        audio_paths=audio_paths,
        names=names,
        workers=config.workers,
        progress_cb=progress_cb,
        langs=langs,
        skip_on_error=config.skip_on_error,
    )
    cur.execute(
        "UPDATE sample_splitting_run SET stage='gen_vocab' WHERE ID=?", (run_id,),
    )
    con.commit()
    return False


def get_vocab_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    assets_path: str,
    vocab_path: str,
    **kwargs,
) -> bool:
    vocab_path = Path(vocab_path)
    vocab_path.mkdir(exist_ok=True, parents=True)

    row = cur.execute(
        "SELECT device FROM sample_splitting_run WHERE ID=?", (run_id,),
    ).fetchone()
    device = row[0]
    device = get_device(device)

    langs = [el.name for el in (Path(data_path) / "raw_data").iterdir()]
    for i, lang in enumerate(langs):
        texts = []
        for (text,) in cur.execute(
            """
            SELECT sample.text AS text FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
            INNER JOIN speaker on speaker.dataset_id = dataset.ID
            INNER JOIN sample on sample.speaker_id = speaker.ID
            WHERE sample_splitting_run.ID=? AND speaker.language=?
            """,
            (run_id, lang),
        ).fetchall():
            texts.append(text)

        lexica_path = str(vocab_path / f"{lang}.txt")
        predicted_phones = generate_vocab(
            texts=texts, lang=lang, assets_path=assets_path, device=device
        )
        punct_set = get_punct(lang=lang)
        with open(lexica_path, "w", encoding="utf-8") as f:
            for word, phones in predicted_phones.items():
                word = word.lower().strip()
                phones = " ".join(phones).strip()
                if len(word) == 0 or len(phones) == 0:
                    continue
                if word in punct_set:
                    continue
                f.write(f"{word.lower()} {phones}\n")

        cur.execute(
            "UPDATE sample_splitting_run SET gen_vocab_progress=? WHERE ID=?",
            ((i + 1) / len(langs), run_id),
        )
        con.commit()

    cur.execute(
        "UPDATE sample_splitting_run SET stage='gen_alignments', gen_vocab_progress=1.0 WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def gen_alignments_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    environment_name: str,
    vocab_path: str,
    **kwargs,
):
    p_config = get_config(cur, run_id)
    vocab_paths = list(Path(vocab_path).iterdir())
    for i, vocab_path in enumerate(vocab_paths):
        lang = vocab_path.name.split(".")[0]
        align(
            environment_name=environment_name,
            in_path=str(Path(data_path) / "raw_data" / lang),
            lexicon_path=str(Path(data_path) / "data" / "vocabs" / f"{lang}.txt"),
            out_path=(str(Path(data_path) / "data" / "textgrid")),
            n_workers=p_config.workers,
            lang=lang,
        )
        cur.execute(
            "UPDATE sample_splitting_run SET gen_align_progress=? WHERE ID=?",
            ((i + 1) / len(vocab_paths), run_id),
        )
        con.commit()
    cur.execute(
        "UPDATE sample_splitting_run SET stage='creating_splits', gen_align_progress=1.0 WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return False


def creating_splits_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    data_path: str,
    splits_path: str,
    datasets_path: str,
    **kwargs,
):
    splits_path = Path(splits_path)
    sample_ids, texts, textgrid_paths, langs = [], [], [], []
    if splits_path.exists():
        shutil.rmtree(splits_path)
    (Path(data_path) / "splits").mkdir(parents=True)
    p_config = get_config(cur, run_id)

    # TODO delete other table
    cur.execute(
        """
        DELETE FROM sample_splitting_run_sample WHERE sample_splitting_run_id=?
        """,
        (run_id,),
    )

    con.commit()
    for (sample_id, text, audio_path, speaker_name, lang) in cur.execute(
        """
        SELECT sample.ID, sample.text, sample.audio_path, 
        speaker.name AS speaker_name, speaker.language
        FROM sample_splitting_run INNER JOIN dataset ON sample_splitting_run.dataset_id = dataset.ID 
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
        langs.append(lang)

    splits = sample_splitting(
        ids=sample_ids, texts=texts, textgrid_paths=textgrid_paths, languages=langs,
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
            Path(datasets_path)
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
    return True


@dataclass
class ApplyChangesSplit:
    full_audio_path: str
    text: str
    split_idx: int


@dataclass
class ApplyChangesInfo:
    sample_id: int
    sample_splitting_run_sample_id: int
    speaker_id: int
    old_sample_txt_path: str
    old_sample_audio_path: str
    old_sample_full_audio_path: str
    splits: List[ApplyChangesSplit]


def apply_changes_stage(
    cur: sqlite3.Cursor,
    con: sqlite3.Connection,
    run_id: int,
    splits_path: str,
    datasets_path: str,
    **kwargs,
) -> bool:
    sample_id_to_info: Dict[int, ApplyChangesInfo] = {}
    for (
        sample_splitting_run_sample_id,
        sample_id,
        txt_path,
        audio_path,
        dataset_id,
        speaker_id,
        new_text,
        split_idx,
    ) in cur.execute(
        """
        SELECT sample_splitting_run_sample.ID AS sample_splitting_run_sample_id, 
        sample.ID AS sampleID, sample.txt_path, sample.audio_path AS sample_audio_path, 
        dataset.ID AS dataset_id, speaker.ID AS speaker_id, sample_splitting_run_split.text,
        sample_splitting_run_split.split_idx
        FROM sample_splitting_run_split
        INNER JOIN sample_splitting_run_sample ON sample_splitting_run_split.sample_splitting_run_sample_id = sample_splitting_run_sample.ID
        INNER JOIN sample on sample_splitting_run_sample.sample_id = sample.ID
        INNER JOIN speaker on sample.speaker_id = speaker.ID
        INNER JOIN dataset ON speaker.dataset_id = dataset.ID
        WHERE sample_splitting_run_sample.sample_splitting_run_id=?
        """,
        (run_id,),
    ).fetchall():
        full_old_audio_path = (
            Path(datasets_path)
            / str(dataset_id)
            / "speakers"
            / str(speaker_id)
            / audio_path
        )
        full_new_audio_path = (
            Path(splits_path)
            / f"{sample_splitting_run_sample_id}_split_{split_idx}.flac"
        )
        apply_changes_split = ApplyChangesSplit(
            full_audio_path=full_new_audio_path, text=new_text, split_idx=split_idx
        )
        if sample_splitting_run_sample_id in sample_id_to_info:
            sample_id_to_info[sample_splitting_run_sample_id].splits.append(
                apply_changes_split
            )
        else:
            sample_id_to_info[sample_splitting_run_sample_id] = ApplyChangesInfo(
                sample_id=sample_id,
                sample_splitting_run_sample_id=sample_splitting_run_sample_id,
                old_sample_txt_path=txt_path,
                old_sample_audio_path=full_old_audio_path,
                old_sample_full_audio_path=full_old_audio_path,
                speaker_id=speaker_id,
                splits=[apply_changes_split],
            )

    for apply_changes_info in sample_id_to_info.values():
        old_sample_txt_path = Path(apply_changes_info.old_sample_txt_path)
        old_sample_full_audio_path = Path(apply_changes_info.old_sample_full_audio_path)
        delete_audio_paths: List[str] = []
        for split in apply_changes_info.splits:
            if not Path(split.full_audio_path).exists():
                continue
            copy_audio_to = (
                Path(old_sample_full_audio_path.parent)
                / f"{old_sample_full_audio_path.stem}_split_{split.split_idx}{split.full_audio_path.suffix}"
            )
            audio_name_to = f"{old_sample_full_audio_path.stem}_split_{split.split_idx}{old_sample_full_audio_path.suffix}"
            text_name_to = f"{old_sample_txt_path.stem}_split_{split.split_idx}{old_sample_txt_path.suffix}"

            shutil.copy2(split.full_audio_path, copy_audio_to)
            cur.execute(
                "INSERT INTO sample (txt_path, audio_path, speaker_id, text) VALUES (?, ?, ? ,?)",
                (
                    text_name_to,
                    audio_name_to,
                    apply_changes_info.speaker_id,
                    split.text,
                ),
            )
            delete_audio_paths.append(split.full_audio_path)
        delete_audio_paths.append(str(old_sample_full_audio_path))
        cur.execute("DELETE FROM sample WHERE ID=?", (apply_changes_info.sample_id,))
        cur.execute(
            "DELETE FROM sample_splitting_run_sample WHERE ID=?",
            (apply_changes_info.sample_splitting_run_sample_id,),
        )
        con.commit()
        for delete_audio_path in delete_audio_paths:
            Path(delete_audio_path).unlink(missing_ok=True)

    cur.execute(
        "UPDATE sample_splitting_run SET stage='finished', applying_changes_progress=1.0 WHERE ID=?",
        (run_id,),
    )
    con.commit()
    return True


def continue_sample_splitting_run(run_id: int,):
    con = get_con(DB_PATH)
    cur = con.cursor()
    save_current_pid(con=con, cur=cur)
    data_path = str(Path(SAMPLE_SPLITTING_RUNS_PATH) / str(run_id))
    splits_path = str(Path(data_path) / "splits")
    vocab_path = str(Path(data_path) / "data" / "vocabs")

    def get_logger():
        con = get_con(DB_PATH)
        cur = con.cursor()
        logger = SQLLogger(
            training_run_id=run_id,
            con=con,
            cursor=cur,
            out_dir=str(data_path),
            stage="sample_splitting_run",
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
            ("gen_vocab", get_vocab_stage),
            ("gen_alignments", gen_alignments_stage),
            ("creating_splits", creating_splits_stage),
            ("apply_changes", apply_changes_stage),
        ],
    )
    runner.run(
        cur=cur,
        con=con,
        run_id=run_id,
        data_path=data_path,
        assets_path=ASSETS_PATH,
        get_logger=get_logger,
        environment_name=ENVIRONMENT_NAME,
        datasets_path=DATASETS_PATH,
        splits_path=splits_path,
        vocab_path=vocab_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()

    continue_sample_splitting_run(run_id=args.run_id,)

