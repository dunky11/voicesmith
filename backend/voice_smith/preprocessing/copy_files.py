from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
import torch
from typing import List, Callable, Optional, Dict, Any
from pathlib import Path
from voice_smith.utils.audio import safe_load
from voice_smith.utils.tools import iter_logger
from voice_smith.utils.audio import save_audio


def write_text_file(src: str, text: str, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    with open(out_path / Path(src).name, "w", encoding="utf-8") as f:
        f.write(text)


def copy_audio_file(src: str, out_dir: str) -> None:
    if not Path(src).exists():
        print(f"Audio file {src} doesn't exist, skipping ...")
        return
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    audio, sr = safe_load(src, sr=None)
    save_audio(str(out_path / (Path(src).stem + ".wav")), torch.FloatTensor(audio), sr)


def copy_files(
    db_id: int,
    table_name: str,
    data_path: str,
    txt_paths: List[str],
    texts: List[str],
    audio_paths: List[str],
    names: List[str],
    get_logger: Optional[Callable],
    preprocess_config: Dict[str, Any],
    log_every: int = 200,
) -> None:
    assert len(txt_paths) == len(texts)
    workers = preprocess_config["workers"]

    def txt_callback(index: int):
        if index % log_every == 0:
            logger = get_logger()
            progress = index / len(txt_paths) / 2
            logger.query(
                f"UPDATE {table_name} SET preprocessing_copying_files_progress=? WHERE id=?",
                [progress, db_id],
            )

    def audio_callback(index: int):
        if index % log_every == 0:
            logger = get_logger()
            progress = (index / len(audio_paths) / 2) + 0.5
            logger.query(
                f"UPDATE {table_name} SET preprocessing_copying_files_progress=? WHERE id=?",
                [progress, db_id],
            )

    print("Writing text files ...")
    Parallel(n_jobs=workers)(
        delayed(write_text_file)(file_path, text, Path(data_path) / "raw_data" / name)
        for file_path, text, name in iter_logger(
            zip(txt_paths, texts, names), cb=txt_callback
        )
    )
    print("Copying audio files ...")
    Parallel(n_jobs=workers)(
        delayed(copy_audio_file)(file_path, Path(data_path) / "raw_data" / name)
        for file_path, name in iter_logger(zip(audio_paths, names), cb=audio_callback)
    )
    logger = get_logger()
    logger.query(
        f"UPDATE {table_name} SET preprocessing_copying_files_progress=? WHERE id=?",
        [1.0, db_id],
    )
