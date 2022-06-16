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
    data_path: str,
    txt_paths: List[str],
    texts: List[str],
    audio_paths: List[str],
    names: List[str],
    langs: List[str],
    workers: int,
    progress_cb: Callable[[int], None],
    log_every: int = 200,
) -> None:
    assert len(txt_paths) == len(texts) == len(langs)

    def txt_callback(index: int):
        if index % log_every == 0:
            progress = index / len(txt_paths) / 2
            progress_cb(progress)

    def audio_callback(index: int):
        if index % log_every == 0:
            progress = (index / len(audio_paths) / 2) + 0.5
            progress_cb(progress)

    print("Writing text files ...")
    Parallel(n_jobs=workers)(
        delayed(write_text_file)(
            file_path, text, Path(data_path) / "raw_data" / lang / name
        )
        for file_path, text, name, lang in iter_logger(
            zip(txt_paths, texts, names, langs), cb=txt_callback
        )
    )
    print("Copying audio files ...")
    Parallel(n_jobs=workers)(
        delayed(copy_audio_file)(file_path, Path(data_path) / "raw_data" / lang / name)
        for file_path, name, lang in iter_logger(
            zip(audio_paths, names, langs), cb=audio_callback
        )
    )
    progress_cb(1.0)
