from joblib import Parallel, delayed
import multiprocessing as mp
import shutil
import torch
from typing import List, Callable, Optional, Dict, Any
from pathlib import Path
from voice_smith.utils.audio import safe_load
from voice_smith.utils.tools import iter_logger
from voice_smith.utils.audio import save_audio


def copy_sample(
    audio_src: str, text_src: str, text: str, out_dir: str, skip_on_error: bool
) -> None:
    if not Path(audio_src).exists():
        print(f"Audio file {audio_src} doesn't exist, skipping ...")
        return
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    audio_out_path = out_path / (Path(audio_src).stem + ".flac")
    txt_out_path = out_path / Path(text_src).name
    if txt_out_path.exists(): 
        return
    try:
        audio, sr = safe_load(audio_src, sr=None)
        save_audio(str(audio_out_path), torch.FloatTensor(audio), sr)
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        if skip_on_error:
            if audio_out_path.exists():
                audio_out_path.unlink()
            print(e)
            return
        else:
            raise e


def copy_files(
    data_path: str,
    txt_paths: List[str],
    texts: List[str],
    audio_paths: List[str],
    names: List[str],
    langs: List[str],
    workers: int,
    skip_on_error: bool,
    progress_cb: Callable[[int], None],
    log_every: int = 200,
) -> None:
    assert len(txt_paths) == len(audio_paths) == len(names) == len(texts) == len(langs)

    def callback(index: int):
        if index % log_every == 0:
            progress = index / len(txt_paths)
            progress_cb(progress)

    print("Copying files ...")
    Parallel(n_jobs=workers)(
        delayed(copy_sample)(
            audio_path,
            txt_path,
            text,
            Path(data_path) / "raw_data" / lang / name,
            skip_on_error,
        )
        for audio_path, txt_path, text, name, lang in iter_logger(
            zip(audio_paths, txt_paths, texts, names, langs), cb=callback
        )
    )
    progress_cb(1.0)
