from joblib import Parallel, delayed
import torch
from typing import List, Callable, Literal, Union
from pathlib import Path
from voice_smith.utils.audio import safe_load, save_audio
from voice_smith.utils.tools import iter_logger


def copy_sample(
    sample_id: int,
    audio_src: str,
    text: str,
    out_dir: str,
    skip_on_error: bool,
    lowercase: bool,
    name_by: Union[Literal["id"], Literal["name"]],
) -> None:
    if not Path(audio_src).exists():
        raise Exception(f"File {audio_src} does not exist ...")
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    audio_out_path = (
        out_path / f"{sample_id if name_by == 'id' else Path(audio_src).stem}.flac"
    )
    txt_out_path = (
        out_path / f"{sample_id if name_by == 'id' else Path(audio_src).stem}.txt"
    )
    if txt_out_path.exists():
        return
    try:
        audio, sr = safe_load(audio_src, sr=None)
        save_audio(str(audio_out_path), torch.FloatTensor(audio), sr)
        with open(txt_out_path, "w", encoding="utf-8") as f:
            f.write(text.lower() if lowercase else text)
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
    sample_ids: List[int],
    texts: List[str],
    audio_paths: List[str],
    names: List[str],
    langs: List[str],
    workers: int,
    skip_on_error: bool,
    name_by: Union[Literal["id"], Literal["name"]],
    progress_cb: Callable[[float], None],
    lowercase: bool = True,
    log_every: int = 200,
) -> None:
    assert len(sample_ids) == len(audio_paths) == len(names) == len(texts) == len(langs)

    def callback(index: int):
        if index % log_every == 0:
            progress = index / len(sample_ids)
            progress_cb(progress)

    print("Copying files ...")
    Parallel(n_jobs=workers)(
        delayed(copy_sample)(
            sample_id,
            audio_path,
            text,
            Path(data_path) / "raw_data" / lang / name,
            skip_on_error,
            lowercase,
            name_by,
        )
        for sample_id, audio_path, text, name, lang in iter_logger(
            zip(sample_ids, audio_paths, texts, names, langs),
            cb=callback,
            total=len(sample_ids),
        )
    )
    progress_cb(1.0)
