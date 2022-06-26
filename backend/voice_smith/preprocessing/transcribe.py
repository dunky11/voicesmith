import torch
import zipfile
import torchaudio
from typing import List, Callable
from glob import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from voice_smith.utils.tools import iter_logger

torch.backends.cudnn.deterministic = True

SUPPORTED_LANGUAGES = ["en", "uk", "es", "de"]


class STTDataset(Dataset):
    def __init__(self, files, audio_load_fn):
        self.files = files
        self.audio_load_fn = audio_load_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.audio_load_fn(self.files[idx])


def transcribe(
    audio_files: List[str],
    lang: str,
    device: torch.device,
    workers: int,
    assets_path: str,
    progress_cb: Callable[[float], None],
    callback_every: int = 5,
    batch_size: int = 256,
) -> List[str]:

    if not lang in SUPPORTED_LANGUAGES:
        raise Exception(
            f"Language '{lang}' is not supported for transcribtion, it has to be in {SUPPORTED_LANGUAGES} ..."
        )
    if lang == "uk":
        lang = "ua"

    model, decoder, utils = torch.hub.load(
        repo_or_dir=str(Path(assets_path) / "silero_models"),
        source="local",
        model="silero_stt",
        language=lang,
        device=device,
    )
    (read_batch, split_into_batches, read_audio, prepare_model_input,) = utils

    ds = STTDataset(audio_files, read_audio)
    loader = DataLoader(
        ds, batch_size=batch_size, collate_fn=lambda x: x, num_workers=workers,
    )

    transcriptions = []

    with torch.no_grad():
        for batch in iter_logger(
            loader,
            cb=progress_cb,
            callback_every=callback_every,
            print_every=callback_every,
        ):
            inp = prepare_model_input(batch, device=device)
            outputs = model(inp)
            for output in outputs.cpu():
                transcriptions.append(decoder(output))

    return transcriptions

