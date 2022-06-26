import torch
from torch.utils.data import Dataset
from typing import Dict, List, Callable
from transformers import pipeline
from voice_smith.utils.tools import iter_logger

lang2model: Dict[str, str] = {
    "bg": "anuragshas/wav2vec2-large-xls-r-300m-bg",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "en": "facebook/wav2vec2-base-960h",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "sv": "viktor-enzell/wav2vec2-large-voxrex-swedish-4gram",
    "th": "airesearch/wav2vec2-large-xlsr-53-th",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv6-turkish",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
}


class STTDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


def transcribe(
    audio_files: List[str],
    lang: str,
    device: torch.device,
    progress_cb: Callable[[float], None],
    callback_every: int = 25,
    batch_size: int = 10,
) -> List[str]:

    dataset = STTDataset(audio_files)

    pipe = pipeline(
        model=lang2model[lang],
        device=0 if "cuda" in device.type else -1,
        chunk_length_s=10,
        stride_length_s=(4, 2),
        framework="pt",
    )

    def cb(progress):
        progress_cb(progress / len(audio_files))

    transcriptions = []

    with torch.no_grad():
        for transcription in iter_logger(
            pipe(dataset, batch_size=batch_size),
            total=len(audio_files),
            print_every=callback_every,
            callback_every=callback_every,
            cb=cb,
        ):
            transcriptions.append(transcription["text"])

    return transcriptions

