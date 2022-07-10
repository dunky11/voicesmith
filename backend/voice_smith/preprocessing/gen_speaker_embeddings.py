import torch
from pathlib import Path
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from typing import Any, List, Tuple, Callable
from voice_smith.utils.tools import iter_logger
from voice_smith.utils.audio import safe_load
import numpy as np
from speechbrain.pretrained import EncoderClassifier


class SpeakerDS(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        file_path = self.file_paths[idx]
        file_name = Path(file_path).stem
        speaker_name = Path(file_path).parent.name
        wav, _ = safe_load(file_path, sr=16000)
        # TODO sample that failed to load shouldn't be included in batch
        if not isinstance(wav, np.ndarray):
            wav = torch.zeros((512,))
        wav = torch.FloatTensor(wav)
        return wav, speaker_name, file_name


class FilesDS(Dataset):
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> None:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path = self.file_paths[idx]
        embedding = torch.load(file_path)
        return embedding


def collate_fn(batch: List[Any]):
    max_len = max(el[0].shape[0] for el in batch)
    relative_lens = torch.FloatTensor([el[0].shape[0] / max_len for el in batch])
    wavs = torch.stack(
        [torch.cat([el[0], torch.zeros((max_len - el[0].shape[0]))]) for el in batch]
    )
    speaker_names = [el[1] for el in batch]
    file_names = [el[2] for el in batch]
    return wavs, relative_lens, speaker_names, file_names


def gen_file_emeddings(
    in_paths: List[str],
    out_dir: str,
    callback: Callable[[int], int],
    device: torch.device,
    batch_size: int = 6,
):
    classifier = EncoderClassifier.from_hparams(
        source=str(Path(".") / "assets" / "ecapa_tdnn")
    )
    classifier.eval()
    classifier.device = device
    classifier.mods.embedding_model.to(device)

    ds = SpeakerDS(in_paths)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=max(mp.cpu_count() - 1, 1),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    def inner_callback(index: int):
        progress = index / (len(in_paths) // batch_size)
        callback(index, progress)

    for i, (wavs, relative_lens, speaker_names, file_names) in iter_logger(
        enumerate(loader), total=len(in_paths) // batch_size, cb=inner_callback
    ):
        with torch.no_grad():
            wavs = wavs.to(device)
            relative_lens = relative_lens.to(device)
            y_pred = classifier.encode_batch(wavs, relative_lens)

        y_pred = y_pred.cpu()
        for emb, speaker_name, file_name in zip(y_pred, speaker_names, file_names):
            directory = Path(out_dir) / speaker_name
            directory.mkdir(exist_ok=True, parents=True)

            torch.save(
                emb.view((-1,)), directory / f"{Path(file_name).stem}.pt",
            )


def gen_speaker_embeddings(
    speaker_paths: List[str],
    out_dir: str,
    callback: Callable[[int, float], None],
    device: torch.device,
    batch_size: int = 20,
):
    def inner_callback(index: int):
        progress = index / len(speaker_paths)
        callback(index, progress)

    for speaker_path in iter_logger(
        speaker_paths, total=len(speaker_paths), cb=inner_callback
    ):
        speaker_name = speaker_path.name
        (Path(out_dir) / speaker_name).mkdir(exist_ok=True, parents=True)
        files = list(speaker_path.iterdir())
        files = [str(file) for file in files]
        if len(files) == 0:
            continue
        ds = FilesDS(files)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=max(mp.cpu_count() - 1, 1),
            shuffle=False,
            drop_last=False,
        )
        embeddings_all = []
        for embeddings in loader:
            embeddings = embeddings.cuda()
            embeddings_all.extend(embeddings)
        embeddings = torch.stack(embeddings_all, 0)
        mean = embeddings.mean(dim=0)
        mean = mean / torch.linalg.vector_norm(mean, ord=2)
        torch.save(mean.cpu(), Path(out_dir) / speaker_name / "embedding.pt")
