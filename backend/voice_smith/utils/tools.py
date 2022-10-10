import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Iterator, List, Union, Tuple, Any, Callable, Literal
from pathlib import Path
import time
import psutil
import shutil
import warnings, sys
import multiprocessing as mp
from tqdm import tqdm


def warnings_to_stdout():
    def customwarn(message, category, filename, lineno, file=None, line=None):
        sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))

    warnings.showwarning = customwarn


def to_device(
    data: Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any],
    device: torch.device,
    is_eval: bool,
):
    if is_eval:
        (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
            wavs,
        ) = data
    else:
        (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
        ) = data

    speakers = torch.tensor(speakers, dtype=torch.int64, device=device)
    texts = torch.tensor(texts, dtype=torch.int64, device=device)
    src_lens = torch.tensor(src_lens, dtype=torch.int64, device=device)
    mels = torch.tensor(mels, dtype=torch.float32, device=device)
    pitches = torch.tensor(pitches, dtype=torch.float32, device=device)
    mel_lens = torch.tensor(mel_lens, dtype=torch.int64, device=device)
    langs = torch.tensor(langs, dtype=torch.int64, device=device)
    attn_priors = torch.tensor(attn_priors, dtype=torch.float32, device=device)

    if is_eval:
        wavs = torch.tensor(wavs, dtype=torch.float32, device=device)
        return (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
            wavs,
        )
    else:
        return (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
        )



def sample_wise_min_max(x: torch.Tensor) -> torch.Tensor:
    maximum = torch.amax(x, dim=(1, 2), keepdim=True)
    minimum = torch.amin(x, dim=(1, 2), keepdim=True)
    return (x - minimum) / (maximum - minimum)


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def pad_1D(inputs: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    def pad_data(x, length):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=pad_value
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len) for x in inputs])

    return padded


def pad_2D(
    inputs: List[np.ndarray], maxlen: Union[int, None] = None, pad_value: float = 0.0
) -> np.ndarray:
    def pad(x, max_len):
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")
        padding = np.ones((x.shape[0], max_len - np.shape(x)[1])) * pad_value
        x = np.concatenate((x, padding), 1)
        return x

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, : np.shape(input_)[0], : np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele: List[torch.Tensor], max_len: int) -> torch.Tensor:
    out_list = torch.jit.annotate(List[torch.Tensor], [])
    for batch in input_ele:
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        else:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def cycle(iterable: Iterable) -> Iterator:
    while True:
        for x in iterable:
            yield x


def cycle_2d(iterable: Iterable) -> Iterator:
    while True:
        for x in iterable:
            for y in x:
                yield y


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def stride_lens(lens: torch.Tensor, stride: int = 2) -> torch.Tensor:
    return torch.ceil(lens / stride).int()


def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


def get_speaker_embeddings(preprocessed_path: str, dim: int) -> torch.Tensor:
    with open(Path(preprocessed_path) / "speakers.json", "r") as f:
        data = json.load(f)

    in_dir = Path(preprocessed_path) / "speaker_embeds"
    speaker_embeddings = torch.zeros((len(data)), dim)
    for speaker in data.keys():
        value = data[speaker]
        embeddings = torch.from_numpy(np.load(in_dir / speaker / "embedding.py.npy"))
        speaker_embeddings[value] = embeddings

    return speaker_embeddings


def iter_logger(
    iterable: Iterable,
    start: Union[int, None] = None,
    total: Union[int, None] = None,
    print_every: int = 20,
    callback_every: int = 1,
    cb: Union[Callable, None] = None,
) -> Iterator:
    for i, it in enumerate(
        tqdm(iterable, initial=0 if start is None else start, total=total, leave=True)
    ):
        if i % callback_every == 0 and cb is not None:
            cb(i)
        # print("\n", flush=True)
        yield it

    """class tqdm_logger(base_tqdm):
        def update(self, n=1):
            super(base_tqdm, self).update(n)
            if n % callback_every == 0 and cb != None:
                cb(n)

    return tqdm_logger(
        iterable, initial=start, total=total, leave=True, file=sys.stdout
    )"""


def bytes_to_gb(bytes: float) -> float:
    return bytes / 1024 ** 3


def get_cpu_usage() -> float:
    return psutil.cpu_percent()


def get_ram_usage() -> Tuple[float, float]:
    info = psutil.virtual_memory()
    total = bytes_to_gb(info.total)
    used = bytes_to_gb(info.used)
    return total, used


def get_disk_usage() -> Tuple[float, float]:
    info = shutil.disk_usage(".")
    total = bytes_to_gb(info.total)
    used = total - bytes_to_gb(info.free)
    return total, used


def get_embeddings(data_path: str, device: torch.device) -> torch.Tensor:
    embeddings = []
    with open(Path(data_path) / "speakers.json", "r", encoding="utf-8") as f:
        for speaker in json.load(f).keys():
            try:
                embedding = torch.load(
                    Path(data_path) / "speaker_embeds" / speaker / "embedding.pt"
                )
                embeddings.append(embedding)
            except Exception as e:
                print(e)
                embeddings.append(torch.randn((192,)))
    embeddings = torch.stack(embeddings).to(device)
    return embeddings


def initialize_embeddings(shape: Tuple[int, ...]) -> torch.Tensor:
    assert len(shape) == 2, "Can only initialize 2-D embedding matrices ..."
    # Kaiming initialization
    return torch.randn(shape) * np.sqrt(2 / shape[1])


def union_shuffled_copies(a: List[Any], b: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a_out = [a[idx] for idx in p]
    b_out = [b[idx] for idx in p]
    return a_out, b_out


def stratified_train_test_split(
    x: List[Any], y: List[Any], train_size: float
) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    label2samples = {}
    for x, y in zip(x, y):
        if y in label2samples:
            label2samples[y].append(x)
        else:
            label2samples[y] = [x]
    train_x_out, train_y_out = [], []
    val_x_out, val_y_out = [], []
    for label, samples in label2samples.items():
        split_at = int(np.round(len(samples) * train_size))
        x_split_train, x_split_val = samples[:split_at], samples[split_at:]
        y_split_train, y_split_val = (
            [label] * len(x_split_train),
            [label] * len(x_split_val),
        )
        train_x_out.extend(x_split_train)
        train_y_out.extend(y_split_train)
        val_x_out.extend(x_split_val)
        val_y_out.extend(y_split_val)
    train_x_out, train_y_out = union_shuffled_copies(train_x_out, train_y_out)
    val_x_out, val_y_out = union_shuffled_copies(val_x_out, val_y_out)
    return train_x_out, val_x_out, train_y_out, val_y_out


class OnlineScaler:
    """Online mean and variance computation, see
    http://www.cs.yale.edu/publications/techreports/tr222.pdf
    equation 1.5a and 1.5b
    """

    t_1_m = None
    s_1_m = None
    m = 0

    def partial_fit(self, x: torch.Tensor) -> None:
        assert len(x.shape) > 1, "First dimension to partial_fit must be batch size"
        if self.m == 0:
            self.t_1_m = x[0]
            self.s_1_m = 0.0
            self.m += 1
            if x.shape[0] > 1:
                self.partial_fit(x[1:])
        else:
            n = x.shape[0]
            x_sum = x.sum(0)
            self.s_1_m = (
                self.s_1_m
                + x.var(0, unbiased=False) * x.shape[0]
                + (self.m / (n * (self.m + n)))
                * ((n / self.m) * self.t_1_m - x_sum) ** 2
            )
            self.t_1_m = self.t_1_m + x_sum
            self.m += n

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.t_1_m / self.m, torch.sqrt(self.s_1_m / self.m)


def get_device(device: Union[Literal["CPU"], Literal["GPU"]]) -> torch.device:
    if device == "CPU":
        return torch.device("cpu")
    elif device == "GPU":
        if not torch.cuda.is_available():
            raise Exception(
                f"Mode was set to 'GPU' but no available GPU could be found ..."
            )
        return torch.device("cuda")
    else:
        raise Exception(f"Device '{device}' is not a valid device type ...")


def get_workers(workers: Union[int, None]) -> int:
    return max(1, mp.cpu_count() - 1) if workers == -1 else workers


def slice_segments(x, ids_str, segment_size):
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths, segment_size):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def byte_encode(word):
    text = word.strip()
    return list(text.encode("utf-8"))

