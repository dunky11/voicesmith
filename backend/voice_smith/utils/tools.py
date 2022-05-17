import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Iterator, List, Union, Tuple, Any, Callable
from pathlib import Path
import time
import psutil
import shutil
import warnings, sys

def warnings_to_stdout():
    def customwarn(message, category, filename, lineno, file=None, line=None):
        sys.stdout.write(warnings.formatwarning(message, category, filename, lineno))
    warnings.showwarning = customwarn

def to_device(
    data: Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any],
    device: torch.device,
    non_blocking: bool = False,
):
    (
        ids,
        raw_texts,
        speakers,
        speaker_names,
        texts,
        src_lens,
        mels,
        pitches,
        durations,
        mel_lens,
        token_ids,
        attention_masks,
    ) = data

    speakers = torch.from_numpy(speakers).to(device, non_blocking=non_blocking)
    texts = torch.from_numpy(texts).long().to(device, non_blocking=non_blocking)
    src_lens = torch.from_numpy(src_lens).to(device, non_blocking=non_blocking)
    mels = torch.from_numpy(mels).float().to(device, non_blocking=non_blocking)
    pitches = torch.from_numpy(pitches).float().to(device, non_blocking=non_blocking)
    durations = torch.from_numpy(durations).long().to(device, non_blocking=non_blocking)
    mel_lens = torch.from_numpy(mel_lens).to(device, non_blocking=non_blocking)
    token_ids = token_ids.to(device, non_blocking=non_blocking)
    attention_masks = attention_masks.to(device, non_blocking=non_blocking)

    return (
        ids,
        raw_texts,
        speakers,
        speaker_names,
        texts,
        src_lens,
        mels,
        pitches,
        durations,
        mel_lens,
        token_ids,
        attention_masks,
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


def pad_1D(inputs: List[np.ndarray], PAD: int = 0) -> np.ndarray:
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs: List[np.ndarray], maxlen: Union[int, None] = None) -> np.ndarray:
    def pad(x, max_len):
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")

        zeros = np.zeros((x.shape[0], max_len - np.shape(x)[1]))
        x = np.concatenate((x, zeros), 1)
        return x

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs: List[np.ndarray], B: int, T: int, L: int) -> np.ndarray:
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
    cb: Union[Callable, None] = None,
) -> Iterator:
    last_time = time.time()
    for i, el in enumerate(iterable):
        if i % print_every == 0 and i != 0:
            message = f"{i if start == None else i + start}"
            if total != None:
                message += f"/{total}"
            this_time = time.time()
            message += f", {round(print_every / (this_time - last_time), 2)}it/s"
            last_time = this_time
            print(message, flush=True)

        if cb != None:
            cb(i)

        yield el


def bytes_to_gb(bytes: float) -> float:
    return bytes / 1024**3


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
    with open(
        Path(data_path) / "speakers.json",
        "r",
    ) as f:
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


def initialize_embeddings(shape: Tuple[int]) -> torch.Tensor:
    assert len(shape) == 2, "Can only initialize 2-D embedding matrices ..."
    # Kaiming initialization
    return torch.randn(shape) * np.sqrt(2 / shape[1])

def union_shuffled_copies(a: List[Any], b: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a_out = [a[idx] for idx in p]
    b_out = [b[idx] for idx in p]
    return a_out, b_out

def stratified_train_test_split(x: List[Any], y: List[Any], train_size: float) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
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
        y_split_train, y_split_val = [label] * len(x_split_train), [label] * len(x_split_val)
        train_x_out.extend(x_split_train)
        train_y_out.extend(y_split_train)
        val_x_out.extend(x_split_val)
        val_y_out.extend(y_split_val)
    train_x_out, train_y_out = union_shuffled_copies(train_x_out, train_y_out)
    val_x_out, val_y_out = union_shuffled_copies(val_x_out, val_y_out)
    return train_x_out, val_x_out, train_y_out, val_y_out

class OnlineScaler():
    """ Online mean and variance computation, see
    http://www.cs.yale.edu/publications/techreports/tr222.pdf
    equation 1.5a and 1.5b
    """
    t_1_m = None
    s_1_m = None
    m = 0

    def partial_fit(self, x: torch.Tensor) -> None:
        assert(len(x.shape) > 1), "First dimension to partial_fit must be batch size"
        if self.m == 0:
            self.t_1_m = x[0]
            self.s_1_m = 0.0
            self.m += 1
            if x.shape[0] > 1:
                self.partial_fit(x[1:])
        else:
            n = x.shape[0]
            x_sum = x.sum(0)
            self.s_1_m = self.s_1_m + x.var(0, unbiased=False) * x.shape[0] + (self.m / (n * (self.m + n))) * ((n / self.m) * self.t_1_m - x_sum) ** 2
            self.t_1_m = self.t_1_m + x_sum
            self.m += n

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.t_1_m / self.m, torch.sqrt(self.s_1_m / self.m)

if __name__ == "__main__":
    def count_labels(xs):
        label2count = {}
        for x in xs:
            if x in label2count:
                label2count[x] += 1
            else:
                label2count[x] = 1
        return label2count

    samples = (["0_sample"] * 800000) + (["1_sample"] * 180000) + (["2_sample"] * 20000) + (["3_sample"] * 10000)
    labels = (["0_label"] * 800000) + (["1_label"] * 180000) + (["2_label"] * 20000) + (["3_label   "] * 10000)
    train_x_out, val_x_out, train_y_out, val_y_out = stratified_train_test_split(samples, labels, train_size=0.9)

    print(count_labels(train_x_out))
    print(count_labels(train_y_out))
    print(count_labels(val_x_out))
    print(count_labels(val_y_out))

    a = torch.randn((10000, 10)) * 10000
    scaler = OnlineScaler()
    for el in a:

        scaler.partial_fit(el.unsqueeze(0))
    
    print(a.mean(0), a.std(0))
    print(scaler.get_mean_std())