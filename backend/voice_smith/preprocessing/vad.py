import torch
from pathlib import Path
from typing import List
from tqdm import tqdm
from joblib import Parallel, delayed
from voice_smith.utils.audio import safe_load, resample, save_audio


def remove_silence(
    in_paths: List[str],
    out_paths: List[str],
    before_silence_sec: float = 0.0,
    after_silence_sec: float = 0.0,
    vad_sr=16000,
) -> None:
    assert len(in_paths) == len(out_paths)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )

    (get_speech_timestamps, _, _, _, _) = utils
    for in_path, out_path in zip(in_paths, out_paths):
        wav, sr_orig = safe_load(in_path, sr=None)
        wav_16k = resample(wav, orig_sr=sr_orig, target_sr=vad_sr)
        tstamps = get_speech_timestamps(
            torch.from_numpy(wav_16k), model, sampling_rate=vad_sr, return_seconds=True
        )
        if len(tstamps) > 0:
            start = max(tstamps[0]["start"] - before_silence_sec, 0)
            end = tstamps[-1]["end"] + after_silence_sec
        else:
            start = 0
            end = -1
        wav = wav[int(start * sr_orig) : int(end * sr_orig if end != -1 else end)]
        Path(out_path).parent.mkdir(exist_ok=True, parents=True)
        save_audio(out_path, torch.from_numpy(wav), sr_orig)


def batched_remove_silence(
    in_paths: List[str],
    out_paths: List[str],
    workers: int,
    before_silence_sec: float = 0.0,
    after_silence_sec: float = 0.0,
    vad_sr=16000,
    chunk_size=500,
):
    Parallel(n_jobs=workers)(
        delayed(remove_silence)(
            in_paths[chunk_start_idx : chunk_start_idx + chunk_size],
            out_paths[chunk_start_idx : chunk_start_idx + chunk_size],
            before_silence_sec,
            after_silence_sec,
            vad_sr,
        )
        for chunk_start_idx in tqdm(range(0, len(in_paths), chunk_size))
    )
