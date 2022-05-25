import torch
from pathlib import Path
from joblib import Parallel, delayed
import multiprocessing
import json
import numpy as np
import tgt
from typing import Dict, Any, Callable, Optional, List, Tuple, Union
from voice_smith.config.symbols import symbol2id
from voice_smith.utils.tools import (
    iter_logger,
    OnlineScaler,
    stratified_train_test_split,
)
from voice_smith.utils.audio import (
    TacotronSTFT,
    get_mel_from_wav,
    compute_yin,
    norm_interp_f0,
    safe_load,
    resample,
)
from voice_smith.utils.tools import warnings_to_stdout

warnings_to_stdout()


def get_lexicon(assets_path) -> Dict[str, List[str]]:
    with open(Path(assets_path) / "lexicon.json", "r", encoding="utf-8") as f:
        lexicon = json.load(f)
    return lexicon


def get_alignment(
    phones_tier, words_tier, text: str, sampling_rate: int, hop_length: int
) -> Tuple[List[str], List[int], float, float]:
    phones = []
    durations = []
    start_time = 0
    end_time = 0
    samples_processed = 0
    words_idx = 0
    next_word_end = (
        words_tier[words_idx].end_time if len(words_tier) > words_idx else -1
    )
    for i, t in enumerate(phones_tier._objects):
        s, e, p = t.start_time, t.end_time, t.text
        if phones == []:
            start_time = s
        else:
            diff = s - end_time
            if diff > 0:
                sil_phone = "SILENCE"
                phones.append(sil_phone)
                best_samples_after = (s - start_time) * sampling_rate
                samples_after_append = samples_processed + diff * sampling_rate
                if samples_after_append >= best_samples_after:
                    duration = int(np.floor((diff * sampling_rate) / hop_length))
                else:
                    duration = int(np.ceil((diff * sampling_rate) / hop_length))
                durations.append(duration)
                samples_processed += duration * hop_length

        phones.append(p)
        if e == next_word_end:
            phones.append("BLANK")
            durations.append(0)
            words_idx += 1
            next_word_end = (
                words_tier[words_idx].end_time if len(words_tier) > words_idx else -1
            )
        diff = e - s
        best_samples_after = (e - start_time) * sampling_rate
        samples_after_append = samples_processed + diff * sampling_rate
        if samples_after_append >= best_samples_after:
            duration = int(np.floor((diff * sampling_rate) / hop_length))
        else:
            duration = int(np.ceil((diff * sampling_rate) / hop_length))
        durations.append(duration)
        samples_processed += duration * hop_length
        end_time = e

    if text[-1] in [".", "?", "!"]:
        phones.append(text[-1])
        durations.append(0)

    return phones, durations, start_time, end_time


def process_utterance(
    in_dir: str,
    out_dir: str,
    speaker: str,
    basename: str,
    sampling_rate: int,
    filter_length: int,
    hop_length: int,
    to_mel: torch.nn.Module,
    min_seconds: float,
    max_seconds: float,
    normalize_loudness: bool,
    ignore_below_hz: Union[int, None],
) -> Union[None, Tuple[str, int]]:
    audio_path = Path(in_dir) / speaker / f"{basename}.wav"
    text_path = Path(in_dir) / speaker / f"{basename}.txt"
    tg_path = Path(out_dir) / "textgrid" / speaker / f"{basename}.TextGrid"

    min_samples = int(sampling_rate * min_seconds)
    max_samples = int(sampling_rate * max_seconds)

    if not audio_path.exists():
        print(f"File {audio_path} does not exist, skipping ...")
        return

    if not text_path.exists():
        print(f"File {text_path} does not exist, skipping ...")
        return

    # Sometimes files cannot be aligned my MFA
    if not tg_path.exists():
        return

    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.readline().strip()

    textgrid = tgt.io.read_textgrid(tg_path)
    phones_tier = textgrid.get_tier_by_name("phones")
    words_tier = textgrid.get_tier_by_name("words")
    phones, durations, start, end = get_alignment(
        phones_tier=phones_tier,
        words_tier=words_tier,
        text=raw_text,
        sampling_rate=sampling_rate,
        hop_length=hop_length,
    )

    if start >= end:
        return

    wav, sr = safe_load(str(audio_path), sr=None)

    if not isinstance(wav, np.ndarray):
        return

    if ignore_below_hz != None and sr < ignore_below_hz:
        return

    if sr != sampling_rate:
        wav = resample(wav, orig_sr=sr, target_sr=sampling_rate)

    wav = wav[
        int(np.round(sampling_rate * start)) : int(np.round(sampling_rate * end))
    ].astype(np.float32)

    if wav.shape[0] < min_samples or wav.shape[0] > max_samples:
        return

    if normalize_loudness:
        wav = wav / np.max(np.abs(wav))

    pitch, _, _, _ = compute_yin(
        wav,
        sr=sampling_rate,
        w_len=filter_length,
        w_step=hop_length,
        f0_min=50,
        f0_max=1000,
        harmo_thresh=0.25,
    )
    if np.sum(pitch != 0) <= 1:
        return

    mel_spectrogram = get_mel_from_wav(wav, to_mel)

    # TODO this shouldnt be necessary, currently pitch sometimes has 1 less frame than spectrogram,
    # We should find out why
    mel_spectrogram = mel_spectrogram[:, : pitch.shape[0]]

    if sum(durations) > mel_spectrogram.shape[1]:
        for i in range(len(durations) - 1, -1, -1):
            if durations[i] != 0:
                durations[i] -= 1
                break

    pitch, _ = norm_interp_f0(pitch)

    assert pitch.shape[0] == sum(durations) == mel_spectrogram.shape[1], (
        pitch.shape,
        sum(durations),
        mel_spectrogram.shape[1],
    )

    # Average per phoneme
    averaged_pitch = np.zeros(len(durations))
    pos = 0
    for i, d in enumerate(durations):
        if d > 0:
            averaged_pitch[i] = np.mean(pitch[pos : pos + d])
        else:
            averaged_pitch[i] = 0
        pos += d

    assert averaged_pitch.shape[0] == len(durations) == len(phones)

    # Save files
    torch.save(
        {
            "mel": torch.from_numpy(mel_spectrogram),
            "pitch": torch.from_numpy(averaged_pitch),
            "phones": phones,
            "raw_text": raw_text,
            "durations": torch.LongTensor(durations),
        },
        Path(out_dir) / "data" / speaker / f"{basename}.pt",
    )

    torch.save(
        {"wav": torch.from_numpy(wav).float(),},
        Path(out_dir) / "wav" / speaker / f"{basename}.pt",
    )

    return (
        "|".join([basename, speaker]),
        mel_spectrogram.shape[1],
    )


def extract_data(
    db_id: int,
    table_name: str,
    training_run_name: str,
    preprocess_config: Dict[str, Any],
    get_logger: Optional[Callable],
    training_runs_path: str,
    assets_path: str,
    log_every: int = 200,
    ignore_below_hz: Union[int, None] = None,
) -> None:
    workers = preprocess_config["workers"]
    print(f"Extracting data with {workers} workers ...")

    in_dir = Path(training_runs_path) / str(training_run_name) / "raw_data"
    out_dir = Path(training_runs_path) / str(training_run_name) / "data"

    min_seconds = preprocess_config["min_seconds"]
    max_seconds = preprocess_config["max_seconds"]
    use_audio_normalization = preprocess_config["use_audio_normalization"]

    hop_length = preprocess_config["stft"]["hop_length"]

    sampling_rate = preprocess_config["sampling_rate"]

    filter_length = preprocess_config["stft"]["filter_length"]
    to_mel = TacotronSTFT(
        filter_length=preprocess_config["stft"]["filter_length"],
        hop_length=preprocess_config["stft"]["hop_length"],
        win_length=preprocess_config["stft"]["win_length"],
        n_mel_channels=preprocess_config["mel"]["n_mel_channels"],
        sampling_rate=sampling_rate,
        mel_fmin=preprocess_config["mel"]["mel_fmin"],
        mel_fmax=preprocess_config["mel"]["mel_fmax"],
        center=False,
        device=torch.device("cpu"),
    )

    out = []
    speaker_names = []
    n_frames_total = 0
    speakers = {speaker.name: i for i, speaker in enumerate(in_dir.iterdir())}

    for speaker_path in in_dir.iterdir():
        (out_dir / "data" / speaker_path.name).mkdir(exist_ok=True, parents=True)
        (out_dir / "wav" / speaker_path.name).mkdir(exist_ok=True, parents=True)

    wav_paths = list(in_dir.glob("*/*.wav"))

    def callback(index: int):
        if index % log_every == 0:
            logger = get_logger()
            progress = index / len(wav_paths) / 2
            logger.query(
                f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
                [progress, db_id],
            )

    rets = Parallel(n_jobs=workers)(
        delayed(process_utterance)(
            in_dir=in_dir,
            out_dir=out_dir,
            speaker=str(wav_path.parent.name),
            basename=wav_path.stem,
            sampling_rate=sampling_rate,
            filter_length=filter_length,
            hop_length=hop_length,
            to_mel=to_mel,
            ignore_below_hz=ignore_below_hz,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            normalize_loudness=use_audio_normalization,
        )
        for wav_path in iter_logger(
            wav_paths, total=len(wav_paths), cb=callback, print_every=log_every
        )
    )

    for ret in rets:
        if ret is None:
            continue
        line, n_frames = ret
        n_frames_total += n_frames
        out.append(line)
        speaker_names.append(line.split("|")[1])

    print("Calculating pitch stats")
    pitch_mean, pitch_std = calculate_pitch_stats(
        table_name=table_name,
        db_id=db_id,
        out_dir=str(out_dir),
        get_logger=get_logger,
        workers=workers,
    )

    print("Normalizing pitch")
    pitch_min, pitch_max = normalize_pitch(
        table_name=table_name,
        db_id=db_id,
        out_dir=str(out_dir),
        mean=pitch_mean,
        std=pitch_std,
        get_logger=get_logger,
        log_every=log_every,
        workers=workers,
    )

    print("Saving stats ... ")
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        stats = {
            "pitch": [
                float(pitch_min),
                float(pitch_max),
                float(pitch_mean),
                float(pitch_std),
            ],
        }
        f.write(json.dumps(stats))

    print("Saving necessary files ... ")
    # Save files
    with open(out_dir / "speakers.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(speakers))

    with open(out_dir / "symbol2id.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(symbol2id))

    with open(out_dir / "lexicon.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(get_lexicon(assets_path)))

    print(
        "Total time: {} hours".format(
            n_frames_total * hop_length / sampling_rate / 3600
        )
    )

    logger = get_logger()
    logger.query(
        f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
        [0.925, db_id],
    )

    print("Creating train and validation splits ... ")
    x_train, x_val, _, _ = stratified_train_test_split(
        x=out, y=speaker_names, train_size=1.0 - preprocess_config["val_size"]
    )

    logger.query(
        f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
        [0.975, db_id],
    )

    print("Creating train.txt ... ")
    # Write metadata
    with open(out_dir / "train.txt", "w", encoding="utf-8") as f:
        for m in x_train:
            f.write(m + "\n")

    print("Creating val.txt ... ")
    with open(out_dir / "val.txt", "w", encoding="utf-8") as f:
        for m in x_val:
            f.write(m + "\n")

    logger.query(
        f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
        [1.0, db_id],
    )


def calculate_pitch_stats(
    table_name: str,
    db_id: int,
    out_dir: str,
    get_logger: Optional[Callable],
    workers: int,
    chunk_size: int = 10000,
) -> Tuple[float, float]:
    scaler = OnlineScaler()

    def _get_pitch(path):
        data = torch.load(path)
        return data["pitch"].numpy()

    files = list((Path(out_dir) / "data").glob("*/*"))

    def callback(index):
        if index % 1 == 0:
            logger = get_logger()
            progress = index / (len(files) // chunk_size + 1) / 5 + 0.5
            logger.query(
                f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
                [progress, db_id],
            )

    for start_idx in iter_logger(
        range(0, len(files), chunk_size),
        total=(len(files) // chunk_size + 1),
        cb=callback,
    ):
        files_slice = files[start_idx : start_idx + chunk_size]
        rets = Parallel(n_jobs=workers)(
            delayed(_get_pitch)(path) for path in files_slice
        )
        for pitch in rets:
            scaler.partial_fit(torch.from_numpy(pitch.reshape(-1, 1)))

    pitch_mean, pitch_std = scaler.get_mean_std()
    return pitch_mean.numpy(), pitch_std.numpy()


def normalize_pitch(
    db_id: int,
    table_name: str,
    out_dir: str,
    mean: float,
    std: float,
    get_logger: Optional[Callable],
    workers: int,
    log_every: int = 200,
) -> Tuple[float, float]:
    paths = list((Path(out_dir) / "data").glob("*/*.pt"))

    def _normalize(path):
        data = torch.load(path)
        data["pitch"] = (data["pitch"] - mean) / std
        min_value = min(data["pitch"])
        max_value = max(data["pitch"])
        torch.save(data, path)
        return min_value, max_value

    def callback(index):
        if get_logger is None:
            return
        if index % log_every == 0:
            logger = get_logger()
            progress = index / len(paths) / 5 + 0.7
            logger.query(
                f"UPDATE {table_name} SET preprocessing_extract_data_progress=? WHERE id=?",
                [progress, db_id],
            )

    rets = Parallel(n_jobs=workers)(
        delayed(_normalize)(path)
        for path in iter_logger(paths, total=len(paths), cb=callback)
    )
    min_value = min([el[0] for el in rets])
    max_value = max([el[1] for el in rets])
    return min_value, max_value


if __name__ == "__main__":

    class NoLogger:
        def query(self, a, b):
            pass

    from voice_smith.config.preprocess_config import preprocess_config

    def get_logger():
        return NoLogger()

    extract_data(
        db_id=None,
        table_name=None,
        training_run_name="pretraining_two_stage_ac",
        get_logger=get_logger,
        ignore_below_hz=22050,
        preprocess_config=preprocess_config,
    )
