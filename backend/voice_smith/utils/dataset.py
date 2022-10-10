import json
import math
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from scipy.stats import betabinom
from voice_smith.config.langs import lang2id
from voice_smith.utils.tools import pad_1D, pad_2D, pad_3D
from voice_smith.config.configs import PreprocessingConfig


def is_phone(string: str) -> bool:
    return len(string) > 1 and string[0] == "@"


class AcousticDataset(Dataset):
    def __init__(
        self,
        filename: str,
        batch_size: int,
        data_path: str,
        assets_path: str,
        is_eval: bool,
        sort: bool = False,
        drop_last: bool = False,
    ):
        self.preprocessed_path = Path(data_path)
        self.batch_size = batch_size
        self.basename, self.speaker = self.process_meta(filename)
        with open(self.preprocessed_path / "speakers.json", encoding="utf-8") as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.is_eval = is_eval

    def __len__(self) -> int:
        return len(self.basename)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        basename = self.basename[idx]
        speaker_name = self.speaker[idx]
        speaker_id = self.speaker_map[speaker_name]
        data = torch.load(
            self.preprocessed_path / "data" / speaker_name / f"{basename}.pt"
        )
        raw_text = data["raw_text"]
        mel = data["mel"]
        pitch = data["pitch"]
        lang = data["lang"]
        phone = torch.LongTensor(data["phones"])
        

        if mel.shape[1] < 20:
            print(
                "Skipping small sample due to the mel-spectrogram containing less than 20 frames"
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        sample = {
            "id": basename,
            "speaker_name": speaker_name,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "lang": lang2id[lang],
            "attn_prior": attn_prior,
        }

        if phone.shape[0] >= mel.shape[1]:
            print(
                "Text is longer than mel, will be skipped due to monotonic alignment search ..."
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        if self.is_eval:
            data = torch.load(
                self.preprocessed_path / "wav" / speaker_name / f"{basename}.pt"
            )
            sample["wav"] = data["wav"].unsqueeze(0)

        return sample

    def process_meta(self, filename: str) -> Tuple[List[str], List[str]]:
        with open(self.preprocessed_path / filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker

    def beta_binomial_prior_distribution(
        self, phoneme_count, mel_count, scaling_factor=1.0
    ):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M + 1):
            a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def reprocess(self, data: List[Dict[str, Any]], idxs: List[int]):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        speaker_names = [data[idx]["speaker_name"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        langs = np.array([data[idx]["lang"] for idx in idxs])
        attn_priors = [data[idx]["attn_prior"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[1] for mel in mels])

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))

        speakers = np.repeat(
            np.expand_dims(np.array(speakers), axis=1), texts.shape[1], axis=1
        )
        langs = np.repeat(
            np.expand_dims(np.array(langs), axis=1), texts.shape[1], axis=1
        )

        if self.is_eval:
            wavs = [data[idx]["wav"] for idx in idxs]
            wavs = pad_2D(wavs)
            return (
                ids,
                raw_texts,
                speakers,
                speaker_names,
                texts,
                text_lens,
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
                text_lens,
                mels,
                pitches,
                mel_lens,
                langs,
                attn_priors,
            )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output


class CodecDataset(Dataset):
    def __init__(self, file_paths: List[Path], speaker_map: Dict[str, int]):
        self.file_paths = file_paths
        self.samples_per_seg = 8192
        self.speaker_map = speaker_map

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]
        speaker_id = self.speaker_map[file_path.parent.name]
        audio = torch.load(file_path)
        audio = torch.clamp(audio, -1.0, 1.0)

        if audio.shape[0] < self.samples_per_seg:
            audio = F.pad(audio, (0, self.samples_per_seg - audio.shape[0]), "constant")

        from_sample = random.randint(0, audio.shape[0] - self.samples_per_seg)
        audio = audio[from_sample : from_sample + self.samples_per_seg]
        return audio, speaker_id

    def __len__(self) -> int:
        return len(self.file_paths)

    def get_sample_to_synth(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rand_idx = np.random.randint(0, self.__len__())
        file_path = self.file_paths[rand_idx]
        speaker_id = self.speaker_map[file_path.parent.name]
        audio = torch.load(file_path)
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio, speaker_id

    def collate_fn(self, datas):
        audios = [data[0] for data in datas]
        speaker_ids = [data[1] for data in datas]
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
        audios = torch.tensor(pad_1D(audios), dtype=torch.float32)
        return audios, speaker_ids


class VocoderDataset(Dataset):
    def __init__(
        self,
        filename: str,
        fine_tuning: bool,
        preprocess_config: PreprocessingConfig,
        preprocessed_path: str,
        segment_size: int,
    ):
        self.preprocessed_path = Path(preprocessed_path)
        self.basename, self.speaker = self.process_meta(filename)
        self.fine_tuning = fine_tuning
        self.segment_size = segment_size
        self.hop_size = preprocess_config.stft.hop_length
        self.frames_per_seg = math.ceil(self.segment_size / self.hop_size)
        with open(self.preprocessed_path / "speakers.json") as f:
            self.speaker_map = json.load(f)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = np.array([self.speaker_map[speaker]])

        try:
            data_path = (
                self.preprocessed_path
                / ("data_gta" if self.fine_tuning else "data")
                / speaker
                / f"{basename}.pt"
            )
            wav_path = self.preprocessed_path / "wav" / speaker / f"{basename}.pt"
            data = torch.load(data_path)
            audio_data = torch.load(wav_path)
            audio = audio_data["wav"]
            mel = data["mel"]

        except Exception as e:
            print(e)
            print("Couldn't find gta, using another file ...")
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        if self.segment_size != -1:
            if audio.shape[0] < self.segment_size:
                audio = F.pad(
                    audio, (0, self.segment_size - audio.shape[0]), "constant"
                )

            if mel.shape[1] < self.frames_per_seg:
                mel = F.pad(mel, (0, self.frames_per_seg - mel.shape[1]), "constant")

            from_frame = random.randint(0, mel.shape[1] - self.frames_per_seg)
            # Skip last frame, otherwise errors are thrown, find out why
            if from_frame > 0:
                from_frame -= 1
            till_frame = from_frame + self.frames_per_seg
            mel = mel[:, from_frame:till_frame]
            audio = audio[from_frame * self.hop_size : till_frame * self.hop_size]
        return mel, audio, speaker_id

    def __len__(self) -> int:
        return len(self.basename)

    def get_sample_to_synth(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rand_idx = np.random.randint(0, self.__len__())
        basename = self.basename[rand_idx]
        speaker = self.speaker[rand_idx]
        speaker_id = np.array([self.speaker_map[speaker]])

        try:
            data_path = (
                self.preprocessed_path
                / ("data_gta" if self.fine_tuning else "data")
                / speaker
                / f"{basename}.pt"
            )
            wav_path = self.preprocessed_path / "wav" / speaker / f"{basename}.pt"
            data = torch.load(data_path)
            audio_data = torch.load(wav_path)
            audio = audio_data["wav"]
            mel = data["mel"]
            return mel, audio, speaker_id
        except Exception as e:
            print("Couldn't find gta, using another file ...")
            return self.get_sample_to_synth()

    def process_meta(self, filename: str) -> Tuple[List[str], List[str]]:
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker

    def collate_fn(self, datas):
        mels = [data[0] for data in datas]
        audios = [data[1] for data in datas]
        speaker_ids = [data[2][0] for data in datas]
        mel_lens = torch.tensor([mel.shape[1] for mel in mels], dtype=torch.int64)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)
        audios = torch.tensor(pad_1D(audios), dtype=torch.float32)
        mels = torch.tensor(pad_2D(mels), dtype=torch.float32)
        return mels, audios, speaker_ids, mel_lens
