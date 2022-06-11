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
from voice_smith.utils.tokenization import BertTokenizer
from voice_smith.utils.text import phones_to_token_ids
from voice_smith.utils.tools import pad_1D, pad_2D
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
        sort: bool = False,
        drop_last: bool = False,
    ):
        self.tokenizer = BertTokenizer(assets_path)
        self.preprocessed_path = Path(data_path)
        self.batch_size = batch_size
        self.basename, self.speaker = self.process_meta(filename)
        with open(self.preprocessed_path / "speakers.json") as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

        with open(self.preprocessed_path / "stats.json") as f:
            stats = json.load(f)
            self.pitch_max, self.pitch_mean, self.pitch_std = (
                stats["pitch"][1],
                stats["pitch"][2],
                stats["pitch"][3],
            )

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
        durations = data["durations"]
        phone = torch.LongTensor(phones_to_token_ids(data["phones"]))

        wav_path = self.preprocessed_path / "wav" / speaker_name / f"{basename}.pt"
        audio = torch.load(wav_path)["wav"]

        sample = {
            "id": basename,
            "speaker_name": speaker_name,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "duration": durations,
            "audio": audio,
        }

        if mel.shape[1] < 64:
            print(
                "Skipping small sample due to the mel-spectrogram containing less than 64 frames"
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

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

    def reprocess(self, data: List[Dict[str, Any]], idxs: List[int]):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        speaker_names = [data[idx]["speaker_name"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        audios = [data[idx]["audio"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[1] for mel in mels])
        encoding = self.tokenizer(raw_texts)
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        durations = pad_1D(durations)
        audios = pad_1D(audios)

        return (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            text_lens,
            mels,
            pitches,
            durations,
            mel_lens,
            encoding["input_ids"],
            encoding["attention_mask"],
            audios,
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
        speaker_id = torch.LongTensor([self.speaker_map[speaker]])

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

        if audio.shape[0] < self.segment_size:
            audio = F.pad(audio, (0, self.segment_size - audio.shape[0]), "constant")

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
        speaker_id = torch.LongTensor([self.speaker_map[speaker]])

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
