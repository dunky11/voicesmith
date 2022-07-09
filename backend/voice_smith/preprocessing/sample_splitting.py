from dataclasses import dataclass
from typing import List, Union, Literal, Any
import sqlite3
import tgt
from pathlib import Path
import torch
from voice_smith.utils.tokenization import SentenceTokenizer, WordTokenizer
from voice_smith.utils.audio import safe_load, save_audio


@dataclass
class Split:
    text: str
    from_msecs: float
    to_msecs: float


@dataclass
class SampleSplit:
    sample_id: int
    text: str
    lang: str
    splits: List[Split]


def flatten(nested_list: List[List[Any]]):
    return [x for sublist in nested_list for x in sublist]


def get_splits(sentences_word, sentences_full, words_tier):
    assert len(sentences_word) == len(sentences_full)
    word_idx = 0
    splits: List[Split] = []
    end_time = None

    if len(words_tier) != len(flatten(sentences_word)):
        # The tokenizers used by MFA and us may be out of sync
        return splits

    word_idx = 0
    for i, (sentence_word, sentence_full) in enumerate(
        zip(sentences_word, sentences_full)
    ):
        if i == 0:
            start_time = 0.0
        else:
            start_time = words_tier[word_idx].start_time
        if i == len(sentences_word) - 1:
            end_time = words_tier.end_time
        else:
            end_time = words_tier[word_idx + len(sentence_word) - 1].end_time
        word_idx += len(sentence_word)
        splits.append(
            Split(text=sentence_full, from_msecs=start_time, to_msecs=end_time)
        )
    return splits


def sample_splitting(
    ids: List[int],
    texts: List[str],
    textgrid_paths: List[str],
    languages: List[Union[Literal["en"]]],
) -> List[Split]:
    assert len(ids) == len(texts) == len(textgrid_paths) == len(languages)
    lang_to_info = {}

    sample_splits: List[SampleSplit] = []

    for sample_id, text, textgrid_path, lang in zip(
        ids, texts, textgrid_paths, languages
    ):
        info = (sample_id, text, textgrid_path)
        if lang not in lang_to_info:
            lang_to_info[lang] = [info]
        else:
            lang_to_info[lang].append(info)

    for lang, infos in lang_to_info.items():
        word_tokenizer = WordTokenizer(lang, remove_punct=True)
        sentence_tokenizer = SentenceTokenizer(lang)

        for sample_id, text, textgrid_path in infos:
            if not Path(textgrid_path).exists():
                continue

            sentences = sentence_tokenizer.tokenize(text)
            if len(sentences) == 1:
                continue

            sentences_words = [word_tokenizer.tokenize(sent) for sent in sentences]
            textgrid = tgt.io.read_textgrid(textgrid_path)
            words_tier = textgrid.get_tier_by_name("words")

            splits = get_splits(
                sentences_word=sentences_words,
                sentences_full=sentences,
                words_tier=words_tier,
            )

            if len(splits) >= 1:
                sample_splits.append(
                    SampleSplit(
                        sample_id=sample_id, text=text, lang=lang, splits=splits
                    )
                )

    return sample_splits


def split_sample(
    sample_split: SampleSplit, audio_path: str, out_dir: str, sample_split_id: int
):
    audio, sr = safe_load(audio_path, sr=None)
    for i, split in enumerate(sample_split.splits):
        audio_split = audio[int(split.from_msecs * sr) : int(split.to_msecs * sr)]
        file_path = Path(out_dir) / f"{sample_split_id}_split_{i}.flac"
        save_audio(file_path=file_path, audio=torch.FloatTensor(audio_split), sr=sr)
