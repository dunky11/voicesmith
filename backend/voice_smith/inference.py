import torch
import numpy as np
from typing import Dict, Iterable, List, Tuple
import numpy as np
from torch.jit._script import ScriptModule
from g2p_en import G2p
import re
from voice_smith.utils.text_normalization import (
    remove_cont_whitespaces,
    EnglishTextNormalizer,
)
from voice_smith.utils.tokenization import (
    WordTokenizer,
    SentenceTokenizer,
    BertTokenizer,
)


def strip_invalid_symbols(text: str, pad_symbol: str, valid_symbols: List[str]) -> str:
    new_token = ""
    for char in text.lower():
        if char in valid_symbols and char != pad_symbol:
            new_token += char
    return new_token


def split_phone_marker(text: str) -> List[str]:
    splits = re.finditer(r"{{(([^}]+}?)*)}}", text)
    out = []
    start = 0
    for split in splits:
        non_arpa = text[start : split.start()]
        arpa = text[split.start() + 2 : split.end() - 2]
        if len(non_arpa) > 0:
            out.append(non_arpa)
        out.append(arpa)
        start = split.end()
    if start < len(text):
        out.append(text[start:])
    return out


def split_context_marker(text: str) -> Iterable[Tuple[str, str]]:
    splits = re.finditer(r"\[\[(([^\]]+\|\|[^\]]+)*)\]\]", text)
    sentences, contexts = [], []
    start = 0
    for split in splits:
        non_context = text[start : split.start()]
        matched_part = text[split.start() + 2 : split.end() - 2].lstrip()
        if len(non_context) > 0:
            sentences.append(non_context)
            contexts.append(non_context)
        s = matched_part.split("||")
        sentence, context = s[0], s[1]
        sentences.append(sentence)
        contexts.append(context)
        start = split.end()
    if start < len(text):
        sentences.append(text[start:])
        contexts.append(text[start:])
    return zip(sentences, contexts)


def is_phone_marker(text: str) -> bool:
    if len(text) < 5:
        return False
    return text[:2] == "{{" and text[-2:] == "}}"


def is_context_marker(text: str) -> bool:
    if len(text) < 7:
        return False
    return text[:2] == "[[" and text[-2:] == "]]" and "||" in text[2:-2]


def is_phone(text: str) -> bool:
    return len(text) > 1 and text[0] == "@"


def synthesize(
    text: str,
    talking_speed: float,
    speaker_id: int,
    model_type: str,
    g2p: G2p,
    symbol2id: Dict[str, int],
    lexicon: Dict[str, List[str]],
    text_normalizer: EnglishTextNormalizer,
    bert_tokenizer: BertTokenizer,
    acoustic_model: ScriptModule,
    style_predictor: ScriptModule,
    vocoder: ScriptModule,
) -> Tuple[np.ndarray, int]:
    text = text.strip()
    text = remove_cont_whitespaces(text)
    word_tokenizer = WordTokenizer(lang="en", remove_punct=False)
    sentence_tokenizer = SentenceTokenizer(lang="en")
    if model_type == "Delighful_FreGANv1_v0.0":
        waves = []
        for sentence in sentence_tokenizer.tokenize(text):
            style_sentences = []
            symbol_ids = []
            for subsentence, context in split_context_marker(sentence):
                for subsubsentence in split_phone_marker(subsentence):
                    if is_phone_marker(subsubsentence):
                        for phone in subsubsentence.strip().split(" "):
                            if f"{phone}" in symbol2id:
                                symbol_ids.append(symbol2id[f"{phone}"])
                            style_sentences.append(context)
                    else:
                        subsubsentence = text_normalizer(subsubsentence)
                        style_sentences.append(text_normalizer(context))
                        for word in word_tokenizer.tokenize(subsubsentence):
                            word = word.lower()
                            if word.strip() == "":
                                continue
                            elif word in [".", "?", "!"]:
                                symbol_ids.append(symbol2id[word])
                            elif word in [",", ";"]:
                                symbol_ids.append(symbol2id["SILENCE"])
                            elif word in lexicon:
                                for phone in lexicon[word]:
                                    symbol_ids.append(symbol2id[phone])
                                symbol_ids.append(symbol2id["BLANK"])
                            else:
                                for phone in g2p(word):
                                    symbol_ids.append(symbol2id[phone])
                                symbol_ids.append(symbol2id["BLANK"])

            sentence_style = " ".join(style_sentences)

            encoding = bert_tokenizer([sentence_style])

            symbol_ids = torch.LongTensor([symbol_ids])
            speaker_ids = torch.LongTensor([speaker_id])

            with torch.no_grad():
                style_embeds = style_predictor(
                    encoding["input_ids"], encoding["attention_mask"]
                )
                mel = acoustic_model(
                    symbol_ids,
                    speaker_ids,
                    style_embeds,
                    1.0,
                    talking_speed,
                )
                wave = vocoder(mel)
                waves.append(wave.view(-1))

        wave_cat = torch.cat(waves).numpy()
        return wave_cat, 22050

    else:
        raise Exception(
            f"Model type '{type}' is not supported for this version of VoiceSmith ..."
        )
