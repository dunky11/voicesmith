import torch
import numpy as np
from typing import Dict, Iterable, List, Tuple
import numpy as np
from torch.jit._script import ScriptModule
from g2p_en import G2p
import re
import time
from voice_smith.utils.text_normalization import (
    remove_cont_whitespaces,
    EnglishTextNormalizer,
)
from voice_smith.utils.tokenization import (
    WordTokenizer,
    SentenceTokenizer,
    BertTokenizer,
)
from voice_smith.g2p.dp.utils.infer import batched_predict


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
    lang: str,
    talking_speed: float,
    speaker_id: int,
    model_type: str,
    g2p: G2p,
    symbol2id: Dict[str, int],
    lang2id: Dict[str, int],
    text_normalizer: EnglishTextNormalizer,
    acoustic_model: ScriptModule,
    vocoder: ScriptModule,
    device: torch.device,
) -> Tuple[np.ndarray, int]:

    text = text.strip()
    text = remove_cont_whitespaces(text)
    word_tokenizer = WordTokenizer(lang=lang, remove_punct=False)
    sentence_tokenizer = SentenceTokenizer(lang=lang)
    acoustic_model = acoustic_model.to(device)
    start_time = time.time()
    if model_type == "Delighful_FreGANv1_v0.0" or model_type == "0.2.3":
        waves = []
        for sentence in sentence_tokenizer.tokenize(text):
            symbol_ids = []
            # sentence = text_normalizer(sentence)
            for word in word_tokenizer.tokenize(sentence):
                word = word.lower()
                if word.strip() == "":
                    continue
                elif word in [".", "?", "!"]:
                    symbol_ids.append(symbol2id[word])
                elif word in [",", ";"]:
                    symbol_ids.append(symbol2id["SILENCE"])
                else:
                    for phone in batched_predict(g2p, [word], [lang])[0]:
                        symbol_ids.append(symbol2id[phone])
                    symbol_ids.append(symbol2id["BLANK"])

            symbol_ids = torch.tensor([symbol_ids], device=device, dtype=torch.int64)
            speaker_ids = torch.tensor([speaker_id], device=device, dtype=torch.int64)
            lang_ids = torch.tensor([lang2id[lang]], device=device, dtype=torch.int64)

            with torch.no_grad():
                mel = acoustic_model(
                    symbol_ids, speaker_ids, lang_ids, 1.0, talking_speed,
                )
                mel_len = torch.tensor([mel.shape[2]], dtype=torch.int64, device=device)
                wave = vocoder(mel.cpu(), mel_len.cpu())
                waves.append(wave.view(-1))

        wave_cat = torch.cat(waves).cpu().numpy()

    else:
        raise Exception(
            f"Model type '{type}' is not supported for this version of VoiceSmith ..."
        )

    time_elapsed = time.time() - start_time
    audio_generated = wave_cat.shape[0] / 22050
    rtf = audio_generated / time_elapsed
    print(
        f"Generating {round(audio_generated, 2)}s of audio took {round(time_elapsed, 2)}s. RFT: {round(rtf, 2)}",
        flush=True,
    )
    return wave_cat, 22050
