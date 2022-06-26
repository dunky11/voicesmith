from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple


def parse_dictionary(
    dictionary_path: str, name: str
) -> Tuple[List[Tuple[str, str, List[str]]], List[str], List[str]]:
    out = []
    all_phones = []
    words_preprocessed = {}
    with open(str(dictionary_path), "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split()
            word = line[0]
            phones = line[1 + 4 :]
            word = word.lower()
            words_preprocessed[word] = 0
            out.append((name, word, phones))
            all_phones.extend(phones)
    unique_phones = list(set(all_phones))
    text_symbols = list(set("".join(words_preprocessed)))
    return out, unique_phones, text_symbols
