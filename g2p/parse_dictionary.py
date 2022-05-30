from pathlib import Path
from tqdm import tqdm

def parse_dictionary(dictionary_name, name):
    this_path = Path(__file__).parent.resolve()
    out = []
    all_phones = []
    words_preprocessed = {}
    with open(this_path / "dictionaries" /  dictionary_name, "r", encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split()
            word = line[0]
            phones = line[1:]
            word = word.lower()
            words_preprocessed[word] = 0
            out.append((name, word, phones))
            all_phones.extend(phones)
    unique_phones = list(set(all_phones))
    text_symbols = list(set("".join(words_preprocessed)))
    return out, unique_phones, text_symbols