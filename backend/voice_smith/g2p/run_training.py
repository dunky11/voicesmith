from pathlib import Path
import random
import argparse
from voice_smith.g2p.dp.preprocess import preprocess
from voice_smith.g2p.dp.train import train
from voice_smith.g2p.dp.utils.io import read_config
from voice_smith.g2p.parse_dictionary import parse_dictionary
import json


perform_benchmark = False

name = "G2P Byte MFA training 6x6 transformer (384, 1536), [15 MFA langs]"

if perform_benchmark:
    SPLIT_SIZE = 12753
else:
    SPLIT_SIZE = 5000

if __name__ == "__main__":
    if perform_benchmark:
        print("Benchmarking on CMUDict ...")
    else:
        print("Training model for production ...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    data, phones, text_symbols = [], [], []
    lang_to_word_to_gold = {}
    config = read_config(Path(".") / "dp" / "configs" / "forward_config.yaml")
    if args.checkpoint is None:
        for dictionary_path in [
            Path(".") / "dictionaries" / "en" / "english_us_mfa.dict",
        ]:
            lang = dictionary_path.parent.name
            d, p, t, word_to_gold = parse_dictionary(dictionary_path, lang)
            data.extend(d)
            phones.extend(p)
            text_symbols.extend(t)
            lang_to_word_to_gold[lang] = word_to_gold

        phones = list(set(phones))
        text_symbols = list(str(el) for el in range(256)) + ["<BLANK>"]

        config["preprocessing"]["phoneme_symbols"] = phones
        config["preprocessing"]["text_symbols"] = text_symbols
        if not perform_benchmark:
            random.shuffle(data)
        train_data, val_data = data[SPLIT_SIZE:], data[:SPLIT_SIZE]

        preprocess(
            config=config,
            train_data=train_data,
            val_data=val_data,
            deduplicate_train_data=False,
        )

        with open(
            Path(".") / "datasets" / "lang_to_word_to_gold.json", "w", encoding="utf-8"
        ) as f:
            f.write(json.dumps(lang_to_word_to_gold))

    else:
        with open(
            Path(".") / "datasets" / "lang_to_word_to_gold.json", "r", encoding="utf-8"
        ) as f:
            lang_to_word_to_gold = json.load(f)

    train(
        config=config,
        checkpoint_file=args.checkpoint,
        name=name,
        lang_to_word_to_gold=lang_to_word_to_gold,
    )
