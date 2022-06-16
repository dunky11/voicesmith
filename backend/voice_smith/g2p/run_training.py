from pathlib import Path
from voice_smith.g2p.dp.preprocess import preprocess
from voice_smith.g2p.dp.train import train
from voice_smith.g2p.dp.utils.io import read_config
from parse_dictionary import parse_dictionary
import random
import argparse

perform_benchmark = False

name = "G2p ARPA training 5x5 transformer (384, 1536), [en], bug fixed"

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

    """
    for dictionary_path, lang in [
        (Path(".") / "dictionaries" / "en" / "english_us_mfa.dict", "en"),
    ]:
    """

    for dictionary_path, lang in [
        (Path(".") / "dictionaries" / "de" / "german_mfa.dict", "de"),
        (Path(".") / "dictionaries" / "en" / "english_us_mfa.dict", "en"),
        (Path(".") / "dictionaries" / "es" / "spanish_mfa.dict", "es"),
        (Path(".") / "dictionaries" / "fr" / "french_mfa.dict", "fr"),
        (Path(".") / "dictionaries" / "ru" / "russian_mfa.dict", "ru"),
    ]:
        d, p, t = parse_dictionary(dictionary_path, lang)
        data.extend(d)
        phones.extend(p)
        text_symbols.extend(t)

    phones = list(set(phones))
    text_symbols = list(set(text_symbols))

    print(phones)
    print(text_symbols)
    print(data[:100])

    config = read_config(Path(".") / "dp" / "configs" / "autoreg_config.yaml")
    config["preprocessing"]["phoneme_symbols"] = phones
    config["preprocessing"]["text_symbols"] = text_symbols
    if not perform_benchmark:
        random.shuffle(data)
    train_data, val_data = data[SPLIT_SIZE:], data[:SPLIT_SIZE]

    if args.checkpoint == None:
        preprocess(
            config=config,
            train_data=train_data,
            val_data=val_data,
            deduplicate_train_data=False,
        )

    train(config=config, checkpoint_file=args.checkpoint, name=name)
