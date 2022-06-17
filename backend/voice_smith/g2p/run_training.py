from pathlib import Path
import random
import argparse
from voice_smith.g2p.dp.preprocess import preprocess
from voice_smith.g2p.dp.train import train
from voice_smith.g2p.dp.utils.io import read_config
from parse_dictionary import parse_dictionary


perform_benchmark = False

name = "G2P MFA training 6x6 transformer (384, 1536), [15 MFA langs]"

if perform_benchmark:
    SPLIT_SIZE = 12753
else:
    SPLIT_SIZE = 10000

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

    if args.checkpoint is None:
        for dictionary_path in [
            Path(".") / "dictionaries" / "bg" / "bulgarian_mfa.dict",
            Path(".") / "dictionaries" / "cs" / "czech_mfa.dict",
            Path(".") / "dictionaries" / "de" / "german_mfa.dict",
            Path(".") / "dictionaries" / "en" / "english_us_mfa.dict",
            Path(".") / "dictionaries" / "es" / "spanish_mfa.dict",
            Path(".") / "dictionaries" / "fr" / "french_mfa.dict",
            Path(".") / "dictionaries" / "hr" / "croatian_mfa.dict",
            Path(".") / "dictionaries" / "pl" / "polish_mfa.dict",
            Path(".") / "dictionaries" / "pt" / "portuguese_portugal_mfa.dict",
            Path(".") / "dictionaries" / "ru" / "russian_mfa.dict",
            Path(".") / "dictionaries" / "sv" / "swedish_mfa.dict",
            Path(".") / "dictionaries" / "th" / "thai_mfa.dict",
            Path(".") / "dictionaries" / "tr" / "turkish_mfa.dict",
            Path(".") / "dictionaries" / "uk" / "ukrainian_mfa.dict",
        ]:
            lang = dictionary_path.parent.name
            d, p, t = parse_dictionary(dictionary_path, lang)
            data.extend(d)
            phones.extend(p)
            text_symbols.extend(t)

        phones = list(set(phones))
        text_symbols = list(set(text_symbols))

        with open(Path(".") / "text_symbols.txt", "w", encoding="utf-8") as f:
            f.write(str(text_symbols))

        with open(Path(".") / "phones.txt", "w", encoding="utf-8") as f:
            f.write(str(phones))

        print(phones)
        print(text_symbols)
        print(data[:100])

        config = read_config(Path(".") / "dp" / "configs" / "autoreg_config.yaml")
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

    train(config=config, checkpoint_file=args.checkpoint, name=name)
