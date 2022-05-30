from pathlib import Path
from voice_smith.g2p.dp.preprocess import preprocess
from voice_smith.g2p.dp.train import train
from voice_smith.g2p.dp.utils.io import read_config
from parse_dictionary import parse_dictionary
import random
import argparse

perform_benchmark = False

name = "G2p ARPA training 4x4 transformer"

if perform_benchmark:
    SPLIT_SIZE = 12753
else:
    SPLIT_SIZE = 5000

if __name__ == '__main__':
    if perform_benchmark:
        print("Benchmarking on CMUDict ...")
    else:
        print("Training model for production ...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    data, phones, text_symbols = parse_dictionary("english_us_arpa.dict", "en")
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
            deduplicate_train_data=False
        )

    train(config=config, checkpoint_file=args.checkpoint, name=name)