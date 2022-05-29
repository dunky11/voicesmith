from pathlib import Path
from dp.preprocess import preprocess
from dp.train import train
from dp.utils.io import read_config
from parse_dictionary import parse_dictionary
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    data, phones, text_symbols = parse_dictionary("english_us_arpa.dict", "en_us", skip_duplicates=True)
    config = read_config(Path(".") / "dp" / "configs" / "autoreg_config.yaml")
    config["preprocessing"]["phoneme_symbols"] = phones
    config["preprocessing"]["text_symbols"] = text_symbols
    random.shuffle(data)
    train_data, val_data = data[5000:], data[:5000]

    if args.checkpoint == None:
        preprocess(
            config=config,
            train_data=train_data,
            val_data=val_data,
            deduplicate_train_data=False
        )

    train(config=config, checkpoint_file=args.checkpoint)