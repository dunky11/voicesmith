from pathlib import Path
import random
from voice_smith import TRAINING_RUNS_PATH

VAL_SIZE = 2000

paths = [
    TRAINING_RUNS_PATH / "pretraining_two_stage_ac" / "data" / "train.txt",
    TRAINING_RUNS_PATH / "pretraining_two_stage_ac" / "data" / "val.txt",
]

lines = []

for path in paths:
    with open(path, "r") as f:
        for line in f:
            lines.append(line)

random.shuffle(lines)

train_lines, val_lines = lines[VAL_SIZE:], lines[:VAL_SIZE]

with open(
    TRAINING_RUNS_PATH
    / "pretraining_two_stage_ac"
    / "data"
    / "train_finetuning.txt",
    "w",
) as f:
    for line in train_lines:
        f.write(line)

with open(
    TRAINING_RUNS_PATH
    / "pretraining_two_stage_ac"
    / "data"
    / "val_finetuning.txt",
    "w",
) as f:
    for line in val_lines:
        f.write(line)
