from pathlib import Path
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from voice_smith.utils.tokenization import WordTokenizer

def get_duration(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)
    duration = audio.shape[0] / sr
    return duration

def get_text(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    return text

tokenizer = WordTokenizer(lang="en", remove_punct=False)

words = set()

with open("en.dict", "r", encoding="utf-8") as f:
    for line in f.readlines():
        words.add(line.split("\t")[0])

for speaker_path in (Path(".") / "in").iterdir():
    speaker_name = speaker_path.name
    durations = Parallel(n_jobs=12)(
        delayed(get_duration)(audio_path)
        for audio_path in tqdm(speaker_path.glob("*.flac"))
    )
    texts = Parallel(n_jobs=12)(
        delayed(get_text)(text_path)
        for text_path in tqdm(speaker_path.glob("*.txt"))
    )

    words_found = 0
    words_total = 0
    for text in texts:
        for token in tokenizer.tokenize(text):
            if token.lower() in words:
                words_found += 1
            words_total += 1
        

    np.random.seed(42)
    x = np.random.normal(size=1000)

    plt.hist(x, density=True, bins=20)  # density=False would make counts
    plt.xlabel('duration')
    plt.savefig(f"{speaker_name}_durations.png")

    print("-" * 20)
    print(f"Speaker: {speaker_name}")
    print(f"Number of files: {len(durations)}")
    print(f"Mean duration: {sum(durations) / len(durations)}")
    print(f"Words in dict: {(words_found / words_total) * 100}%")
    print("-" * 20)