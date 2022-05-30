from dp.phonemizer import Phonemizer
from typing import List

def batched_predict(model: Phonemizer, texts: List[str], batch_size=32):
    return model.phonemise_list(texts, lang="en", batch_size=batch_size)