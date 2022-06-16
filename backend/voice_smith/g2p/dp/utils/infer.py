from voice_smith.g2p.dp.phonemizer import Phonemizer
from typing import List


def batched_predict(
    model: Phonemizer, texts: List[str], langs: List[str], batch_size=32
):
    assert len(texts) == len(langs)
    return model.phonemise_list(texts, langs=langs, batch_size=batch_size)
