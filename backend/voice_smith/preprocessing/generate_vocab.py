import torch
from typing import List, Dict
from voice_smith.utils.shell import run_conda_in_shell
from voice_smith.preprocessing.g2p import batched_predict, get_g2p
from voice_smith.utils.tokenization import WordTokenizer


def generate_vocab(
    texts: List[str], lang: str, assets_path: str, device: torch.device
) -> Dict[str, List[str]]:
    tokenizer = WordTokenizer(lang=lang)
    words_to_tokenize = set()
    for text in texts:
        for word in tokenizer.tokenize(text):
            words_to_tokenize.add(word)
    words_to_tokenize = list(words_to_tokenize)
    g2p = get_g2p(assets_path=assets_path, device=device)
    predicted_phones = batched_predict(model=g2p, texts=words_to_tokenize)
    vocab = {word: phones for word, phones in zip(words_to_tokenize, predicted_phones)}
    return vocab
