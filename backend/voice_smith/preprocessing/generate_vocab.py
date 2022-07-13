import torch
from typing import List, Dict
from voice_smith.config.configs import PreprocessLangType
from voice_smith.utils.shell import run_conda_in_shell
from voice_smith.preprocessing.g2p import batched_predict, get_g2p
from voice_smith.utils.tokenization import WordTokenizer
from voice_smith.utils.mfa import lang_to_mfa_g2p


def generate_vocab(
    texts: List[str],
    lang: str,
    assets_path: str,
    language_type: PreprocessLangType,
    device: torch.device,
) -> Dict[str, List[str]]:
    tokenizer = WordTokenizer(lang=lang, remove_punct=False)
    words_to_tokenize = set()
    for text in texts:
        for word in tokenizer.tokenize(text):
            words_to_tokenize.add(word)
    words_to_tokenize = list(words_to_tokenize)
    g2p = get_g2p(assets_path=assets_path, device=device)
    predicted_phones = batched_predict(
        model=g2p,
        texts=words_to_tokenize,
        langs=[lang for _ in range(len(words_to_tokenize))],
    )
    vocab = {word: phones for word, phones in zip(words_to_tokenize, predicted_phones)}
    return vocab


def generate_vocab_mfa(
    lexicon_path: str,
    n_workers: int,
    lang: str,
    corpus_path: str,
    environment_name: str,
    language_type: PreprocessLangType,
):
    cmd = f"mfa g2p --clean -j {n_workers} {lang_to_mfa_g2p(lang, language_type)} {corpus_path} {lexicon_path}"
    run_conda_in_shell(cmd, environment_name, stderr_to_stdout=True)
