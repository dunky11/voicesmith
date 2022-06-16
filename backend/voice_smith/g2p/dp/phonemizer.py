import re
from itertools import zip_longest
from typing import Dict, Union, List, Set

from voice_smith.g2p.dp import PhonemizerResult
from voice_smith.g2p.dp.model.model import load_checkpoint
from voice_smith.g2p.dp.model.predictor import Predictor
from voice_smith.g2p.dp.utils.logging import get_logger
from voice_smith.g2p.dp.preprocessing.text import Preprocessor
from voice_smith.utils.punctuation import get_punct


class Phonemizer:
    def __init__(
        self, predictor: Predictor, lang_phoneme_dict: Dict[str, Dict[str, str]] = None
    ) -> None:
        """
        Initializes a phonemizer with a ready predictor.

        Args:
            predictor (Predictor): Predictor object carrying the trained transformer model.
            lang_phoneme_dict (Dict[str, Dict[str, str]], optional): Word-phoneme dictionary for each language.
        """

        self.predictor = predictor
        self.lang_phoneme_dict = lang_phoneme_dict

    def phonemise_list(
        self, words: List[str], langs: List[str], batch_size: int = 8,
    ) -> PhonemizerResult:
        """Phonemizes a list of texts and returns tokenized texts,
        phonemes and word predictions with probabilities.

        Args:
          texts (List[str]): List texts to phonemize.
          lang (List[str]): Languages used for phonemization.
          punctuation (str): Punctuation symbols by which the texts are split. (Default value = DEFAULT_PUNCTUATION)
          batch_size (int): Batch size of model to speed up inference. (Default value = 8)

        Returns:
          PhonemizerResult: Object containing original texts, phonemes, split texts, split phonemes, and predictions.

        """
        if len(words) == 0:
            return []
        # TODO currently works because get_punct returns same set for every language,
        # has to be changed when get_punct gets language specific
        punc_set = get_punct(lang=langs[0])
        # collect dictionary phonemes for words and hyphenated words
        word_phonemes = {
            (word, lang): self._get_dict_entry(word=word, lang=lang, punc_set=punc_set)
            for word, lang in zip(words, langs)
        }
        # predict all subwords that are missing in the phoneme dict
        words_to_predict = [
            (word, lang)
            for (word, lang), phons in word_phonemes.items()
            if phons is None
        ]
        words_pred = [tup[0] for tup in words_to_predict]
        langs_pred = [tup[1] for tup in words_to_predict]
        predictions = self.predictor(
            words=words_pred, langs=langs_pred, batch_size=batch_size,
        )
        word_phonemes.update(
            {
                (pred.word, lang): pred.phonemes_list
                for pred, lang in zip(predictions, langs_pred)
            }
        )
        phoneme_lists = []
        for word, lang in zip(words, langs):
            phoneme_lists.append(word_phonemes[(word, lang)])

        return phoneme_lists

    def _get_dict_entry(
        self, word: str, lang: str, punc_set: Set[str]
    ) -> Union[str, None]:
        if word in punc_set or len(word) == 0:
            return [word]
        if not self.lang_phoneme_dict or lang not in self.lang_phoneme_dict:
            return None
        phoneme_dict = self.lang_phoneme_dict[lang]
        if word in phoneme_dict:
            return phoneme_dict[word].split(" ")
        elif word.lower() in phoneme_dict:
            return phoneme_dict[word.lower()].split(" ")
        elif word.title() in phoneme_dict:
            return phoneme_dict[word.title()].split(" ")
        else:
            return None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device="cpu",
        lang_phoneme_dict: Dict[str, Dict[str, str]] = None,
    ) -> "Phonemizer":
        """Initializes a Phonemizer object from a model checkpoint (.pt file).

        Args:
          checkpoint_path (str): Path to the .pt checkpoint file.
          device (str): Device to send the model to ('cpu' or 'cuda'). (Default value = 'cpu')
          lang_phoneme_dict (Dict[str, Dict[str, str]], optional): Word-phoneme dictionary for each language.

        Returns:
          Phonemizer: Phonemizer object carrying the loaded model and, optionally, a phoneme dictionary.
        """

        model, checkpoint = load_checkpoint(checkpoint_path, device=device)
        applied_phoneme_dict = None
        if lang_phoneme_dict is not None:
            applied_phoneme_dict = lang_phoneme_dict
        elif "phoneme_dict" in checkpoint:
            applied_phoneme_dict = checkpoint["phoneme_dict"]
        preprocessor = Preprocessor.from_config(checkpoint["config"])
        predictor = Predictor(model=model, preprocessor=preprocessor)
        return Phonemizer(predictor=predictor, lang_phoneme_dict=applied_phoneme_dict)
