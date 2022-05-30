from typing import List, Dict


class Prediction:
    """
    Container for single word prediction result.
    """

    def __init__(self,
                 word: str,
                 phonemes: str,
                 phonemes_list: List[str],
                 phoneme_tokens: List[str],
                 confidence: float,
                 token_probs: List[float]) -> None:
        """
        Initializes a Prediction object.

        Args:
          word (str): Original word to predict.
          phonemes (str): Predicted phonemes (without special tokens).
          phonemes_list (List[str]): Predicted phoneme tokens (withou special tokens).
          phoneme_tokens (List[str]): Predicted phoneme tokens (including special tokens).
          confidence (float): Total confidence of result.
          token_probs (List[float]): Probability of each phoneme token.
        """

        self.word = word
        self.phonemes = phonemes
        self.phonemes_list = phonemes_list
        self.phoneme_tokens = phoneme_tokens
        self.confidence = confidence
        self.token_probs = token_probs


class PhonemizerResult:
    """
    Container for phonemizer output.
    """

    def __init__(self,
                 text: List[str],
                 split_text: List[List[str]],
                 split_phonemes: List[List[str]],
                 predictions: Dict[str, Prediction]) -> None:
        """
        Initializes a PhonemizerResult object.

        Args:
          text (List[str]): List of input texts.
          split_text (List[List[str]]): List of texts, where each text is split into words and special chars.
          split_phonemes (List[List[str]]): List of phonemes corresponding to split_text.
          predictions (Dict[str, Prediction]): Dictionary with entries word to Tuple (phoneme, probability).
        """

        self.text = text
        self.split_text = split_text
        self.split_phonemes = split_phonemes
        self.predictions = predictions