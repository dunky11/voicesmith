import torch
from typing import Dict, List, Tuple
import unicodedata
from pathlib import Path
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ru import Russian
from spacy.lang.de import German
from voice_smith.utils.text import strip_cont_whitespaces
from voice_smith.utils.punctuation import get_punct


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    # \t, \n, and \r are technically control characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `char` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (
        (cp >= 33 and cp <= 47)
        or (cp >= 58 and cp <= 64)
        or (cp >= 91 and cp <= 96)
        or (cp >= 123 and cp <= 126)
    ):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


class BertTokenizer:
    """COPIED FROM huggingface/transformers: https://github.com/huggingface/transformers"""

    def __init__(self, assets_path: str):
        self.max_length = 150
        self.vocab = self.load_vocab(assets_path)
        self.do_lower_case = True
        self.tokenize_chinese_chars = True
        self.tokenizer = BertBasicTokenizer(self.tokenize_chinese_chars)
        self.word_piece_tokenizer = WordpieceTokenizer(self.vocab, "[UNK]", 200)
        self.cls_token = self.vocab["[CLS]"]
        self.sep_token = self.vocab["[SEP]"]

    def __call__(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        ids, attentions = [], []
        for text in texts:
            tokens: List[str] = self.tokenize(text)
            token_ids: List[int] = self.convert_token_to_ids(tokens)
            token_ids_tensor, attention_mask = self.prepare_for_model(token_ids)
            ids.append(token_ids_tensor)
            attentions.append(attention_mask)
        max_len_ids = max(el.shape[0] for el in ids)
        max_len_attentions = max(el.shape[0] for el in attentions)
        ids = [
            torch.cat(
                [el, torch.zeros((max_len_ids - el.shape[0],), dtype=torch.int64)]
            )
            for el in ids
        ]
        attentions = [
            torch.cat(
                [el, torch.zeros((max_len_ids - el.shape[0],), dtype=torch.int64)]
            )
            for el in attentions
        ]
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(attentions),
        }

    def tokenize(self, text: str) -> List[str]:
        if self.do_lower_case:
            text = text.lower()
        split_tokens: List[str] = []
        tokens: List[str] = self.tokenizer.tokenize(text)
        for token in tokens:
            tmp: List[str] = self.word_piece_tokenizer.tokenize(token)
            for s in tmp:
                split_tokens.append(s)
        return split_tokens

    def convert_token_to_ids(self, tokens: List[str]) -> List[int]:
        token_ids: List[int] = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab["[UNK]"])
        return token_ids

    def prepare_for_model(
        self, token_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(token_ids) > self.max_length:
            token_ids = token_ids[:-1]

        token_ids.insert(0, self.cls_token)
        token_ids.append(self.sep_token)
        attention_mask = torch.ones((len(token_ids),), dtype=torch.int64)
        token_ids_tensor = torch.LongTensor(token_ids)
        return token_ids_tensor, attention_mask

    def load_vocab(self, assets_path: str) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        with open(Path(assets_path) / "tiny_bert" / "vocab.txt", "r") as f:
            for i, line in enumerate(f):
                line_stripped = line.rstrip()
                vocab[line_stripped] = i
        return vocab


class BertBasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).
    Args:
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (`Iterable`, *optional*):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            `do_basic_tokenize=True`
        tokenize_chinese_chars (`bool`, *optional*, defaults to `True`):
            Whether or not to tokenize Chinese characters.
            This should likely be deactivated for Japanese (see this
            [issue](https://github.com/huggingface/transformers/issues/328)).
        strip_accents: (`bool`, *optional*):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for `lowercase` (as in the original BERT).
    """

    def __init__(
        self,
        do_lower_case=True,
        never_split=None,
        tokenize_chinese_chars=True,
        strip_accents=None,
    ):
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.
        Args:
            never_split (`List[str]`, *optional*)
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                [`PreTrainedTokenizer.tokenize`]) List of token not to split.
        """
        # union() returns a new set by concatenating the two sets.
        never_split = (
            self.never_split.union(set(never_split))
            if never_split
            else self.never_split
        )
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:
                if self.do_lower_case:
                    token = token.lower()
                    if self.strip_accents is not False:
                        token = self._run_strip_accents(token)
                elif self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.
        For example, `input = "unaffable"` wil return as output `["un", "##aff", "##able"]`.
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through *BasicTokenizer*.
        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _get_nlp(lang):
    if lang == "en":
        nlp = English()
    elif lang == "es":
        nlp = Spanish()
    elif lang == "de":
        nlp = German()
    elif lang == "ru":
        nlp = Russian()
    else:
        raise Exception(
            f"No case selected in switch-statement, '{lang}' is not a valid case ..."
        )
    return nlp


class WordTokenizer:
    def __init__(self, lang: str):
        self.nlp = _get_nlp(lang)
        self.tokenizer = self.nlp.tokenizer

    def tokenize(self, text: str) -> List[str]:
        tokens = self.tokenizer(text)
        tokens = [str(token) for token in tokens]
        return tokens


class SentenceTokenizer:
    def __init__(self, lang: str):
        self.nlp = _get_nlp(lang)
        self.nlp.add_pipe("sentencizer", config={"punct_chars": get_punct(lang=lang)})

    def tokenize(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sents = [str(sent) for sent in doc.sents]
        return sents
