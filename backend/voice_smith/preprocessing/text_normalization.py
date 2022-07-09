from pathlib import Path
from typing import Tuple, List, Dict, Callable, Optional, Set, Union, Literal
from dataclasses import dataclass
from nemo_text_processing.text_normalization.normalize import Normalizer
from voice_smith.utils.tokenization import WordTokenizer
from voice_smith.utils.exceptions import InvalidLangException
from voice_smith.utils.number_normalization import (
    NumberNormalizerBase,
    get_number_normalizer,
)

# CHARACTERS
LATIN_CHARACTERS = list("abcdefghijklmnopqrstuvwxyz")
GERMAN_CHARACTERS = list("öüäß")
SPANISH_CHARACTERS = list("üúóñíéá")
RUSSIAN_CHARACTERS = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".lower())

# PUNCTUATION
MULTILINGUAL_PUNCTUATION = list(".,!?:-")
ENGLISH_PUNCTUATION = list("")
SPANISH_PUNCTUATION = list("¿¡«»—")
GERMAN_PUNCTUATION = list("")
RUSSIAN_PUNCTUATION = list("")

# SPECIAL SYMBOLS
SPECIALS = list("\"' ")

# ABBREVIATIONS
MULTILINGUAL_ABBREVIATIONS = [
    "kg",
    "cal",
    "corp.",
    "dept",
    "dr",
    "oz",
    "ft",
    "Ft",
    "gal",
    "hr",
    "inc",
    "jr",
    "km",
    "ltd",
    "mg",
    "mph",
    "mr",
    "ms",
    "oz",
    "sr",
    "sec",
    "vol",
    "vs",
    "pc",
    "ing",
    "prof",
    "gov",
    "assoc",
    "co",
    "corp",
]
MULTILINGUAL_ABBREVIATIONS = MULTILINGUAL_ABBREVIATIONS + [
    f"{el}." for el in MULTILINGUAL_ABBREVIATIONS
]
ENGLISH_ABBREVIATIONS = []
SPANISH_ABBREVIATIONS = []
GERMAN_ABBREVIATIONS = []
RUSSIAN_ABBREVIATIONS = []

# INITIALISMS
MULTILINGUAL_INITIALISMS = [
    "FBI",
    "GOP",
    "BBC",
    "CNN",
    "PD",
    "TV",
    "F.B.I.",
    "EU",
    "USA",
    "PDT",
    "PT",
    "PST",
    "UTC",
    "GMT",
    "MSNBC",
    "CBS",
    "CNBC",
    "MMORPG",
    "mmorpg",
    "RPG",
    "rpg",
    "PVP",
    "pvp",
    "QR",
    "PTD",
    "PTS",
    "BMI",
    "BMX",
    "BMW",
    "BDSM",
    "CTC",
]
ENGLISH_INITIALISMS = []
GERMAN_INITIALISMS = []
SPANISH_INITIALISMS = []
RUSSIAN_INITIALISMS = []


class CharNormalizer:
    rules = set(
        [
            ("’", "'"),
            ("“", '"'),
            ("”", '"'),
            ("..", "."),
            ("...", "."),
            ("....", "."),
            (".....", "."),
            ("…", "."),
            ("!!", "!"),
            ("!!!", "!"),
            ("!!!!", "!"),
            ("!!!!!", "!"),
            ("??", "?"),
            ("???", "?"),
            ("????", "?"),
            ("?????", "?"),
            ("„", '"'),
            ("–", "-"),
            ("«", '"'),
            ("»", '"'),
        ]
    )

    def remove_cont_whitespaces(self, text):
        new_string = ""
        last_char = None
        for char in text:
            if char == " " and last_char == " ":
                continue
            last_char = char
            new_string += char
        return new_string

    def normalize(self, text: str) -> Tuple[str, List[str]]:
        rules = []
        text_stripped = self.remove_cont_whitespaces(text.strip())
        if text_stripped != text:
            text = text_stripped
            rules.append("The text contains unnecessary whitespaces")
        for r_from, r_to in self.rules:
            if r_from in text:
                text = text.replace(r_from, r_to)
                rules.append(f"'{r_from}' should be normalized to '{r_to}'")
        return text, rules


class DetShouldNormalizeBase:
    """
    Nemo's text normalizer unfortunately produces a large amount of false positives.
    For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'.
    To reduce the amount of false postives we will do a check for unusual symbols
    or words inside the text and only normalize if necessary.
    """

    def should_normalize(
        self, text: str, tokenizer: WordTokenizer
    ) -> Tuple[bool, List[str]]:
        raise NotImplementedError()

    def get_reasons(
        self,
        text: str,
        abbreviations: Set[str],
        initialisms: Set[str],
        allowed_characters: Set[str],
        tokenizer: WordTokenizer,
    ) -> List[str]:
        reasons, abbr_detected, inits_detected, unusual_chars_detected = [], [], [], []
        for token in tokenizer.tokenize(text):
            token = token.word
            if token.lower() in abbreviations:
                abbr_detected.append(token.lower())
            if token in initialisms:
                inits_detected.append(token)

        abbr_detected = list(set(abbr_detected))
        if len(abbr_detected) > 0:
            reasons.append(f"The text contains the abbreviations: {str(abbr_detected)}")

        inits_detected = list(set(inits_detected))
        if len(inits_detected) > 0:
            reasons.append(f"The text contains the initialisms: {str(inits_detected)}")

        for char in text.lower():
            if char not in allowed_characters:
                unusual_chars_detected.append(char)

        unusual_chars_detected = list(set(unusual_chars_detected))
        if len(unusual_chars_detected) > 0:
            reasons.append(
                f"The text contains characters not recognised by most TTS systems: {str(unusual_chars_detected)}"
            )

        return reasons

    def fetch_word_list(self, name: str, assets_path: str):
        lines = []
        with open(Path(assets_path) / "word_lists" / name, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(line.strip())
        return lines


class DetShouldNormalizeEN(DetShouldNormalizeBase):
    def __init__(self, assets_path: str):
        super().__init__()
        en_punctuation = MULTILINGUAL_PUNCTUATION + ENGLISH_PUNCTUATION
        self.allowed_characters = set(LATIN_CHARACTERS + SPECIALS + en_punctuation)
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + ENGLISH_ABBREVIATIONS
            + self.fetch_word_list("english_abbreviations.txt", assets_path=assets_path)
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + ENGLISH_INITIALISMS
            + self.fetch_word_list("english_initialisms.txt", assets_path=assets_path)
        )

        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: WordTokenizer
    ) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text,
            self.abbreviations,
            self.initialisms,
            self.allowed_characters,
            tokenizer,
        )
        should_normalize = len(reasons) > 0
        return should_normalize, reasons


class DetShouldNormalizeES(DetShouldNormalizeBase):
    def __init__(self, assets_path: str):
        super().__init__()
        es_punctuation = MULTILINGUAL_PUNCTUATION + SPANISH_PUNCTUATION
        self.allowed_characters = set(
            LATIN_CHARACTERS + SPECIALS + SPANISH_CHARACTERS + es_punctuation
        )
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + SPANISH_ABBREVIATIONS
            + self.fetch_word_list("spanish_abbreviations.txt", assets_path=assets_path)
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + self.fetch_word_list("spanish_initialisms.txt", assets_path=assets_path)
            + SPANISH_INITIALISMS
        )
        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: WordTokenizer
    ) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text,
            self.abbreviations,
            self.initialisms,
            self.allowed_characters,
            tokenizer,
        )
        should_normalize = len(reasons) > 0
        return should_normalize, reasons


class DetShouldNormalizeDE(DetShouldNormalizeBase):
    def __init__(self, assets_path: str):
        super().__init__()
        de_punctuation = MULTILINGUAL_PUNCTUATION + GERMAN_PUNCTUATION
        self.allowed_characters = set(
            LATIN_CHARACTERS + SPECIALS + GERMAN_CHARACTERS + de_punctuation
        )
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + GERMAN_ABBREVIATIONS
            + self.fetch_word_list("german_abbreviations.txt", assets_path=assets_path)
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + self.fetch_word_list("german_initialisms.txt", assets_path=assets_path)
            + GERMAN_INITIALISMS
        )
        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: WordTokenizer
    ) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text,
            self.abbreviations,
            self.initialisms,
            self.allowed_characters,
            tokenizer,
        )
        should_normalize = len(reasons) > 0
        return should_normalize, reasons


class DetShouldNormalizeRU(DetShouldNormalizeBase):
    def __init__(self, assets_path: str):
        super().__init__()
        ru_punctuation = MULTILINGUAL_PUNCTUATION + RUSSIAN_PUNCTUATION
        self.allowed_characters = set(RUSSIAN_CHARACTERS + SPECIALS + ru_punctuation)
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + RUSSIAN_ABBREVIATIONS
            + self.fetch_word_list("russian_abbreviations.txt", assets_path=assets_path)
        )
        initialisms = (
            self.fetch_word_list("russian_initialisms.txt", assets_path=assets_path)
            + RUSSIAN_INITIALISMS
        )
        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: WordTokenizer
    ) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text,
            self.abbreviations,
            self.initialisms,
            self.allowed_characters,
            tokenizer,
        )
        should_normalize = len(reasons) > 0
        return should_normalize, reasons


def apply_nemo_normalization(text: str, normalizer: Normalizer):
    text_in = text
    text_out = normalizer.normalize(
        text=text_in, verbose=False, punct_post_process=True,
    )
    if (
        len(text_in) > 0
        and len(text_out) > 0
        and text_in[0].lower() == text_out[0].lower()
    ):
        if text_in[0].isupper():
            if len(text_out) == 1:
                text_out = text_out.upper()
            else:
                text_out = text_out[0].upper() + text_out[1:]
        else:
            if len(text_out) == 1:
                text_out = text_out.lower()
            else:
                text_out = text_out[0].lower() + text_out[1:]
    return text_out


def replace_text(text: str, replacement: str, from_idx: int, to_idx: int):
    return text[:from_idx] + replacement + text[to_idx:]


def normalize_sample(
    text_in: str,
    char_normalizer: Optional[CharNormalizer],
    detector: Optional[DetShouldNormalizeBase],
    normalizer: Optional[Normalizer],
    tokenizer: WordTokenizer,
    number_normalizer: Optional[NumberNormalizerBase],
):
    reasons_number_norm = []
    reasons_det = []
    if char_normalizer is None:
        text_out, reasons_char_norm = text_in, []
    else:
        text_out, reasons_char_norm = char_normalizer.normalize(text_in)
    if detector is None:
        if number_normalizer is not None:
            words = tokenizer.tokenize(text_out)
            bias = 0
            for i in range(len(words)):
                prev_word = None if i == 0 else words[i - 1]
                next_word = None if i >= len(words) - 1 else words[i + 1]
                word = words[i]
                assert word.offset is not None
                if number_normalizer.is_number(word.word):
                    result = number_normalizer.normalize(
                        prev_word=None if prev_word is None else prev_word.word,
                        word=word.word,
                        next_word=None if next_word is None else next_word.word,
                    )
                    if result.has_normalized:
                        if prev_word is not None and result.collapsed_prev:
                            assert prev_word.offset is not None
                            from_idx = bias + prev_word.offset
                        else:
                            from_idx = word.offset + bias

                        if next_word is not None and result.collapsed_next:
                            assert next_word.offset is not None
                            to_idx = bias + next_word.offset + len(next_word.word)
                        else:
                            to_idx = bias + word.offset + len(word.word)

                        len_before = len(text_out)
                        text_out = replace_text(
                            text_out, result.word, from_idx=from_idx, to_idx=to_idx
                        )
                        bias += len(text_out) - len_before
                        reasons_number_norm.append(
                            f"The text contains the number: {word.word}"
                        )
    else:
        should_normalize, reasons_det = detector.should_normalize(text_out, tokenizer)
        if should_normalize:
            text_out = apply_nemo_normalization(text=text_out, normalizer=normalizer)

    reason = ". ".join(reasons_char_norm + reasons_det + reasons_number_norm)
    if len(reason) > 0:
        reason = f"{reason}."

    has_normalized = text_in != text_out

    return has_normalized, text_in, text_out, reason


def load_text(text_path: Path):
    if not text_path.exists():
        return None
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


@dataclass
class NormalizationUtils:
    tokenizer: WordTokenizer
    char_normalizer: Optional[CharNormalizer]
    detector: Optional[DetShouldNormalizeBase]
    normalizer: Optional[Normalizer]
    number_normalizer: Optional[NumberNormalizerBase]


def get_normalization_utils(
    lang: str,
    normalize_characters: bool,
    assets_path: str,
    mode: Union[Literal["fast"], Literal["slow"]],
) -> NormalizationUtils:
    if lang == "bg":
        detector = None
        normalizer = None
        number_normalizer = None
    elif lang == "cs":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("cz")
    elif lang == "de":
        if mode == "fast":
            detector = None
            normalizer = None
            number_normalizer = get_number_normalizer("de")
        else:
            detector = DetShouldNormalizeDE(assets_path)
            normalizer = Normalizer(
                input_case="cased",
                lang=lang,
                overwrite_cache=False,
                deterministic=True,
            )
            number_normalizer = None
    elif lang == "en":
        if mode == "fast":
            detector = None
            normalizer = None
            number_normalizer = get_number_normalizer("en")
        else:
            detector = DetShouldNormalizeEN(assets_path)
            normalizer = Normalizer(
                input_case="cased",
                lang=lang,
                overwrite_cache=False,
                deterministic=True,
            )
            number_normalizer = None
    elif lang == "es":
        if mode == "fast":
            detector = None
            normalizer = None
            number_normalizer = get_number_normalizer("es")
        else:
            detector = DetShouldNormalizeES(assets_path)
            normalizer = Normalizer(
                input_case="cased",
                lang=lang,
                overwrite_cache=False,
                deterministic=True,
            )
            number_normalizer = None
    elif lang == "fr":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("fr")
    elif lang == "hr":
        detector = None
        normalizer = None
        number_normalizer = None
    elif lang == "pl":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("pl")
    elif lang == "pt":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("pt")
    elif lang == "ru":
        if mode == "fast":
            detector = None
            normalizer = None
            number_normalizer = get_number_normalizer("ru")
        else:
            detector = DetShouldNormalizeRU(assets_path)
            normalizer = Normalizer(
                input_case="cased",
                lang=lang,
                overwrite_cache=False,
                deterministic=False,
            )
            number_normalizer = None
    elif lang == "sv":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("sv")
    elif lang == "th":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("th")
    elif lang == "tr":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("tr")
    elif lang == "uk":
        detector = None
        normalizer = None
        number_normalizer = get_number_normalizer("uk")
    else:
        raise InvalidLangException(f"Language '{lang}' is not a supported language ...")

    return NormalizationUtils(
        tokenizer=WordTokenizer(lang, remove_punct=False),
        char_normalizer=CharNormalizer() if normalize_characters else None,
        detector=detector,
        normalizer=normalizer,
        number_normalizer=number_normalizer,
    )


def text_normalize(
    sample_ids: List[str],
    texts: List[str],
    langs: List[str],
    assets_path: str,
    progress_cb: Optional[Callable[[float], None]],
    callback_every: int = 50,
    normalize_characters: bool = True,
) -> List[Tuple[int, str, str, str]]:
    assert len(sample_ids) == len(texts) == len(langs)

    normalizations = []
    lang2utils: Dict[str, NormalizationUtils] = {}

    for i, (sample_id, text, lang) in enumerate(zip(sample_ids, texts, langs)):
        if not lang in lang2utils:
            lang2utils[lang] = get_normalization_utils(
                lang=lang,
                normalize_characters=normalize_characters,
                assets_path=assets_path,
                mode="slow",
            )
        utils = lang2utils[lang]
        ret = normalize_sample(
            text_in=text,
            char_normalizer=utils.char_normalizer,
            detector=utils.detector,
            normalizer=utils.normalizer,
            tokenizer=utils.tokenizer,
            number_normalizer=utils.number_normalizer,
        )
        is_normalized, text_in, text_out, reason = ret
        if is_normalized:
            normalizations.append((sample_id, text_in, text_out, reason))
            print("Text in: ", text_in, flush=True)
            print("Text normalized: ", text_out, flush=True)
            print("Reason: ", reason, flush=True)
            print("", flush=True)

        if progress_cb is not None and i % callback_every == 0 and i != 0:
            progress_cb((i + 1) / len(sample_ids))

    if progress_cb is not None:
        progress_cb(1.0)

    return normalizations

