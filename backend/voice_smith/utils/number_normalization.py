from decimal import InvalidOperation
from typing import List, Set, Union, Literal
from dataclasses import dataclass
from xml.dom import InvalidAccessErr
from num2words import num2words, CONVERTER_CLASSES
from voice_smith.utils.currencies import iso_4217_to_symbols

# NUMBERS
ARABIC_NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
ARABIC_NUMBERS_PUNCT = [".", ",", "-"]
# Thai has its own numeric symbols. However, outside of government documents, those are rarely used.
# Usually arabic number are used.
THAI_NUMBERS = ["๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙"]

NumberNormLangType = Union[
    Literal["cz"],
    Literal["de"],
    Literal["en"],
    Literal["es"],
    Literal["fr"],
    Literal["pl"],
    Literal["pt"],
    Literal["ru"],
    Literal["sv"],
    Literal["th"],
    Literal["tr"],
    Literal["uk"],
]


@dataclass
class NumberNormalizationResult:
    has_normalized: bool
    collapsed_prev: bool
    word: str
    collapsed_next: bool


class NumberNormalizerBase:
    def _get_currency_symbols(self, currency_codes: List[str]) -> Set[str]:
        symbols = []
        for currency_code in currency_codes:
            symbols.extend(iso_4217_to_symbols[currency_code])
        return set(symbols)

    def _is_number(
        self, word: str, valid_chars: Set[str], iso_4217_currency_codes: List[str]
    ) -> bool:
        for code in iso_4217_currency_codes:
            for symbol in iso_4217_to_symbols[code]:
                if word.startswith(symbol):
                    word = word.lstrip(symbol)
                if word.endswith(symbol):
                    word = word.rstrip(symbol)
        for char in word:
            if char not in valid_chars:
                return False
        return True

    def is_number(self, word: str) -> bool:
        raise NotImplementedError()

    def _normalize(
        self,
        prev_word: Union[str, None],
        word: str,
        next_word: Union[str, None],
        lang: NumberNormLangType,
        iso_4217_currency_codes: List[str],
    ) -> NumberNormalizationResult:
        # search for currency_symbols
        collapsed_prev = False
        collapsed_next = False
        to = "cardinal"
        currency = None
        for code in iso_4217_currency_codes:
            for symbol in iso_4217_to_symbols[code]:
                if word.startswith(symbol):
                    word = word.lstrip(symbol)
                    to = "currency"
                    currency = code
                    break
                elif word.endswith(symbol):
                    word = word.rstrip(symbol)
                    to = "currency"
                    currency = code
                    break
                elif prev_word is not None and prev_word == symbol:
                    collapsed_prev = True
                    to = "currency"
                    currency = code
                    break
                elif next_word is not None and next_word == symbol:
                    collapsed_next = True
                    to = "currency"
                    currency = code
                    break

        try:

            if to == "currency":
                with_cents = "." in word or "," in word
                word = num2words(
                    word, lang=lang, to=to, currency=currency, cents=True, separator="|"
                )
                if with_cents:
                    word = word.replace("|", "")
                else:
                    splits = word.split("|")
                    if len(splits) == 0:
                        word = splits[0]
                    else:
                        word = "".join(splits[:-1])
            else:
                word = num2words(word, lang=lang, to=to)
        except (ValueError, OverflowError, InvalidOperation) as e:
            return NumberNormalizationResult(
                has_normalized=False,
                collapsed_prev=False,
                word=word,
                collapsed_next=False,
            )
        return NumberNormalizationResult(
            has_normalized=True,
            collapsed_prev=collapsed_prev,
            word=word,
            collapsed_next=collapsed_next,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        raise NotImplementedError()


class NumberNormalizerCSCZ(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["cz"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="cz",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerDEDE(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["de"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="de",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerRURU(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["ru"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="ru",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerFRFR(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["fr"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="fr",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerENEN(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["en"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="en",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerESES(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["es"].CURRENCY_FORMS.keys()
        self.iso_4217_currency_codes = list(
            filter(lambda x: x in iso_4217_to_symbols, self.iso_4217_currency_codes)
        )

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="es",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerPLPL(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["pl"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="pl",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerPTPT(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["pt"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="pt",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerSVSV(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["sv"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="sv",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerTHTH(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT + THAI_NUMBERS)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["th"].CURRENCY_FORMS.keys()

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="th",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerTRTR(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = []

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="tr",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


class NumberNormalizerUKUK(NumberNormalizerBase):
    def __init__(self):
        self.valid_chars = set(ARABIC_NUMBERS + ARABIC_NUMBERS_PUNCT)
        self.iso_4217_currency_codes = CONVERTER_CLASSES["uk"].CURRENCY_FORMS.keys()
        self.iso_4217_currency_codes = list(
            filter(lambda x: x in iso_4217_to_symbols, self.iso_4217_currency_codes)
        )

    def is_number(self, word: str) -> bool:
        return self._is_number(
            word,
            valid_chars=self.valid_chars,
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )

    def normalize(
        self, prev_word: Union[str, None], word: str, next_word: Union[str, None],
    ) -> NumberNormalizationResult:
        return self._normalize(
            prev_word=prev_word,
            word=word,
            next_word=next_word,
            lang="uk",
            iso_4217_currency_codes=self.iso_4217_currency_codes,
        )


def get_number_normalizer(lang: NumberNormLangType) -> NumberNormalizerBase:
    if lang == "cz":
        return NumberNormalizerCSCZ()
    elif lang == "de":
        return NumberNormalizerDEDE()
    elif lang == "en":
        return NumberNormalizerENEN()
    elif lang == "es":
        return NumberNormalizerESES()
    elif lang == "fr":
        return NumberNormalizerFRFR()
    elif lang == "pl":
        return NumberNormalizerPLPL()
    elif lang == "pt":
        return NumberNormalizerPTPT()
    elif lang == "ru":
        return NumberNormalizerRURU()
    elif lang == "sv":
        return NumberNormalizerSVSV()
    elif lang == "th":
        return NumberNormalizerTHTH()
    elif lang == "tr":
        return NumberNormalizerTRTR()
    elif lang == "uk":
        return NumberNormalizerUKUK()
    else:
        raise Exception(
            f"No case selected in switch-statement, '{lang}' is not a valid case ..."
        )


if __name__ == "__main__":
    normalizer = get_number_normalizer("cz")
    output = normalizer.normalize(None, "-100", None)
    print(output)
