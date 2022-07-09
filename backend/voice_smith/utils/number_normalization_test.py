from typing import List
from voice_smith.utils.number_normalization import (
    get_number_normalizer,
    NumberNormLangType,
)

_LANGUAGES_TO_CHECK: List[NumberNormLangType] = [
    "cz",
    "de",
    "en",
    "es",
    "fr",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tr",
    "uk",
]

_NUMBER_INPUTS = [
    "100.00",
    "22",
    "10000.0",
    "9.00100",
    "00.00",
    "987654210.123456789",
    "00.921",
    "100",
]

_NO_NUMBER_INPUTS = [
    "Parent00.1Safa",
    "0000Mother",
    "Mike Tyson",
    "?-!;%21+",
]


def test_should_get_number_normalizers():
    for lang in _LANGUAGES_TO_CHECK:
        get_number_normalizer(lang)


def test_should_normalize_number():
    for lang in _LANGUAGES_TO_CHECK:
        for number in _NUMBER_INPUTS:
            normalizer = get_number_normalizer(lang)
            output = normalizer.normalize(None, number, None)
            assert (
                output.has_normalized
            ), f"Failed check for language {lang}, input: {number}, output: {output}"
            assert (
                output.word != number
            ), f"Failed for language {lang}, input: {number}, output: {output}"
            assert (
                not output.collapsed_prev
            ), f"Failed check for language {lang}, input: {number}, output: {output}"
            assert (
                not output.collapsed_next
            ), f"Failed check for language {lang}, input: {number}, output: {output}"


def test_should_not_normalize_number():
    for lang in _LANGUAGES_TO_CHECK:
        for number in _NO_NUMBER_INPUTS:
            normalizer = get_number_normalizer(lang)
            output = normalizer.normalize(None, number, None)
            assert (
                not output.has_normalized
            ), f"Failed check for language {lang}, input: {number}, output: {output}"
            assert (
                not output.collapsed_prev
            ), f"Failed check for language {lang}, input: {number}, output: {output}"
            assert (
                not output.collapsed_next
            ), f"Failed check for language {lang}, input: {number}, output: {output}"

