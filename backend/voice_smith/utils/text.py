from voice_smith.config.symbols import symbol2id, pad
from typing import List


def phones_to_token_ids(phones: List[str]) -> List[int]:
    token_ids = []
    for phone in phones:
        if _should_keep_symbol(phone):
            token_ids.append(symbol2id[phone])
        else:
            raise Exception(f"Symbol {phone} is not a valid phone ...")
    return token_ids


def _should_keep_symbol(s: str) -> bool:
    return s in symbol2id and s != pad


def strip_cont_whitespaces(string: str) -> str:
    new_string = ""
    last_whitespace = False
    for char in string:
        if char == " " and last_whitespace:
            continue
        new_string += char
        last_whitespace = char == " "
    return new_string
