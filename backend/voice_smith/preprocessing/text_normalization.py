from pathlib import Path
import fire
from nemo_text_processing.text_normalization.normalize import Normalizer
import multiprocessing as mp
from typing import Tuple, List, Dict, Any
import multiprocessing as mp
from joblib import Parallel, delayed
import sqlite3
import argparse
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ru import Russian
from spacy.lang.de import German

LATIN_CHARACTERS = list("abcdefghijklmnopqrstuvwxyz")
GERMAN_CHARACTERS = list("öüäß")
SPANISH_CHARACTERS = list("üúóñíéá")
RUSSIAN_CHARACTERS = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".lower())

MULTILINGUAL_PUNCTUATION = list(".,!?:-")
ENGLISH_PUNCTUATION = list("")
SPANISH_PUNCTUATION = list("¿¡«»—")
GERMAN_PUNCTUATION = list("")
RUSSIAN_PUNCTUATION = list("")

SPECIALS = list("\"' ")

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
            ("…", "..."),
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

    def normalize(self, text: str) -> List[str]:
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
    Nemos text normalizer unfortunately produces a large amount of false positives.
    For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'.
    To reduce the amount of false postives we will do a check for unusual symbols
    or words inside the text and only normalize if necessary.
    """

    def should_normalize(self, text: str, tokenizer: Tokenizer) -> Tuple[bool, str]:
        raise NotImplementedError()

    def get_reasons(
        self,
        text: str,
        abbreviations: Dict[str, Any],
        initialisms: Dict[str, Any],
        allowed_characters: Dict[str, Any],
        tokenizer: Tokenizer,
    ) -> List[str]:
        reasons, abbr_detected, inits_detected, unusual_chars_detected = [], [], [], []
        for token in tokenizer(text):
            token = str(token)
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

    def fetch_word_list(self, name: str):
        lines = []
        with open(Path(".") / "word_lists" / name, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(line.strip())
        return lines


class DetShouldNormalizeEN(DetShouldNormalizeBase):
    def __init__(self):
        super().__init__()
        en_punctuation = MULTILINGUAL_PUNCTUATION + ENGLISH_PUNCTUATION
        self.allowed_characters = set(LATIN_CHARACTERS + SPECIALS + en_punctuation)
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + ENGLISH_ABBREVIATIONS
            + self.fetch_word_list("english_abbreviations.txt")
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + ENGLISH_INITIALISMS
            + self.fetch_word_list("english_initialisms.txt")
        )

        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: Tokenizer
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
    def __init__(self):
        super().__init__()
        es_punctuation = MULTILINGUAL_PUNCTUATION + SPANISH_PUNCTUATION
        self.allowed_characters = set(
            LATIN_CHARACTERS + SPECIALS + SPANISH_CHARACTERS + es_punctuation
        )
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + SPANISH_ABBREVIATIONS
            + self.fetch_word_list("spanish_abbreviations.txt")
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + self.fetch_word_list("spanish_initialisms.txt")
            + SPANISH_INITIALISMS
        )
        self.abbreviations = set(abbr, es_punctuation)
        self.initialisms = set(initialisms, es_punctuation)

    def should_normalize(
        self, text: str, tokenizer: Tokenizer
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
    def __init__(self):
        super().__init__()
        de_punctuation = MULTILINGUAL_PUNCTUATION + GERMAN_PUNCTUATION
        self.allowed_characters = set(
            LATIN_CHARACTERS + SPECIALS + GERMAN_CHARACTERS + de_punctuation
        )
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + GERMAN_ABBREVIATIONS
            + self.fetch_word_list("german_abbreviations.txt")
        )
        initialisms = (
            MULTILINGUAL_INITIALISMS
            + self.fetch_word_list("german_initialisms.txt")
            + GERMAN_INITIALISMS
        )
        self.abbreviations = set(abbr)
        self.initialisms = set(initialisms)

    def should_normalize(
        self, text: str, tokenizer: Tokenizer
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
    def __init__(self):
        super().__init__()
        ru_punctuation = MULTILINGUAL_PUNCTUATION + RUSSIAN_PUNCTUATION
        self.allowed_characters = set(RUSSIAN_CHARACTERS + SPECIALS + ru_punctuation)
        abbr = (
            MULTILINGUAL_ABBREVIATIONS
            + RUSSIAN_ABBREVIATIONS
            + self.fetch_word_list("russian_abbreviations.txt")
        )
        initialisms = (
            self.fetch_word_list("russian_initialisms.txt") + RUSSIAN_INITIALISMS
        )
        self.abbreviations = set(abbr, ru_punctuation)
        self.initialisms = set(initialisms, ru_punctuation)

    def should_normalize(
        self, text: str, tokenizer: Tokenizer
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


def normalize_text(
    text_in: str,
    char_normalizer: CharNormalizer,
    detector: DetShouldNormalizeEN,
    normalizer: Normalizer,
    deterministic: bool,
    tokenizer: Tokenizer,
):
    text_out, reasons_char_norm = char_normalizer.normalize(text_in)
    should_normalize, reasons_det = detector.should_normalize(text_out, tokenizer)
    if should_normalize:
        text_out = apply_nemo_normalization(text=text_out, normalizer=normalizer)

    reason = ". ".join(reasons_char_norm + reasons_det)
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


def normalize(ID: int, run_type: str, lang: str):
    con = sqlite3.connect(Path(".") / "db" / "voice_smith.db")
    cur = con.cursor()
    id_text_pairs = []
    if run_type == "textNormalizationRun":
        cur.execute(
            "DELETE FROM text_normalization_sample WHERE text_normalization_run_id = ?",
            (ID,),
        )
        con.commit()

        for (sample_id, text) in cur.execute(
            """
            SELECT sample.ID AS sampleID, sample.text FROM sample
            INNER JOIN speaker ON sample.speaker_id = speaker.ID
            INNER JOIN dataset on speaker.dataset_id = dataset.ID
            INNER JOIN text_normalization_run ON text_normalization_run.dataset_id = dataset.ID
            WHERE text_normalization_run.ID = ?
            """,
            (ID,),
        ).fetchall():
            id_text_pairs.append((sample_id, text))
    else:
        raise Exception(
            f"No case selected in switch-statement, '{run_type}' is not a valid case ..."
        )

    if lang == "en":
        detector = DetShouldNormalizeEN()
        deterministic = True
        nlp = English()
    elif lang == "es":
        detector = DetShouldNormalizeES()
        deterministic = True
        nlp = Spanish()
    elif lang == "de":
        detector = DetShouldNormalizeDE()
        deterministic = True
        nlp = German()
    elif lang == "ru":
        detector = DetShouldNormalizeRU()
        deterministic = False
        nlp = Russian()
    else:
        raise Exception(
            f"No case selected in switch-statement, '{lang}' is not a valid case ..."
        )
    tokenizer = nlp.tokenizer

    char_normalizer = CharNormalizer()
    normalizer = Normalizer(
        input_case="cased",
        lang=lang,
        overwrite_cache=False,
        deterministic=deterministic,
    )

    rets = []

    for i, (_, text) in enumerate(id_text_pairs):
        ret = normalize_text(
            text_in=text,
            char_normalizer=char_normalizer,
            detector=detector,
            normalizer=normalizer,
            deterministic=deterministic,
            tokenizer=tokenizer,
        )
        rets.append(ret)
        normalized, text_in, text_out, reason = ret
        if normalized:
            print("Text in: ", text_in, flush=True)
            print("Text normalized: ", text_out, flush=True)
            print("Reason: ", reason, flush=True)
            print("", flush=True)

        if i % 100 == 0 and i != 0:
            progress = ((i + 1) / len(id_text_pairs)) * 0.9
            cur.execute(
                "UPDATE text_normalization_run SET text_normalization_progress = ? WHERE ID=?",
                (progress, ID),
            )
            con.commit()

    cur.execute(
        "UPDATE text_normalization_run SET text_normalization_progress=0.9 WHERE ID=?",
        (ID,),
    )
    con.commit()

    for (sample_id, _), (has_normalized, text_in, text_out, reason) in zip(
        id_text_pairs, rets
    ):
        if run_type == "textNormalizationRun":
            if has_normalized:
                cur.execute(
                    """
                    INSERT INTO text_normalization_sample (old_text, new_text, reason, sample_id, text_normalization_run_id) 
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (text_in, text_out, reason, sample_id, ID),
                )
        else:
            raise Exception(
                f"No case selected in switch-statement, '{run_type}' is not a valid case ..."
            )

    cur.execute(
        "UPDATE text_normalization_run SET text_normalization_progress=1.0 WHERE ID=?",
        (ID,),
    )
    con.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_run_id", type=int, required=True)
    parser.add_argument("--run_type", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()

    normalize(ID=args.training_run_id, run_type=args.run_type, lang=args.lang)
