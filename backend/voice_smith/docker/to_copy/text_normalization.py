from pathlib import Path
import fire
from nemo_text_processing.text_normalization.normalize import Normalizer
import multiprocessing as mp
from typing import Tuple, List, Dict, Any
import multiprocessing as mp
from joblib import Parallel, delayed
import sqlite3

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
    "corp",
    "dept",
    "dr",
    "oz",
    "ft",
    "gal",
    "hr",
    "inc",
    "jr",
    "km",
    "ltd",
    "mg",
    "mm",
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
ENGLISH_ABBREVIATIONS = [
    "i.e",
    "e.g",
    "ed",
    "est",
    "fl",
    "oz",
    "sq",
    "mon",
    "tu",
    "tue",
    "tues",
    "wed",
    "th",
    "thu",
    "thur",
    "thurs",
    "fri",
    "sat",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sept",
    "oct",
    "nov",
    "dec",
]
SPANISH_ABBREVIATIONS = [
    "feb",
    "abr",
    "jun",
    "jul",
    "set",
    "oct",
    "nov",
    "dic",
    "dra",
    "profa",
    "pdta",
    "arq",
    "mtro",
    "mtra",
    "psic",
    "lu",
    "Ma",
    "Mi",
    "Ju",
    "Vi",
    "Sa",
    "Do",
]
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
ENGLISH_INITIALISMS = ["PM", "pm", "a.m.", "p.m.", "A.M", "P.M."]
GERMAN_INITIALISMS = []
SPANISH_INITIALISMS = [
    "a.D.g.",
    "A.H.",
    "A.T.",
    "ACS",
    "ACV",
    "ADE",
    "ADN",
    "AEC",
    "AL",
    "ALDF",
    "ALV",
    "alv",
    "ANHQV",
    "ANSV",
    "ARN",
    "AT",
    "AUE",
    "AUH",
    "AXJ",
    "BCB",
    "BCE",
    "bdd",
    "BID",
    "BM",
    "BOE",
    "BOPE",
    "bpd",
    "BS",
    "BTT",
    "C.A.",
    "CA",
    "CAE",
    "CAISS",
    "CCD",
    "CCNCC",
    "CDS",
    "CEI",
    "CEPAL",
    "CES",
    "CF",
    "CFC",
    "CGPJ",
    "CHE",
    "CI",
    "CJNG",
    "CNI",
    "CNMC",
    "CNT",
    "CSD",
    "CyL",
    "D.F.",
    "dana",
    "DANA",
    "DCV",
    "DD.HH.",
    "DEP",
    "DF",
    "DGT",
    "DLE",
    "DNI",
    "DPN",
    "DRAE",
    "DT",
    "e.d.",
    "EAU",
    "ECG",
    "EDAR",
    "EE.UU.",
    "EGDE",
    "ELN",
    "EPD",
    "EPOC",
    "ERC",
    "ETS",
    "ETT",
    "EUA",
    "FA",
    "FCF",
    "FMI",
    "FMLN",
    "FRA",
    "FSLN",
    "GC",
    "GH",
    "GNL",
    "HBP",
    "HDA",
    "HDB",
    "HDLGP",
    "HDP",
    "IA",
    "IBEX",
    "ICEX",
    "ICFT",
    "IDH",
    "IES",
    "IFE",
    "IGN",
    "IMAO",
    "IMC",
    "IME",
    "IMSS",
    "INAH",
    "INSS",
    "IRA",
    "IRPF",
    "ISRS",
    "ITS",
    "ITU",
    "IU",
    "JCE",
    "JLB",
    "JRG",
    "LATAM",
    "LQSA",
    "LSE",
    "mcd",
    "mcm",
    "mdd",
    "mde",
    "MDP",
    "mdp",
    "MDQ",
    "MIR",
    "MPR",
    "msnm",
    "MYHYV",
    "NIE",
    "NMC",
    "NNA",
    "NOM",
    "NPI",
    "OEA",
    "OGM",
    "OMC",
    "OMI",
    "OMS",
    "OMT",
    "PA",
    "PAN",
    "PBC",
    "PCC",
    "PCR",
    "PCUS",
    "PFM",
    "PIN",
    "PNL",
    "PNV",
    "PP",
    "PPE",
    "PPK",
    "PPP",
    "PRI",
    "PRM",
    "PSUV",
    "PUC",
    "q.e.p.d.",
    "QR",
    "RAAN",
    "RAAS",
    "RAE",
    "RCN",
    "RCP",
    "RDSI",
    "RPC",
    "RU",
    "S.A.",
    "S.A.",
    "S.T.D.",
    "SAG",
    "SAI",
    "SARM",
    "SCA",
    "SD",
    "SHCP",
    "SMI",
    "SRL",
    "TAC",
    "TAPO",
    "TC",
    "TCA",
    "TDAH",
    "TEDH",
    "TEP",
    "THC",
    "TIC",
    "TKM",
    "TLC",
    "TLCAN",
    "TMA",
    "TOC",
    "TU",
    "TV3",
    "TVE",
    "UBA",
    "UE",
    "UGT",
    "URSS",
    "VMP",
    "ZEC",
]
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

    def should_normalize(self, text: str) -> Tuple[bool, str]:
        raise NotImplementedError()

    def get_reasons(
        self,
        text: str,
        abbreviations: Dict[str, Any],
        initialisms: Dict[str, Any],
        allowed_characters: Dict[str, Any],
    ) -> List[str]:
        reasons, abbr_detected, inits_detected, unusual_chars_detected = [], [], [], []
        for token in text.split(" "):
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

    def expand_with_punct(
        self, to_expand: List[str], punctutations: List[str]
    ) -> List[str]:
        out = to_expand
        for punct in punctutations:
            out.extend([f"{el}{punct}" for el in to_expand])
        return out

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

        self.abbreviations = set(self.expand_with_punct(abbr, en_punctuation))
        self.initialisms = set(self.expand_with_punct(initialisms, en_punctuation))

    def should_normalize(self, text: str) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text, self.abbreviations, self.initialisms, self.allowed_characters
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
        self.abbreviations = set(self.expand_with_punct(abbr, es_punctuation))
        self.initialisms = set(self.expand_with_punct(initialisms, es_punctuation))

    def should_normalize(self, text: str) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text, self.abbreviations, self.initialisms, self.allowed_characters
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
        self.abbreviations = set(self.expand_with_punct(abbr, de_punctuation))
        self.initialisms = set(self.expand_with_punct(initialisms, de_punctuation))

    def should_normalize(self, text: str) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text, self.abbreviations, self.initialisms, self.allowed_characters
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
        self.abbreviations = set(self.expand_with_punct(abbr, ru_punctuation))
        self.initialisms = set(self.expand_with_punct(initialisms, ru_punctuation))

    def should_normalize(self, text: str) -> Tuple[bool, List[str]]:
        reasons = self.get_reasons(
            text, self.abbreviations, self.initialisms, self.allowed_characters
        )
        should_normalize = len(reasons) > 0
        return should_normalize, reasons


def apply_nemo_normalization(text: str, normalizer: Normalizer):
    text_in = text
    text_out = normalizer.normalize(
        text=text_in,
        verbose=False,
        punct_post_process=True,
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
):
    text_out, reasons_char_norm = char_normalizer.normalize(text_in)
    should_normalize, reasons_det = detector.should_normalize(text_out)
    if should_normalize:
        text_out = apply_nemo_normalization(
            text=text_out,
            normalizer=normalizer,
        )

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
    elif lang == "es":
        detector = DetShouldNormalizeES()
        deterministic = True
    elif lang == "de":
        detector = DetShouldNormalizeDE()
        deterministic = True
    elif lang == "ru":
        detector = DetShouldNormalizeRU()
        deterministic = False
    else:
        raise Exception(
            f"No case selected in switch-statement, '{lang}' is not a valid case ..."
        )

    char_normalizer = CharNormalizer()
    normalizer = Normalizer(
        input_case="cased",
        lang=lang,
        overwrite_cache=False,
        deterministic=deterministic,
    )

    rets = []

    for _, text in id_text_pairs:
        ret = normalize_text(
            text_in=text,
            char_normalizer=char_normalizer,
            detector=detector,
            normalizer=normalizer,
            deterministic=deterministic,
        )
        rets.append(ret)

        normalized, text_in, text_out, reason = ret
        if normalized:
            print("Text in: ", text_in)
            print("Text normalized: ", text_out)
            print("Reason: ", reason)
            print("", flush=True)

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

    con.commit()


if __name__ == "__main__":
    fire.Fire(normalize)
