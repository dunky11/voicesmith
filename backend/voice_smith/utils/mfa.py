from voice_smith.config.configs import PreprocessLangType


def lang_to_mfa_acoustic(lang: str, language_type: PreprocessLangType):
    if lang == "bg":
        return "bulgarian_mfa"
    elif lang == "cs":
        return "czech_mfa"
    elif lang == "de":
        return "german_mfa"
    elif lang == "en":
        if language_type == "english_only":
            return "english_us_arpa"
        else:
            return "english_mfa"
    elif lang == "es":
        return "spanish_mfa"
    elif lang == "fr":
        return "french_mfa"
    elif lang == "hr":
        return "croatian_mfa"
    elif lang == "pl":
        return "polish_mfa"
    elif lang == "pt":
        return "portuguese_mfa"
    elif lang == "ru":
        return "russian_mfa"
    elif lang == "sv":
        return "swedish_mfa"
    elif lang == "th":
        return "thai_mfa"
    elif lang == "tr":
        return "turkish_mfa"
    elif lang == "uk":
        return "ukrainian_mfa"
    raise Exception(
        f"No case selected in switch-statement - language '{lang}' is not supported ..."
    )


def lang_to_mfa_g2p(lang, language_type: PreprocessLangType):
    if lang == "bg":
        return "bulgarian_mfa"
    elif lang == "cs":
        return "czech_mfa"
    elif lang == "de":
        return "german_mfa"
    elif lang == "en":
        if language_type == "english_only":
            return "english_us_arpa"
        else:
            return "english_mfa"
    elif lang == "es":
        return "spanish_mfa"
    elif lang == "fr":
        return "french_mfa"
    elif lang == "hr":
        return "croatian_mfa"
    elif lang == "pl":
        return "polish_mfa"
    elif lang == "pt":
        return "portuguese_mfa"
    elif lang == "ru":
        return "russian_mfa"
    elif lang == "sv":
        return "swedish_mfa"
    elif lang == "th":
        return "thai_mfa"
    elif lang == "tr":
        return "turkish_mfa"
    elif lang == "uk":
        return "ukrainian_mfa"
    raise Exception(
        f"No case selected in switch-statement - language '{lang}' is not supported ..."
    )
