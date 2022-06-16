def lang_to_mfa_acoustic(lang):
    if lang == "en":
        return "english_mfa"
    elif lang == "de":
        return "german_mfa"
    elif lang == "ru":
        return "russian_mfa"
    elif lang == "fr":
        return "french_mfa"
    elif lang == "es":
        return "spanish_mfa"
    raise Exception(f"No case selected in switch-statement - language '{lang}' is not supported ...")
