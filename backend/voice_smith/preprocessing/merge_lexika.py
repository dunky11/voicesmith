from pathlib import Path


def lex_to_dict(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    dic = {}
    for line in lines:
        if line.strip() == "":
            continue
        split = line.split()
        key = split[0].strip().lower()
        value = " ".join(split[1:])
        # Some dicts contain lower case phones, which is incorrect
        value = value.upper()
        dic[key] = value.strip()
    return dic


def merge_lexica(base_lexica_path: str, lang: str, assets_path: str, out_path: str):
    dic_final = {}
    for lex in list((Path(assets_path) / "lexica" / lang).iterdir()) + [
        base_lexica_path
    ]:
        dic = lex_to_dict(str(lex))
        for key in dic.keys():
            if not key in dic_final:
                dic_final[key] = dic[key]

    with open(out_path, "w", encoding="utf-8") as f:
        for key in dic_final.keys():
            f.write(key + " " + dic_final[key] + "\n")

