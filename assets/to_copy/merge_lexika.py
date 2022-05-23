from pathlib import Path
import fire


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


def merge_lexica(training_run_name: str):
    training_run_name = str(training_run_name)
    dic_final = {}
    training_run_path = Path(".") / "data" / "training_runs" / training_run_name / "data"
    for lex in list((Path(".") / "lexica" / "english").iterdir()) + [
        training_run_path / "lexicon_pre.txt"
    ]:
        dic = lex_to_dict(str(lex))
        for key in dic.keys():
            if not key in dic_final:
                dic_final[key] = dic[key]

    with open(training_run_path / "lexicon_post.txt", "w", encoding="utf-8") as f:
        for key in dic_final.keys():
            f.write(key + " " + dic_final[key] + "\n")


if __name__ == "__main__":
    fire.Fire(merge_lexica)
