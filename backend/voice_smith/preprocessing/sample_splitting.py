from dataclasses import dataclass
from typing import List, Union, Literal, Dict
import tgt
from voice_smith.utils.tokenization import SentenceTokenizer

@dataclass
class Split():
    text: str
    from_msecs: float
    to_msecs: float

@dataclass
class SampleSplit():
    sample_id: int
    text: str
    lang: str
    splits: List[Split]

def sample_splitting(
    ids: List[int], 
    texts: List[str], 
    textgrid_paths: List[str], 
    languages: List[Union[Literal["en"]]]
) -> List[Split]:
    assert len(ids) == len(texts) == len(textgrid_paths) == len(languages)
    lang_to_info = {}

    for sample_id, text, textgrid_path, lang in zip(ids, texts, textgrid_paths, languages):
        info = (sample_id, text, textgrid_path)
        if lang not in lang_to_info:
            lang_to_info[lang] = [info]
        else:
            lang_to_info[lang].append(info)
    
    for lang, infos in lang_to_info.items():
        tokenizer = SentenceTokenizer(lang)
        for sample_id, text, textgrid_path in infos:
            sentences = tokenizer.tokenize(text)
        textgrid = tgt.io.read_textgrid(textgrid_path)
        words_tier = textgrid.get_tier_by_name("words")
        print(sentences, words_tier)