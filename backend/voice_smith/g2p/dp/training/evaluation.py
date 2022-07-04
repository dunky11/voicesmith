from typing import List, Tuple, Dict, Any
from voice_smith.g2p.dp.training.metrics import phoneme_error, word_error


def evaluate_samples(
    lang_samples: Dict[str, List[Tuple[List[str], List[str], List[str]]]],
    lang_to_word_to_gold: Dict[str, Dict[str, List[List[str]]]],
) -> Dict[str, Any]:
    """Calculates word and phoneme error rates per language and their mean across languages

    Args:
      lang_samples (Dict): Data to evaluate. Contains languages as keys and list of result samples as values.
                           Prediction samples is given as a List of Tuples, where each Tuple is a tokenized representation of
                           (text, result, target).

    Returns:
      Dict: Evaluation result carrying word and phoneme error rates per language.

    """

    evaluation_result = dict()
    lang_phon_err, lang_phon_count, lang_word_err = dict(), dict(), dict()
    languages = sorted(lang_samples.keys())
    for lang in languages:
        for word, generated, _ in lang_samples[lang]:
            lang_to_word_to_gold
            word = "".join(word)

    phon_errors, phon_counts, word_errors, word_counts = [], [], [], []
    for lang in languages:
        phon_err = sum(lang_phon_err[lang].values())
        phon_errors.append(phon_err)
        phon_count = sum(lang_phon_count[lang].values())
        phon_counts.append(phon_count)
        word_err = sum(lang_word_err[lang].values())
        word_errors.append(word_err)
        word_count = len(lang_word_err[lang])
        word_counts.append(word_count)
        per = phon_err / phon_count
        wer = word_err / word_count
        evaluation_result.setdefault(lang, {}).update({"per": per})
        evaluation_result.setdefault(lang, {}).update({"wer": wer})
    mean_per = sum(phon_errors) / sum(phon_counts)
    mean_wer = sum(word_errors) / sum(word_counts)
    evaluation_result["mean_per"] = mean_per
    evaluation_result["mean_wer"] = mean_wer

    return evaluation_result
