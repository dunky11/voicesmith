import numpy
from typing import List, Union, Tuple, Dict, Set


def compute_validation_errors(
    self, gold_values: Dict[str, Set[str]], hypothesis_values: Dict[str, List[str]],
):
    """ From MFA: https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/blob/a99c93730e4d9d2f6f5562cfb831c98b96f69c0e/montreal_forced_aligner/g2p/generator.py
    Computes validation errors
    Parameters
    ----------
    gold_values: dict[str, set[str]]
        Gold pronunciations
    hypothesis_values: dict[str, list[str]]
        Hypothesis pronunciations
    """
    # Word-level measures.
    correct = 0
    incorrect = 0
    # Label-level measures.
    total_edits = 0
    total_length = 0
    # Since the edit distance algorithm is quadratic, let's do this with
    # multiprocessing.
    self.log_debug(f"Processing results for {len(hypothesis_values)} hypotheses")
    to_comp = []
    indices = []
    hyp_pron_count = 0
    gold_pron_count = 0
    output = []
    for word, gold_pronunciations in gold_values.items():
        if word not in hypothesis_values:
            incorrect += 1
            gold_length = statistics.mean(len(x.split()) for x in gold_pronunciations)
            total_edits += gold_length
            total_length += gold_length
            output.append(
                {
                    "Word": word,
                    "Gold pronunciations": ", ".join(gold_pronunciations),
                    "Hypothesis pronunciations": "",
                    "Accuracy": 0,
                    "Error rate": 1.0,
                    "Length": gold_length,
                }
            )
            continue
        hyp = hypothesis_values[word]
        for h in hyp:
            if h in gold_pronunciations:
                correct += 1
                total_length += len(h)
                output.append(
                    {
                        "Word": word,
                        "Gold pronunciations": ", ".join(gold_pronunciations),
                        "Hypothesis pronunciations": ", ".join(hyp),
                        "Accuracy": 1,
                        "Error rate": 0.0,
                        "Length": len(h),
                    }
                )
                break
        else:
            incorrect += 1
            indices.append(word)
            to_comp.append((gold_pronunciations, hyp))  # Multiple hypotheses to compare
        self.log_debug(
            f"For the word {word}: gold is {gold_pronunciations}, hypothesized are: {hyp}"
        )
        hyp_pron_count += len(hyp)
        gold_pron_count += len(gold_pronunciations)
    self.log_debug(
        f"Generated an average of {hyp_pron_count /len(hypothesis_values)} variants "
        f"The gold set had an average of {gold_pron_count/len(hypothesis_values)} variants."
    )
    with mp.Pool(self.num_jobs) as pool:
        gen = pool.starmap(score_g2p, to_comp)
        for i, (edits, length) in enumerate(gen):
            word = indices[i]
            gold_pronunciations = gold_values[word]
            hyp = hypothesis_values[word]
            output.append(
                {
                    "Word": word,
                    "Gold pronunciations": ", ".join(gold_pronunciations),
                    "Hypothesis pronunciations": ", ".join(hyp),
                    "Accuracy": 1,
                    "Error rate": edits / length,
                    "Length": length,
                }
            )
            total_edits += edits
            total_length += length

    wer = 100 * incorrect / (correct + incorrect)
    per = 100 * total_edits / total_length

    return wer, per


def word_error(
    predicted: List[Union[str, int]], target: List[Union[str, int]]
) -> float:
    """Calculates the word error rate of a single word result.

    Args:
      predicted: Predicted word.
      target: Target word.
      predicted: List[Union[str: 
      int]]: 
      target: List[Union[str: 

    Returns:
      Word error

    """

    return int(predicted != target)


def phoneme_error(
    predicted: List[Union[str, int]], target: List[Union[str, int]]
) -> Tuple[int, int]:
    """Calculates the phoneme error rate of a single result based on the Levenshtein distance.

    Args:
      predicted: Predicted word.
      target: Target word.
      predicted: List[Union[str: 
      int]]: 
      target: List[Union[str: 

    Returns:
      Phoneme error.

    """

    d = numpy.zeros((len(target) + 1) * (len(predicted) + 1), dtype=numpy.uint8)
    d = d.reshape((len(target) + 1, len(predicted) + 1))
    for i in range(len(target) + 1):
        for j in range(len(predicted) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(target) + 1):
        for j in range(1, len(predicted) + 1):
            if target[i - 1] == predicted[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(target)][len(predicted)], len(target)

