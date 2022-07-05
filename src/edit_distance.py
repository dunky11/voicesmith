import numpy as np
from typing import List


def levenshtein_distance(s1, s2):
    """ https://stackoverflow.com/questions/2460177/edit-distance-in-python
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def normalized_levenshtein_distance(x: str, y: str):
    max_len = max(len(x), len(y))
    return levenshtein_distance(x, y) / max_len

