import numpy as np
import sys
from typing import Union, List


def set_stream_location(location: str) -> None:
    sys.stdout = open(location, "w")
    sys.stderr = open(location, "w")


class Logger:
    def __init__(self):
        # self.cm = matplotlib.cm.get_cmap("plasma")
        pass
    
    def map_image_color(self, image: np.ndarray) -> None:
        normed_data = (image - np.min(image)) / (np.max(image) - np.min(image))
        # mapped_data = self.cm(normed_data)
        return normed_data

    def log_image(self, name: str, image: np.ndarray, step: int) -> None:
        raise NotImplementedError

    def log_graph(self, name: str, value: float, step: int) -> None:
        raise NotImplementedError

    def log_audio(self, name: str, audio: np.ndarray, step: int, sr: int) -> None:
        raise NotImplementedError

    def query(self, query: str, args: List[Union[str, int]]) -> None:
        raise NotImplementedError
