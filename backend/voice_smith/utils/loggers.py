import numpy as np
import sys
from typing import Union, List
import matplotlib

class DualLogger(object):
    # https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    # Writes to both stdout and terminal
    def __init__(self, location: str, mode: str):
        self.terminal = sys.stdout
        self.log = open(location, mode, encoding="utf-8")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass  

def set_stream_location(location: str) -> None:
    sys.stdout = DualLogger(location, "w")
    sys.stderr = sys.stdout


class Logger:
    def __init__(self):
        self.cm = matplotlib.cm.get_cmap("plasma")

    def map_image_color(self, image: np.ndarray) -> None:
        normed_data = (image - np.min(image)) / (np.max(image) - np.min(image))
        mapped_data = self.cm(normed_data)
        return mapped_data

    def log_image(self, name: str, image: np.ndarray, step: int) -> None:
        raise NotImplementedError

    def log_graph(self, name: str, value: float, step: int) -> None:
        raise NotImplementedError

    def log_audio(self, name: str, audio: np.ndarray, step: int, sr: int) -> None:
        raise NotImplementedError

    def query(self, query: str, args: List[Union[str, int]]) -> None:
        raise NotImplementedError
