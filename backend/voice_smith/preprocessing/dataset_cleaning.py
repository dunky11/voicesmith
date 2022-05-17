from torch import nn
import torch
import numpy as np
from cleanlab.classification import CleanLearning
from skorch import NeuralNetClassifier
import multiprocessing as mp
from typing import List
from joblib import Parallel, delayed
from voice_smith.utils.tools import iter_logger


class LogisticRegression(nn.Module):
    def __init__(self, in_dim: int, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.softmax(x)
        return x


def get_issues(x: np.ndarray, y: np.ndarray, n_classes: int) -> None:
    model = LogisticRegression(x.shape[1], n_classes)
    model = NeuralNetClassifier(model)
    df = CleanLearning(model).find_label_issues(x, y)
    df = df[df["is_label_issue"]]
    df = df.sort_values(by=["label_quality"])
    indices, label_qualities = df.index.values, df.label_quality.values
    return indices, label_qualities


def _load_embedding(file_path):
    try:
        return torch.load(file_path)
    except Exception as e:
        print(e)
        return torch.zeros(
            192,
        )


def load_embeddings(file_paths: List[str]) -> List[torch.Tensor]:
    return Parallel(n_jobs=max(1, mp.cpu_count() - 1))(
        delayed(_load_embedding)(file_path) for file_path in iter_logger(file_paths)
    )
