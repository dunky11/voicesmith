import numpy as np
from typing import List, Any, Tuple
import torch

def unison_shuffled_copies(a: List[Any], b: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    a_out = [a[idx] for idx in p]
    b_out = [b[idx] for idx in p]
    return a_out, b_out

def stratified_train_test_split(x: List[Any], y: List[Any], train_size: float) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    label2samples = {}
    for x, y in zip(x, y):
        if y in label2samples:
            label2samples[y].append(x)
        else:
            label2samples[y] = [x]
    train_x_out, train_y_out = [], []
    val_x_out, val_y_out = [], []
    for label, samples in label2samples.items():
        split_at = int(np.round(len(samples) * train_size))
        x_split_train, x_split_val = samples[:split_at], samples[split_at:]
        y_split_train, y_split_val = [label] * len(x_split_train), [label] * len(x_split_val)
        train_x_out.extend(x_split_train)
        train_y_out.extend(y_split_train)
        val_x_out.extend(x_split_val)
        val_y_out.extend(y_split_val)
    train_x_out, train_y_out = unison_shuffled_copies(train_x_out, train_y_out)
    val_x_out, val_y_out = unison_shuffled_copies(val_x_out, val_y_out)
    return train_x_out, val_x_out, train_y_out, val_y_out

class OnlineScaler():
    """ Online mean and variance computation, see
    http://www.cs.yale.edu/publications/techreports/tr222.pdf
    equation 1.5a and 1.5b
    """
    t_1_m = None
    s_1_m = None
    m = 0

    def partial_fit(self, x: torch.Tensor) -> None:
        assert(len(x.shape) > 1), "First dimension to partial_fit must be batch size"
        if self.m == 0:
            self.t_1_m = x[0]
            self.s_1_m = 0.0
            if x.shape[0] > 1:
                self.m += 1
                self.partial_fit(x[1:])
        else:
            n = x.shape[0]
            x_sum = x.sum(0)
            self.s_1_m = self.s_1_m + x.var(0) + (self.m / (n * (self.m + n))) * ((n / self.m) * self.t_1_m - x_sum) ** 2
            self.t_1_m = self.t_1_m + x_sum
            self.m += n

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.t_1_m / self.m, torch.sqrt(self.s_1_m)

if __name__ == "__main__":
    def count_labels(xs):
        label2count = {}
        for x in xs:
            if x in label2count:
                label2count[x] += 1
            else:
                label2count[x] = 1
        return label2count

    samples = (["0_sample"] * 800000) + (["1_sample"] * 180000) + (["2_sample"] * 20000) + (["3_sample"] * 10000)
    labels = (["0_label"] * 800000) + (["1_label"] * 180000) + (["2_label"] * 20000) + (["3_label   "] * 10000)
    train_x_out, val_x_out, train_y_out, val_y_out = stratified_train_test_split(samples, labels, train_size=0.9)

    print(count_labels(train_x_out))
    print(count_labels(train_y_out))
    print(count_labels(val_x_out))
    print(count_labels(val_y_out))

    a = torch.randn((100000, 10))
    scaler = OnlineScaler()
    scaler.partial_fit(a)
    
    print(scaler_sklearn._mean, scaler_sklearn._std)

    print(a.mean(0), a.var(0))
    print(scaler.get_mean_var())