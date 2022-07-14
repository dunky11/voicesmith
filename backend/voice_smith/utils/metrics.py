from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import euclidean
from typing import Tuple
import torch

# from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
# from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from voice_smith.utils.audio import resample


def mcd(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the mel cepstrum

    :param a: np.ndarray of shape (timesteps_1, n_mels)
    :param b: np.ndarray of shape (timesteps_2, n_mels)
    :return: (np.ndarray, np.ndarray), which are the aligned versions of a and b,
        both of shape (max(timesteps_1, timesteps_2), n_mels)
    """
    K = 10 / np.log(10) * np.sqrt(2)
    return K * np.mean(np.sqrt(np.sum((a - b) ** 2, axis=1)))


def dtw_align(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Uses dynamic time warping to align two 2-dimensional numpy arrays.
    Returns the aligned versions of both a and b.

    :param a: np.ndarray of shape (timesteps_1, n_mels)
    :param b: np.ndarray of shape (timesteps_2, n_mels)
    :return: (np.ndarray, np.ndarray), which are the aligned versions of a and b,
        both of shape (max(timesteps_1, timesteps_2), n_mels)
    """
    _, warp_path = fastdtw(a, b, dist=euclidean)
    a_aligned = np.zeros((len(warp_path), max(a.shape[1], b.shape[1])))
    b_aligned = np.zeros((len(warp_path), max(a.shape[1], b.shape[1])))
    for i, (a_index, b_index) in enumerate(warp_path):
        a_aligned[i] = a[min(a_index, a.shape[0])]
        b_aligned[i] = b[min(b_index, b.shape[0])]
    return a_aligned, b_aligned


def mcd_dtw(a: np.ndarray, b: np.ndarray) -> float:
    a_aligned, b_aligned = dtw_align(a, b)
    distortion = mcd(a_aligned, b_aligned)
    return distortion


def calc_estoi(audio_real, audio_fake, sampling_rate):
    return torch.tensor([0.0], device=audio_fake.device, dtype=torch.float32)
    """return torch.mean(
        short_time_objective_intelligibility(
            audio_fake, audio_real, sampling_rate
        ).float()
    )"""


def calc_pesq(audio_real_16k, audio_fake_16k):
    return torch.tensor([0.0], device=audio_fake_16k.device, dtype=torch.float32)
    """return torch.mean(
        perceptual_evaluation_speech_quality(
            audio_fake_16k, audio_real_16k, 16000, "wb"
        )
    )"""


def calc_rmse(audio_real, audio_fake, stft):
    spec_real = stft.linear_spectrogram(audio_real.squeeze(1))
    spec_fake = stft.linear_spectrogram(audio_fake.squeeze(1))
    mse = torch.nn.functional.mse_loss(spec_fake, spec_real)
    rmse = torch.sqrt(mse)
    return rmse
