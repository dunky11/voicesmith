import torch
import math
import numpy as np
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
import librosa
import soundfile as sf
from voice_smith.utils.librosa import mel as librosa_mel_fn


def save_audio(file_path: str, audio: torch.Tensor, sr: int):
    sf.write(file_path, audio.numpy(), sr)


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    return np.mean(audio, axis=0, keepdims=True)


def get_mel_from_wav(audio: np.ndarray, to_mel: torch.nn.Module) -> np.ndarray:
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    with torch.no_grad():
        _, melspec = to_mel(audio_tensor)
    melspec = melspec.squeeze(0).cpu().numpy().astype(np.float32)
    return melspec


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    return wav


def safe_load(
    path: str, sr: Union[int, None], verbose: bool = True
) -> Tuple[np.ndarray, int]:
    try:
        audio, sr_actual = librosa.load(path, sr=sr)

    except Exception as e:
        import sys

        raise type(e)(
            f"The following error happened loading the file {path} ... \n" + str(e)
        ).with_traceback(sys.exc_info()[2])

    return audio, sr_actual


class MelSpec2MFCC(torch.nn.Module):
    def __init__(self, n_mfcc: int, n_mels: int):
        super().__init__()
        self.dct_mat = create_dct(n_mfcc, n_mels, "ortho")

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        mfcc = torch.matmul(mel_spec.transpose(-1, -2), self.dct_mat).transpose(-1, -2)
        return mfcc


def create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> torch.Tensor:
    """FROM https://pytorch.org/audio/main/_modules/torchaudio/functional/functional.html
    Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    .. devices:: CPU

    .. properties:: TorchScript

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either 'ortho' or None)

    Returns:
        Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """
    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = torch.arange(float(n_mels))
    k = torch.arange(float(n_mfcc)).unsqueeze(1)
    dct = torch.cos(math.pi / float(n_mels) * (n + 0.5) * k)  # size (n_mfcc, n_mels)
    if norm is None:
        dct *= 2.0
    else:
        assert norm == "ortho"
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.t()


def differenceFunction(x: np.ndarray, N: int, tau_max: int) -> np.ndarray:
    """
    Compute difference function of data x. This corresponds to equation (6) in [1]
    This solution is implemented directly with Numpy fft.
    :param x: audio data
    :param N: length of data
    :param tau_max: integration window size
    :return: difference function
    :rtype: list
    """

    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2 ** p2 for x in nice_numbers if x * 2 ** p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w : w - tau_max : -1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df: np.ndarray, N: int) -> np.ndarray:
    """
    Compute cumulative mean normalized difference function (CMND).
    This corresponds to equation (8) in [1]
    :param df: Difference function
    :param N: length of data
    :return: cumulative mean normalized difference function
    :rtype: list
    """
    cmndf = (
        df[1:] * range(1, N) / (np.cumsum(df[1:]).astype(float) + 1e-8)
    )  # scipy method
    return np.insert(cmndf, 0, 1)


def getPitch(cmdf: np.ndarray, tau_min: int, tau_max: int, harmo_th=0.1) -> int:
    """
    Return fundamental period of a frame based on CMND function.
    :param cmdf: Cumulative Mean Normalized Difference function
    :param tau_min: minimum period for speech
    :param tau_max: maximum period for speech
    :param harmo_th: harmonicity threshold to determine if it is necessary to compute pitch frequency
    :return: fundamental period if there is values under threshold, 0 otherwise
    :rtype: float
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0  # if unvoiced


def compute_yin(
    sig: np.ndarray,
    sr: int,
    w_len: int = 512,
    w_step: int = 256,
    f0_min: int = 100,
    f0_max: int = 500,
    harmo_thresh: float = 0.1,
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    """
    Compute the Yin Algorithm. Return fundamental frequency and harmonic rate.
    :param sig: Audio signal (list of float)
    :param sr: sampling rate (int)
    :param w_len: size of the analysis window (samples)
    :param w_step: size of the lag between two consecutives windows (samples)
    :param f0_min: Minimum fundamental frequency that can be detected (hertz)
    :param f0_max: Maximum fundamental frequency that can be detected (hertz)
    :param harmo_tresh: Threshold of detection. The yalgorithmù return the first minimum of the CMND function below this treshold.
    :returns:
        * pitches: list of fundamental frequencies,
        * harmonic_rates: list of harmonic rate values for each fundamental frequency value (= confidence value)
        * argmins: minimums of the Cumulative Mean Normalized DifferenceFunction
        * times: list of time of each estimation
    :rtype: tuple
    """

    sig_torch = torch.from_numpy(sig)
    sig_torch = sig_torch.view(1, 1, -1)
    sig_torch = F.pad(
        sig_torch.unsqueeze(1),
        (int((w_len - w_step) / 2), int((w_len - w_step) / 2), 0, 0),
        mode="reflect",
    )
    sig_torch = sig_torch.view(-1).numpy()

    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    timeScale = range(
        0, len(sig_torch) - w_len, w_step
    )  # time values for each analysis window
    times = [t / float(sr) for t in timeScale]
    frames = [sig_torch[t : t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)
    argmins = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):
        # Compute YIN
        df = differenceFunction(frame, w_len, tau_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        p = getPitch(cmdf, tau_min, tau_max, harmo_thresh)

        # Get results
        if np.argmin(cmdf) > tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0:  # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else:  # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)

    return np.array(pitches), harmonic_rates, argmins, times


def norm_interp_f0(f0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    uv: np.ndarray = f0 == 0
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return f0, uv


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        sampling_rate: int,
        mel_fmin: Union[int, None],
        mel_fmax: Union[int, None],
        device: torch.device,
        center: bool,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center

        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )

        mel_basis = torch.from_numpy(mel).float().to(device)
        hann_window = torch.hann_window(win_length).to(device)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def linear_spectrogram(self, y):
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )
        spec = torch.norm(spec, p=2, dim=-1)

        return spec

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]
        RETURNS
        -------
        spectrogram: torch.FloatTensor of shape (B, n_spech_channels, T)
        mel_spectrogram: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        mel = torch.matmul(self.mel_basis, spec)
        mel = self.spectral_normalize_torch(mel)

        return spec, mel

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
