import torch
from voice_smith.utils.audio import save_audio, safe_load, resample
from voice_smith.config.file_extensions import SUPPORTED_AUDIO_EXTENSIONS

_SAMPLING_RATES_TO_TEST = [
    800,
    2000,
    1999,
    22050,
    22001,
    22057,
    44100,
    48000,
    79999,
    80000,
]


def test_should_save_audio(tmp_path):
    for n_samples in _SAMPLING_RATES_TO_TEST:
        for sr in _SAMPLING_RATES_TO_TEST:
            audio = torch.randn((n_samples,))
            for audio_extension in SUPPORTED_AUDIO_EXTENSIONS:
                audio_path = tmp_path / f"audio{audio_extension}"
                save_audio(file_path=audio_path, audio=audio, sr=sr)
                assert (
                    audio_path.exists()
                ), f"Failed to write audio for extension {audio_extension}"


def test_should_load_audio(tmp_path):
    for n_samples in _SAMPLING_RATES_TO_TEST:
        for sr_in in _SAMPLING_RATES_TO_TEST:
            audio_shape = (n_samples,)
            audio_in = torch.randn(audio_shape)
            for audio_extension in SUPPORTED_AUDIO_EXTENSIONS:
                audio_path = tmp_path / f"audio{audio_extension}"
                save_audio(file_path=audio_path, audio=audio_in, sr=sr_in)
                audio_out, sr_out = safe_load(path=str(audio_path), sr=None)
                assert (
                    sr_in == sr_out
                ), f"Invalid sampling rate loaded for extension {audio_extension}. sr_in: {sr_in}, sr_out: {sr_out}"
                assert (
                    audio_in.shape == audio_out.shape
                ), f"Invalid shape loaded for extension {audio_extension}. shape_in: {audio_in.shape}, shape_out: {audio_out.shape}"


def test_should_resample():
    for n_samples in [1000, 22049, 22050, 22051, 44100]:
        for orig_sr in _SAMPLING_RATES_TO_TEST:
            for target_sr in _SAMPLING_RATES_TO_TEST:
                audio_shape = (n_samples,)
                audio_in = torch.randn(audio_shape)
                resample(wav=audio_in.numpy(), orig_sr=orig_sr, target_sr=target_sr)
