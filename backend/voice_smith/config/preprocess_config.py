from typing import Dict, Any

preprocess_config: Dict[str, Any] = {
    "val_size": 0.05,
    "min_seconds": 0.5,
    "max_seconds": 10.0,
    "sampling_rate": 22050,
    "stft": {
        "filter_length": 1024,
        "hop_length": 256,
        "win_length": 1024,
    },
    "mel": {"n_mel_channels": 100, "mel_fmin": 20, "mel_fmax": 11025},
}
