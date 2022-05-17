from typing import Dict, Any

vocoder_model_config: Dict[str, Any] = {
    "gen": {
        "noise_dim": 64,
        "channel_size": 32,
        "dilations": [1, 3, 9, 27],
        "strides": [8, 8, 4],
        "lReLU_slope": 0.2,
        "kpnet_conv_size": 3,
    },
    "mpd": {
        "periods": [2, 3, 5, 7, 11],
        "kernel_size": 5,
        "stride": 3,
        "use_spectral_norm": False,
        "lReLU_slope": 0.2,
    },
    "mrd": {
        "resolutions": [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)],
        "use_spectral_norm": False,
        "lReLU_slope": 0.2,
    },
}
