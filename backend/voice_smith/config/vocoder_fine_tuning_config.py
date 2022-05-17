from typing import Dict, Any

vocoder_fine_tuning_config: Dict[str, Any] = {
    "segment_size": 16384,
    "learning_rate": 0.0001,
    "adam_b1": 0.5,
    "adam_b2": 0.9,
    "lr_decay": 0.995,
    "batch_size": 5,
    "grad_accum_steps": 3,
    "train_steps": 10000,
    "stdout_interval": 100,
    "synth_interval": 250,
    "validation_interval": 4000,
    "checkpoint_interval": 250,
    "stft_lamb": 2.5,
}
