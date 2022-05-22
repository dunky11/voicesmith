from typing import Dict, Any

vocoder_pre_training_config: Dict[str, Any] = {
    "segment_size": 16384,
    "learning_rate": 0.0002,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.99975,
    "batch_size": 2,
    "grad_accum_steps": 5,
    "train_steps": 1000000,
    "stdout_interval": 25,
    "synth_interval": 250,
    "validation_interval": 2000,
    "checkpoint_interval": 250,
    "stft_lamb": 2.5,
}
 