from typing import Dict, Any

acoustic_fine_tuning_config: Dict[str, Any] = {
    "batch_size": 3,
    "grad_acc_step": 5,
    "optimizer": {
        "betas": [0.9, 0.98],
        "eps": 0.000000001,
        "weight_decay": 0.01,
        "grad_clip_thresh": 1.0,
        "warm_up_step": 4000,
        "anneal_steps": [],
        "anneal_rate": 0.3,
        "learning_rate": 0.0004,
        "lr_decay": 0.99999,
    },
    "step": {
        "train_steps": 30000,
        "log_step": 100,
        "synth_step": 250,
        "val_step": 4000,
        "save_step": 250,
        "freeze_bert_until": 4000,
        "mcd_gen_max_samples": 400,
        "only_train_speaker_until": 5000,
    },
}
