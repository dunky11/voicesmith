from typing import Dict, Any

acoustic_pre_training_config: Dict[str, Any] = {
    "batch_size": 5,
    "grad_acc_step": 3,
    "optimizer": {
        "betas": [0.9, 0.98],
        "eps": 0.000000001,
        "weight_decay": 0.01,
        "grad_clip_thresh": 1.0,
        "warm_up_step": 4000,
        "anneal_steps": [],
        "anneal_rate": 0.3,
        "learning_rate": 0.051,  # 384^(-0.5) (BERT-default)
    },
    "step": {
        "train_steps": 500000,
        "log_step": 20,
        "synth_step": 250,
        "val_step": 4000,
        "save_step": 1000,
        "freeze_bert_until": 4000,
        "mcd_gen_max_samples": 400,
        "only_train_speaker_until": 0,
    },
}
