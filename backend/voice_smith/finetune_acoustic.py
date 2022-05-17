import torch
import fire
from pathlib import Path
from voice_smith.acoustic_training import train_acoustic

if __name__ == "__main__":
    from voice_smith.config.preprocess_config import preprocess_config
    from voice_smith.config.acoustic_fine_tuning_config import (
        acoustic_fine_tuning_config,
    )
    from voice_smith.config.acoustic_model_config import acoustic_model_config
    from voice_smith.utils.wandb_logger import WandBLogger
    import wandb

    def pass_args(training_run_name, checkpoint_path=None):
        training_run_name = str(training_run_name)
        if checkpoint_path == None:
            reset = True
            checkpoint_path = str(Path(".") / "assets" / "acoustic_pretrained.pt")
        else:
            checkpoint_path = str(checkpoint_path)
            reset = False
        device = (
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        )
        logger = WandBLogger(training_run_name)
        wandb.config.update(
            {
                "preprocess_config": preprocess_config,
                "model_config": acoustic_model_config,
                "training_config": acoustic_fine_tuning_config,
            },
            allow_val_change=True,
        )
        train_acoustic(
            training_run_name=training_run_name,
            preprocess_config=preprocess_config,
            model_config=acoustic_model_config,
            train_config=acoustic_fine_tuning_config,
            checkpoint_path=checkpoint_path,
            logger=logger,
            device=device,
            fine_tuning=True,
            reset=reset,
        )

    fire.Fire(pass_args)
