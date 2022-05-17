import torch
import fire
from typing import Union
from voice_smith.acoustic_training import train_acoustic

if __name__ == "__main__":
    from voice_smith.config.preprocess_config import preprocess_config
    from voice_smith.config.acoustic_pre_training_config import (
        acoustic_pre_training_config,
    )
    from voice_smith.config.acoustic_model_config import acoustic_model_config
    from voice_smith.utils.wandb_logger import WandBLogger
    import wandb
    from pathlib import Path

    def pass_args(
        training_run_name: str,
        checkpoint_acoustic: Union[str, None] = None,
        checkpoint_style: Union[str, None] = None,
    ):
        training_run_name = str(training_run_name)
        if checkpoint_acoustic != None:
            checkpoint_acoustic = str(checkpoint_acoustic)

        if checkpoint_style != None:
            checkpoint_style = str(checkpoint_style)

        device = (
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        )
        logger = WandBLogger("Acoustic Model Pretraining 100 mel_bins v2")
        wandb.config.update(
            {
                "preprocess_config": preprocess_config,
                "model_config": acoustic_model_config,
                "training_config": acoustic_pre_training_config,
            },
            allow_val_change=True,
        )
        train_acoustic(
            db_id=None,
            training_run_name=training_run_name,
            preprocess_config=preprocess_config,
            model_config=acoustic_model_config,
            train_config=acoustic_pre_training_config,
            checkpoint_acoustic=checkpoint_acoustic,
            checkpoint_style=checkpoint_style,
            logger=logger,
            device=device, 
            fine_tuning=False,
            reset=False,
            overwrite_saves=False,
            assets_path=Path(".") / ".." / "assets",
            training_runs_path=Path(".") / ".." / "training_runs"
        )

    fire.Fire(pass_args)
