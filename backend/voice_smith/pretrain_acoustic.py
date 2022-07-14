import torch
import wandb
from voice_smith.acoustic_training import train_acoustic
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticPretrainingConfig,
    AcousticENModelConfig,
)
from voice_smith.utils.wandb_logger import WandBLogger
import argparse
from voice_smith.config.globals import TRAINING_RUNS_PATH, ASSETS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = WandBLogger("DelightfulTTS 120M parameters with UnsupDurAligner")
    p_config = PreprocessingConfig(language="english_only")
    m_config = AcousticENModelConfig()
    t_config = AcousticPretrainingConfig()
    wandb.config.update(
        {
            "preprocess_config": p_config,
            "model_config": m_config,
            "training_config": t_config,
        },
        allow_val_change=True,
    )
    train_acoustic(
        db_id=args.run_id,
        training_run_name=str(args.run_id),
        preprocess_config=p_config,
        model_config=m_config,
        train_config=t_config,
        logger=logger,
        device=device,
        reset=False,
        checkpoint_acoustic=args.checkpoint,
        fine_tuning=False,
        overwrite_saves=True,
        assets_path=ASSETS_PATH,
        training_runs_path=TRAINING_RUNS_PATH,
    )

