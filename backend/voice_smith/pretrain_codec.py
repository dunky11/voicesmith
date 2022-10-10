import torch
import wandb
import argparse
from voice_smith.codec_training import train_codec
from voice_smith.config.configs import (
    PreprocessingConfig,
    VocoderPretrainingConfig,
    VocoderModelConfig,
)
from voice_smith.utils.wandb_logger import WandBLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger = WandBLogger("Codec Pretraining")
    t_config = VocoderPretrainingConfig()
    p_config = PreprocessingConfig(language="english_only")
    m_config = VocoderModelConfig()

    wandb.config.update(
        {
            "preprocess_config": p_config,
            "model_config": m_config,
            "training_config": t_config,
        },
        allow_val_change=True,
    )
    train_codec(
        db_id=args.run_id,
        train_config=t_config,
        model_config=m_config,
        preprocess_config=p_config,
        logger=logger,
        device=device,
        checkpoint_path=args.checkpoint,
    )

