import argparse
from pathlib import Path
import torch
from voice_smith.utils.model import get_acoustic_models
from voice_smith.config.globals import ASSETS_PATH
from voice_smith.config.configs import PreprocessingConfig, AcousticFinetuningConfig, AcousticModelConfig

def prep_acoustic_for_fine(checkpoint: str, data_path: str):
    device = torch.device("cpu")
    gen, optim, step = get_acoustic_models(
        data_path=data_path,
        checkpoint_acoustic=checkpoint,
        train_config=AcousticFinetuningConfig(),
        preprocess_config=PreprocessingConfig(),
        model_config=AcousticModelConfig(),
        fine_tuning=True,
        device=device,
        reset=True,
        assets_path=ASSETS_PATH,
    )
    torch.save({"gen": gen.state_dict(), "steps": 0}, Path(ASSETS_PATH) / "acoustic_pretrained.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)    
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    prep_acoustic_for_fine(checkpoint=args.checkpoint, data_path=args.data_path)
