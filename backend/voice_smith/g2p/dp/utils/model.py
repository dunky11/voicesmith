from pathlib import Path
from dp.phonemizer import Phonemizer
import torch

def get_g2p(assets_path: str, device: torch.device) -> Phonemizer:
    checkpoint_path = Path(assets_path) / 1
    phonemizer = Phonemizer.from_checkpoint(str(checkpoint_path), device=device)
    return phonemizer
