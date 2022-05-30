from pathlib import Path
import torch
from voice_smith.g2p.dp.phonemizer import Phonemizer

def get_g2p(assets_path: str, device: torch.device) -> Phonemizer:
    checkpoint_path = Path(assets_path) / "g2p" / "en" / "g2p.pt"
    phonemizer = Phonemizer.from_checkpoint(str(checkpoint_path), device=device)
    return phonemizer
