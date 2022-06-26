import torch
from typing import List
from voice_smith.g2p.dp.utils.model import get_g2p
from voice_smith.g2p.dp.utils.infer import batched_predict


def grapheme_to_phonemes(
    texts: List[str], langs: List[str], assets_path: str, device: torch.device
) -> List[List[str]]:
    assert len(texts) == len(langs)
    model = get_g2p(assets_path=assets_path, device=device)
    phonemes_list = batched_predict(model, texts)
    return phonemes_list


if __name__ == "__main__":
    texts = ["This", "is", "a", "test", "hehehehee", "!"]
    assets_path = "/home/media/main_volume/datasets/voice-smith/assets"
    phones = grapheme_to_phonemes(
        texts=texts, assets_path=assets_path, device=torch.device("cuda")
    )
    print(phones)
