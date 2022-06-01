import torch
from typing import Tuple
from torch.jit._script import script, ScriptModule
from torch.jit._trace import trace
from voice_smith.utils.model import get_acoustic_models, get_vocoder
from voice_smith.config.acoustic_model_config import acoustic_model_config
from voice_smith.config.acoustic_fine_tuning_config import acoustic_fine_tuning_config
from voice_smith.config.preprocess_config import preprocess_config
from voice_smith.config.vocoder_fine_tuning_config import vocoder_fine_tuning_config
from voice_smith.utils.text_normalization import EnglishTextNormalizer

def acoustic_to_torchscript(
    checkpoint_acoustic: str, 
    checkpoint_style: str, 
    data_path: str
) -> Tuple[ScriptModule, ScriptModule]:
    device = torch.device("cpu")
    acoustic, style_predictor, _, _ = get_acoustic_models(
        checkpoint_acoustic=checkpoint_acoustic,
        checkpoint_style=checkpoint_style,
        data_path=data_path,
        train_config=acoustic_fine_tuning_config,
        preprocess_config=preprocess_config,
        model_config=acoustic_model_config,
        fine_tuning=False,
        device=device,
        reset=False,
        embeddings=embeddings,
    )
    acoustic.prepare_for_export()
    acoustic.eval()
    style_predictor.eval()
    acoustic_torch = script(
        acoustic,
    )
    return acoustic_torch, style_predictor


def vocoder_to_torchscript(ckpt_path: str, data_path: str) -> ScriptModule:
    device = torch.device("cpu")
    embeddings = get_embeddings(data_path=data_path, device=device)
    vocoder, _, _, _, _, _, _, _ = get_vocoder(
        checkpoint=ckpt_path,
        train_config=vocoder_fine_tuning_config,
        reset=False,
        device=device,
        fine_tuning=False,
        embeddings=embeddings,
    )
    vocoder.eval()
    vocoder.remove_weight_norm()
    mels = torch.randn((2, 80, 50))
    speaker_ids = torch.LongTensor([0, 0])
    vocoder_torch = trace(vocoder, (mels, speaker_ids))
    return vocoder_torch
