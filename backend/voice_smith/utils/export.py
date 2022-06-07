import torch
from typing import Tuple, Union
from torch.jit._script import script, ScriptModule
from torch.jit._trace import trace
from voice_smith.utils.model import get_acoustic_models
from voice_smith.config.configs import (
    AcousticPretrainingConfig,
    AcousticFinetuningConfig,
    PreprocessingConfig,
    AcousticModelConfig,
    VocoderModelConfig,
)


def acoustic_to_torchscript(
    checkpoint_acoustic: str,
    checkpoint_style: str,
    data_path: str,
    train_config: Union[AcousticPretrainingConfig, AcousticFinetuningConfig],
    preprocess_config: PreprocessingConfig,
    model_config: AcousticModelConfig,
    assets_path: str,
) -> Tuple[ScriptModule, ScriptModule]:
    device = torch.device("cpu")
    acoustic, style_predictor, _, _ = get_acoustic_models(
        checkpoint_acoustic=checkpoint_acoustic,
        checkpoint_style=checkpoint_style,
        data_path=data_path,
        train_config=train_config,
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=False,
        device=device,
        reset=False,
        assets_path=assets_path,
    )
    acoustic.prepare_for_export()
    acoustic.eval()
    style_predictor.eval()
    acoustic_torch = script(acoustic,)
    return acoustic_torch, style_predictor

