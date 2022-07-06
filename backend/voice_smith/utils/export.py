import torch
from typing import Tuple, Union
from torch.jit._script import script, ScriptModule
from torch.jit._trace import trace
from voice_smith.utils.model import get_acoustic_models, get_vocoder
from voice_smith.model.univnet import TracedGenerator
from voice_smith.config.configs import (
    AcousticPretrainingConfig,
    AcousticFinetuningConfig,
    PreprocessingConfig,
    AcousticModelConfig,
    VocoderPretrainingConfig,
    VocoderFinetuningConfig,
    VocoderModelConfig,
)


def acoustic_to_torchscript(
    checkpoint_acoustic: str,
    data_path: str,
    train_config: Union[AcousticPretrainingConfig, AcousticFinetuningConfig],
    preprocess_config: PreprocessingConfig,
    model_config: AcousticModelConfig,
    assets_path: str,
) -> Tuple[ScriptModule, ScriptModule]:
    device = torch.device("cpu")
    acoustic, _, _ = get_acoustic_models(
        checkpoint_acoustic=checkpoint_acoustic,
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
    acoustic_torch = script(acoustic,)
    return acoustic_torch


def vocoder_to_torchscript(
    ckpt_path: str,
    data_path: str,
    train_config: Union[VocoderPretrainingConfig, VocoderFinetuningConfig],
    preprocess_config: PreprocessingConfig,
    model_config: VocoderModelConfig,
) -> ScriptModule:
    device = torch.device("cpu")
    vocoder, _, _, _, _, _, _ = get_vocoder(
        checkpoint=ckpt_path,
        train_config=train_config,
        reset=False,
        device=device,
        preprocess_config=preprocess_config,
        model_config=model_config,
    )
    vocoder.eval(True)
    mels = torch.randn((2, preprocess_config.stft.n_mel_channels, 50))
    mel_lens = torch.tensor([50], dtype=torch.int64)
    vocoder = TracedGenerator(vocoder, (mels,))
    vocoder_torch = trace(
        vocoder,
        (mels, mel_lens),
        check_inputs=[
            (
                torch.randn(3, preprocess_config.stft.n_mel_channels, 64),
                torch.tensor([64], dtype=torch.int64),
            )
        ],
    )
    quit()
    return vocoder_torch
