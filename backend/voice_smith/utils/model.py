import torch
from torch.jit._serialization import load
from torch.jit._script import ScriptModule
from itertools import chain
from pathlib import Path
from typing import Dict, Any, Tuple, Union
import json
from voice_smith.utils.optimizer import (
    ScheduledOptimPretraining,
    ScheduledOptimFinetuning,
)
from voice_smith.model import acoustic_model
from voice_smith.model.univnet import Discriminator
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticPretrainingConfig,
    AcousticFinetuningConfig,
    AcousticModelConfig,
    VocoderModelConfig,
)


def get_acoustic_models(
    checkpoint_acoustic: Union[str, None],
    checkpoint_style: Union[str, None],
    data_path: str,
    train_config: Union[AcousticPretrainingConfig, AcousticFinetuningConfig],
    preprocess_config: PreprocessingConfig,
    model_config: AcousticModelConfig,
    fine_tuning: bool,
    device: torch.device,
    reset: bool,
    assets_path: str,
) -> Tuple[
    acoustic_model.AcousticModel,
    ScriptModule,
    Discriminator,
    Union[ScheduledOptimFinetuning, ScheduledOptimPretraining],
    Union[ScheduledOptimFinetuning, ScheduledOptimPretraining],
    int,
]:
    with open(Path(data_path) / "speakers.json", "r", encoding="utf-8") as f:
        n_speakers = len(json.load(f))

    gen = acoustic_model.AcousticModel(
        data_path=data_path,
        preprocess_config=preprocess_config,
        model_config=model_config,
        n_speakers=n_speakers,
    ).to(device)
    disc = Discriminator(model_config=VocoderModelConfig()).to(device)
    if checkpoint_acoustic is not None:
        ckpt = torch.load(checkpoint_acoustic)
        if reset:
            del ckpt["gen"]["speaker_embed"]
            del ckpt["gen"]["pitch_adaptor.pitch_bins"]
            # del ckpt["gen"]["pitch_adaptor.pitch_embedding.embeddings"]
            step = 0
            ckpt_vocoder = torch.load(str(Path(assets_path) / "vocoder_pretrained.pt"))
            gen.vocoder.load_state_dict(ckpt_vocoder["generator"])
            disc.load_state_dict(ckpt_vocoder["discriminator"])
            gen.load_state_dict(ckpt["gen"], strict=False)
        else:
            step = ckpt["steps"] + 1
            gen.load_state_dict(ckpt["gen"], strict=False)
            disc.load_state_dict(ckpt["disc"])

    else:
        step = 0

    if checkpoint_style is None:
        checkpoint_style = str(Path(assets_path) / "tiny_bert.pt")

    style_predictor = load(checkpoint_style).to(device)

    if fine_tuning:
        scheduled_optim_g = ScheduledOptimFinetuning(
            parameters=chain(gen.parameters(), style_predictor.parameters()),
            train_config=train_config,
            current_step=step,
        )
        scheduled_optim_d = ScheduledOptimPretraining(
            parameters=chain(gen.parameters(), style_predictor.parameters()),
            train_config=train_config,
            current_step=step,
        )
    else:
        scheduled_optim_g = ScheduledOptimPretraining(
            parameters=chain(gen.parameters(), style_predictor.parameters()),
            train_config=train_config,
            current_step=step,
        )
        scheduled_optim_d = ScheduledOptimPretraining(
            parameters=disc.parameters(), train_config=train_config, current_step=step,
        )

    if checkpoint_acoustic is not None and not reset:
        scheduled_optim_g.load_state_dict(ckpt["optim_g"])
        scheduled_optim_d.load_state_dict(ckpt["optim_d"])

    gen.train()
    style_predictor.train()
    disc.train()

    return gen, style_predictor, disc, scheduled_optim_g, scheduled_optim_d, step


def get_param_num(model: torch.nn.Module) -> int:
    num_param = sum(param.numel() for param in model.parameters())
    num_buffers = sum(buffer.numel() for buffer in model.buffers())
    return num_param + num_buffers


def save_torchscript(
    name: str, model: ScriptModule, ckpt_dir: str, step: int, overwrite: bool
) -> None:
    if overwrite:
        files = Path(ckpt_dir).glob(f"{name}*.pt")

    model_name = f"{name}_{step}.pt"

    model.save(str(Path(ckpt_dir) / model_name))

    if overwrite:
        for file in files:
            if file.name == model_name:
                continue
            file.unlink(missing_ok=True)


def save_model(
    name: str, ckpt_dict: Dict[str, Any], ckpt_dir: str, step: int, overwrite: bool
) -> None:
    if overwrite:
        files = Path(ckpt_dir).glob(f"{name}*.pt")

    model_name = f"{name}_{step}.pt"

    torch.save(ckpt_dict, str(Path(ckpt_dir) / model_name))

    if overwrite:
        for file in files:
            if file.name == model_name:
                continue
            file.unlink(missing_ok=True)
