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
from voice_smith.config.acoustic_model_config import acoustic_model_config
from voice_smith.config.preprocess_config import preprocess_config
from voice_smith.model import acoustic_model
from voice_smith.config.vocoder_model_config import vocoder_model_config
from voice_smith.model.univnet import Discriminator
from voice_smith.model.natural_speech import NaturalSpeech


def get_acoustic_models(
    checkpoint_acoustic: Union[str, None],
    checkpoint_style: Union[str, None],
    data_path: str,
    train_config: Dict[str, Any],
    preprocess_config: Dict[str, Any],
    model_config: Dict[str, Any],
    fine_tuning: bool,
    device: torch.device,
    reset: bool,
    assets_path: str,
) -> Tuple[
    acoustic_model.AcousticModel,
    ScriptModule,
    Union[ScheduledOptimFinetuning, ScheduledOptimPretraining],
    int,
]:
    with open(Path(data_path) / "speakers.json", "r", encoding="utf-8") as f:
        n_speakers = len(json.load(f))

    gen = acoustic_model.AcousticModel(
        data_path=data_path,
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=fine_tuning,
        n_speakers=n_speakers,
    ).to(device)
    if checkpoint_acoustic != None:
        ckpt = torch.load(checkpoint_acoustic)
        if reset:
            del ckpt["gen"]["speaker_embed"]
            del ckpt["gen"]["pitch_adaptor.pitch_bins"]
            del ckpt["gen"]["pitch_adaptor.pitch_embedding.embeddings"]
            step = 0
        else:
            step = ckpt["steps"] + 1
        gen.load_state_dict(ckpt["gen"], strict=False)
    else:
        step = 0

    if checkpoint_style == None:
        checkpoint_style = str(Path(assets_path) / "tiny_bert.pt")

    style_predictor = load(checkpoint_style).to(device)

    if fine_tuning:
        scheduled_optim = ScheduledOptimFinetuning(
            parameters=chain(gen.parameters(), style_predictor.parameters()),
            train_config=train_config,
            current_step=step,
        )
    else:
        scheduled_optim = ScheduledOptimPretraining(
            parameters=chain(gen.parameters(), style_predictor.parameters()),
            train_config=train_config,
            current_step=step,
        )

    if checkpoint_acoustic != None and not reset:
        scheduled_optim.load_state_dict(ckpt["optim"])

    gen.train()
    style_predictor.train()

    return gen, style_predictor, scheduled_optim, step


def get_param_num(model: torch.nn.Module) -> int:
    num_param = sum(param.numel() for param in model.parameters())
    num_buffers = sum(buffer.numel() for buffer in model.buffers())
    return num_param + num_buffers


def get_nat_speech(
    checkpoint: str,
    preprocessed_path: str,
    train_config: Dict[str, Any],
    reset: bool,
    device: torch.device,
) -> Tuple[
    NaturalSpeech,
    Discriminator,
    int,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ExponentialLR,
    torch.optim.lr_scheduler.ExponentialLR,
]:
    with open(preprocessed_path / "speakers.json", "r", encoding="utf-8") as f:
        n_speakers = len(json.load(f))

    generator = NaturalSpeech(
        preprocess_config=preprocess_config,
        model_config=acoustic_model_config,
        n_speakers=n_speakers,
    )
    discriminator = Discriminator()

    if checkpoint is None:
        state_dict = None
        steps = 0
    else:
        state_dict = torch.load(checkpoint, map_location=device)
        if reset:
            steps = 0
        else:
            steps = state_dict["steps"] + 1

        generator.load_state_dict(state_dict["generator"], strict=False)
        discriminator.load_state_dict(state_dict["discriminator"], strict=False)

    generator.to(device)
    discriminator.to(device)

    optim_g = torch.optim.AdamW(
        generator.parameters(),
        train_config["learning_rate"],
        betas=(train_config["adam_b1"], train_config["adam_b2"]),
    )
    optim_d = torch.optim.AdamW(
        discriminator.parameters(),
        train_config["learning_rate"],
        betas=(train_config["adam_b1"], train_config["adam_b2"]),
    )

    if checkpoint is not None and not reset:
        optim_g.load_state_dict(state_dict["optim_g"])
        optim_d.load_state_dict(state_dict["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=train_config["lr_decay"], last_epoch=-1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=train_config["lr_decay"], last_epoch=-1
    )

    assert len(optim_g.param_groups) == 1
    optim_g.param_groups[0]["lr"] = train_config["learning_rate"]
    assert len(optim_d.param_groups) == 1
    optim_d.param_groups[0]["lr"] = train_config["learning_rate"]
    for _ in range(steps // 1000):
        scheduler_g.step()
        scheduler_d.step()

    return generator, discriminator, steps, optim_g, optim_d, scheduler_g, scheduler_d


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
