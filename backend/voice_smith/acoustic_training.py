from logging import Logger
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain
import numpy as np
from torch.jit._script import ScriptModule
from typing import Dict, Any, Union, Tuple, Generator
from voice_smith.utils.optimizer import (
    ScheduledOptimPretraining,
    ScheduledOptimFinetuning,
)
from voice_smith.utils.model import (
    get_acoustic_models,
    get_param_num,
    save_model,
    save_torchscript,
)
from voice_smith.utils.tools import (
    to_device,
    cycle_2d,
    get_mask_from_lengths,
    iter_logger,
    get_embeddings,
)
from voice_smith.utils.loss import FastSpeech2LossGen
from voice_smith.utils.dataset import AcousticDataset
from voice_smith.utils.metrics import mcd_dtw
from voice_smith.model.acoustic_model import AcousticModel
from voice_smith.utils.model import get_infer_vocoder
from voice_smith.utils.loggers import Logger

def unfreeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = True


def freeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = False


torch.backends.cudnn.benchmark = True


def synth_iter(
    gen: AcousticModel,
    style_predictor: ScriptModule,
    step: int,
    preprocess_config: Dict[str, Any],
    device: torch.device,
    logger: Logger,
    embeddings: torch.Tensor,
    data_path: str,
    assets_path: str
) -> None:
    dataset = AcousticDataset(
        filename="val.txt",
        batch_size=1,
        sort=True,
        drop_last=False,
        data_path=data_path,
        assets_path=assets_path
    )
    loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    sampling_rate = preprocess_config["sampling_rate"]
    vocoder = get_infer_vocoder(
        checkpoint=str(Path(assets_path) / "vocoder_pretrained.pt"),
        device=device,
    )
    with torch.no_grad():
        for batches in loader:
            for batch in batches:
                batch = to_device(batch, device)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    durations,
                    mel_lens,
                    token_ids,
                    attention_masks,
                ) = batch
                style_embeds_pred = style_predictor(token_ids, attention_masks)
                for (speaker, text, src_len, mel, mel_len, style_pred,) in zip(
                    speakers,
                    texts,
                    src_lens,
                    mels,
                    mel_lens,
                    style_embeds_pred,
                ):
                    y_pred = gen(
                        x=text[: src_len.item()].unsqueeze(0),
                        speakers=speaker.unsqueeze(0),
                        style_embeds_pred=style_pred.unsqueeze(0),
                        p_control=1.0,
                        d_control=1.0,
                    )

                    wav_prediction = vocoder(y_pred)
                    wav_prediction = wav_prediction.cpu().numpy()

                    wav_reconstruction = vocoder(
                        mel[:, : mel_len.item()].unsqueeze(0)
                    )
                    wav_reconstruction = wav_reconstruction.cpu().numpy()

                    logger.log_audio(
                        name="val_wav_reconstructed",
                        audio=wav_reconstruction,
                        sr=sampling_rate,
                        step=step,
                    )
                    logger.log_audio(
                        name="val_wav_synthesized",
                        audio=wav_prediction,
                        sr=sampling_rate,
                        step=step,
                    )

                    logger.log_image(
                        name="mel_spec_ground_truth",
                        image=np.flip(mel[:, : mel_len.item()].cpu().numpy(), axis=0),
                        step=step,
                    )
                    logger.log_image(
                        name="mel_spec_synth",
                        image=np.flip(y_pred[0].cpu().numpy(), axis=0),
                        step=step,
                    )
                    return


def get_data_loaders(
    batch_size: int, group_size: int, data_path: str, assets_path: str
) -> Tuple[DataLoader, DataLoader]:

    dataset = AcousticDataset(
        filename="train.txt",
        batch_size=batch_size,
        sort=True,
        drop_last=True,
        data_path=data_path,
        assets_path=assets_path
    )
    # TODO check if this assertion is necessary
    # assert batch_size * group_size < len(dataset)
    train_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size * group_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    dataset = AcousticDataset(
        filename="val.txt",
        batch_size=batch_size * group_size,
        sort=True,
        drop_last=False,
        data_path=data_path,
        assets_path=assets_path
    )
    eval_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    return train_loader, eval_loader


def train_iter(
    db_id: int,
    gen: AcousticModel,
    style_predictor: ScriptModule,
    optim: Union[ScheduledOptimPretraining, ScheduledOptimFinetuning],
    train_loader: Generator,
    criterion: FastSpeech2LossGen,
    device: torch.device,
    grad_acc_steps: int,
    grad_clip_thresh: float,
    step: int,
    log_step: int,
    total_step: int,
    logger: Logger,
    model_is_frozen: bool,
) -> None:

    gen.train()
    style_predictor.train()

    losses = {
        "reconstruction_loss": torch.FloatTensor([0.0]).to(device),
        "mel_loss": torch.FloatTensor([0.0]).to(device),
        "ssim_loss": torch.FloatTensor([0.0]).to(device),
        "duration_loss": torch.FloatTensor([0.0]).to(device),
        "p_prosody_loss": torch.FloatTensor([0.0]).to(device),
        "pitch_loss": torch.FloatTensor([0.0]).to(device),
    }

    total_batch_size = 0

    for j in range(grad_acc_steps):
        batch = next(train_loader)
        batch = to_device(batch, device)
        (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            durations,
            mel_lens,
            token_ids,
            attention_masks,
        ) = batch
        style_embeds_pred = style_predictor(token_ids, attention_masks)

        src_mask = get_mask_from_lengths(src_lens)
        mel_mask = get_mask_from_lengths(mel_lens)
        outputs = gen.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=src_lens,
            mels=mels,
            mel_lens=mel_lens,
            style_embeds_pred=style_embeds_pred,
            attention_mask=attention_masks,
            pitches=pitches,
            durations=durations,
        )
        y_pred = outputs["y_pred"]
        log_duration_prediction = outputs["log_duration_prediction"]
        p_prosody_ref = outputs["p_prosody_ref"]
        p_prosody_pred = outputs["p_prosody_pred"]
        pitch_prediction = outputs["pitch_prediction"]

        (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            p_prosody_loss,
            pitch_loss,
        ) = criterion(
            src_masks=src_mask,
            mel_masks=mel_mask,
            mel_targets=mels,
            mel_predictions=y_pred,
            log_duration_predictions=log_duration_prediction,
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_predictions=pitch_prediction,
            p_targets=pitches,
            durations=durations,
        )

        batch_size = mels.shape[0]
        losses["reconstruction_loss"] += total_loss * batch_size
        losses["mel_loss"] += mel_loss * batch_size
        losses["ssim_loss"] += ssim_loss * batch_size
        losses["duration_loss"] += duration_loss * batch_size
        losses["p_prosody_loss"] += p_prosody_loss * batch_size
        losses["pitch_loss"] += pitch_loss * batch_size
        total_batch_size += batch_size

        (total_loss / grad_acc_steps).backward()

    # Clipping gradients to avoid gradient explosion
    clip_grad_norm_(
        chain(gen.parameters(), style_predictor.parameters()), grad_clip_thresh
    )

    # Update weights
    optim.step_and_update_lr(step)
    optim.zero_grad()

    if step % log_step == 0:

        losses["reconstruction_loss"] /= total_batch_size
        losses["mel_loss"] /= total_batch_size
        losses["ssim_loss"] /= total_batch_size
        losses["duration_loss"] /= total_batch_size
        losses["p_prosody_loss"] /= total_batch_size
        losses["pitch_loss"] /= total_batch_size

        message = "Step {}/{}, ".format(step, total_step)
        for j, loss_name in enumerate(losses.keys()):
            if j != 0:
                message += ", "
            loss_value = losses[loss_name]
            message += f"{loss_name}: {round(loss_value.item(), 4)}"

        for key in losses.keys():
            logger.log_graph(name=f"train_{key}", value=losses[key].item(), step=step)

        logger.log_graph(
            name="lr", value=optim._optimizer.param_groups[0]["lr"], step=step
        )
        logger.log_graph(
            name="only_train_speaker_emb", value=1 if model_is_frozen else 0, step=step
        )

        logger.log_image(
            name="bert_attention",
            image=outputs["bert_attention"][0, 0, :, :].T.detach().cpu().numpy(),
            step=step,
        )

    if step % 10 == 0:
        logger.query(
            "UPDATE training_run SET acoustic_fine_tuning_progress=? WHERE ID=?",
            [step / total_step, db_id],
        )


def eval_iter(
    gen: AcousticModel,
    style_predictor: ScriptModule,
    step: int,
    train_config: Dict[str, Any],
    batch_size: int,
    loader: DataLoader,
    criterion: FastSpeech2LossGen,
    device: torch.device,
    logger: Logger,
) -> None:
    gen.eval()
    style_predictor.eval()

    losses = {
        "reconstruction_loss": torch.FloatTensor([0.0]).to(device),
        "mel_loss": torch.FloatTensor([0.0]).to(device),
        "ssim_loss": torch.FloatTensor([0.0]).to(device),
        "duration_loss": torch.FloatTensor([0.0]).to(device),
        "p_prosody_loss": torch.FloatTensor([0.0]).to(device),
        "pitch_loss": torch.FloatTensor([0.0]).to(device),
    }

    len_ds = 0
    with torch.no_grad():
        for i, batches in iter_logger(enumerate(loader)):
            for batch in batches:
                batch = to_device(batch, device)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    durations,
                    mel_lens,
                    token_ids,
                    attention_masks,
                ) = batch
                style_embeds_pred = style_predictor(token_ids, attention_masks)

                src_mask = get_mask_from_lengths(src_lens)
                mel_mask = get_mask_from_lengths(mel_lens)

                outputs = gen.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=src_lens,
                    mels=mels,
                    mel_lens=mel_lens,
                    style_embeds_pred=style_embeds_pred,
                    attention_mask=attention_masks,
                    pitches=pitches,
                    durations=durations,
                )

                y_pred = outputs["y_pred"]
                log_duration_prediction = outputs["log_duration_prediction"]
                p_prosody_ref = outputs["p_prosody_ref"]
                p_prosody_pred = outputs["p_prosody_pred"]
                pitch_prediction = outputs["pitch_prediction"]

                (
                    total_loss,
                    mel_loss,
                    ssim_loss,
                    duration_loss,
                    p_prosody_loss,
                    pitch_loss,
                ) = criterion(
                    src_masks=src_mask,
                    mel_masks=mel_mask,
                    mel_targets=mels,
                    mel_predictions=y_pred,
                    log_duration_predictions=log_duration_prediction,
                    p_prosody_ref=p_prosody_ref,
                    p_prosody_pred=p_prosody_pred,
                    pitch_predictions=pitch_prediction,
                    p_targets=pitches,
                    durations=durations,
                )

                batch_size = mels.shape[0]
                losses["reconstruction_loss"] += total_loss * batch_size
                losses["mel_loss"] += mel_loss * batch_size
                losses["ssim_loss"] += ssim_loss * batch_size
                losses["duration_loss"] += duration_loss * batch_size
                losses["p_prosody_loss"] += p_prosody_loss * batch_size
                losses["pitch_loss"] += pitch_loss * batch_size
                len_ds += batch_size

    samples_to_gen = train_config["step"]["mcd_gen_max_samples"]
    samples_generated = 0
    mcds = []

    with torch.no_grad():
        for batches in iter_logger(loader):
            for batch in batches:
                batch = to_device(batch, device)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    durations,
                    mel_lens,
                    token_ids,
                    attention_masks,
                ) = batch
                style_embeds_pred = style_predictor(token_ids, attention_masks)
                for (speaker, text, src_len, mel, mel_len, style_pred,) in zip(
                    speakers,
                    texts,
                    src_lens,
                    mels,
                    mel_lens,
                    style_embeds_pred,
                ):
                    y_pred = gen(
                        x=text[: src_len.item()].unsqueeze(0),
                        speakers=speaker.unsqueeze(0),
                        style_embeds_pred=style_pred.unsqueeze(0),
                        p_control=1.0,
                        d_control=1.0,
                    )

                    samples_generated += 1
                    mcd = mcd_dtw(
                        y_pred[0].T.cpu().numpy(),
                        mel[:, : mel_len.item()].T.cpu().numpy(),
                    )
                    mcds.append(mcd)
                    if samples_generated >= samples_to_gen:
                        break
                if samples_generated >= samples_to_gen:
                    break
            if samples_generated >= samples_to_gen:
                break

    losses["reconstruction_loss"] /= len_ds
    losses["mel_loss"] /= len_ds
    losses["ssim_loss"] /= len_ds
    losses["duration_loss"] /= len_ds
    losses["p_prosody_loss"] /= len_ds
    losses["pitch_loss"] /= len_ds
    losses["mcd_dtw"] = sum(mcds) / len(mcds)

    message = f"Validation Step {step}: "
    for j, loss_name in enumerate(losses.keys()):
        if j != 0:
            message += ", "
        loss_value = losses[loss_name]
        message += f"{loss_name}: {round(loss_value.item(), 4)}"

    print(message)

    for key in losses.keys():
        logger.log_graph(name=f"val_{key}", value=losses[key].item(), step=step)


def train_acoustic(
    db_id: int,
    training_run_name: str,
    preprocess_config: Dict[str, Any],
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    logger: Logger,
    device: torch.device,
    reset: bool,
    checkpoint_acoustic: Union[str, None],
    checkpoint_style: Union[str, None],
    fine_tuning: bool,
    overwrite_saves: bool,
    assets_path: str,
    training_runs_path: str
) -> None:
    batch_size = train_config["batch_size"]
    data_path = Path(training_runs_path) / str(training_run_name) / "data"

    embeddings = get_embeddings(data_path=str(data_path), device=device)

    # Prepare model
    gen, style_predictor, optim, step = get_acoustic_models(
        data_path=str(data_path),
        checkpoint_acoustic=checkpoint_acoustic,
        checkpoint_style=checkpoint_style,
        train_config=train_config,
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=fine_tuning,
        device=device,
        reset=reset,
        embeddings=embeddings,
    )

    group_size = 5  # Set this larger than 1 to enable sorting in Dataset
    train_loader, validation_loader = get_data_loaders(
        batch_size=batch_size, group_size=group_size, data_path=data_path, assets_path=assets_path
    )
    train_loader = cycle_2d(train_loader)

    criterion = FastSpeech2LossGen(fine_tuning=fine_tuning, device=device)

    gen_pars = get_param_num(gen)
    style_pars = get_param_num(style_predictor)
    prosody_encoder_pars = get_param_num(gen.phoneme_prosody_encoder)

    print(f"Number of acoustic model parameters: {gen_pars}")
    print(f"Number of style predictor parameters: {style_pars}")

    print(f"Total number of parameters: {gen_pars + style_pars}")
    print(
        f"Total number of parameters during inference: {gen_pars + style_pars - prosody_encoder_pars}"
    )

    # Training
    grad_acc_steps = train_config["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    total_step = train_config["step"]["train_steps"]
    only_train_speaker_until = train_config["step"]["only_train_speaker_until"]
    freeze_bert_until = 0 if fine_tuning else train_config["step"]["freeze_bert_until"]
    freeze_model_until = only_train_speaker_until if fine_tuning else 0
    ckpt_dir = Path(training_runs_path) / training_run_name / "ckpt" / "acoustic"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    bert_is_frozen = step < freeze_bert_until
    model_is_frozen = step < freeze_model_until

    if bert_is_frozen or model_is_frozen:
        freeze_torchscript(style_predictor)
    else:
        unfreeze_torchscript(style_predictor)

    if model_is_frozen:
        gen.freeze()
    else:
        gen.unfreeze()

    for step in iter_logger(
        iterable=range(step, total_step + 1),
        start=step,
        total=total_step,
    ):
        if bert_is_frozen and step >= freeze_bert_until:
            bert_is_frozen = False
            if not model_is_frozen:
                unfreeze_torchscript(style_predictor)

        if model_is_frozen and step >= freeze_model_until:
            model_is_frozen = False
            gen.unfreeze()
            unfreeze_torchscript(style_predictor)

        train_iter(
            db_id=db_id,
            gen=gen,
            style_predictor=style_predictor,
            optim=optim,
            train_loader=train_loader,
            criterion=criterion,
            device=device,
            grad_acc_steps=grad_acc_steps,
            grad_clip_thresh=grad_clip_thresh,
            step=step,
            log_step=log_step,
            total_step=total_step,
            logger=logger,
            model_is_frozen=model_is_frozen,
        )

        if step % val_step == 0 and step != 0:
            eval_iter(
                gen=gen,
                style_predictor=style_predictor,
                step=step,
                train_config=train_config,
                batch_size=batch_size,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                logger=logger,
            )

        if step % synth_step == 0 and step != 0:
            synth_iter(
                gen=gen,
                style_predictor=style_predictor,
                step=step,
                preprocess_config=preprocess_config,
                device=device,
                embeddings=embeddings,
                logger=logger,
                data_path=str(data_path),
                assets_path=assets_path
            )

        if step % save_step == 0 and step != 0 or step >= total_step:
            save_model(
                name="acoustic",
                ckpt_dict={
                    "gen": gen.state_dict(),
                    "optim": optim._optimizer.state_dict(),
                    "steps": step,
                },
                ckpt_dir=str(ckpt_dir),
                step=step,
                overwrite=overwrite_saves,
            )
            save_torchscript(
                name="style",
                model=style_predictor,
                ckpt_dir=str(ckpt_dir),
                step=step,
                overwrite=overwrite_saves,
            )
            if step >= total_step:
                break

    logger.query(
        "UPDATE training_run SET acoustic_fine_tuning_progress=? WHERE ID=?",
        [1.0, db_id],
    )

