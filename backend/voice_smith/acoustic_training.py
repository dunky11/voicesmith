from logging import Logger
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain
import numpy as np
from torch.jit._script import ScriptModule
from typing import Dict, Any, Union, Tuple, Generator
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticModelConfig,
    AcousticPretrainingConfig,
    AcousticFinetuningConfig,
    VocoderModelConfig,
)
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
    slice_segments,
)
from voice_smith.utils.audio import TacotronSTFT
from voice_smith.model.univnet import MultiResolutionSTFTLoss
from voice_smith.utils.dataset import AcousticDataset
from voice_smith.utils.metrics import calc_rmse, calc_pesq, calc_estoi
from voice_smith.model.acoustic_model import AcousticModel
from voice_smith.utils.loss import duration_loss, pitch_loss, prosody_loss
from voice_smith.model.univnet import Discriminator
from voice_smith.utils.loggers import Logger


def unfreeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = True


def freeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = False


def synth_iter(
    gen: AcousticModel,
    style_predictor: ScriptModule,
    step: int,
    preprocess_config: PreprocessingConfig,
    device: torch.device,
    logger: Logger,
    data_path: str,
    assets_path: str,
) -> None:
    dataset = AcousticDataset(
        filename="val.txt",
        batch_size=1,
        sort=True,
        drop_last=False,
        data_path=data_path,
        assets_path=assets_path,
    )
    loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    sampling_rate = preprocess_config.sampling_rate

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
                    audio,
                ) = batch
                style_embeds_pred = style_predictor(token_ids, attention_masks)
                for (speaker, text, src_len, mel, mel_len, style_pred,) in zip(
                    speakers, texts, src_lens, mels, mel_lens, style_embeds_pred,
                ):
                    y_pred = gen(
                        x=text[: src_len.item()].unsqueeze(0),
                        speakers=speaker.unsqueeze(0),
                        style_embeds_pred=style_pred.unsqueeze(0),
                        p_control=1.0,
                        d_control=1.0,
                    )

                    logger.log_audio(
                        name="val_wav_reconstructed",
                        audio=audio[0].reshape((-1,)).cpu().numpy(),
                        sr=sampling_rate,
                        step=step,
                    )
                    logger.log_audio(
                        name="val_wav_synthesized",
                        audio=y_pred.reshape((-1,)).cpu().numpy(),
                        sr=sampling_rate,
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
        assets_path=assets_path,
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
        assets_path=assets_path,
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
    disc: Discriminator,
    optim_g: Union[ScheduledOptimPretraining, ScheduledOptimFinetuning],
    optim_d: Union[ScheduledOptimPretraining, ScheduledOptimFinetuning],
    train_loader: Generator,
    device: torch.device,
    stft_criterion: MultiResolutionSTFTLoss,
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
    disc.train()

    losses = {
        "reconstruction_loss": torch.FloatTensor([0.0]).to(device),
        "duration_loss": torch.FloatTensor([0.0]).to(device),
        "p_prosody_loss": torch.FloatTensor([0.0]).to(device),
        "pitch_loss": torch.FloatTensor([0.0]).to(device),
        "disc_loss": torch.FloatTensor([0.0]).to(device),
        "gen_adv_loss": torch.FloatTensor([0.0]).to(device),
        "mel_loss": torch.FloatTensor([0.0]).to(device),
    }

    total_batch_size = 0
    audios, fake_audios = [], []
    optim_g.zero_grad()

    # Train Generator
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
            audio,
        ) = batch
        audio = audio.unsqueeze(1)
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
        audio = slice_segments(
            audio,
            outputs["ids_slice"] * PreprocessingConfig().stft.hop_length,
            PreprocessingConfig().segment_size * PreprocessingConfig().stft.hop_length,
        )  # slice
        fake_audio = outputs["y_pred"]
        sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * AcousticFinetuningConfig().stft_lamb

        res_fake, period_fake = disc(fake_audio)

        score_loss = 0.0

        for (_, score_fake) in res_fake + period_fake:
            score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

        score_loss = score_loss / len(res_fake + period_fake)

        l_pitch = pitch_loss(
            outputs["pitch_prediction"], y_true=pitches, src_masks=src_mask
        )
        l_dur = duration_loss(
            log_y_pred=outputs["log_duration_prediction"],
            y_true=durations,
            src_masks=src_mask,
        )
        l_prosody = (
            prosody_loss(
                y_pred=outputs["p_prosody_pred"],
                y_ref=outputs["p_prosody_ref"],
                src_masks=src_mask,
            )
            * 0.5
        )

        total_loss = stft_loss + score_loss + l_pitch + l_dur + l_prosody

        batch_size = mels.shape[0]
        losses["reconstruction_loss"] += total_loss * batch_size
        losses["duration_loss"] += l_dur * batch_size
        losses["p_prosody_loss"] += l_prosody * batch_size
        losses["pitch_loss"] += l_pitch * batch_size
        losses["mel_loss"] += stft_loss * batch_size
        losses["gen_adv_loss"] += score_loss * batch_size

        total_batch_size += batch_size

        (total_loss / grad_acc_steps).backward()

        audios.append(audio)
        fake_audios.append(fake_audio.detach())

    clip_grad_norm_(
        chain(gen.parameters(), style_predictor.parameters()), grad_clip_thresh
    )

    optim_g.step_and_update_lr(step)
    optim_d.zero_grad()

    # Train Discriminator
    for audio, fake_audio in zip(audios, fake_audios):

        res_fake, period_fake = disc(fake_audio.detach())
        res_real, period_real = disc(audio)

        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(
            res_fake + period_fake, res_real + period_real
        ):
            loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
            loss_d += torch.mean(torch.pow(score_fake, 2))

        loss_d = loss_d / len(res_fake + period_fake)

        batch_size = audio.shape[0]
        losses["disc_loss"] += loss_d * batch_size

        (loss_d / grad_acc_steps).backward()

    clip_grad_norm_(disc.parameters(), grad_clip_thresh)

    optim_d.step_and_update_lr(step)

    if step % log_step == 0:

        for loss_name in losses.keys():
            losses[loss_name] /= total_batch_size

        message = "Step {}/{}, ".format(step, total_step)
        for j, loss_name in enumerate(losses.keys()):
            if j != 0:
                message += ", "
            loss_value = losses[loss_name]
            message += f"{loss_name}: {round(loss_value.item(), 4)}"

        for key in losses.keys():
            logger.log_graph(name=f"train_{key}", value=losses[key].item(), step=step)

        logger.log_graph(
            name="lr", value=optim_g._optimizer.param_groups[0]["lr"], step=step
        )
        logger.log_graph(
            name="only_train_speaker_emb", value=1 if model_is_frozen else 0, step=step
        )

        logger.log_image(
            name="bert_attention",
            image=outputs["bert_attention"][0, 0, :, :].T.detach().cpu().numpy(),
            step=step,
        )

        print(message)

    if step % 10 == 0:
        logger.query(
            "UPDATE training_run SET acoustic_fine_tuning_progress=? WHERE ID=?",
            [step / total_step, db_id],
        )


def eval_iter(
    gen: AcousticModel,
    style_predictor: ScriptModule,
    step: int,
    train_config: Union[AcousticFinetuningConfig, AcousticPretrainingConfig],
    batch_size: int,
    stft_criterion: MultiResolutionSTFTLoss,
    loader: DataLoader,
    device: torch.device,
    logger: Logger,
    preprocess_config: PreprocessingConfig,
) -> None:
    gen.eval()
    style_predictor.eval()

    losses = {
        "mel_loss": torch.FloatTensor([0.0]).to(device),
        "duration_loss": torch.FloatTensor([0.0]).to(device),
        "p_prosody_loss": torch.FloatTensor([0.0]).to(device),
        "pitch_loss": torch.FloatTensor([0.0]).to(device),
        "estoi": torch.FloatTensor([0.0]).to(device),
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
                    audio,
                ) = batch
                audio = audio.unsqueeze(1)
                style_embeds_pred = style_predictor(token_ids, attention_masks)

                src_mask = get_mask_from_lengths(src_lens)
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
                audio = slice_segments(
                    audio,
                    outputs["ids_slice"] * PreprocessingConfig().stft.hop_length,
                    PreprocessingConfig().segment_size
                    * PreprocessingConfig().stft.hop_length,
                )  # slice
                fake_audio = outputs["y_pred"]
                sc_loss, mag_loss = stft_criterion(
                    fake_audio.squeeze(1), audio.squeeze(1)
                )
                stft_loss = (sc_loss + mag_loss) * AcousticFinetuningConfig().stft_lamb

                l_pitch = pitch_loss(
                    outputs["pitch_prediction"], y_true=pitches, src_masks=src_mask
                )
                l_dur = duration_loss(
                    log_y_pred=outputs["log_duration_prediction"],
                    y_true=durations,
                    src_masks=src_mask,
                )
                l_prosody = (
                    prosody_loss(
                        y_pred=outputs["p_prosody_pred"],
                        y_ref=outputs["p_prosody_ref"],
                        src_masks=src_mask,
                    )
                    * 0.5
                )

                estoi = calc_estoi(audio, fake_audio, preprocess_config.sampling_rate)

                batch_size = mels.shape[0]
                losses["duration_loss"] += l_dur * batch_size
                losses["p_prosody_loss"] += l_prosody * batch_size
                losses["pitch_loss"] += l_pitch * batch_size
                losses["mel_loss"] += stft_loss * batch_size
                losses["estoi"] += estoi * batch_size
                len_ds += batch_size

    message = f"Validation Step {step}: "
    for j, loss_name in enumerate(losses.keys()):

        if j != 0:
            message += ", "
        loss_value = losses[loss_name]
        loss_value /= len_ds
        message += f"{loss_name}: {round(loss_value.item(), 4)}"

    print(message)

    for key in losses.keys():
        logger.log_graph(name=f"val_{key}", value=losses[key].item(), step=step)


def train_acoustic(
    db_id: int,
    training_run_name: str,
    preprocess_config: PreprocessingConfig,
    model_config: AcousticModelConfig,
    train_config: Union[AcousticFinetuningConfig, AcousticPretrainingConfig],
    logger: Logger,
    device: torch.device,
    reset: bool,
    checkpoint_acoustic: Union[str, None],
    checkpoint_style: Union[str, None],
    fine_tuning: bool,
    overwrite_saves: bool,
    assets_path: str,
    training_runs_path: str,
) -> None:
    batch_size = train_config.batch_size
    data_path = Path(training_runs_path) / str(training_run_name) / "data"

    # Prepare model
    gen, style_predictor, disc, optim_g, optim_d, step = get_acoustic_models(
        data_path=str(data_path),
        checkpoint_acoustic=checkpoint_acoustic,
        checkpoint_style=checkpoint_style,
        train_config=train_config,
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=fine_tuning,
        device=device,
        reset=reset,
        assets_path=assets_path,
    )

    group_size = 5  # Set this larger than 1 to enable sorting in Dataset
    train_loader, validation_loader = get_data_loaders(
        batch_size=batch_size,
        group_size=group_size,
        data_path=data_path,
        assets_path=assets_path,
    )
    train_loader = cycle_2d(train_loader)

    gen_pars = get_param_num(gen)
    style_pars = get_param_num(style_predictor)
    prosody_encoder_pars = get_param_num(gen.phoneme_prosody_encoder)

    print(f"Number of acoustic model parameters: {gen_pars}")
    print(f"Number of style predictor parameters: {style_pars}")

    print(f"Total number of parameters: {gen_pars + style_pars}")
    print(
        f"Total number of parameters during inference: {gen_pars + style_pars - prosody_encoder_pars}"
    )

    stft_criterion = MultiResolutionSTFTLoss(
        device, VocoderModelConfig().mrd.resolutions
    )

    # Training
    grad_acc_steps = train_config.grad_acc_step
    grad_clip_thresh = train_config.optimizer_config.grad_clip_thresh
    log_step = train_config.log_step
    save_step = train_config.save_step
    synth_step = train_config.synth_step
    val_step = train_config.val_step
    total_step = train_config.train_steps
    only_train_speaker_until = train_config.only_train_speaker_until
    freeze_bert_until = 0 if fine_tuning else train_config.freeze_bert_until
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
        iterable=range(step, total_step + 1), start=step, total=total_step,
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
            disc=disc,
            optim_g=optim_g,
            optim_d=optim_d,
            train_loader=train_loader,
            stft_criterion=stft_criterion,
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
                stft_criterion=stft_criterion,
                step=step,
                train_config=train_config,
                batch_size=batch_size,
                loader=validation_loader,
                device=device,
                logger=logger,
                preprocess_config=preprocess_config,
            )

        if step % synth_step == 0 and step != 0:
            synth_iter(
                gen=gen,
                style_predictor=style_predictor,
                step=step,
                preprocess_config=preprocess_config,
                device=device,
                logger=logger,
                data_path=str(data_path),
                assets_path=assets_path,
            )

        if step % save_step == 0 and step != 0 or step >= total_step:
            save_model(
                name="acoustic",
                ckpt_dict={
                    "gen": gen.state_dict(),
                    "disc": disc.state_dict(),
                    "optim_g": optim_g._optimizer.state_dict(),
                    "optim_d": optim_d._optimizer.state_dict(),
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
