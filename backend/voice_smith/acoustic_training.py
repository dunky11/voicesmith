
from logging import Logger
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
import torchaudio
from pathlib import Path
from itertools import chain
import numpy as np
from torch.jit._script import ScriptModule
from typing import Dict, Any, Union, Tuple, Generator
from voice_smith.config.configs import (
    PreprocessingConfig,
    AcousticPretrainingConfig,
    AcousticFinetuningConfig,
    AcousticModelConfigType,
)
from voice_smith.utils.optimizer import (
    ScheduledOptimPretraining,
    ScheduledOptimFinetuning,
)
from voice_smith.utils.model import (
    get_acoustic_models,
    get_param_num,
    save_model,
)
from voice_smith.utils.tools import (
    to_device,
    cycle_2d,
    get_mask_from_lengths,
    iter_logger,
)
from voice_smith.utils.loss import FastSpeech2LossGen
from voice_smith.utils.dataset import AcousticDataset
from voice_smith.model.acoustic_model import AcousticModel
from voice_smith.utils.model import get_infer_vocoder
from voice_smith.utils.loggers import Logger
from voice_smith.utils.metrics import calc_estoi, calc_pesq


def unfreeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = True


def freeze_torchscript(model: ScriptModule) -> None:
    for par in model.parameters():
        par.requires_grad = False


def synth_iter(
    gen: AcousticModel,
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
        is_eval=True,
    )
    loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    sampling_rate = preprocess_config.sampling_rate
    vocoder = get_infer_vocoder(
        checkpoint=str(Path(assets_path) / "vocoder_pretrained.pt"),
        preprocess_config=preprocess_config,
        device=device,
    )
    with torch.no_grad():
        for batches in loader:
            for batch in batches:
                batch = to_device(batch, device, is_eval=True)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    mel_lens,
                    langs,
                    attn_priors,
                    wavs,
                ) = batch
                for (speaker, text, src_len, mel, mel_len, lang, wav) in zip(
                    speakers, texts, src_lens, mels, mel_lens, langs, wavs
                ):
                    y_pred = gen(
                        x=text[: src_len.item()].unsqueeze(0),
                        speakers=speaker.unsqueeze(0),
                        p_control=1.0,
                        d_control=1.0,
                        langs=lang.unsqueeze(0),
                    )
                    wav_prediction = vocoder.infer(
                        y_pred,
                        mel_lens=torch.tensor(
                            [y_pred.shape[2]], dtype=torch.int32, device=device
                        ),
                    )
                    wav_prediction = wav_prediction[0, 0].cpu().numpy()

                    logger.log_audio(
                        name="val_wav_reconstructed",
                        audio=wav[0, : mel_len * preprocess_config.stft.hop_length]
                        .cpu()
                        .numpy(),
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
        assets_path=assets_path,
        is_eval=False,
    )
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
        is_eval=True,
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

    losses = {
        "reconstruction_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "mel_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "ssim_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "duration_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "u_prosody_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "p_prosody_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "pitch_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "ctc_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "bin_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
    }

    total_batch_size = 0

    for j in range(grad_acc_steps):
        batch = next(train_loader)
        batch = to_device(batch, device, is_eval=False)
        (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
        ) = batch

        src_mask = get_mask_from_lengths(src_lens)
        mel_mask = get_mask_from_lengths(mel_lens)
        outputs = gen.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=src_lens,
            mels=mels,
            mel_lens=mel_lens,
            pitches=pitches,
            langs=langs,
            attn_priors=attn_priors,
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
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
        ) = criterion(
            src_masks=src_mask,
            mel_masks=mel_mask,
            mel_targets=mels,
            mel_predictions=y_pred,
            log_duration_predictions=log_duration_prediction,
            u_prosody_ref=outputs["u_prosody_ref"],
            u_prosody_pred=outputs["u_prosody_pred"],
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_predictions=pitch_prediction,
            p_targets=outputs["pitch_target"],
            durations=outputs["attn_hard_dur"],
            attn_logprob=outputs["attn_logprob"],
            attn_soft=outputs["attn_soft"],
            attn_hard=outputs["attn_hard"],
            src_lens=src_lens,
            mel_lens=mel_lens,
            step=step,
        )

        batch_size = mels.shape[0]
        losses["reconstruction_loss"] += total_loss * batch_size
        losses["mel_loss"] += mel_loss * batch_size
        losses["ssim_loss"] += ssim_loss * batch_size
        losses["duration_loss"] += duration_loss * batch_size
        losses["u_prosody_loss"] += u_prosody_loss * batch_size
        losses["p_prosody_loss"] += p_prosody_loss * batch_size
        losses["pitch_loss"] += pitch_loss * batch_size
        losses["ctc_loss"] += ctc_loss * batch_size
        losses["bin_loss"] += bin_loss * batch_size
        total_batch_size += batch_size

        (total_loss / grad_acc_steps).backward()

    clip_grad_norm_(gen.parameters(), grad_clip_thresh)

    optim.step_and_update_lr(step)
    optim.zero_grad()

    if step % log_step == 0:

        for loss_name in losses.keys():
            losses[loss_name] /= total_batch_size

        message = f"Step {step}/{total_step}, "
        for j, loss_name in enumerate(losses.keys()):
            if j != 0:
                message += ", "
            loss_value = losses[loss_name]
            message += f"{loss_name}: {round(loss_value.item(), 4)}"
        print(message)

        for key in losses.keys():
            logger.log_graph(name=f"train_{key}", value=losses[key].item(), step=step)

        logger.log_graph(
            name="lr", value=optim._optimizer.param_groups[0]["lr"], step=step
        )
        logger.log_graph(
            name="only_train_speaker_emb", value=1 if model_is_frozen else 0, step=step
        )
        logger.log_image(
            name="attn_soft",
            image=outputs["attn_soft"][0, 0, :mel_lens[0].item(), :src_lens[0].item()].detach().T.cpu().numpy(),
            step=step,
        )
        logger.log_image(
            name="attn_hard",
            image=outputs["attn_hard"][0, 0, :mel_lens[0].item(), :src_lens[0].item()].detach().T.cpu().numpy(),
            step=step,
        )

    if step % 10 == 0:
        logger.query(
            "UPDATE training_run SET acoustic_fine_tuning_progress=? WHERE ID=?",
            (step / total_step, db_id),
        )


def evaluate(
    gen: AcousticModel,
    step: int,
    train_config: Union[AcousticFinetuningConfig, AcousticPretrainingConfig],
    batch_size: int,
    loader: DataLoader,
    criterion: FastSpeech2LossGen,
    device: torch.device,
    logger: Logger,
    assets_path: str,
    preprocess_config: PreprocessingConfig,
) -> None:
    gen.eval()

    losses = {
        "reconstruction_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "mel_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "ssim_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "duration_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "u_prosody_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "p_prosody_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "pitch_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "pesq": torch.tensor([0.0], dtype=torch.float32, device=device),
        "estoi": torch.tensor([0.0], dtype=torch.float32, device=device),
        "ctc_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "bin_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
    }

    len_ds = 0
    with torch.no_grad():
        for batches in iter_logger(loader):
            for batch in batches:
                batch = to_device(batch, device, is_eval=True)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    mel_lens,
                    langs,
                    attn_priors,
                    wavs,
                ) = batch
                src_mask = get_mask_from_lengths(src_lens)
                mel_mask = get_mask_from_lengths(mel_lens)
                outputs = gen.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=src_lens,
                    mels=mels,
                    mel_lens=mel_lens,
                    pitches=pitches,
                    langs=langs,
                    attn_priors=attn_priors,
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
                    u_prosody_loss,
                    p_prosody_loss,
                    pitch_loss,
                    ctc_loss,
                    bin_loss,
                ) = criterion(
                    src_masks=src_mask,
                    mel_masks=mel_mask,
                    mel_targets=mels,
                    mel_predictions=y_pred,
                    log_duration_predictions=log_duration_prediction,
                    u_prosody_ref=outputs["u_prosody_ref"],
                    u_prosody_pred=outputs["u_prosody_ref"],
                    p_prosody_ref=p_prosody_ref,
                    p_prosody_pred=p_prosody_pred,
                    pitch_predictions=pitch_prediction,
                    p_targets=outputs["pitch_target"],
                    durations=outputs["attn_hard_dur"],
                    attn_logprob=outputs["attn_logprob"],
                    attn_soft=outputs["attn_soft"],
                    attn_hard=outputs["attn_hard"],
                    src_lens=src_lens,
                    mel_lens=mel_lens,
                    step=step,
                )
                batch_size = mels.shape[0]
                losses["reconstruction_loss"] += total_loss * batch_size
                losses["mel_loss"] += mel_loss * batch_size
                losses["ssim_loss"] += ssim_loss * batch_size
                losses["duration_loss"] += duration_loss * batch_size
                losses["u_prosody_loss"] += u_prosody_loss * batch_size
                losses["p_prosody_loss"] += p_prosody_loss * batch_size
                losses["pitch_loss"] += pitch_loss * batch_size
                losses["ctc_loss"] += ctc_loss * batch_size
                losses["bin_loss"] += bin_loss * batch_size
                len_ds += batch_size

    vocoder = get_infer_vocoder(
        checkpoint=str(Path(assets_path) / "vocoder_pretrained.pt"),
        preprocess_config=preprocess_config,
        device=device,
    )
    sampling_rate = preprocess_config.sampling_rate

    resampler_16k = torchaudio.transforms.Resample(
        orig_freq=sampling_rate, new_freq=16000
    ).to(device)

    with torch.no_grad():
        for batches in iter_logger(loader):
            for batch in batches:
                batch = to_device(batch, device, is_eval=True)
                (
                    ids,
                    raw_texts,
                    speakers,
                    speaker_names,
                    texts,
                    src_lens,
                    mels,
                    pitches,
                    mel_lens,
                    langs,
                    attn_priors,
                    wavs,
                ) = batch
                outputs = gen.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=src_lens,
                    mels=mels,
                    mel_lens=mel_lens,
                    pitches=pitches,
                    langs=langs,
                    attn_priors=attn_priors,
                    use_ground_truth=False,
                )
                y_pred = vocoder.infer(outputs["y_pred"], mel_lens=mel_lens)
                wavs = wavs[:, :, : y_pred.shape[2]]
                estoi = calc_estoi(
                    audio_real=wavs,
                    audio_fake=y_pred,
                    sampling_rate=preprocess_config.sampling_rate,
                )
                pesq = calc_pesq(
                    audio_real_16k=resampler_16k(wavs),
                    audio_fake_16k=resampler_16k(y_pred),
                )
                batch_size = mels.shape[0]
                losses["estoi"] += estoi * batch_size
                losses["pesq"] += pesq * batch_size

    for loss_name in losses.keys():
        losses[loss_name] /= len_ds

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
    preprocess_config: PreprocessingConfig,
    model_config: AcousticModelConfigType,
    train_config: Union[AcousticFinetuningConfig, AcousticPretrainingConfig],
    logger: Logger,
    device: torch.device,
    reset: bool,
    checkpoint_acoustic: Union[str, None],
    fine_tuning: bool,
    overwrite_saves: bool,
    assets_path: str,
    training_runs_path: str,
) -> None:
    batch_size = train_config.batch_size
    data_path = Path(training_runs_path) / str(training_run_name) / "data"

    gen, optim, step = get_acoustic_models(
        data_path=str(data_path),
        checkpoint_acoustic=checkpoint_acoustic,
        train_config=train_config,
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=fine_tuning,
        device=device,
        reset=reset,
        assets_path=assets_path,
    )

    group_size = 5
    train_loader, validation_loader = get_data_loaders(
        batch_size=batch_size,
        group_size=group_size,
        data_path=data_path,
        assets_path=assets_path,
    )
    train_loader = cycle_2d(train_loader)

    criterion = FastSpeech2LossGen(fine_tuning=fine_tuning, device=device)

    gen_pars = get_param_num(gen)
    prosody_encoder_pars = get_param_num(gen.phoneme_prosody_encoder)

    print(f"Number of acoustic model parameters: {gen_pars}")
    print(
        f"Total number of parameters during inference: {gen_pars - prosody_encoder_pars}"
    )

    grad_acc_steps = train_config.grad_acc_step
    grad_clip_thresh = train_config.optimizer_config.grad_clip_thresh
    log_step = train_config.log_step
    save_step = train_config.save_step
    synth_step = train_config.synth_step
    val_step = train_config.val_step
    total_step = train_config.train_steps
    only_train_speaker_until = train_config.only_train_speaker_until
    freeze_model_until = only_train_speaker_until if fine_tuning else 0
    ckpt_dir = Path(training_runs_path) / training_run_name / "ckpt" / "acoustic"
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    model_is_frozen = step < freeze_model_until

    if model_is_frozen:
        gen.freeze()
    else:
        gen.unfreeze(freeze_text_embed=fine_tuning, freeze_lang_embed=fine_tuning)

    for step in iter_logger(
        iterable=range(step, total_step + 1), start=step, total=total_step,
    ):

        if model_is_frozen and step >= freeze_model_until:
            model_is_frozen = False
            gen.unfreeze(freeze_text_embed=fine_tuning, freeze_lang_embed=fine_tuning)

        train_iter(
            db_id=db_id,
            gen=gen,
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

        # Don't evaluate on pretraining step 0 since duration predictor could
        # predict too high durations
        if step % val_step == 0 and not (step == 0 and not fine_tuning):
            evaluate(
                gen=gen,
                step=step,
                train_config=train_config,
                batch_size=batch_size,
                loader=validation_loader,
                criterion=criterion,
                device=device,
                logger=logger,
                assets_path=assets_path,
                preprocess_config=preprocess_config,
            )

        if step % synth_step == 0 and step != 0:
            synth_iter(
                gen=gen,
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
                    "optim": optim._optimizer.state_dict(),
                    "steps": step,
                },
                ckpt_dir=str(ckpt_dir),
                step=step,
                overwrite=overwrite_saves,
            )
            if step >= total_step:
                break

    logger.query(
        "UPDATE training_run SET acoustic_fine_tuning_progress=? WHERE ID=?",
        (1.0, db_id),
    )