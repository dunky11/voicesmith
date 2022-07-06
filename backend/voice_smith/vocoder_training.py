import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from typing import Union
from voice_smith.utils.dataset import VocoderDataset
from voice_smith.utils.audio import TacotronSTFT
from voice_smith.utils.model import get_param_num, save_model, get_vocoder
from voice_smith.model.univnet import MultiResolutionSTFTLoss
from voice_smith.utils.tools import (
    cycle,
    iter_logger,
)
from voice_smith.utils.metrics import calc_rmse, calc_pesq, calc_estoi
from voice_smith.config.configs import (
    PreprocessingConfig,
    VocoderFinetuningConfig,
    VocoderPretrainingConfig,
    VocoderModelConfig,
)

torch.backends.cudnn.benchmark = True


def synth_iter(
    eval_ds: torch.utils.data.DataLoader, vocoder, step, sampling_rate, device, logger
):
    vocoder.eval()
    with torch.no_grad():
        mel, audio, _ = eval_ds.get_sample_to_synth()
        mel = mel.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)

        audio_fake = vocoder(
            mel.unsqueeze(0),
            mel_lens=torch.tensor([mel.shape[1]], dtype=torch.int64, device=device),
        )
        logger.log_audio(
            name="audio_fake",
            audio=audio_fake[0, 0].cpu().numpy(),
            sr=sampling_rate,
            step=step,
        )
        logger.log_audio(
            name="audio_real", audio=audio.cpu().numpy(), sr=sampling_rate, step=step,
        )


def get_data_loaders(
    preprocess_config: PreprocessingConfig,
    fine_tuning,
    segment_size,
    batch_size,
    num_workers,
    preprocessed_path,
):
    trainset = VocoderDataset(
        filename="train.txt",
        fine_tuning=fine_tuning,
        preprocess_config=preprocess_config,
        preprocessed_path=preprocessed_path,
        segment_size=segment_size,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=num_workers,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )

    validset = VocoderDataset(
        filename="val.txt",
        fine_tuning=fine_tuning,
        preprocess_config=preprocess_config,
        preprocessed_path=preprocessed_path,
        segment_size=-1,
    )

    validation_loader = DataLoader(
        validset,
        num_workers=num_workers,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=validset.collate_fn,
    )

    return train_loader, validation_loader, validset


def train_iter(
    generator,
    discriminator,
    loader,
    device,
    grad_accum_steps,
    stft_criterion,
    optim_g,
    optim_d,
    stft_lamb,
):
    generator.train()
    discriminator.train()

    loss_means = {
        "total_loss_gen": torch.tensor([0.0], dtype=torch.float32, device=device),
        "total_loss_disc": torch.tensor([0.0], dtype=torch.float32, device=device),
        "mel_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
        "score_loss": torch.tensor([0.0], dtype=torch.float32, device=device),
    }

    len_group = 0
    optim_g.zero_grad()
    audios, fake_audios = [], []

    # Train Generator
    for _ in range(grad_accum_steps):
        batch = next(loader)
        mel, audio, _ = batch
        mel = mel.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        audio = audio.unsqueeze(1)

        fake_audio = generator.forward_train(mel)

        sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * stft_lamb

        res_fake, period_fake = discriminator(fake_audio)

        score_loss = 0.0

        for (_, score_fake) in res_fake + period_fake:
            score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

        score_loss = score_loss / len(res_fake + period_fake)
        loss_g = score_loss + stft_loss

        batch_size = mel.shape[0]
        len_group += batch_size
        loss_means["total_loss_gen"] += loss_g * batch_size
        loss_means["mel_loss"] += stft_loss * batch_size
        loss_means["score_loss"] += score_loss * batch_size

        (loss_g / grad_accum_steps).backward()

        audios.append(audio)
        fake_audios.append(fake_audio.detach())

    optim_g.step()
    optim_d.zero_grad()

    # Train Discriminator
    for audio, fake_audio in zip(audios, fake_audios):

        res_fake, period_fake = discriminator(fake_audio.detach())
        res_real, period_real = discriminator(audio)

        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(
            res_fake + period_fake, res_real + period_real
        ):
            loss_d += torch.mean(torch.pow(score_real - 1.0, 2))
            loss_d += torch.mean(torch.pow(score_fake, 2))

        loss_d = loss_d / len(res_fake + period_fake)

        batch_size = audio.shape[0]
        loss_means["total_loss_disc"] += loss_d * batch_size

        (loss_d / grad_accum_steps).backward()

    optim_d.step()

    for loss_name in loss_means.keys():
        loss_means[loss_name] /= len_group

    return loss_means


def evaluate(
    generator,
    loader,
    device,
    stft_criterion,
    stft_lamb,
    preprocess_config: PreprocessingConfig,
):
    generator.eval()
    loss_means = {
        "pesq": torch.tensor([0.0], dtype=torch.float32, device=device),
        "estoi": torch.tensor([0.0], dtype=torch.float32, device=device),
        "rmse": torch.tensor([0.0], dtype=torch.float32, device=device),
    }
    len_ds = 0
    stft = TacotronSTFT(
        filter_length=preprocess_config.stft.filter_length,
        hop_length=preprocess_config.stft.hop_length,
        win_length=preprocess_config.stft.win_length,
        n_mel_channels=preprocess_config.stft.n_mel_channels,
        sampling_rate=preprocess_config.sampling_rate,
        mel_fmin=preprocess_config.stft.mel_fmin,
        mel_fmax=preprocess_config.stft.mel_fmax,
        device=device,
        center=False,
    )
    resampler_16k = torchaudio.transforms.Resample(
        orig_freq=preprocess_config.sampling_rate, new_freq=16000
    ).to(device)

    with torch.no_grad():
        for batch in iter_logger(loader):
            mel, audio, speaker_ids, mel_lens = batch
            mel = mel.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            mel_lens = mel_lens.to(device, non_blocking=True)
            audio = audio.unsqueeze(1)
            fake_audio = generator(mel, mel_lens=mel_lens)

            audio = audio[:, :, : fake_audio.shape[2]]

            estoi = calc_estoi(
                audio_real=audio,
                audio_fake=fake_audio,
                sampling_rate=preprocess_config.sampling_rate,
            )
            rmse = calc_rmse(audio_real=audio, audio_fake=fake_audio, stft=stft)
            pesq = calc_pesq(
                audio_real_16k=resampler_16k(audio),
                audio_fake_16k=resampler_16k(fake_audio),
            )
            batch_size = mel.shape[0]
            len_ds += batch_size
            loss_means["estoi"] += estoi * batch_size
            loss_means["pesq"] += pesq * batch_size
            loss_means["rmse"] += rmse * batch_size

    for loss_name in loss_means.keys():
        loss_means[loss_name] /= len_ds

    return loss_means


def train_vocoder(
    db_id: int,
    training_run_name,
    train_config: Union[VocoderPretrainingConfig, VocoderFinetuningConfig],
    model_config: VocoderModelConfig,
    preprocess_config: PreprocessingConfig,
    logger,
    device,
    reset,
    checkpoint_path,
    fine_tuning,
    overwrite_saves,
    training_runs_path: str,
    stop_after_hour=None,
):

    preprocessed_path = Path(training_runs_path) / str(training_run_name) / "data"

    if stop_after_hour is not None:
        stop_at = time.time() + float(stop_after_hour) * 3600
    else:
        stop_at = np.Inf

    checkpoint_dir = (
        Path(training_runs_path) / str(training_run_name) / "ckpt" / "vocoder"
    )
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    print("checkpoints directory : ", checkpoint_path)

    stft_lamb = train_config.stft_lamb
    (
        generator,
        discriminator,
        steps,
        optim_g,
        optim_d,
        scheduler_g,
        scheduler_d,
    ) = get_vocoder(
        checkpoint=checkpoint_path,
        train_config=train_config,
        model_config=model_config,
        preprocess_config=preprocess_config,
        reset=reset,
        device=device,
    )

    stft_criterion = MultiResolutionSTFTLoss(device, model_config.mrd.resolutions)
    train_loader, validation_loader, validset = get_data_loaders(
        preprocess_config=preprocess_config,
        fine_tuning=fine_tuning,
        segment_size=train_config.segment_size,
        batch_size=train_config.batch_size,
        num_workers=4,
        preprocessed_path=preprocessed_path,
    )

    train_loader = cycle(train_loader)

    print("Generator has a total of " + str(get_param_num(generator)) + " parameters")
    print("MRD has a total of " + str(get_param_num(discriminator.MRD)) + " parameters")
    print("MPD has a total of " + str(get_param_num(discriminator.MPD)) + " parameters")

    while True:

        start_b = time.time()

        loss_means = train_iter(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            device=device,
            stft_criterion=stft_criterion,
            grad_accum_steps=train_config.grad_accum_steps,
            optim_g=optim_g,
            optim_d=optim_d,
            stft_lamb=stft_lamb,
        )

        if steps % train_config.stdout_interval == 0:
            assert optim_g.param_groups[0]["lr"] == optim_d.param_groups[0]["lr"]
            message = f"Train step {steps}: "
            for j, loss_name in enumerate(loss_means.keys()):
                if j != 0:
                    message += ", "
                loss_value = loss_means[loss_name]
                message += f"{loss_name}: {round(loss_value.item(), 4)}"
            message += f" s/b : {round(time.time() - start_b, 4)}"
            print(message)
            for key in loss_means.keys():
                logger.log_graph(
                    name=f"train_{key}", value=loss_means[key].item(), step=steps
                )
            logger.log_graph(name="lr", value=optim_g.param_groups[0]["lr"], step=steps)

        if steps % train_config.synth_interval == 0:
            synth_iter(
                eval_ds=validset,
                vocoder=generator,
                step=steps,
                sampling_rate=preprocess_config.sampling_rate,
                device=device,
                logger=logger,
            )

        if steps % train_config.validation_interval == 0 and steps != 0:
            print("\nEvaluating ...\n")
            loss_means = evaluate(
                generator=generator,
                loader=validation_loader,
                device=device,
                stft_criterion=stft_criterion,
                stft_lamb=stft_lamb,
                preprocess_config=preprocess_config,
            )
            message = f"Validation step {steps}: "
            for j, loss_name in enumerate(loss_means.keys()):
                if j != 0:
                    message += ", "
                loss_value = loss_means[loss_name]
                message += f"{loss_name}: {round(loss_value.item(), 4)}"
            message += f" s/b : {round(time.time() - start_b, 4)}"
            print(message)
            for key in loss_means.keys():
                logger.log_graph(
                    name=f"val_{key}", value=loss_means[key].item(), step=steps
                )

        if steps % 10 == 0:
            logger.query(
                "UPDATE training_run SET vocoder_fine_tuning_progress=? WHERE ID=?",
                [steps / train_config.train_steps, db_id],
            )

        if steps % train_config.checkpoint_interval == 0 and steps != 0:
            save_model(
                name="vocoder",
                ckpt_dict={
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "steps": steps,
                },
                ckpt_dir=str(checkpoint_dir),
                step=steps,
                overwrite=overwrite_saves,
            )

        if steps >= train_config.train_steps:
            save_model(
                name="vocoder",
                ckpt_dict={
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "steps": steps,
                },
                ckpt_dir=str(checkpoint_dir),
                step=steps,
                overwrite=overwrite_saves,
            )
            break

        if stop_after_hour is not None and time.time() >= stop_at:
            save_model(
                name="vocoder",
                ckpt_dict={
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "steps": steps,
                },
                ckpt_dir=str(checkpoint_dir),
                step=steps,
                overwrite=overwrite_saves,
            )
            break

        if steps % 1000 == 0 and steps != 0:
            scheduler_g.step()
            scheduler_d.step()

        steps += 1

    logger.query(
        "UPDATE training_run SET vocoder_fine_tuning_progress=? WHERE ID=?",
        [1.0, db_id],
    )

