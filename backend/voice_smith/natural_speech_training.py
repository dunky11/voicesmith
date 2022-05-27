import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from voice_smith.utils.dataset import NaturalSpeechDataset
from voice_smith.utils.audio import TacotronSTFT
from voice_smith.utils.model import get_param_num, save_model, get_nat_speech
from voice_smith.model.univnet import MultiResolutionSTFTLoss
from voice_smith.utils.tools import (
    cycle_2d,
    iter_logger,
)
from voice_smith.config.preprocess_config import preprocess_config
from voice_smith.config.vocoder_model_config import vocoder_model_config as hp
from voice_smith.utils.metrics import calc_rmse, calc_pesq, calc_estoi
from voice_smith.model.vits_orig import slice_segments, rand_slice_segments

# torch.backends.cudnn.benchmark = True


def duration_loss(log_duration_pred, duration_targets, mask):
    mask = ~mask.squeeze(1)
    log_duration_true = torch.log(duration_targets.float() + 1)
    log_duration_pred = log_duration_pred.masked_select(mask)
    log_duration_true = log_duration_true.masked_select(mask)
    mse = F.mse_loss(log_duration_pred, log_duration_true)
    return mse


def synth_iter(
    eval_loader: torch.utils.data.DataLoader,
    generator,
    step,
    sampling_rate,
    device,
    logger,
):
    generator.eval()
    with torch.no_grad():
        for group in eval_loader:
            for batch in group:
                # TODO create directly on cuda
                (
                    speakers,
                    texts,
                    mel,
                    audio,
                    text_lens,
                    mel_lens,
                    durations,
                    token_ids,
                    attention_masks,
                ) = to_device(batch, device)
                audio_fake = generator(
                    texts[0:1, : text_lens[0].item()],
                    speakers=speakers[0:1],
                    src_lens=text_lens[0:1],
                    token_ids=token_ids[0:1],
                    attention_mask=attention_masks[0:1],
                )

                logger.log_audio(
                    name="audio_fake",
                    audio=audio_fake.squeeze(0).squeeze(0).cpu().numpy(),
                    sr=sampling_rate,
                    step=step,
                )
                logger.log_audio(
                    name="audio_real",
                    audio=audio[0].cpu().numpy(),
                    sr=sampling_rate,
                    step=step,
                )
                return


def get_data_loaders(
    assets_path,
    preprocess_config,
    fine_tuning,
    segment_size,
    batch_size,
    num_workers,
    preprocessed_path,
):
    trainset = NaturalSpeechDataset(
        assets_path=assets_path,
        filename="train.txt",
        batch_size=batch_size,
        data_path=preprocessed_path,
        sort=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=num_workers,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=trainset.collate_fn,
    )

    validset = NaturalSpeechDataset(
        assets_path=assets_path,
        filename="val.txt",
        batch_size=batch_size,
        data_path=preprocessed_path,
        sort=False,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validset,
        num_workers=num_workers,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,  # For debugging
        collate_fn=validset.collate_fn,
    )

    return train_loader, validation_loader, validset


def to_device(batch, device):
    (
        speakers,
        texts,
        mels,
        audios,
        text_lens,
        mel_lens,
        durations,
        token_ids,
        attention_masks,
    ) = batch
    speakers = torch.LongTensor(speakers).to(device)
    texts = torch.LongTensor(texts).to(device)
    mels = torch.FloatTensor(mels).to(device)
    audios = torch.FloatTensor(audios).to(device)
    text_lens = torch.LongTensor(text_lens).to(device)
    mel_lens = torch.LongTensor(mel_lens).to(device)
    durations = torch.LongTensor(durations).to(device)
    token_ids = torch.LongTensor(token_ids).to(device)
    attention_masks = torch.LongTensor(attention_masks).to(device)
    return (
        speakers,
        texts,
        mels,
        audios,
        text_lens,
        mel_lens,
        durations,
        token_ids,
        attention_masks,
    )


def get_batch(loader, device):
    return to_device(next(loader), device)


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


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
        "total_loss_gen": 0,
        "total_loss_disc": 0,
        "mel_loss_dec": 0,
        "score_loss": 0,
        "kl_loss_backward": 0,
        "kl_loss_forward": 0,
        "l_dur": 0,
    }

    len_group = 0
    optim_g.zero_grad()
    audios, fake_audios = [], []

    # Train Generator
    for _ in range(grad_accum_steps):
        (
            speakers,
            texts,
            mel,
            audio,
            text_lens,
            mel_lens,
            durations,
            token_ids,
            attention_masks,
        ) = get_batch(loader, device=device)
        audio = audio.unsqueeze(1)

        outputs = generator.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=text_lens,
            specs=mel,
            spec_lens=mel_lens,
            durations=durations,
            token_ids=token_ids,
            attention_mask=attention_masks,
        )

        fake_audio = outputs["wave"]
        ids_slice = outputs["sample_q_ids_slice"]

        mel = slice_segments(mel, ids_slice, 32)
        audio = slice_segments(audio, ids_slice * 256, 8192)  # slice

        res_fake, period_fake = discriminator(fake_audio)

        loss_dur = duration_loss(
            log_duration_pred=outputs["log_duration_pred"],
            duration_targets=durations,
            mask=outputs["src_mask"],
        )

        loss_kl_backward = kl_loss(
            outputs["z_p"],
            outputs["log_std_q"],
            outputs["mu_p"],
            outputs["log_std_p"],
            ~outputs["spec_mask"],
        )
        loss_kl_forward = kl_loss(
            outputs["z_q"],
            outputs["log_std_p"],
            outputs["mu_q"],
            outputs["log_std_q"],
            ~outputs["spec_mask"],
        )

        sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * stft_lamb

        score_loss = 0.0

        for (_, score_fake) in res_fake + period_fake:
            score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

        score_loss = score_loss / len(res_fake + period_fake)
        loss_g = score_loss + stft_loss + loss_dur + loss_kl_backward + loss_kl_forward
        batch_size = mel.shape[0]
        len_group += batch_size
        loss_means["total_loss_gen"] += loss_g * batch_size
        loss_means["mel_loss_dec"] += stft_loss * batch_size
        loss_means["score_loss"] += score_loss * batch_size
        loss_means["kl_loss_backward"] += loss_kl_backward * batch_size
        loss_means["kl_loss_forward"] += loss_kl_forward * batch_size
        loss_means["l_dur"] += loss_dur * batch_size

        (loss_g / grad_accum_steps).backward()

        audios.append(audio)
        fake_audios.append(fake_audio.detach())

    optim_g.step()
    optim_d.zero_grad()

    # Train Discriminator
    for audio, fake_audio in zip(audios, fake_audios):

        res_fake, period_fake = discriminator(fake_audio)
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


def evaluate(generator, loader, device, stft_criterion, stft_lamb):
    generator.eval()

    loss_means = {
        "mel_loss_dec": 0,
        "kl_loss_backward": 0,
        "kl_loss_forward": 0,
        "l_dur": 0,
        "pesq": 0,
        "estoi": 0,
        "rmse": 0,
    }
    len_group = 0
    len_group_pesq = 0
    batch_size = 0

    stft = TacotronSTFT(
        filter_length=preprocess_config["stft"]["filter_length"],
        hop_length=preprocess_config["stft"]["hop_length"],
        win_length=preprocess_config["stft"]["win_length"],
        n_mel_channels=preprocess_config["mel"]["n_mel_channels"],
        sampling_rate=preprocess_config["sampling_rate"],
        mel_fmin=preprocess_config["mel"]["mel_fmin"],
        mel_fmax=preprocess_config["mel"]["mel_fmax"],
        device=device,
        center=False,
    )

    with torch.no_grad():
        for group in loader:
            for batch in group:
                (
                    speakers,
                    texts,
                    mel,
                    audio,
                    text_lens,
                    mel_lens,
                    durations,
                    token_ids,
                    attention_masks,
                ) = to_device(batch, device)
                audio = audio.unsqueeze(1)

                outputs = generator.forward_train(
                    x=texts,
                    speakers=speakers,
                    src_lens=text_lens,
                    specs=mel,
                    spec_lens=mel_lens,
                    durations=durations,
                    token_ids=token_ids,
                    attention_mask=attention_masks,
                )

                fake_audio = outputs["wave"]
                ids_slice = outputs["sample_q_ids_slice"]

                mel = slice_segments(mel, ids_slice, 32)
                audio = slice_segments(audio, ids_slice * 256, 8192)  # slice

                loss_dur = duration_loss(
                    log_duration_pred=outputs["log_duration_pred"],
                    duration_targets=durations,
                    mask=outputs["src_mask"],
                )

                loss_kl_backward = kl_loss(
                    outputs["z_p"],
                    outputs["log_std_q"],
                    outputs["mu_p"],
                    outputs["log_std_p"],
                    ~outputs["spec_mask"],
                )
                loss_kl_forward = kl_loss(
                    outputs["z_q"],
                    outputs["log_std_p"],
                    outputs["mu_q"],
                    outputs["log_std_q"],
                    ~outputs["spec_mask"],
                )

                sc_loss, mag_loss = stft_criterion(
                    fake_audio.squeeze(1), audio.squeeze(1)
                )
                stft_loss = (sc_loss + mag_loss) * stft_lamb
                estoi = calc_estoi(
                    audio_real=audio,
                    audio_fake=fake_audio,
                    sampling_rate=preprocess_config["sampling_rate"],
                )
                pesq = calc_pesq(
                    audio_real=audio,
                    audio_fake=fake_audio,
                    sampling_rate=preprocess_config["sampling_rate"],
                )
                rmse = calc_rmse(audio_real=audio, audio_fake=fake_audio, stft=stft)

                batch_size = mel.shape[0]
                if pesq != None:
                    loss_means["pesq"] += pesq * batch_size
                    len_group_pesq += batch_size

                len_group += batch_size
                loss_means["l_dur"] += loss_dur * batch_size
                loss_means["kl_loss_backward"] += loss_kl_backward * batch_size
                loss_means["kl_loss_forward"] += loss_kl_forward * batch_size
                loss_means["mel_loss_dec"] += stft_loss * batch_size
                loss_means["estoi"] += estoi * batch_size
                loss_means["rmse"] += rmse * batch_size

    if len_group_pesq > 0:
        loss_means["pesq"] /= len_group_pesq

    if len_group > 0:
        loss_means["estoi"] /= len_group
        loss_means["rmse"] /= len_group
        loss_means["mel_loss_dec"] /= len_group
        loss_means["kl_loss_backward"] /= len_group
        loss_means["kl_loss_forward"] /= len_group
        loss_means["l_dur"] /= len_group

    return loss_means


def train_vocoder(
    db_id: int,
    assets_path,
    training_run_name,
    train_config,
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

    stft_lamb = train_config["stft_lamb"]
    (
        generator,
        discriminator,
        steps,
        optim_g,
        optim_d,
        scheduler_g,
        scheduler_d,
    ) = get_nat_speech(
        assets_path=assets_path,
        checkpoint=checkpoint_path,
        preprocessed_path=preprocessed_path,
        train_config=train_config,
        reset=reset,
        device=device,
    )

    stft_criterion = MultiResolutionSTFTLoss(device, hp["mrd"]["resolutions"])
    train_loader, validation_loader, validset = get_data_loaders(
        assets_path=assets_path,
        preprocess_config=preprocess_config,
        fine_tuning=fine_tuning,
        segment_size=train_config["segment_size"],
        batch_size=train_config["batch_size"],
        num_workers=4,
        preprocessed_path=preprocessed_path,
    )

    train_loader = cycle_2d(train_loader)

    print("Generator has a total of " + str(get_param_num(generator)) + " parameters")
    print("MRD has a total of " + str(get_param_num(discriminator.MRD)) + " parameters")
    print("MPD has a total of " + str(get_param_num(discriminator.MPD)) + " parameters")
    print(
        "Total number of parameters during inference: "
        + str(get_param_num(generator) - get_param_num(generator.enc_q))
        + " parameters"
    )

    while True:

        start_b = time.time()

        loss_means = train_iter(
            generator=generator,
            discriminator=discriminator,
            loader=train_loader,
            device=device,
            stft_criterion=stft_criterion,
            grad_accum_steps=train_config["grad_accum_steps"],
            optim_g=optim_g,
            optim_d=optim_d,
            stft_lamb=stft_lamb,
        )

        if steps % train_config["stdout_interval"] == 0:
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

        if steps % train_config["synth_interval"] == 0 and steps != 0:
            synth_iter(
                eval_loader=validation_loader,
                generator=generator,
                step=steps,
                sampling_rate=preprocess_config["sampling_rate"],
                device=device,
                logger=logger,
            )

        if steps % train_config["validation_interval"] == 0 and steps != 0:
            print("\nEvaluating ...\n")
            loss_means = evaluate(
                generator=generator,
                loader=validation_loader,
                device=device,
                stft_criterion=stft_criterion,
                stft_lamb=stft_lamb,
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
                [steps / train_config["train_steps"], db_id],
            )

        if steps % train_config["checkpoint_interval"] == 0 and steps != 0:
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

        if steps >= train_config["train_steps"]:
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


if __name__ == "__main__":
    from voice_smith.utils.wandb_logger import WandBLogger
    from voice_smith.config.vocoder_pre_training_config import (
        vocoder_pre_training_config,
    )

    training_runs_path = Path(".") / ".." / ".." / "training_runs"

    logger = WandBLogger("Delight VITS NatSpeech Combo v2")
    train_vocoder(
        db_id=None,
        assets_path=str(Path(".") / ".." / ".." / "assets"),
        training_run_name="univnet_pretraining",
        train_config=vocoder_pre_training_config,
        logger=logger,
        device=torch.device("cuda"),
        reset=False,
        training_runs_path=training_runs_path,
        checkpoint_path=Path(".")
        / ".."
        / ".."
        / "training_runs"
        / "univnet_pretraining"
        / "ckpt"
        / "vocoder"
        / "vocoder_56750.pt",
        fine_tuning=False,
        overwrite_saves=False,
        stop_after_hour=None,
    )

