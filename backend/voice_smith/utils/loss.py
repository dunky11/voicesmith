import torch
import torch.nn as nn
from piq import SSIMLoss
from voice_smith.utils.tools import sample_wise_min_max
from typing import Dict, Tuple, Any, List


class FastSpeech2LossGen(nn.Module):
    def __init__(self, fine_tuning: bool, device: torch.device):
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.fine_tuning = fine_tuning
        self.device = device

    def forward(
        self,
        src_masks: torch.Tensor,
        mel_masks: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_predictions: torch.Tensor,
        log_duration_predictions: torch.Tensor,
        u_prosody_ref: torch.Tensor,
        u_prosody_pred: torch.Tensor,
        p_prosody_ref: torch.Tensor,
        p_prosody_pred: torch.Tensor,
        durations: torch.Tensor,
        pitch_predictions: torch.Tensor,
        p_targets: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        log_duration_targets = torch.log(durations.float() + 1)

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False
        p_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(~src_masks)
        log_duration_targets = log_duration_targets.masked_select(~src_masks)

        mel_masks_expanded = mel_masks.unsqueeze(1)

        mel_predictions_normalized = sample_wise_min_max(mel_predictions)
        mel_targets_normalized = sample_wise_min_max(mel_targets)

        ssim_loss = self.ssim_loss(
            mel_predictions_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1)
        )

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            print(
                f"Overflow in ssim loss detected, which was {ssim_loss.item()}, setting to 1.0"
            )
            ssim_loss = torch.FloatTensor([1.0]).to(self.device)

        masked_mel_predictions = mel_predictions.masked_select(~mel_masks_expanded)

        mel_targets = mel_targets.masked_select(~mel_masks_expanded)

        mel_loss = self.mae_loss(masked_mel_predictions, mel_targets)

        p_prosody_ref = p_prosody_ref.permute((0, 2, 1))
        p_prosody_pred = p_prosody_pred.permute((0, 2, 1))

        p_prosody_ref = p_prosody_ref.masked_fill(src_masks.unsqueeze(1), 0.0)
        p_prosody_pred = p_prosody_pred.masked_fill(src_masks.unsqueeze(1), 0.0)

        p_prosody_ref = p_prosody_ref.detach()

        p_prosody_loss = 0.5 * self.mae_loss(
            p_prosody_ref.masked_select(~src_masks.unsqueeze(1)),
            p_prosody_pred.masked_select(~src_masks.unsqueeze(1)),
        )

        u_prosody_ref = u_prosody_ref.detach()
        u_prosody_loss = 0.5 * self.mae_loss(u_prosody_ref, u_prosody_pred)

        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        pitch_predictions = pitch_predictions.masked_select(~src_masks)
        p_targets = p_targets.masked_select(~src_masks)

        pitch_loss = self.mse_loss(pitch_predictions, p_targets)

        total_loss = (
            mel_loss
            + duration_loss
            + u_prosody_loss
            + p_prosody_loss
            + ssim_loss
            + pitch_loss
        )

        return (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
        )


def feature_loss(
    fmap_r: List[torch.Tensor], fmap_g: List[torch.Tensor]
) -> torch.Tensor:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, List[float], List[float]]:
    loss = torch.FloatTensor([0]).to(device)
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor], device: torch.device
) -> Tuple[torch.Tensor, List[float]]:
    loss = torch.FloatTensor([0]).to(device)
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses
