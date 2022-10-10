import torch
import torch.nn as nn
import torch.nn.functional as F
from piq import SSIMLoss
from voice_smith.utils.tools import sample_wise_min_max
from typing import Dict, Tuple, Any, List


class FastSpeech2LossGen(nn.Module):
    def __init__(self, fine_tuning: bool, device: torch.device):
        super().__init__()

        self.mse_loss = nn.L1Loss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
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
        attn_logprob: torch.Tensor,
        attn_soft: torch.Tensor,
        attn_hard: torch.Tensor,
        step: int,
        src_lens: torch.Tensor,
        mel_lens: torch.Tensor,
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

        ctc_loss = self.sum_loss(
            attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens
        )

        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
        # bin_loss = torch.FloatTensor([0.0]).to(self.device)

        total_loss = (
            mel_loss
            + duration_loss
            + u_prosody_loss
            + p_prosody_loss
            + ssim_loss
            + pitch_loss
            + ctc_loss
            + bin_loss
        )

        return (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(
            input=attn_logprob, pad=(1, 0), value=self.blank_logprob
        )

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[
                : query_lens[bid], :, : key_lens[bid] + 1
            ]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(
            torch.clamp(soft_attention[hard_attention == 1], min=1e-12)
        ).sum()
        return -log_sum / hard_attention.sum()


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
