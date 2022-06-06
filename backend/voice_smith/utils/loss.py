import torch
import torch.nn as nn
import torch.nn.functional as F
from voice_smith.utils.tools import sample_wise_min_max
from typing import Dict, Tuple, Any, List
import torch.nn.functional as F

_SIM_LOSS_FUNC = None


def duration_loss(
    log_y_pred: torch.Tensor,
    y_true: torch.Tensor,
    src_masks: torch.Tensor,
):
    y_true.requires_grad = False
    log_y_true = torch.log(y_true.float() + 1)
    log_y_pred = log_y_pred.masked_select(~src_masks)
    log_y_true = log_y_true.masked_select(~src_masks)
    loss = F.mse_loss(log_y_pred, log_y_true)
    return loss


def pitch_loss(y_pred: torch.Tensor, y_true: torch.Tensor, src_masks):
    y_true.requires_grad = False
    y_pred = y_pred.masked_select(~src_masks)
    y_true = y_true.masked_select(~src_masks)
    loss = F.mse_loss(y_pred, y_true)
    return loss


def prosody_loss(y_pred: torch.Tensor, y_ref: torch.Tensor, src_masks: torch.Tensor):
    y_ref = y_ref.permute((0, 2, 1))
    y_pred = y_pred.permute((0, 2, 1))

    y_ref = y_ref.detach()

    p_prosody_loss = F.mae_loss(
        y_ref.masked_select(~src_masks.unsqueeze(1)),
        y_pred.masked_select(~src_masks.unsqueeze(1)),
    )
    return p_prosody_loss


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
        g_loss = torch.mean(dg**2)
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
