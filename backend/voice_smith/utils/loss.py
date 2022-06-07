import torch
import torch.nn as nn
import torch.nn.functional as F
from voice_smith.utils.tools import sample_wise_min_max
from typing import Dict, Tuple, Any, List
import torch.nn.functional as F


def duration_loss(
    log_y_pred: torch.Tensor, y_true: torch.Tensor, src_masks: torch.Tensor,
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

    p_prosody_loss = F.l1_loss(
        y_ref.masked_select(~src_masks.unsqueeze(1)),
        y_pred.masked_select(~src_masks.unsqueeze(1)),
    )
    return p_prosody_loss

