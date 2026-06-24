"""Loss functions for DA-SAM3 training."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.sigmoid() if pred.dtype != torch.float16 else pred.sigmoid()
        pred = pred.reshape(pred.size(0), -1)
        target = target.reshape(target.size(0), -1).float()
        inter = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DASAM3Loss(nn.Module):
    """
    L = L_seg + λ1 * L_balance + λ2 * L_sparse  (Eq. 7)
    L_seg = L_dice + L_focal
    """

    def __init__(
        self,
        lambda_balance: float = 0.01,
        lambda_sparse: float = 0.001,
    ):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.lambda_balance = lambda_balance
        self.lambda_sparse = lambda_sparse

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        moe_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        seg = self.dice(pred_masks, target_masks) + self.focal(pred_masks, target_masks)
        balance = moe_aux.get("balance", torch.tensor(0.0, device=pred_masks.device))
        sparse = moe_aux.get("sparse", torch.tensor(0.0, device=pred_masks.device))
        total = seg + self.lambda_balance * balance + self.lambda_sparse * sparse
        return {
            "total": total,
            "seg": seg,
            "dice": self.dice(pred_masks, target_masks),
            "focal": self.focal(pred_masks, target_masks),
            "balance": balance,
            "sparse": sparse,
        }
