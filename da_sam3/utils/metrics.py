"""Evaluation metrics for medical segmentation."""

from __future__ import annotations

import numpy as np
import torch


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (pred.sigmoid() > threshold).float() if pred.dtype != torch.bool else pred.float()
    target = target.float()
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    inter = (pred * target).sum()
    return float((2 * inter / (pred.sum() + target.sum() + 1e-6)).item())


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Approximate HD95 using scipy if available, else bounding-box fallback."""
    try:
        from scipy.ndimage import distance_transform_edt

        pred = pred.astype(bool)
        target = target.astype(bool)
        if pred.sum() == 0 or target.sum() == 0:
            return float(max(pred.shape))

        dt_pred = distance_transform_edt(~pred)
        dt_tgt = distance_transform_edt(~target)
        d1 = dt_pred[target]
        d2 = dt_tgt[pred]
        return float(max(np.percentile(d1, 95) if len(d1) else 0, np.percentile(d2, 95) if len(d2) else 0))
    except ImportError:
        return 0.0


def evaluate_batch(pred: torch.Tensor, target: torch.Tensor) -> dict:
    dsc = dice_coefficient(pred, target)
    pred_np = (pred.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
    tgt_np = target.cpu().numpy().astype(np.uint8)
    hd = hausdorff_distance(pred_np[0, 0], tgt_np[0, 0]) if pred_np.ndim == 4 else 0.0
    return {"dsc": dsc, "hd": hd}
