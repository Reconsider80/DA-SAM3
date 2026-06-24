"""
DA-SAM3 model builder: SAM3 backbone + Dual-Adaptive MoE fusion layers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from da_sam3.integration.sam3_patcher import (
    MoEContextStore,
    collect_moe_aux_losses,
    configure_trainable_parameters,
    inject_da_moe_into_sam3,
)
from da_sam3.models.da_moe import DAMoEConfig


def _ensure_sam3_on_path(sam3_root: Optional[str] = None) -> Path:
    root = Path(
        sam3_root
        or os.environ.get("SAM3_ROOT", "")
        or os.environ.get("MEDSAM3_ROOT", "")
    )
    if not root:
        candidate = Path(__file__).resolve().parents[2].parent / "Medical-SAM3" / "Medical-SAM3-main"
        if candidate.exists():
            root = candidate
    if not root or not root.exists():
        raise FileNotFoundError(
            "SAM3 source not found. Set SAM3_ROOT to Medical-SAM3-main "
            "(e.g. ../Medical-SAM3/Medical-SAM3-main)."
        )
    root = root.resolve()
    for p in (root, root / "sam3"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root


class DASAM3Model(nn.Module):
    """Wrapper around Sam3Image with DA-MoE fusion and context injection."""

    def __init__(self, sam3_model: nn.Module, moe_indices: list[int]):
        super().__init__()
        self.sam3 = sam3_model
        self.moe_indices = moe_indices

    def forward(self, batched_input, **kwargs):
        self._set_moe_context(batched_input)
        try:
            return self.sam3(batched_input, **kwargs)
        finally:
            MoEContextStore.clear()

    def _set_moe_context(self, batched_input) -> None:
        """Extract concept/visual signals from SAM3 backbone for DER routing."""
        with torch.no_grad():
            images = batched_input.img_batch
            text_batch = batched_input.text_batch
            backbone_out = self.sam3.backbone(images, text_batch)
            visual_memory = backbone_out["vision_features"][-1]
            text_memory = backbone_out["language_features"]
            concept_token = text_memory[:, :1, :]
            concept_emb = text_memory.mean(dim=1)

        if visual_memory.dim() == 4:
            b, c, h, w = visual_memory.shape
            visual_memory = visual_memory.flatten(2).transpose(1, 2)
        MoEContextStore.set(concept_token, visual_memory, concept_emb)

    def moe_aux_losses(self) -> Dict[str, torch.Tensor]:
        return collect_moe_aux_losses(self.sam3)

    def set_training_stage(self, stage: str) -> None:
        configure_trainable_parameters(self.sam3, stage=stage)


def build_da_sam3_model(
    checkpoint_path: Optional[str] = None,
    sam3_root: Optional[str] = None,
    moe_config: Optional[DAMoEConfig] = None,
    device: str = "cuda",
    load_from_hf: bool = True,
) -> DASAM3Model:
    root = _ensure_sam3_on_path(sam3_root)
    from sam3.model_builder import build_sam3_image_model

    bpe_path = root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    sam3 = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf and checkpoint_path is None,
        device=device,
    )

    config = moe_config or DAMoEConfig()
    sam3, moe_indices, _ = inject_da_moe_into_sam3(sam3, config)
    configure_trainable_parameters(sam3, stage="warmup")

    return DASAM3Model(sam3, moe_indices)


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }
