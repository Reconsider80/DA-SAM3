#!/usr/bin/env python3
"""
DA-SAM3 inference with text prompts.

Usage:
  python infer.py --config configs/da_sam3_default.yaml \\
      --checkpoint outputs/da_sam3/best_warmup.pt \\
      --image path/to/image.png --prompt "left ventricle" --output result.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from da_sam3.models.da_moe import DAMoEConfig
from da_sam3.models.da_sam3_model import build_da_sam3_model
from da_sam3.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="DA-SAM3 inference")
    parser.add_argument("--config", type=str, default="configs/da_sam3_default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--sam3-root", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def build_datapoint(image_path: str, prompt: str):
    from da_sam3.data.sam3_datapoint import medical_sample_to_datapoint
    from torchvision.transforms.functional import normalize, resize, to_tensor

    pil = Image.open(image_path).convert("RGB")
    img = resize(pil, [256, 256])
    tensor = normalize(to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    mask = torch.zeros(1, 256, 256)
    return medical_sample_to_datapoint(tensor, mask, prompt)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["hardware"]["device"]
    if not torch.cuda.is_available():
        device = "cpu"

    sam3_root = args.sam3_root or cfg.get("sam3_root")
    if sam3_root:
        os.environ["SAM3_ROOT"] = str(Path(sam3_root).resolve())

    moe_cfg = DAMoEConfig(**{k: cfg["model"][k] for k in (
        "d_model", "dim_feedforward", "num_experts", "top_k", "rank", "dropout", "activation"
    )})

    model = build_da_sam3_model(
        checkpoint_path=cfg.get("checkpoint_path"),
        sam3_root=sam3_root,
        moe_config=moe_cfg,
        device=device,
        load_from_hf=cfg.get("load_from_hf", True),
    )
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    model.to(device)

    from sam3.train.data.collator import collate_fn_api

    dp = build_datapoint(args.image, args.prompt)
    batched = collate_fn_api([dp]).to(device)

    with torch.no_grad():
        output = model(batched)

    if hasattr(output, "pred_masks"):
        pred = output.pred_masks.sigmoid().cpu().numpy()
    elif isinstance(output, dict) and "pred_masks" in output:
        pred = torch.sigmoid(output["pred_masks"]).cpu().numpy()
    else:
        pred = np.zeros((1, 1, 256, 256), dtype=np.float32)

    mask = (pred[0, 0] > args.threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(Image.open(args.image))
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f'Prompt: "{args.prompt}"')
    axes[1].axis("off")
    overlay = np.array(Image.open(args.image).resize((mask.shape[1], mask.shape[0])))
    overlay[mask > 0] = [255, 0, 0]
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
