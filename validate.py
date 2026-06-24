#!/usr/bin/env python3
"""
Validate DA-SAM3 on medical benchmarks (DSC / HD).

Usage:
  python validate.py --config configs/da_sam3_default.yaml --checkpoint outputs/da_sam3/best_warmup.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from da_sam3.data.medical_dataset import build_dataloader
from da_sam3.data.sam3_datapoint import batch_to_datapoints
from da_sam3.models.da_moe import DAMoEConfig
from da_sam3.models.da_sam3_model import build_da_sam3_model
from da_sam3.utils.checkpoint import load_checkpoint
from da_sam3.utils.metrics import evaluate_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DA-SAM3")
    parser.add_argument("--config", type=str, default="configs/da_sam3_default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sam3-root", type=str, default=None)
    return parser.parse_args()


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
        sam3_root=sam3_root,
        moe_config=moe_cfg,
        device=device,
        load_from_hf=cfg.get("load_from_hf", True),
    )
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    model.to(device)

    train_cfg = cfg["training"]
    loader = build_dataloader(
        train_cfg["data_root"],
        args.split,
        train_cfg["dataset"],
        batch_size=1,
        num_workers=train_cfg["num_workers"],
        image_size=train_cfg.get("image_size"),
    )

    from sam3.train.data.collator import collate_fn_api

    dsc_list, hd_list = [], []
    for batch in tqdm(loader, desc="validate"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        batch = {**batch, "image": images, "mask": masks}

        try:
            datapoints = batch_to_datapoints(batch)
            batched = collate_fn_api(datapoints).to(device)
            with torch.no_grad():
                output = model(batched)
            pred = output.pred_masks if hasattr(output, "pred_masks") else output["pred_masks"]
            if pred.shape[-2:] != masks.shape[-2:]:
                pred = torch.nn.functional.interpolate(
                    pred, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
            metrics = evaluate_batch(pred, masks)
            dsc_list.append(metrics["dsc"])
            hd_list.append(metrics["hd"])
        except Exception as exc:
            print(f"Skip sample due to: {exc}")

    if dsc_list:
        print(f"DSC: {sum(dsc_list)/len(dsc_list):.4f} | HD: {sum(hd_list)/len(hd_list):.4f} (n={len(dsc_list)})")
    else:
        print("No samples evaluated.")


if __name__ == "__main__":
    main()
