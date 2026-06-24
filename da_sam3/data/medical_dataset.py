"""Medical segmentation dataset utilities for DA-SAM3."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from da_sam3.data.transforms import build_train_transforms, build_val_transforms


DATASET_DEFAULTS: Dict[str, Dict] = {
    "synapse": {"image_size": 224, "split": {"train": 18, "test": 12}},
    "mmwhs": {"image_size": 256, "split": {"train": 16, "test": 4}},
    "btcv": {"image_size": 256, "split": {"train": 24, "test": 6}},
    "acdc": {"image_size": 256, "split": None},
}


class MedicalSegDataset(Dataset):
    """
    Generic 2D medical segmentation dataset.

    Expected layout (per dataset root):
      images/{case_id}.png
      masks/{case_id}.png
      prompts.json  (optional) mapping case_id -> text prompt

    Or COCO-style via annotation file.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        dataset_name: str = "synapse",
        transform: Optional[Callable] = None,
        annotation_file: Optional[str] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.dataset_name = dataset_name
        self.transform = transform
        self.samples: List[Dict] = []

        if annotation_file and Path(annotation_file).exists():
            self._load_coco(annotation_file)
        else:
            self._load_folder_split()

        prompts_path = self.root / "prompts.json"
        self.prompts = {}
        if prompts_path.exists():
            self.prompts = json.loads(prompts_path.read_text())

    def _load_folder_split(self) -> None:
        list_file = self.root / f"{self.split}.txt"
        if list_file.exists():
            ids = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
        else:
            img_dir = self.root / "images"
            ids = sorted(p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            defaults = DATASET_DEFAULTS.get(self.dataset_name, {})
            split_cfg = defaults.get("split")
            if split_cfg and self.split in split_cfg:
                n = split_cfg[self.split]
                ids = ids[:n] if self.split == "train" else ids[-n:]

        for case_id in ids:
            img_path = self._find_file("images", case_id)
            mask_path = self._find_file("masks", case_id)
            if img_path and mask_path:
                self.samples.append({"id": case_id, "image": img_path, "mask": mask_path})

    def _find_file(self, subdir: str, case_id: str) -> Optional[Path]:
        base = self.root / subdir
        for ext in (".png", ".jpg", ".jpeg", ".npy"):
            p = base / f"{case_id}{ext}"
            if p.exists():
                return p
        return None

    def _load_coco(self, annotation_file: str) -> None:
        with open(annotation_file) as f:
            coco = json.load(f)
        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        for ann in coco["annotations"]:
            if ann.get("split", self.split) != self.split:
                continue
            img_id = ann["image_id"]
            self.samples.append(
                {
                    "id": str(img_id),
                    "image": self.root / "images" / id_to_file[img_id],
                    "mask": None,
                    "rle": ann.get("segmentation"),
                    "prompt": ann.get("query_text") or ann.get("category_name"),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        prompt = sample.get("prompt") or self.prompts.get(sample["id"], "organ")

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = (mask > 0).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "prompt": prompt,
            "id": sample["id"],
        }


def build_dataloader(
    root: str,
    split: str,
    dataset_name: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    defaults = DATASET_DEFAULTS.get(dataset_name, {"image_size": 256})
    size = image_size or defaults["image_size"]
    transform = build_train_transforms(size) if split == "train" else build_val_transforms(size)
    dataset = MedicalSegDataset(root, split, dataset_name, transform)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
    )
