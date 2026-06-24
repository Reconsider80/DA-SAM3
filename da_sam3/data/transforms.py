"""Data augmentation for medical segmentation."""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class MedicalTrainTransform:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        scale = random.uniform(0.85, 1.15)
        new_size = int(self.image_size * scale)
        image = TF.resize(image, [new_size, new_size])
        mask = TF.resize(mask, [new_size, new_size], interpolation=TF.InterpolationMode.NEAREST)
        image = TF.center_crop(image, [self.image_size, self.image_size])
        mask = TF.center_crop(mask, [self.image_size, self.image_size])

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        return image, (mask > 0.5).float()


class MedicalValTransform:
    def __init__(self, image_size: int = 256):
        self.image_size = image_size

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.NEAREST)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = TF.to_tensor(mask)
        return image, (mask > 0.5).float()


def build_train_transforms(image_size: int = 256) -> MedicalTrainTransform:
    return MedicalTrainTransform(image_size)


def build_val_transforms(image_size: int = 256) -> MedicalValTransform:
    return MedicalValTransform(image_size)
