"""Convert medical samples to SAM3 Datapoint format."""

from __future__ import annotations

from typing import List

import torch
from PIL import Image as PILImage

from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
    Object,
)


def medical_sample_to_datapoint(
    image: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    image_id: int = 0,
) -> Datapoint:
    """Build a SAM3 Datapoint from a medical training sample."""
    if image.dim() == 3:
        pil = PILImage.fromarray(
            (image.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        )
    else:
        pil = image

    h, w = mask.shape[-2], mask.shape[-1]
    sam_image = SAMImage(data=pil, objects=[])

    obj = Object(
        bbox=torch.tensor([0, 0, w, h], dtype=torch.float32),
        area=float(mask.sum()),
        object_id=0,
        frame_index=0,
    )
    sam_image.objects.append(obj)

    query = FindQueryLoaded(
        query_text=prompt,
        image_id=0,
        object_ids_output=[0],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=image_id,
            original_image_id=image_id,
            original_category_id=0,
            original_size=[h, w],
            object_id=0,
            frame_index=0,
        ),
    )

    dp = Datapoint(find_queries=[query], images=[sam_image])
    dp.images[0].objects = [obj]
    return dp


def batch_to_datapoints(batch: dict, start_id: int = 0) -> List[Datapoint]:
    datapoints = []
    for i in range(batch["image"].size(0)):
        dp = medical_sample_to_datapoint(
            batch["image"][i],
            batch["mask"][i],
            batch["prompt"][i] if isinstance(batch["prompt"], list) else batch["prompt"],
            image_id=start_id + i,
        )
        datapoints.append(dp)
    return datapoints
