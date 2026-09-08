from __future__ import annotations

import math

import numpy as np
import torch


def binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = torch.sigmoid(logits) >= threshold
    references = targets >= 0.5
    dimensions = tuple(range(1, references.ndim))
    intersection = (predictions & references).sum(dim=dimensions).float()
    prediction_size = predictions.sum(dim=dimensions).float()
    reference_size = references.sum(dim=dimensions).float()
    union = (predictions | references).sum(dim=dimensions).float()
    dice = (2.0 * intersection + smooth) / (prediction_size + reference_size + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def normal_confidence_interval(values: list[float], confidence_z: float = 1.96) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric list")
    mean = float(array.mean())
    if array.size == 1:
        return {"mean": mean, "lower": mean, "upper": mean, "n": 1}
    standard_error = float(array.std(ddof=1) / math.sqrt(array.size))
    return {
        "mean": mean,
        "lower": mean - confidence_z * standard_error,
        "upper": mean + confidence_z * standard_error,
        "n": int(array.size),
    }
