from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import resolve_path
from .data import TextSegmentationDataset
from .losses import BoundaryAwareContrastiveLoss
from .model import BACLModel


def build_model(config: dict[str, Any]) -> BACLModel:
    model_config = config.get("model", {})
    return BACLModel(
        clip_model=model_config.get("clip_model", "openai/clip-vit-base-patch32"),
        local_files_only=bool(model_config.get("local_files_only", False)),
        freeze_clip=bool(model_config.get("freeze_clip", False)),
        pretrained_backbone=bool(model_config.get("pretrained_backbone", True)),
    )


def build_contrastive_loss(config: dict[str, Any]) -> BoundaryAwareContrastiveLoss:
    loss_config = config.get("contrastive", {})
    return BoundaryAwareContrastiveLoss(
        temperature=float(loss_config.get("temperature", 0.07)),
        samples_per_level=int(loss_config.get("samples_per_level", 10)),
    )


def build_dataset(config: dict[str, Any], split: str, *, augment: bool) -> TextSegmentationDataset:
    data_config = config["data"]
    split_config = data_config[split]
    return TextSegmentationDataset(
        resolve_path(split_config["images_dir"], config),
        resolve_path(split_config["masks_dir"], config),
        resolve_path(split_config["metadata"], config),
        image_column=data_config.get("image_column", "image"),
        text_column=data_config.get("text_column", "text"),
        mask_column=data_config.get("mask_column"),
        image_size=int(data_config.get("image_size", 224)),
        augment=augment,
        rotation_limit=int(data_config.get("rotation_limit", 20)),
        flip_probability=float(data_config.get("flip_probability", 0.5)),
    )


def contrastive_weight(epoch: int, config: dict[str, Any]) -> float:
    loss_config = config.get("contrastive", {})
    initial = float(loss_config.get("initial_weight", 0.05))
    maximum = float(loss_config.get("max_weight", 0.5))
    warmup_epochs = int(loss_config.get("warmup_epochs", 20))
    if warmup_epochs <= 0:
        return maximum
    progress = min(1.0, max(0.0, epoch / warmup_epochs))
    return initial + (maximum - initial) * progress


def load_checkpoint(path: str | Path, model: BACLModel, contrastive_loss=None, device="cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError("Expected a release-format checkpoint containing a 'model' state dict")
    model.load_state_dict(checkpoint["model"])
    if contrastive_loss is not None and checkpoint.get("contrastive_loss"):
        contrastive_loss.load_state_dict(checkpoint["contrastive_loss"])
    return checkpoint
