from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


def swap_laterality(text: str) -> str:
    """Swap left/right terms after horizontal image flipping."""

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        replacement = "right" if word.lower() == "left" else "left"
        if word.isupper():
            return replacement.upper()
        if word[:1].isupper():
            return replacement.capitalize()
        return replacement

    swapped = re.sub(r"\b(?:left|right)\b", replace, str(text), flags=re.IGNORECASE)
    return swapped.translate(str.maketrans({"左": "右", "右": "左"}))


def _read_metadata(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported metadata format: {path.suffix}")


class TextSegmentationDataset(Dataset):
    """Paired 2-D image, binary mask, and text-description dataset."""

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        metadata_file: str | Path,
        *,
        image_column: str = "image",
        text_column: str = "text",
        mask_column: str | None = None,
        image_size: int = 224,
        augment: bool = False,
        rotation_limit: int = 20,
        flip_probability: float = 0.5,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.metadata_file = Path(metadata_file)
        self.data = _read_metadata(self.metadata_file)
        self.image_column = image_column
        self.text_column = text_column
        self.mask_column = mask_column
        self.image_size = image_size
        self.augment = augment
        self.rotation_limit = rotation_limit
        self.flip_probability = flip_probability

        if image_column not in self.data.columns or text_column not in self.data.columns:
            if len(self.data.columns) < 2:
                raise ValueError("Metadata needs at least image and text columns")
            self.image_column = str(self.data.columns[0])
            self.text_column = str(self.data.columns[1])
        if mask_column is not None and mask_column not in self.data.columns:
            raise ValueError(f"Mask column '{mask_column}' is missing")

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, image: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, bool]:
        flipped = self.augment and np.random.random() < self.flip_probability
        transforms: list[Any] = []
        if self.augment:
            transforms.append(
                A.Rotate(
                    limit=self.rotation_limit,
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                )
            )
        if flipped:
            transforms.append(A.HorizontalFlip(p=1.0))
        transforms.extend(
            [
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        transformed = A.Compose(transforms)(image=image, mask=mask)
        return transformed["image"], transformed["mask"].float(), flipped

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        image_name = str(row[self.image_column])
        mask_name = str(row[self.mask_column]) if self.mask_column else image_name
        text = "" if pd.isna(row[self.text_column]) else str(row[self.text_column])
        image_path = self.images_dir / image_name
        mask_path = self.masks_dir / mask_name
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not mask_path.is_file():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.float32)
        image_tensor, mask_tensor, flipped = self._transform(image, mask)
        if flipped:
            text = swap_laterality(text)
        return {
            "image": image_tensor,
            "mask": mask_tensor.unsqueeze(0),
            "text": text,
            "id": image_name,
        }
