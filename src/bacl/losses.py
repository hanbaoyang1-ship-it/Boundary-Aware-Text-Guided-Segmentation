from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    dimensions = tuple(range(1, targets.ndim))
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    return (1.0 - (2.0 * intersection + smooth) / (denominator + smooth)).mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + soft_dice_loss(logits, targets)


class SampleProjectionHead(nn.Module):
    """GAP plus two-layer MLP applied to sampled 1x1 feature patches."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(samples[:, :, None, None]).flatten(1)
        return self.mlp(pooled)


class BoundaryAwareContrastiveLoss(nn.Module):
    """Multi-scale foreground/mask-positive/background-negative objective."""

    def __init__(
        self,
        feature_dims: Sequence[int] = (64, 128, 256),
        temperature: float = 0.07,
        samples_per_level: int = 10,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if samples_per_level < 1:
            raise ValueError("samples_per_level must be at least 1")
        self.temperature = temperature
        self.samples_per_level = samples_per_level
        self.projections = nn.ModuleList(SampleProjectionHead(dim) for dim in feature_dims)

    def _sample_loss(
        self,
        image_features: torch.Tensor,
        mask_features: torch.Tensor,
        binary_mask: torch.Tensor,
        projection: SampleProjectionHead,
    ) -> torch.Tensor:
        foreground = torch.nonzero(binary_mask > 0.5, as_tuple=False).flatten()
        background = torch.nonzero(binary_mask <= 0.5, as_tuple=False).flatten()
        count = min(self.samples_per_level, foreground.numel(), background.numel())
        if count == 0:
            return image_features.sum() * 0.0

        fg_order = torch.randperm(foreground.numel(), device=foreground.device)[:count]
        bg_order = torch.randperm(background.numel(), device=background.device)[:count]
        fg_indices = foreground[fg_order]
        bg_indices = background[bg_order]

        anchors = projection(image_features[fg_indices])
        positives = projection(mask_features[fg_indices])
        negatives = projection(image_features[bg_indices])
        anchors = F.normalize(anchors, dim=1)
        positives = F.normalize(positives, dim=1)
        negatives = F.normalize(negatives, dim=1)
        positive_similarity = (anchors * positives).sum(dim=1)
        negative_similarity = torch.mm(anchors, negatives.transpose(0, 1)).mean(dim=1)
        logits = torch.stack([positive_similarity, negative_similarity], dim=1) / self.temperature
        labels = torch.zeros(count, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def forward(
        self,
        image_features: Sequence[torch.Tensor],
        mask_features: Sequence[torch.Tensor],
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if len(image_features) != 3 or len(mask_features) != 3:
            raise ValueError("Expected three encoder feature levels")
        losses: list[torch.Tensor] = []
        for image_level, mask_level, projection in zip(
            image_features, mask_features, self.projections, strict=True
        ):
            resized_masks = F.interpolate(masks, size=image_level.shape[-2:], mode="nearest")
            image_flat = image_level.flatten(2).transpose(1, 2)
            mask_flat = mask_level.flatten(2).transpose(1, 2)
            for batch_index in range(image_level.shape[0]):
                losses.append(
                    self._sample_loss(
                        image_flat[batch_index],
                        mask_flat[batch_index],
                        resized_masks[batch_index, 0].flatten(),
                        projection,
                    )
                )
        return torch.stack(losses).mean()
