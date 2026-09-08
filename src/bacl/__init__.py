"""Boundary-Aware Contrastive Learning for text-guided segmentation."""

from .losses import BoundaryAwareContrastiveLoss, bce_dice_loss
from .metrics import binary_segmentation_metrics
from .model import BACLModel

__all__ = [
    "BACLModel",
    "BoundaryAwareContrastiveLoss",
    "bce_dice_loss",
    "binary_segmentation_metrics",
]

__version__ = "1.0.0"
