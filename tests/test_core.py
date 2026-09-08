import torch

from bacl.data import swap_laterality
from bacl.losses import BoundaryAwareContrastiveLoss, bce_dice_loss
from bacl.metrics import binary_segmentation_metrics


def test_swap_laterality_is_atomic():
    assert swap_laterality("Left and RIGHT; 左右") == "Right and LEFT; 右左"


def test_metrics_score_per_case_and_handle_empty_masks():
    logits = torch.tensor([[[[-20.0, 20.0]]], [[[-20.0, -20.0]]]])
    targets = torch.tensor([[[[0.0, 1.0]]], [[[0.0, 0.0]]]])
    dice, iou = binary_segmentation_metrics(logits, targets)
    assert torch.allclose(dice, torch.ones_like(dice))
    assert torch.allclose(iou, torch.ones_like(iou))


def test_boundary_loss_is_finite_and_backpropagates():
    image_features = [
        torch.randn(2, 64, 8, 8, requires_grad=True),
        torch.randn(2, 128, 4, 4, requires_grad=True),
        torch.randn(2, 256, 2, 2, requires_grad=True),
    ]
    mask_features = [
        torch.randn(2, 64, 8, 8, requires_grad=True),
        torch.randn(2, 128, 4, 4, requires_grad=True),
        torch.randn(2, 256, 2, 2, requires_grad=True),
    ]
    masks = torch.zeros(2, 1, 16, 16)
    masks[:, :, :8] = 1
    criterion = BoundaryAwareContrastiveLoss(samples_per_level=4)
    loss = criterion(image_features, mask_features, masks)
    assert torch.isfinite(loss)
    loss.backward()
    assert criterion.projections[0].mlp[0].weight.grad is not None


def test_segmentation_loss_is_finite_for_empty_mask():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = bce_dice_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
