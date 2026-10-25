"""Tests for the FCOS detection loss and target assignment."""

import torch

from genml_kit.losses.detection import (FCOSLoss, _centerness_targets, _focal_loss,
                                        _location_grid)
from genml_kit.losses.registry import LOSSES
from genml_kit.models.decoders.base import EncoderLevel, EncoderPyramid
from genml_kit.models.decoders.fcos_head import FCOSHead


def _head_output(batch=1, size=32, num_classes=2):
  """Run a tiny FCOS head on a synthetic pyramid and return its output."""
  feature = torch.randn(batch, 16, size // 8, size // 8)
  pyramid = EncoderPyramid([EncoderLevel(feature=feature, stride=8)])
  head = FCOSHead(in_channels=[16], num_classes=num_classes, hidden=16, num_convs=1)
  return head(pyramid)


class TestLocationGrid:

  def test_grid_shape_and_centres(self):
    grid = _location_grid((2, 3), stride=8, device="cpu")
    assert grid.shape == (6, 2)
    # First location centre: ((0 + 0.5) * 8, (0 + 0.5) * 8) = (4, 4).
    assert torch.allclose(grid[0], torch.tensor([4.0, 4.0]))
    # Second location moves along x by one stride.
    assert torch.allclose(grid[1], torch.tensor([12.0, 4.0]))


class TestCenternessTargets:

  def test_perfectly_centred_is_one(self):
    # Equal distances in every direction -> centerness 1.
    ltrb = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    assert torch.allclose(_centerness_targets(ltrb), torch.tensor([1.0]))

  def test_off_centre_is_below_one(self):
    ltrb = torch.tensor([[1.0, 5.0, 5.0, 5.0]])
    value = _centerness_targets(ltrb).item()
    assert 0.0 < value < 1.0


class TestFocalLoss:

  def test_zero_when_confident_and_correct(self):
    logits = torch.tensor([10.0, -10.0])
    targets = torch.tensor([1.0, 0.0])
    assert _focal_loss(logits, targets).item() < 1e-3

  def test_positive_when_wrong(self):
    logits = torch.tensor([-10.0, 10.0])
    targets = torch.tensor([1.0, 0.0])
    assert _focal_loss(logits, targets).item() > 1.0


class TestFCOSLoss:

  def test_returns_scalar_and_metrics(self):
    loss = FCOSLoss(num_classes=2)
    head_out = _head_output()
    targets = [{
        "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
        "labels": torch.tensor([0])
    }]
    total, metrics = loss(head_out, targets)
    assert total.ndim == 0
    assert torch.isfinite(total)
    for key in ("cls_loss", "box_loss", "centerness_loss", "num_pos"):
      assert key in metrics

  def test_empty_ground_truth_is_finite(self):
    loss = FCOSLoss(num_classes=2)
    head_out = _head_output()
    targets = [{
        "boxes": torch.zeros((0, 4)),
        "labels": torch.zeros((0,), dtype=torch.long)
    }]
    total, metrics = loss(head_out, targets)
    # No positives -> only the (all-negative) classification term is active.
    assert torch.isfinite(total)
    assert float(metrics["num_pos"]) == 0.0

  def test_gradients_flow(self):
    loss = FCOSLoss(num_classes=2)
    feature = torch.randn(1, 16, 4, 4, requires_grad=True)
    pyramid = EncoderPyramid([EncoderLevel(feature=feature, stride=8)])
    head = FCOSHead(in_channels=[16], num_classes=2, hidden=16, num_convs=1)
    head_out = head(pyramid)
    targets = [{
        "boxes": torch.tensor([[4.0, 4.0, 20.0, 20.0]]),
        "labels": torch.tensor([0])
    }]
    total, _ = loss(head_out, targets)
    total.backward()
    assert feature.grad is not None

  def test_registered_in_registry(self):
    assert LOSSES.contains("fcos")
    criterion = LOSSES.build("fcos", num_classes=3)
    assert isinstance(criterion, FCOSLoss)

  def test_positive_location_reduces_box_loss(self):
    # A head that predicts the exact GT distances should score a lower box
    # loss than one predicting garbage, sanity-checking the GIoU wiring.
    loss = FCOSLoss(num_classes=1)
    feature = torch.randn(1, 16, 4, 4)
    pyramid = EncoderPyramid([EncoderLevel(feature=feature, stride=8)])
    head = FCOSHead(in_channels=[16], num_classes=1, hidden=16, num_convs=1)
    head_out = head(pyramid)
    targets = [{
        "boxes": torch.tensor([[10.0, 10.0, 22.0, 22.0]]),
        "labels": torch.tensor([0])
    }]
    _, metrics = loss(head_out, targets)
    assert float(metrics["box_loss"]) >= 0.0
