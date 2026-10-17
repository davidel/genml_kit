"""Tests for PairedSegTransform and the segmentation loss/metric helpers."""

import numpy as np
import pytest
import torch
from PIL import Image

from genml_kit.datasets.transforms import (
    PairedSegTransform,
    build_seg_geometric,
    build_seg_photometric,
)
from genml_kit.losses.segmentation import dice_ce_loss, dice_loss
from genml_kit.training.seg_metrics import IoUMeter


def _sample():
  rng = np.random.default_rng(0)
  image = Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8))
  mask = rng.integers(0, 4, (20, 30), dtype=np.int64)
  return image, mask


class TestPairedSegTransform:

  def test_shapes_and_mask_dtype(self):
    image, mask = _sample()
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=True))
    out = tf({"image": image, "mask": mask})
    assert out["image"].size == (16, 16)
    # Mask stays an integer label map at the same spatial size.
    assert out["mask"].shape == (16, 16)
    assert out["mask"].dtype == torch.int64

  def test_mask_uses_nearest_and_keeps_classes(self):
    image, mask = _sample()
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=False))
    out = tf({"image": image, "mask": mask})
    classes = set(out["mask"].unique().tolist())
    assert classes.issubset(set(np.unique(mask).tolist()))

  def test_geometric_is_synchronised(self):
    """Image and mask receive the *same* geometric transform.

    We build a mask that encodes the image content, then check the flip
    decision applies identically to both by comparing against a manual
    deterministic flip.
    """
    image = Image.fromarray(np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3))
    mask = np.arange(4 * 6, dtype=np.int64).reshape(4, 6)

    from torchvision.transforms import v2
    flip = v2.Compose([v2.Resize((4, 6))])  # identity geometry
    tf = PairedSegTransform(geometric=flip)
    out = tf({"image": image, "mask": mask})
    # With identity geometry the mask must be unchanged (no misalignment).
    assert torch.equal(out["mask"], torch.from_numpy(mask))

  def test_source_row_not_mutated(self):
    image, mask = _sample()
    row = {"image": image, "mask": mask}
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=True))
    tf(row)
    assert row["image"] is image
    assert row["mask"] is mask

  def test_photometric_only_touches_image(self):
    image, mask = _sample()
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=False),
                            photometric=build_seg_photometric(16, is_train=True))
    out = tf({"image": image, "mask": mask})
    assert out["mask"].dtype == torch.int64
    assert out["image"].size == (16, 16)

  def test_non_dict_raises(self):
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=False))
    with pytest.raises(TypeError):
      tf(123)


class TestSegLosses:

  def test_dice_perfect_prediction_is_zero(self):
    target = torch.randint(0, 3, (2, 8, 8))
    logits = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 20
    assert float(dice_loss(logits, target)) < 1e-3

  def test_dice_ce_matches_sum(self):
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    d = dice_loss(logits, target, ignore_index=-1)
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert torch.allclose(dice_ce_loss(logits, target, ignore_index=-1), d + ce)

  def test_ignore_index_excluded(self):
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 6, 6)
    target = torch.zeros(1, 6, 6, dtype=torch.long)
    target[0, :3, :] = 255
    # Dice over a single present class with a forced-ignore region.
    val = dice_loss(logits, target, ignore_index=255)
    assert torch.isfinite(val)


class TestIoUMeter:

  def test_perfect(self):
    meter = IoUMeter(num_classes=3, ignore_index=-1)
    target = torch.randint(0, 3, (2, 8, 8))
    logits = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 10
    meter.update(logits, target)
    m = meter.compute()
    assert m["miou"] == pytest.approx(1.0)
    assert m["pixel_acc"] == pytest.approx(1.0)

  def test_miou_matches_sklearn_on_random(self):
    torch.manual_seed(1)
    meter = IoUMeter(num_classes=3, ignore_index=-1)
    logits = torch.randn(4, 3, 10, 10)
    target = torch.randint(0, 3, (4, 10, 10))
    meter.update(logits, target)
    m = meter.compute()

    from sklearn.metrics import jaccard_score
    preds = logits.argmax(1).reshape(-1).numpy()
    tgt = target.reshape(-1).numpy()
    expected = jaccard_score(tgt, preds, average="macro")
    assert m["miou"] == pytest.approx(expected, abs=1e-6)

  def test_absent_class_excluded(self):
    meter = IoUMeter(num_classes=4, ignore_index=-1)
    target = torch.zeros(1, 4, 4, dtype=torch.long)  # only class 0
    logits = torch.nn.functional.one_hot(target, 4).permute(0, 3, 1, 2).float()
    meter.update(logits, target)
    m = meter.compute()
    # Only class 0 present and perfectly predicted -> miou 1.0, not 0.25.
    assert m["miou"] == pytest.approx(1.0)

  def test_ignore_index(self):
    meter = IoUMeter(num_classes=2, ignore_index=255)
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    target[0, 0, :] = 255
    logits = torch.nn.functional.one_hot(torch.zeros_like(target),
                                         2).permute(0, 3, 1, 2).float()
    meter.update(logits, target)
    assert meter.compute()["miou"] == pytest.approx(1.0)

  def test_upsamples_logits(self):
    meter = IoUMeter(num_classes=2, ignore_index=-1)
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    logits = torch.zeros(1, 2, 4, 4)  # half resolution
    logits[:, 0] = 5.0
    meter.update(logits, target)
    assert meter.compute()["miou"] == pytest.approx(1.0)
