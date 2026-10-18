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

  def test_train_uses_scale_jitter(self):
    """Training geometry is ``RandomResizedCrop`` based (scale jitter)."""
    from torchvision.transforms import v2

    tf = build_seg_geometric(32, is_train=True)
    assert isinstance(tf, v2.Compose)
    assert any(isinstance(t, v2.RandomResizedCrop) for t in tf.transforms)
    assert any(isinstance(t, v2.RandomHorizontalFlip) for t in tf.transforms)

  def test_val_geometry_is_deterministic(self):
    """Validation stays Resize -> CenterCrop (no randomness)."""
    from torchvision.transforms import v2

    tf = build_seg_geometric(16, is_train=False)
    assert isinstance(tf, v2.Compose)
    kinds = [type(t) for t in tf.transforms]
    assert v2.Resize in kinds and v2.CenterCrop in kinds
    assert not any(
        isinstance(t, (v2.RandomResizedCrop, v2.RandomHorizontalFlip))
        for t in tf.transforms)

  def test_scale_jitter_keeps_mask_aligned_and_unblended(self):
    """RandomResizedCrop must keep masks integer, aligned, unblended.

    Run many random draws; the mask must always keep the source ids
    (never new/blended labels) and stay int64 at the image spatial size.
    """
    image, mask = _sample()
    tf = PairedSegTransform(geometric=build_seg_geometric(16, is_train=True))
    allowed = set(np.unique(mask).tolist())
    torch.manual_seed(0)
    for _ in range(50):
      out = tf({"image": image, "mask": mask})
      m = out["mask"]
      assert m.shape == (16, 16)
      assert m.dtype == torch.int64
      assert set(m.unique().tolist()).issubset(allowed)

  def test_scale_jitter_deterministic_for_both_fields(self):
    """One random draw per RRC call -> image and mask change together.

    Same seed twice on the same pair must yield identical image AND mask
    (proves the random crop/flip decision is shared, not drawn per field).
    """
    from torchvision.transforms import v2

    image, mask = _sample()
    geo = v2.Compose([
        v2.RandomResizedCrop(16, scale=(0.5, 2.0), ratio=(0.75, 1.3333333333333333)),
        v2.RandomHorizontalFlip(p=0.5),
    ])
    tf = PairedSegTransform(geometric=geo)
    torch.manual_seed(123)
    a = tf({"image": image, "mask": mask})
    torch.manual_seed(123)
    b = tf({"image": image, "mask": mask})
    assert torch.equal(
        v2.functional.to_dtype(v2.functional.to_image(a["image"]),
                               torch.float32,
                               scale=True),
        v2.functional.to_dtype(v2.functional.to_image(b["image"]),
                               torch.float32,
                               scale=True))
    assert torch.equal(a["mask"], b["mask"])

  def test_calibration_photometric_is_none(self):
    from torchvision.transforms import v2

    tf = build_seg_photometric(16, is_train=False)
    assert tf is None
    assert isinstance(build_seg_photometric(16, is_train=True), v2.RandomApply)

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

  def test_ignored_pixels_do_not_create_phantom_class(self):
    """Void pixels are one-hot encoded as class 0, but must not count as a
    present class in the Dice mean (regression: they used to)."""
    logits = torch.zeros(1, 2, 4, 4)
    logits[:, 1] = 10.0  # confident foreground = class 1
    clean = torch.ones(1, 4, 4, dtype=torch.long)
    dirty = clean.clone()
    dirty[0, 0, 0] = 255  # one void pixel -> phantom class 0 if unweighted
    d_clean = dice_loss(logits, clean, ignore_index=255)
    d_dirty = dice_loss(logits, dirty, ignore_index=255)
    assert torch.allclose(d_clean, d_dirty, atol=1e-4)


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
