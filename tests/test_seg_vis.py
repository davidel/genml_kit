"""Tests for segmentation validation overlays (B5: ``--vis_every`` no-op)."""

import numpy as np
import torch

from genml_kit.methods.segmentation import (
    SegmentationMethod,
    _denormalize_image,
    _overlay_row,
)
from genml_kit.utils.seg import class_mask_to_rgb


class _FakeLogger:
  """Minimal MetricsLogger stand-in (records image artifacts)."""

  def __init__(self):
    self.images = {}  # tag -> (grid ndarray, global_step)

  def log_artifact(self, tag, payload_kind, payload, *, step=None):
    self.images[tag] = (payload, step)


class _FakeModel(torch.nn.Module):
  """Splits the input into ``num_classes`` logits with a fixed ``argmax``."""

  def __init__(self, num_classes=3):
    super().__init__()
    self.num_classes = num_classes

  def forward(self, pixel_values):
    # (B, 3, H, W) -> (B, C, H, W); class = last channel value mod C.
    B, _, H, W = pixel_values.shape
    logits = torch.zeros(B, self.num_classes, H, W, device=pixel_values.device)
    logits[:, 1] = 1.0  # always predict class 1
    return type("Out", (), {"logits": logits})()


class _FakeBlob:

  def __init__(self, images, mask):
    self.data = images
    self.meta = {"mask": mask}


class _FakeLoader:
  """Iterates once over a small list of blobs."""

  def __init__(self, blobs):
    self.blobs = blobs

  def __iter__(self):
    yield from self.blobs


def _make_method(num_classes=3):
  method = SegmentationMethod()
  method._num_classes = num_classes
  return method


def _sample_batch(batch=2):
  # Images: (B, 3, 16, 16) in [0, 1] (normalized), masks (B, 16, 16) long.
  images = torch.rand(batch, 3, 16, 16)
  mask = torch.randint(0, 3, (batch, 16, 16), dtype=torch.long)
  return images, mask


def test_class_mask_to_rgb_shape_and_palette():
  mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
  rgb = class_mask_to_rgb(mask)
  assert rgb.shape == (2, 3, 3)
  assert rgb.dtype == np.uint8
  # First class renders black.
  assert rgb[0, 0].tolist() == [0, 0, 0]
  # Distinct classes get distinct colors.
  assert rgb[0, 1].tolist() != rgb[0, 2].tolist()


def test_class_mask_to_rgb_wraps_large_ids():
  mask = np.array([[0, 100]], dtype=np.int64)
  rgb = class_mask_to_rgb(mask)
  assert rgb.shape == (1, 2, 3)
  # id 100 % 20 == 0 -> black wrap (no crash for out-of-palette ids).
  assert rgb[0, 0].tolist() == rgb[0, 1].tolist() == [0, 0, 0]


def test_class_mask_to_rgb_ignore_index_black():
  mask = np.array([[0, 255]], dtype=np.int64)
  rgb = class_mask_to_rgb(mask, ignore_index=255)
  assert rgb[0, 1].tolist() == [0, 0, 0]


def test_class_mask_to_rgb_ignore_none_keeps_palette():
  # With ignore_index=None, 255 maps via the palette (255 % 20 == 15).
  mask = np.array([[255]], dtype=np.int64)
  rgb = class_mask_to_rgb(mask, ignore_index=None)
  assert rgb[0, 0].tolist() == [255, 0, 128]


def test_denormalize_image_shape_and_range():
  img = torch.zeros(3, 8, 8)
  out = _denormalize_image(img)
  assert out.shape == (3, 8, 8)
  # Mean added back: denorm(0) = mean.
  assert torch.allclose(out,
                        torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                        atol=1e-3)


def test_overlay_row_concat_shape():
  img = torch.rand(3, 8, 8)
  mask = np.zeros((8, 8, 3), dtype=np.uint8)
  pred = np.ones((8, 8, 3), dtype=np.uint8) * 255
  row = _overlay_row(img, mask, pred)
  assert row.shape == (8, 24, 3)  # H x 3W x 3
  assert row.dtype == np.uint8


def test_log_validation_writes_overlay(tmp_path):
  method = _make_method(num_classes=3)
  logger = _FakeLogger()
  images, mask = _sample_batch(batch=2)
  loader = _FakeLoader([_FakeBlob(images, mask)])
  model = _FakeModel(num_classes=3)

  method.log_validation(model,
                        loader,
                        lambda x, d: x,
                        logger,
                        global_step=7,
                        device="cpu",
                        num_samples=2)

  assert "seg/overlay" in logger.images
  grid, step = logger.images["seg/overlay"]
  assert step == 7
  assert isinstance(grid, np.ndarray)
  # 2 rows x (16 x 3) wide x 3 channels.
  assert grid.shape == (2 * 16, 3 * 16, 3)
  assert grid.dtype == np.uint8


def test_log_validation_none_logger_noop():
  method = _make_method()
  logger = None
  images, mask = _sample_batch()
  loader = _FakeLoader([_FakeBlob(images, mask)])
  model = _FakeModel()
  # Should not raise and write nothing.
  method.log_validation(model,
                        loader,
                        lambda x, d: x,
                        logger,
                        global_step=1,
                        device="cpu")


def test_log_validation_restores_model_mode():
  method = _make_method()
  logger = _FakeLogger()
  images, mask = _sample_batch()
  loader = _FakeLoader([_FakeBlob(images, mask)])
  model = _FakeModel()
  model.train()
  method.log_validation(model,
                        loader,
                        lambda x, d: x,
                        logger,
                        global_step=1,
                        device="cpu",
                        num_samples=1)
  # Original training flag should be restored.
  assert model.training
