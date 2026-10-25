"""Tests for the box-aware detection transforms (letterbox + hflip)."""

import numpy as np
import torch
from PIL import Image

from genml_kit.datasets.det_transforms import (Letterbox, RandomHorizontalFlip,
                                               ToImageTensor, build_det_transform)


def _image(width, height, value=128):
  arr = np.full((height, width, 3), value, dtype="uint8")
  return Image.fromarray(arr)


class TestLetterbox:

  def test_square_image_scales_uniformly(self):
    row = {"image": _image(100, 100), "boxes": torch.tensor([[0.0, 0.0, 50.0, 50.0]])}
    out = Letterbox(200)(row)
    assert out["image"].size == (200, 200)
    # scale = 2, no padding -> boxes double.
    assert torch.allclose(out["boxes"], torch.tensor([[0.0, 0.0, 100.0, 100.0]]))

  def test_wide_image_pads_top_bottom(self):
    # 200x100 -> scale to fit 200 wide => scale 1, height 100, pad 50 top/bottom.
    row = {"image": _image(200, 100), "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]])}
    out = Letterbox(200)(row)
    assert out["image"].size == (200, 200)
    # y is shifted by pad_top=50, x unchanged (scale 1, pad_left 0).
    assert torch.allclose(out["boxes"], torch.tensor([[10.0, 60.0, 30.0, 80.0]]))

  def test_tall_image_pads_left_right(self):
    # 100x200 -> scale 1, width 100, pad 50 left/right.
    row = {"image": _image(100, 200), "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]])}
    out = Letterbox(200)(row)
    assert torch.allclose(out["boxes"], torch.tensor([[60.0, 10.0, 80.0, 30.0]]))

  def test_boxes_clamped_to_canvas(self):
    # A box that would fall outside after padding must be clamped in-bounds.
    row = {"image": _image(50, 50), "boxes": torch.tensor([[40.0, 40.0, 60.0, 60.0]])}
    out = Letterbox(100)(row)
    assert out["boxes"].max().item() <= 100.0
    assert out["boxes"].min().item() >= 0.0

  def test_empty_boxes_preserved(self):
    row = {"image": _image(64, 64), "boxes": torch.zeros((0, 4))}
    out = Letterbox(128)(row)
    assert out["boxes"].shape == (0, 4)


class TestRandomHorizontalFlip:

  def test_flip_maps_x_coordinates(self):
    row = {
        "image": _image(100, 50),
        "boxes": torch.tensor([[10.0, 5.0, 30.0, 20.0]]),
    }
    out = RandomHorizontalFlip(p=1.0)(row)
    # width=100: x1' = 100 - 30 = 70, x2' = 100 - 10 = 90; y untouched.
    assert torch.allclose(out["boxes"], torch.tensor([[70.0, 5.0, 90.0, 20.0]]))

  def test_no_flip_when_p_zero(self):
    boxes = torch.tensor([[10.0, 5.0, 30.0, 20.0]])
    row = {"image": _image(100, 50), "boxes": boxes}
    out = RandomHorizontalFlip(p=0.0)(row)
    assert torch.allclose(out["boxes"], boxes)

  def test_flip_preserves_box_width(self):
    boxes = torch.tensor([[10.0, 5.0, 30.0, 20.0]])
    row = {"image": _image(100, 50), "boxes": boxes}
    out = RandomHorizontalFlip(p=1.0)(row)
    width_before = boxes[0, 2] - boxes[0, 0]
    width_after = out["boxes"][0, 2] - out["boxes"][0, 0]
    assert torch.allclose(width_before, width_after)

  def test_empty_boxes_preserved(self):
    row = {"image": _image(64, 64), "boxes": torch.zeros((0, 4))}
    out = RandomHorizontalFlip(p=1.0)(row)
    assert out["boxes"].shape == (0, 4)


class TestToImageTensor:

  def test_converts_to_chw_float01(self):
    row = {"image": _image(8, 4, value=255), "boxes": torch.zeros((0, 4))}
    out = ToImageTensor()(row)
    assert out["image"].shape == (3, 4, 8)
    assert out["image"].dtype == torch.float32
    assert float(out["image"].max()) == 1.0

  def test_boxes_untouched(self):
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = ToImageTensor()({"image": _image(8, 8), "boxes": boxes})
    assert torch.allclose(out["boxes"], boxes)


class TestBuildDetTransform:

  def test_train_flips_and_letterboxes(self):
    transform = build_det_transform(64, is_train=True)
    row = {"image": _image(100, 50), "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]])}
    out = transform(row)
    assert out["image"].shape == (3, 64, 64)
    assert out["boxes"].shape == (1, 4)

  def test_val_is_deterministic(self):
    transform = build_det_transform(64, is_train=False)
    row = {"image": _image(100, 50), "boxes": torch.tensor([[10.0, 5.0, 30.0, 20.0]])}
    first = transform(row)
    second = transform(dict(row))
    assert torch.allclose(first["boxes"], second["boxes"])

  def test_boxes_stay_inside_canvas(self):
    transform = build_det_transform(96, is_train=True)
    for _ in range(20):
      row = {
          "image": _image(200, 80),
          "boxes": torch.tensor([[0.0, 0.0, 200.0, 80.0]]),
      }
      out = transform(row)
      assert out["boxes"].max().item() <= 96.0
      assert out["boxes"].min().item() >= 0.0
