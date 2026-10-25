"""Box-aware transforms for object detection.

Detection cannot reuse the segmentation transform stack: ``torchvision.v2``
remaps :class:`~torchvision.tv_tensors.Mask` automatically through geometric
ops, but *boxes* are plain tensors, so every resize / flip must move the
coordinates itself or the targets silently drift away from the image.

The two transforms here are therefore hand-written and operate on the same
item dict the dataset emits (``{"image", "boxes", "labels"}``):

* :class:`Letterbox` -- aspect-preserving resize + pad to a square canvas,
  then remap boxes.  Letterboxing keeps the aspect ratio intact, so boxes do
  not shear; the padding offset is folded into the coordinates.
* :class:`RandomHorizontalFlip` -- mirror the image and remap
  ``x1 -> W - x2`` / ``x2 -> W - x1``.

All boxes entering and leaving these transforms are ``xyxy`` in absolute
pixels of the *current* image, matching :mod:`genml_kit.datasets.det_folder`.
"""

import random

import numpy as np
import torch
from PIL import Image

from genml_kit.utils.logging import fatal


class Letterbox:
  """Aspect-preserving resize + pad to ``(size, size)`` with box remap.

  Parameters
  ----------
  size : int
      Output canvas side length (square).
  fill : int
      Padding value written into the added border (0 = black).
  """

  def __init__(self, size, fill=0):
    self.size = int(size)
    self.fill = int(fill)

  def __call__(self, row):
    """Return *row* with a letterboxed ``image`` and remapped ``boxes``."""
    image = row["image"]
    boxes = row["boxes"]
    if not isinstance(image, Image.Image):
      fatal(
          f"Letterbox expects a PIL image, got {type(image).__name__}",
          TypeError,
      )

    # Original (W, H) before any resize; boxes are expressed in this frame.
    width, height = image.size
    scale = min(self.size / width, self.size / height)
    # New content size after the aspect-preserving scale. At least 1 px so a
    # degenerate input cannot produce a zero-sized canvas.
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    # (W, H) -> (new_w, new_h): bilinear resize of the content region.
    resized = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # Centre the content on the square canvas; pad_left/top is where the
    # image origin lands, so it is the offset added to every box coordinate.
    pad_left = (self.size - new_w) // 2
    pad_top = (self.size - new_h) // 2
    # (new_w, new_h) -> (size, size): paste into a constant-filled canvas.
    canvas = Image.new("RGB", (self.size, self.size), (self.fill, self.fill, self.fill))
    canvas.paste(resized, (pad_left, pad_top))

    out = dict(row)
    out["image"] = canvas
    out["boxes"] = self._remap(boxes, scale, pad_left, pad_top)
    return out

  def _remap(self, boxes, scale, pad_left, pad_top):
    """Map ``(N, 4)`` boxes from the original frame to the padded canvas.

    ``x' = x * scale + pad`` for every coordinate; the order of the two
    operations matters (scale first, then translate) because the scale is
    about the *original* origin, not the padded one.
    """
    if boxes.numel() == 0:
      # (0, 4) -> (0, 4): nothing to remap, keep the dtype/shape contract.
      return boxes.clone()
    # (N, 4) -> (N, 4): scale every coordinate, then add the pad offset.
    scaled = boxes * scale
    offset = boxes.new_tensor([pad_left, pad_top, pad_left, pad_top])
    # Clamp into the canvas so a box that fell partly outside the padded
    # region (only possible with rounding) stays a valid, in-bounds target.
    remapped = scaled + offset
    return remapped.clamp(min=0.0, max=float(self.size))


class RandomHorizontalFlip:
  """Mirror the image left/right and remap boxes, at probability ``p``.

  Parameters
  ----------
  p : float
      Probability of applying the flip.
  """

  def __init__(self, p=0.5):
    self.p = float(p)

  def __call__(self, row):
    """Return *row* flipped horizontally with probability ``p``."""
    if random.random() >= self.p:
      return row
    image = row["image"]
    if not isinstance(image, Image.Image):
      fatal(
          f"RandomHorizontalFlip expects a PIL image, got {type(image).__name__}",
          TypeError,
      )
    width = image.size[0]
    # (W, H) -> (W, H): mirror across the vertical axis.
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

    boxes = row["boxes"]
    out = dict(row)
    out["image"] = flipped
    out["boxes"] = self._remap(boxes, width)
    return out

  def _remap(self, boxes, width):
    """Map ``(N, 4)`` boxes through the mirror ``x -> width - x``.

    A mirrored box's new ``x1`` is the mirror of its old ``x2`` and vice
    versa, so the two x-coordinates are swapped as well as reflected:
    ``[x1, y1, x2, y2] -> [width - x2, y1, width - x1, y2]``.  The y-axis is
    untouched.
    """
    if boxes.numel() == 0:
      return boxes.clone()
    # (N, 4) -> (N, 4): build the reflected x pair, keep y as-is.
    x1 = width - boxes[:, 2]
    x2 = width - boxes[:, 0]
    return torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=1)


class ToImageTensor:
  """Convert the PIL image to a ``(C, H, W)`` float32 tensor in ``[0, 1]``.

  Boxes are untouched: they are already in absolute pixels of the letterboxed
  image, which is the same coordinate frame as the tensor.
  """

  def __call__(self, row):
    image = row["image"]
    if not isinstance(image, Image.Image):
      fatal(
          f"ToImageTensor expects a PIL image, got {type(image).__name__}",
          TypeError,
      )
    # (W, H, 3) uint8 -> (3, H, W) float32 in [0, 1] via numpy.
    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    out = dict(row)
    out["image"] = tensor
    return out


def build_det_transform(image_size, *, is_train):
  """Return the default detection transform pipeline.

  Training adds a random horizontal flip (a label-preserving augmentation for
  detection: the box labels are unchanged, only the coordinates move).
  Validation is deterministic.  Both letterbox to ``image_size`` so train and
  eval share one input geometry.

  Parameters
  ----------
  image_size : int
      Square canvas side length.
  is_train : bool
      Whether to build the training variant.

  Returns
  -------
  callable
      A function mapping an item dict to a transformed item dict.
  """
  steps = []
  if is_train:
    steps.append(RandomHorizontalFlip(p=0.5))
  steps.append(Letterbox(image_size))
  steps.append(ToImageTensor())

  def _apply(row):
    for step in steps:
      row = step(row)
    return row

  return _apply
