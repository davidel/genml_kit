"""Dict-dataset transform utilities shared across tasks.

Promoted from ``pretrain/cli.py::_TransformWrapper`` (plan §12.6, item 1)
and extended for multi-field (pair) datasets: the VO pipeline applies one
augmentation to both frames of a pair through the same wrapper the
pre-training harness uses for its single image field.

:class:`PairedSegTransform` adds the segmentation variant of the
two-phase contract: geometric ops go to *image and mask* together (so
they stay pixel-aligned) while photometric ops and normalization touch
the image only.
"""

import numpy as np
import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2

from genml_kit.utils.logging import fatal


class DictFieldTransform(nn.Module):
  """Apply a transform to selected fields of a dict-returning dataset.

  Parameters
  ----------
  dataset : Dataset
      A dataset whose ``__getitem__`` returns a ``dict``.
  transform : callable
      Applied to every field listed in *fields* that is present in the
      item; absent fields are passed through untouched.
  fields : tuple[str, ...]
      The item keys to transform.

  Notes
  -----
  The single-field case used by the pre-training harness is
  ``DictFieldTransform(ds, t, fields=("image",))``; a VO pair dataset
  uses ``fields=("A", "B")`` so both frames see identical augmentation.
  """

  def __init__(self, dataset, transform, fields=("image",)):
    super().__init__()
    if not fields:
      fatal("DictFieldTransform requires a non-empty 'fields' tuple.", ValueError)
    self._ds = dataset
    self._t = transform
    self._fields = tuple(fields)

  def __len__(self):
    return len(self._ds)

  @property
  def fields(self):
    """The transformed field names."""
    return self._fields

  def forward(self, idx):
    # Copy: never mutate the source row.
    row = dict(self._ds[idx])
    for f in self._fields:
      if f in row:
        row[f] = self._t(row[f])
    return row

  # For backwards compatibility with Dataset interface
  def __getitem__(self, idx):
    return self.forward(idx)


class PairedSegTransform(nn.Module):
  """Co-transform an ``{"image", "mask"}`` item for semantic segmentation.

  The segmentation two-phase contract is split by op *semantics* rather
  than by field:

  - **geometric** ops (resize, crop, flip, rotate) are applied to the
    image and the mask together so they stay pixel-aligned;
  - **photometric** ops (colour jitter, blur) and normalization are
    applied to the image only.

  This is implemented with ``torchvision.transforms.v2``: both fields are
  wrapped as ``tv_tensors`` (the mask as :class:`tv_tensors.Mask`) and
  passed through a single :class:`v2.Compose`.  The v2 kernels route the
  mask to nearest-neighbour interpolation and skip any op that does not
  apply to it, so a single transform list is sufficient.

  Parameters
  ----------
  geometric : callable | None
      Transform applied to the image **and** mask (must be v2-aware when
      the mask is present, e.g. wrap it as ``tv_tensors.Mask``).
  photometric : callable | None
      Transform applied to the image only (e.g. ``v2.ColorJitter``).
  normalize : callable | None
      Normalization applied to the image only, after the other two.
  """

  def __init__(self, geometric=None, photometric=None, normalize=None):
    super().__init__()
    self._geometric = geometric
    self._photometric = photometric
    self._normalize = normalize

  @staticmethod
  def _as_mask(mask):
    """Wrap *mask* as a ``tv_tensors.Mask`` (no-op if already one)."""
    if isinstance(mask, tv_tensors.Mask):
      return mask
    if isinstance(mask, np.ndarray):
      mask = torch.from_numpy(np.ascontiguousarray(mask))
    if not isinstance(mask, torch.Tensor):
      fatal(
          f"PairedSegTransform: unsupported mask type "
          f"{type(mask).__name__}; expected tensor or ndarray", TypeError)
    return tv_tensors.Mask(mask)

  def forward(self, row):
    """Return *row* with transformed ``image`` / ``mask`` fields.

    ``row`` is the item dict produced by the dataset (``{"image", "mask"}``).
    The dict is copied before mutation so callers can share a source
    dataset safely, matching :class:`DictFieldTransform`.
    """
    if not isinstance(row, dict):
      fatal(f"PairedSegTransform expects an item dict, got "
            f"{type(row).__name__}", TypeError)

    row = dict(row)
    image = row["image"]
    mask = row.get("mask")

    if self._geometric is not None:
      if mask is not None:
        # Single pass over both fields keeps v2's random ops in sync
        # (one flip / crop decision shared by image and mask), which keeps
        # them pixel-aligned: image (C, H, W) and mask (H, W) -> (C, H', W')
        # and (H', W').
        out = self._geometric({"image": image, "mask": self._as_mask(mask)})
        image, mask = out["image"], out["mask"]
      else:
        image = self._geometric(image)

    if isinstance(mask, tv_tensors.Mask):
      mask = mask.as_subclass(torch.Tensor)

    if self._photometric is not None:
      image = self._photometric(image)
    if self._normalize is not None:
      image = self._normalize(image)

    row["image"] = image
    if mask is not None:
      row["mask"] = mask
    return row

  def __call__(self, row):
    return self.forward(row)


def build_seg_geometric(image_size, *, is_train):
  """Default geometric transforms (image + mask) for segmentation.

  Args:
    image_size: Target spatial size.
    is_train: When True, add a horizontal flip.

  Returns:
    A ``torchvision.transforms.v2`` callable usable on the paired dict.
  """
  from torchvision.transforms.functional import InterpolationMode

  ops = [
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
  ]
  if is_train:
    ops.append(v2.RandomHorizontalFlip(p=0.5))
  return v2.Compose(ops)


def build_seg_photometric(image_size, *, is_train):
  """Default photometric transforms (image only) for segmentation.

  Args:
    image_size: Target spatial size (kept for symmetry with the geometric
      builder; photometric ops are size-agnostic).
  Returns:
    A ``torchvision.transforms.v2`` callable, or ``None`` when no
    augmentation applies (validation).
  """
  del image_size  # documented as unused; kept for call symmetry
  if not is_train:
    return None
  return v2.RandomApply(
      [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.3)
