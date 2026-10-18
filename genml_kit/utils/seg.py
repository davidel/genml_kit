"""Shared segmentation helpers.

Kept free of ``genml_kit.models`` imports so that light modules (the metrics
accumulator, the training method) can use them without pulling in the whole
model package.
"""

import torch.nn.functional as F


def upsample_to(logits, size):
  """Bilinearly resize *logits* to ``(H, W) = size`` when they differ.

  Segmentation decoders (and HF models) commonly emit low-resolution logits
  such as ``(B, C, H/4, W/4)``; the loss and metric need full-resolution maps
  to match the mask grid.  Returns *logits* unchanged when already aligned.

  Parameters
  ----------
  logits : torch.Tensor
    ``(B, C, H, W)`` raw scores.
  size : tuple[int, int]
    Target ``(H, W)``.

  Returns
  -------
  torch.Tensor
    ``(B, C, H', W')`` logits at *size*.
  """
  if logits.shape[-2:] == tuple(size):
    return logits
  return F.interpolate(logits, size=tuple(size), mode="bilinear", align_corners=False)
