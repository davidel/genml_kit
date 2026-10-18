"""Segmentation losses: Dice, cross-entropy, and their combination.

All losses take ``logits`` of shape ``(B, C, H, W)`` and a ``target``
label map of shape ``(B, H, W)`` with integer classes in ``[0, C)`` (plus
an optional ``ignore_index``).  They return a scalar tensor.

The combination :func:`dice_ce_loss` follows the common practice of
averaging a region-overlap term (Dice, robust to the class imbalance
caused by background pixels dominating) with a per-pixel cross-entropy
term (stable gradients early in training).
"""

import torch
import torch.nn.functional as F


def _one_hot(target, num_classes, ignore_index):
  """One-hot encode ``(B, H, W)`` -> ``(B, C, H, W)`` and an ignore mask.

  Returns ``(one_hot, valid_mask)`` where ``valid_mask`` is ``(B, 1, H, W)``
  with zeros at ignored pixels.  When ``ignore_index`` is negative every
  pixel is valid.
  """
  if ignore_index is not None and ignore_index >= 0:
    valid = (target != ignore_index)
    safe_target = target.clone()
    safe_target[~valid] = 0
  else:
    valid = torch.ones_like(target, dtype=torch.bool)
    safe_target = target

  # One-hot encode: (B, H, W) -> (B, H, W, C).
  one_hot = F.one_hot(safe_target, num_classes=num_classes)
  # Move the class axis to channels: (B, H, W, C) -> (B, C, H, W).
  one_hot = one_hot.permute(0, 3, 1, 2).to(torch.float32)
  # Valid mask gains a channel axis: (B, H, W) -> (B, 1, H, W).
  return one_hot, valid.unsqueeze(1).to(torch.float32)


def dice_loss(logits, target, *, ignore_index=255, eps=1.0):
  """Soft Dice loss averaged over classes present in *target*.

  Args:
    logits: ``(B, C, H, W)`` raw scores.
    target: ``(B, H, W)`` integer label map.
    ignore_index: Label value excluded from the loss (``< 0`` disables).
    eps: Smoothing constant in the numerator and denominator.

  Returns:
    Scalar tensor ``1 - mean_class_dice``.
  """
  num_classes = logits.size(1)
  probs = torch.softmax(logits, dim=1)
  one_hot, valid = _one_hot(target, num_classes, ignore_index)

  dims = (0, 2, 3)  # sum over batch + spatial
  intersection = torch.sum(probs * one_hot * valid, dim=dims)
  cardinality = torch.sum((probs + one_hot) * valid, dim=dims)

  dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
  # Average only over classes that actually occur *in the valid region*
  # (else background-only batches would pull the mean toward 1 and hide
  # poor foreground Dice).  Weighting by ``valid`` is essential: ignored
  # pixels are one-hot encoded as class 0, so summing ``one_hot`` alone
  # would count class 0 as present whenever the batch has any void pixel.
  present = (torch.sum(one_hot * valid, dim=dims) > 0).to(torch.float32)
  denom = torch.clamp(present.sum(), min=1.0)
  mean_dice = torch.sum(dice_per_class * present) / denom
  return 1.0 - mean_dice


def dice_ce_loss(logits, target, *, weight=1.0, ignore_index=255, eps=1.0):
  """Dice + cross-entropy: ``dice + weight * ce``.

  Args:
    logits: ``(B, C, H, W)`` raw scores.
    target: ``(B, H, W)`` integer label map.
    weight: Relative weight of the cross-entropy term.
    ignore_index: Label value excluded from both terms (``< 0`` disables).
    eps: Smoothing constant forwarded to :func:`dice_loss`.

  Returns:
    Scalar tensor.
  """
  dice = dice_loss(logits, target, ignore_index=ignore_index, eps=eps)
  # cross_entropy uses -100 as its "no ignore" sentinel.
  ce_ignore = ignore_index if (ignore_index is not None and ignore_index >= 0) else -100
  ce = F.cross_entropy(logits, target, ignore_index=ce_ignore)
  return dice + weight * ce


class DiceCELoss(torch.nn.Module):
  """Module wrapper so ``load_loss("dice_ce")`` returns a callable module.

  Keeping this a module (rather than a bare function) mirrors how the
  other registered supervised criteria (``focal``) are exposed and lets
  the loss be moved to a device or serialized like any other criterion.
  """

  def __init__(self, weight=1.0, ignore_index=255, eps=1.0):
    super().__init__()
    self.weight = weight
    self.ignore_index = ignore_index
    self.eps = eps

  def forward(self, logits, target):
    """Return the scalar Dice+CE loss.

    Shapes: *logits* ``(B, C, H, W)``, *target* ``(B, H, W)``.
    """
    return dice_ce_loss(logits,
                        target,
                        weight=self.weight,
                        ignore_index=self.ignore_index,
                        eps=self.eps)
