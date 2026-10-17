"""Segmentation evaluation metrics (IoU / Dice / pixel accuracy).

The metrics are accumulated incrementally over a validation loader through
a confusion matrix, then reduced once per epoch.  This keeps memory
constant regardless of dataset size and avoids storing every prediction.

Usage::

    meter = IoUMeter(num_classes=4, ignore_index=255)
    for blob in loader:
      logits = model(blob.data).logits
      meter.update(logits, blob.meta["mask"])
    metrics = meter.compute()   # {"miou": ..., "mdice": ..., "pixel_acc": ...}
"""

import numpy as np
import torch


class IoUMeter:
  """Incremental per-class IoU / Dice / accuracy accumulator.

  Parameters
  ----------
  num_classes : int
      Number of segmentation classes (excluding any ignore label).
  ignore_index : int
      Label value excluded from the metrics (``< 0`` disables).
  """

  def __init__(self, num_classes, ignore_index=255):
    self.num_classes = num_classes
    self.ignore_index = ignore_index
    self.reset()

  def reset(self):
    """Zero the confusion matrix (call at the start of each epoch)."""
    self._confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

  @torch.no_grad()
  def update(self, logits, target):
    """Accumulate one batch.

    Args:
      logits: ``(B, C, H, W)`` raw scores (any spatial size; upsampled to
        match *target* when needed).
      target: ``(B, H, W)`` integer label map.
    """
    if logits.shape[-2:] != target.shape[-2:]:
      # Align to the mask grid: (B, C, h, w) -> (B, C, H, W).
      logits = torch.nn.functional.interpolate(logits,
                                               size=target.shape[-2:],
                                               mode="bilinear",
                                               align_corners=False)
    # Per-pixel class ids: (B, C, H, W) -> (B, H, W).
    preds = logits.argmax(dim=1)
    self._add_confusion(preds.reshape(-1), target.reshape(-1))

  def _add_confusion(self, preds, target):
    preds = preds.detach().to("cpu").numpy().astype(np.int64)
    target = target.detach().to("cpu").numpy().astype(np.int64)

    if self.ignore_index is not None and self.ignore_index >= 0:
      valid = target != self.ignore_index
      preds = preds[valid]
      target = target[valid]

    # Guard against out-of-range labels (e.g. a stray class id) so a single
    # bad pixel cannot corrupt the histogram.
    in_range = ((target >= 0) & (target < self.num_classes) & (preds >= 0) &
                (preds < self.num_classes))
    if not in_range.all():
      preds = preds[in_range]
      target = target[in_range]

    if target.size == 0:
      return
    flat = target * self.num_classes + preds
    counts = np.bincount(flat, minlength=self.num_classes * self.num_classes)
    self._confusion += counts.reshape(self.num_classes, self.num_classes)

  def compute(self):
    """Return ``{"miou", "mdice", "pixel_acc", "per_class_iou"}``.

    Classes absent from both prediction and ground truth are excluded from
    the means (they carry no signal and would otherwise bias the score).
    """
    cm = self._confusion
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    denom_iou = tp + fp + fn
    iou = np.where(denom_iou > 0, tp / np.maximum(denom_iou, 1.0), np.nan)

    denom_dice = 2.0 * tp + fp + fn
    dice = np.where(denom_dice > 0, 2.0 * tp / np.maximum(denom_dice, 1.0), np.nan)

    present = denom_iou > 0
    if present.any():
      miou = float(np.nanmean(iou[present]))
      mdice = float(np.nanmean(dice[present]))
    else:
      miou = 0.0
      mdice = 0.0

    total = cm.sum()
    pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

    return {
        "miou": miou,
        "mdice": mdice,
        "pixel_acc": pixel_acc,
        "per_class_iou": iou.tolist(),
    }
