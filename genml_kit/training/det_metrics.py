"""Detection evaluation metrics (COCO-style mAP).

Like :mod:`genml_kit.training.seg_metrics`, the metric is accumulated
incrementally over the validation loader so memory stays constant regardless
of dataset size.  Unlike segmentation (a confusion matrix), detection needs
the *full* precision/recall curve per class, so the meter stores the decoded
detections and ground-truth boxes and reduces them once per epoch.

The AP definition is the COCO/VOC one:

* For a class and an IoU threshold, sort detections by descending score.
* Greedily match each detection to an unmatched ground-truth box of the same
  class whose IoU is ``>= threshold``; a match is a true positive (TP), the
  rest are false positives (FP).
* Precision/recall are computed along that ranked list; AP is the
  all-point-interpolated area under the curve.
* ``mAP@.5`` is the mean over classes at IoU 0.5; ``mAP`` (COCO's
  ``mAP@[.5:.95]``) averages over the ten IoU thresholds 0.5, 0.55, ..., 0.95.

Classes absent from the ground truth are excluded from the mean (they have no
recall to measure), mirroring how :class:`IoUMeter` only averages over
classes present in the target.
"""

import numpy as np
import torch
from torchvision.ops import box_iou

# COCO's ten IoU thresholds, used for the headline mAP.
_DEFAULT_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


class MAPMeter:
  """Incremental COCO-style mean-average-precision accumulator.

  Parameters
  ----------
  num_classes : int
      Number of foreground classes.
  iou_thresholds : array-like, optional
      IoU thresholds to average over (default: COCO's 0.5:0.95:0.05).
  """

  def __init__(self, num_classes, iou_thresholds=None):
    self.num_classes = int(num_classes)
    self.iou_thresholds = (np.asarray(iou_thresholds, dtype=np.float64)
                           if iou_thresholds is not None else _DEFAULT_THRESHOLDS)
    self.reset()

  def reset(self):
    """Clear all accumulated detections and ground truth."""
    # One record per image: {"boxes", "scores", "labels"} (detections, already
    # NMS-decoded) and {"boxes", "labels"} (ground truth).
    self._detections = []
    self._ground_truth = []

  def update(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels):
    """Accumulate one image's predictions and ground truth.

    Parameters
    ----------
    pred_boxes : torch.Tensor
      ``(N, 4)`` decoded ``xyxy`` boxes (post-NMS).
    pred_scores : torch.Tensor
      ``(N,)`` confidence scores aligned with *pred_boxes*.
    pred_labels : torch.Tensor
      ``(N,)`` predicted class ids.
    gt_boxes : torch.Tensor
      ``(M, 4)`` ground-truth ``xyxy`` boxes.
    gt_labels : torch.Tensor
      ``(M,)`` ground-truth class ids.
    """
    self._detections.append({
        "boxes": pred_boxes.detach().cpu(),
        "scores": pred_scores.detach().cpu(),
        "labels": pred_labels.detach().cpu(),
    })
    self._ground_truth.append({
        "boxes": gt_boxes.detach().cpu(),
        "labels": gt_labels.detach().cpu(),
    })

  def compute(self):
    """Return ``{"map", "map_50", "map_75"}`` over the accumulated data."""
    per_threshold = [self._ap_at(threshold) for threshold in self.iou_thresholds]
    # per_threshold[i][c] is the AP of class c at threshold i (NaN if the
    # class is absent from the ground truth at that threshold).
    stacked = np.array(per_threshold, dtype=np.float64)
    if stacked.size == 0:
      return {"map": 0.0, "map_50": 0.0, "map_75": 0.0}

    def _mean_over_classes(values):
      # Average the per-class APs, ignoring NaN (absent) classes. If every
      # class is absent, the metric is defined as 0.
      finite = values[~np.isnan(values)]
      return float(finite.mean()) if finite.size else 0.0

    # Headline mAP: mean of the per-threshold class-means.
    per_threshold_means = np.array([_mean_over_classes(row) for row in stacked])
    map_value = float(per_threshold_means.mean())
    map_50 = self._threshold_mean(0.5, _mean_over_classes)
    map_75 = self._threshold_mean(0.75, _mean_over_classes)
    return {"map": map_value, "map_50": map_50, "map_75": map_75}

  def _threshold_mean(self, target, reduce_fn):
    """Mean AP over classes at IoU *target*, or 0.0 when it is not tracked."""
    matches = np.isclose(self.iou_thresholds, target, atol=1e-6)
    if not matches.any():
      return 0.0
    index = int(np.argmax(matches))
    return reduce_fn(self._ap_at(self.iou_thresholds[index]))

  def _ap_at(self, threshold):
    """Return the per-class AP vector at one IoU *threshold*.

    Returns a ``(num_classes,)`` float array where absent classes are NaN.
    """
    aps = np.full(self.num_classes, np.nan, dtype=np.float64)
    for cls in range(self.num_classes):
      aps[cls] = self._ap_for_class(cls, threshold)
    return aps

  def _ap_for_class(self, cls, threshold):
    """Return the average precision of one class at one IoU threshold."""
    # Gather this class's detections across every image, remembering which
    # image each came from so matching stays within an image.
    det_scores = []
    det_image = []
    det_box = []
    for img_idx, det in enumerate(self._detections):
      mask = det["labels"] == cls
      if mask.any():
        det_scores.append(det["scores"][mask])
        det_box.append(det["boxes"][mask])
        det_image.append(torch.full((int(mask.sum()),), img_idx, dtype=torch.long))
    # Ground-truth boxes per image for this class, plus the total count.
    gt_per_image = {}
    num_gt = 0
    for img_idx, gt in enumerate(self._ground_truth):
      mask = gt["labels"] == cls
      boxes = gt["boxes"][mask]
      gt_per_image[img_idx] = boxes
      num_gt += int(boxes.shape[0])

    if num_gt == 0:
      # No ground truth for this class anywhere: AP is undefined (NaN).
      return float("nan")
    if not det_scores:
      # Ground truth exists but nothing was detected: AP is zero.
      return 0.0

    scores = torch.cat(det_scores)
    boxes = torch.cat(det_box)
    images = torch.cat(det_image)
    # Rank all detections by descending score (the precision/recall order).
    order = torch.argsort(scores, descending=True)
    scores = scores[order]
    boxes = boxes[order]
    images = images[order]

    # Track which ground-truth boxes have already been matched (per image), so
    # a second detection cannot claim the same box.
    matched = {
        img_idx: torch.zeros(boxes_per_img.shape[0], dtype=torch.bool)
        for img_idx, boxes_per_img in gt_per_image.items()
    }
    tp = np.zeros(scores.shape[0], dtype=np.float64)
    fp = np.zeros(scores.shape[0], dtype=np.float64)
    for i in range(scores.shape[0]):
      img_idx = int(images[i])
      gt_boxes = gt_per_image.get(img_idx)
      if gt_boxes is None or gt_boxes.shape[0] == 0:
        fp[i] = 1.0
        continue
      # IoU of this detection against every GT box in its image: (M,).
      ious = box_iou(boxes[i:i + 1], gt_boxes)[0]
      # Exclude already-matched boxes by zeroing their IoU.
      ious = ious.masked_fill(matched[img_idx], -1.0)
      best_iou, best_gt = torch.max(ious, dim=0)
      if float(best_iou) >= threshold:
        tp[i] = 1.0
        matched[img_idx][best_gt] = True
      else:
        fp[i] = 1.0

    # Cumulative TP/FP along the ranked list -> precision/recall curve.
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    return _average_precision(recall, precision)


def _average_precision(recall, precision):
  """All-point-interpolated AP (VOC2010+ / COCO).

  ``precision`` is made monotonically non-increasing from the right, then
  integrated over ``recall``.  This is the standard area under the
  precision-recall curve.
  """
  mrec = np.concatenate([[0.0], recall, [1.0]])
  mpre = np.concatenate([[0.0], precision, [0.0]])
  for i in range(mpre.shape[0] - 2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i + 1])
  # Sum of rectangular slices where recall increases.
  idx = np.where(mrec[1:] != mrec[:-1])[0]
  return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
