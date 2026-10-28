"""Object-detection method: encoder + FCOS head, focal/GIoU loss, mAP metric.

The method owns the criterion (:class:`~genml_kit.losses.detection.FCOSLoss`)
and the validation metric (COCO-style mAP).  ``METRIC_KEY`` is ``map``
(maximize), so the best-checkpoint value is stored under ``best_map``.

The class count is inferred from the pipeline's dataset during ``wire_data``
(the COCO categories are compacted to ``[0, C)`` by the dataset), so the same
CLI works for any corpus without a separate ``--num_classes`` flag.  The head
predicts per-level dense outputs; this method decodes them to ``xyxy`` boxes,
applies per-class NMS, and feeds the result to the mAP meter.
"""

import logging

import torch
from torchvision.ops import batched_nms

from genml_kit.losses import load_loss
from genml_kit.losses.detection import _location_grid
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.training.det_metrics import MAPMeter
from genml_kit.utils.logging import fatal


def _sigmoid_focal_scores(cls_logits):
  """Return per-class scores from raw logits as a probability-like map.

  ``cls_logits`` is ``(B, P, C)``; the head uses independent sigmoids per
  class (FCOS is not softmax-normalised), so the score is just the sigmoid.
  """
  return torch.sigmoid(cls_logits)


def decode_predictions(head_output, score_thresh=0.05, topk=1000, nms_thresh=0.5):
  """Decode a head output into per-image ``(boxes, scores, labels)`` lists.

  Parameters
  ----------
  head_output : DetHeadOutput
    The head's per-level predictions.
  score_thresh : float
    Discard locations whose best-class score is below this.
  topk : int
    Keep at most this many detections per image *before* NMS (guards the
    NMS cost on dense maps).
  nms_thresh : float
    IoU threshold for per-class non-maximum suppression.

  Returns
  -------
  list[dict]
    One entry per image with ``boxes`` ``(N, 4)`` ``xyxy``, ``scores``
    ``(N,)`` and ``labels`` ``(N,)``.
  """
  cls_logits, box_pred, centerness, points = [], [], [], []
  for level_idx, logits in enumerate(head_output.cls_logits):
    stride = head_output.strides[level_idx]
    batch, _, height, width = logits.shape
    # (B, C, H, W) -> (B, H*W, C) so all levels concatenate along one axis.
    cls_logits.append(logits.permute(0, 2, 3, 1).reshape(batch, -1, logits.shape[1]))
    # (B, 4, H, W) -> (B, H*W, 4).
    box_pred.append(head_output.box_pred[level_idx].permute(0, 2, 3,
                                                            1).reshape(batch, -1, 4))
    # (B, 1, H, W) -> (B, H*W).
    centerness.append(head_output.centerness[level_idx].permute(0, 2, 3,
                                                                1).reshape(batch, -1))
    # (H*W, 2) location centres in input pixels (shared across the batch).
    points.append(_location_grid((height, width), stride, logits.device))
  # (B, P_total, C) / (B, P_total, 4) / (B, P_total) / (P_total, 2).
  cls_logits = torch.cat(cls_logits, dim=1)
  box_pred = torch.cat(box_pred, dim=1)
  centerness = torch.cat(centerness, dim=1)
  points = torch.cat(points, dim=0)

  # FCOS score = sqrt(cls_prob * centerness_prob): centerness suppresses the
  # low-quality locations that classify confidently but localise poorly.
  cls_prob = _sigmoid_focal_scores(cls_logits)
  ctr_prob = torch.sigmoid(centerness).unsqueeze(-1)
  # (B, P, C) * (B, P, 1) -> (B, P, C).
  scores = torch.sqrt((cls_prob * ctr_prob).clamp(min=0.0))

  batch_size = scores.shape[0]
  results = []
  for img in range(batch_size):
    # Best class per location: (B, P, C) -> (P,) / (P,).
    img_scores, img_labels = scores[img].max(dim=-1)
    keep = img_scores > score_thresh
    if int(keep.sum()) > topk:
      # Keep the top-scoring locations only, to bound the NMS cost.
      topk_idx = torch.topk(img_scores, topk).indices
      keep = torch.zeros_like(keep)
      keep[topk_idx] = True
    if not keep.any():
      results.append({
          "boxes": points.new_zeros((0, 4)),
          "scores": points.new_zeros((0,)),
          "labels": torch.zeros((0,), dtype=torch.long, device=points.device),
      })
      continue
    sel_points = points[keep]
    sel_ltrb = box_pred[img][keep]
    # distances -> xyxy: x1 = x - l, y1 = y - t, x2 = x + r, y2 = y + b.
    boxes = torch.stack([
        sel_points[:, 0] - sel_ltrb[:, 0],
        sel_points[:, 1] - sel_ltrb[:, 1],
        sel_points[:, 0] + sel_ltrb[:, 2],
        sel_points[:, 1] + sel_ltrb[:, 3],
    ],
                        dim=1)
    sel_scores = img_scores[keep]
    sel_labels = img_labels[keep]
    # Per-class NMS: batched_nms offsets boxes by class so two classes never
    # suppress each other.
    keep_idx = batched_nms(boxes, sel_scores, sel_labels, nms_thresh)
    results.append({
        "boxes": boxes[keep_idx],
        "scores": sel_scores[keep_idx],
        "labels": sel_labels[keep_idx],
    })
  return results


@METHODS.register
class DetectionMethod(Method):
  """Supervised object detection (encoder + FCOS head)."""

  NAME = "detection"
  METRIC_KEY = "map"
  METRIC_MINIMIZE = False
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self._num_classes = None
    self._id2label = None
    self._criterion = None
    self.train_transforms = None
    self.val_transforms = None
    # Decode thresholds, filled from args in ``wire_data`` (``evaluate`` has
    # no access to ``args``).  Defaults match the CLI defaults so a missing
    # ``wire_data`` still yields sane behaviour.
    self._score_thresh = 0.05
    self._nms_thresh = 0.5
    self._max_detections = 100

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("detection method")
    group.add_argument("--det_num_classes",
                       type=int,
                       help="Override the inferred class count.")
    group.add_argument("--det_score_thresh",
                       type=float,
                       default=0.05,
                       help="Discard detections below this score.")
    group.add_argument("--det_nms_thresh",
                       type=float,
                       default=0.5,
                       help="IoU threshold for per-class NMS.")
    group.add_argument("--det_max_detections",
                       type=int,
                       default=100,
                       help="Keep at most this many detections per image "
                       "after NMS (for the metric).")

  def prepare_transforms(self, args, device):
    """Build the train/val detection transforms.

    Lifecycle position 1.  Detection has no processor-driven normalization
    (the encoder is a plain timm backbone), so both splits share the default
    letterbox + tensor pipeline; train additionally flips.
    """
    from genml_kit.datasets.det_transforms import build_det_transform

    image_size = args.image_size or 224
    self.train_transforms = build_det_transform(image_size, is_train=True)
    self.val_transforms = build_det_transform(image_size, is_train=False)

  def wire_data(self, args, pipeline):
    """Infer the class count from the dataset and build the criterion.

    Lifecycle position 2.  The loaders exist, so the dataset's compacted
    label space is available.
    """
    self._num_classes = self._resolve_num_classes(args, pipeline)
    self._id2label = self._resolve_id2label(pipeline)
    self._criterion = self._build_criterion(args)
    self._score_thresh = getattr(args, "det_score_thresh", 0.05)
    self._nms_thresh = getattr(args, "det_nms_thresh", 0.5)
    self._max_detections = getattr(args, "det_max_detections", 100)

  def build_model(self, args, device):
    """Load the detection model (encoder + head) via the registry."""
    if self._num_classes is None:
      fatal(
          "DetectionMethod.build_model requires wire_data(args, pipeline) "
          "to have inferred the class count; check the run order "
          "(wire_data before build_model).", RuntimeError)
    from genml_kit.models import load_model

    model = load_model(
        args.model,
        num_labels=self._num_classes,
        id2label=self._id2label,
        image_size=args.image_size or 224,
        device=device,
        cache_dir=args.cache_dir,
        **args.model_arg,
    )
    return self._apply_model_extras(args, model, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    boxes = blob.meta.get("boxes")
    if boxes is None:
      fatal("DetectionMethod.train_step requires boxes in blob.meta['boxes'].",
            ValueError)
    out = model(pixel_values=blob.data)
    targets = [{
        "boxes": b,
        "labels": lab
    } for b, lab in zip(blob.meta["boxes"], blob.meta["labels"])]
    loss, extra = self._criterion(out.logits, targets)
    metrics = {"loss": loss.detach()}
    metrics.update(extra)
    return LossOutput(loss=loss, metrics=metrics)

  def evaluate(self, model, loader, device, to_device):
    """Return ``{"map", "map_50", "map_75", "loss"}`` over *loader*."""
    meter = MAPMeter(self._num_classes)
    was_training = getattr(model, "training", False)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    max_det = self._max_detections
    with torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        out = model(pixel_values=blob.data)
        targets = [{
            "boxes": b,
            "labels": lab
        } for b, lab in zip(blob.meta["boxes"], blob.meta["labels"])]
        loss, _ = self._criterion(out.logits, targets)
        total_loss += float(loss)
        n_batches += 1
        decoded = decode_predictions(out.logits,
                                     score_thresh=self._score_thresh,
                                     nms_thresh=self._nms_thresh)
        for img, det in enumerate(decoded):
          # Cap detections for the metric: extra low-score boxes can only add
          # false positives, and COCO evaluates with a 100-detection cap too.
          order = torch.argsort(det["scores"], descending=True)[:max_det]
          meter.update(det["boxes"][order], det["scores"][order], det["labels"][order],
                       targets[img]["boxes"], targets[img]["labels"])
    model.train(was_training)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics

  def has_metric_improved(self, new_metric, best_metric):
    return new_metric > best_metric

  def get_checkpoint_state(self, model, args):
    return {
        "method": "detection",
        "num_classes": self._num_classes,
        "id2label": self._id2label,
    }

  def load_checkpoint_state(self, model, state, args):
    """Restore the class count / label map carried in the checkpoint."""
    if state.get("num_classes") is not None:
      self._num_classes = int(state["num_classes"])
    if state.get("id2label") is not None:
      self._id2label = {int(k): v for k, v in state["id2label"].items()}

  def _resolve_num_classes(self, args, pipeline):
    explicit = getattr(args, "det_num_classes", None)
    if explicit:
      logging.info("DetectionMethod: class count = %d (explicit)", explicit)
      return int(explicit)
    inferred = getattr(pipeline, "num_classes", None)
    if not inferred:
      fatal(
          "DetectionMethod could not infer the class count from the dataset; "
          "pass --det_num_classes.", ValueError)
    logging.info("DetectionMethod: class count = %d (inferred)", inferred)
    return int(inferred)

  def _resolve_id2label(self, pipeline):
    id2label = getattr(pipeline, "id2label", None)
    if id2label:
      return {int(k): v for k, v in id2label.items()}
    return {i: f"class_{i}" for i in range(self._num_classes)}

  def _build_criterion(self, args):
    """Build the detection criterion via the loss registry.

    Defaults to ``fcos``; ``--loss`` / ``--loss_args`` override it.  The
    inferred ``num_classes`` is injected here (it is not a CLI flag).
    """
    spec = args.loss or "fcos"
    loss_args = dict(args.loss_args or {})
    loss_args.setdefault("num_classes", self._num_classes)
    criterion = load_loss(spec, **loss_args)
    logging.info("DetectionMethod criterion: %s %s", spec, loss_args)
    return criterion
