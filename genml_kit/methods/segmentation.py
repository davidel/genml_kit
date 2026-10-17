"""Semantic segmentation method: encoder + decoder, Dice+CE, mIoU metric.

The method owns the criterion (:class:`~genml_kit.losses.segmentation.DiceCELoss`)
and the validation metric (mean IoU).  ``METRIC_KEY`` is ``miou``
(maximize), so the best-checkpoint value is stored under ``best_miou``.

The class count is inferred from the training masks during ``wire_data``
(the masks are already built by the pipeline), so the same CLI works for
any dataset without a separate ``--num_classes`` flag.  An explicit
``--seg_num_classes`` overrides inference when needed.
"""

import logging

import torch
from torchvision.transforms import v2

from genml_kit.datasets.transforms import (
    PairedSegTransform,
    build_seg_geometric,
    build_seg_photometric,
)
from genml_kit.losses import load_loss
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.training.seg_metrics import IoUMeter
from genml_kit.utils.logging import fatal


@METHODS.register
class SegmentationMethod(Method):
  """Supervised semantic segmentation (encoder + pluggable decoder)."""

  NAME = "segmentation"
  METRIC_KEY = "miou"
  METRIC_MINIMIZE = False
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self._num_classes = None
    self._id2label = None
    self._criterion = None
    self._ignore_index_raw = 255

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("segmentation method")
    group.add_argument("--seg_num_classes",
                       type=int,
                       default=None,
                       help="Override the inferred class count.")
    group.add_argument("--seg_ignore_index",
                       type=int,
                       default=255,
                       help="Mask label excluded from loss/metrics "
                       "(-1 disables).")
    group.add_argument("--seg_augmentation_script",
                       type=str,
                       default=None,
                       help="Path/URL to a script defining "
                       "create_seg_transform(image_size, is_train).")

  # -- Lifecycle hooks -------------------------------------------------------

  def prepare_transforms(self, args, device):
    """Build the paired train/val transforms from the model processor.

    Lifecycle position 1.  The processor supplies ``image_mean`` /
    ``image_std``; geometric ops are shared with the mask, photometric and
    normalization ops touch the image only.  Stored on the method (never
    in ``args``) for the pipeline to read.
    """
    from genml_kit.models import load_processor

    processor = load_processor(
        args.model,
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
    )
    self.train_transforms = self._build_transforms(args, processor, is_train=True)
    self.val_transforms = self._build_transforms(args, processor, is_train=False)

  def _build_transforms(self, args, processor, *, is_train):
    image_size = args.image_size
    if getattr(args, "seg_augmentation_script", None):
      # A user script owns the whole paired transform (image + mask); the
      # processor normalization remains the caller's responsibility in the
      # script.  Loaded lazily so a plain run never imports it.
      from genml_kit.utils.script import load_extern

      builder = load_extern(args.seg_augmentation_script, "create_seg_transform")
      return builder(image_size=image_size, is_train=is_train)

    geometric = build_seg_geometric(image_size, is_train=is_train)
    photometric = build_seg_photometric(image_size, is_train=is_train)
    normalize = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
    return PairedSegTransform(geometric=geometric,
                              photometric=photometric,
                              normalize=normalize)

  def wire_data(self, args, pipeline):
    """Infer the class count from the training masks and build the criterion.

    Lifecycle position 2.  The loaders exist, so we can scan the training
    masks for the maximum label id.
    """
    self._ignore_index_raw = getattr(args, "seg_ignore_index", 255)
    self._num_classes = self._resolve_num_classes(args, pipeline)
    self._id2label = {i: f"class_{i}" for i in range(self._num_classes)}
    self._criterion = self._build_criterion(args)

  def build_model(self, args, device):
    """Load the segmentation model (encoder + decoder) via the registry."""
    if self._num_classes is None:
      fatal(
          "SegmentationMethod.build_model requires wire_data(args, pipeline) "
          "to have inferred the class count; check the run order "
          "(wire_data before build_model).", RuntimeError)
    from genml_kit.models import load_model

    model = load_model(
        args.model,
        num_labels=self._num_classes,
        id2label=self._id2label,
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        task="segmentation",
        **getattr(args, "model_arg", {}),
    )
    return self._apply_model_extras(args, model, device)

  # -- Core steps ------------------------------------------------------------

  def train_step(self, model, blob, global_step, *, labels=None):
    images = blob.data
    mask = blob.meta.get("mask")
    if mask is None:
      mask = labels
    if mask is None:
      fatal("SegmentationMethod.train_step requires the mask in "
            "blob.meta['mask'].", ValueError)

    out = model(pixel_values=images)
    # Decoder logits (B, C, h, w) -> full-resolution (B, C, H, W) to match
    # the mask grid.
    logits = _match_mask_size(out.logits, mask)
    loss = self._criterion(logits, mask)
    with torch.no_grad():
      # Per-pixel class ids: (B, C, H, W) -> (B, H, W).
      preds = logits.argmax(dim=1)
      valid = mask != self._ignore_index()
      acc = (preds[valid] == mask[valid]).float().mean() if valid.any() else 0.0
    return LossOutput(loss=loss, metrics={"loss": loss.detach(), "pixel_acc": acc})

  def evaluate(self, model, loader, device, to_device):
    """Return ``{"miou", "mdice", "pixel_acc"}`` accumulated over *loader*."""
    meter = IoUMeter(self._num_classes, ignore_index=self._ignore_index())
    was_training = getattr(model, "training", False)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        out = model(pixel_values=blob.data)
        logits = _match_mask_size(out.logits, blob.meta["mask"])
        mask = blob.meta["mask"]
        total_loss += float(self._criterion(logits, mask))
        n_batches += 1
        meter.update(logits, mask)
    model.train(was_training)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics

  # -- Checkpointing ---------------------------------------------------------

  def get_checkpoint_state(self, model, args):
    return {
        "method": "segmentation",
        "num_classes": self._num_classes,
        "id2label": self._id2label,
    }

  def load_checkpoint_state(self, model, state, args):
    if state and state.get("num_classes"):
      self._num_classes = state["num_classes"]
      self._id2label = state.get("id2label", self._id2label)

  # -- Helpers ---------------------------------------------------------------

  def _ignore_index(self):
    """Return the CE ignore index (``-100`` when ignoring is disabled)."""
    return self._ignore_index_raw if self._ignore_index_raw >= 0 else -100

  def _build_criterion(self, args):
    """Construct the criterion from ``--loss`` / ``--loss_args``.

    Defaults to ``dice_ce`` for segmentation; ignore index follows
    ``--seg_ignore_index`` unless the user overrides it via ``--loss``.
    """
    spec = getattr(args, "loss", None) or "dice_ce"
    loss_args = dict(getattr(args, "loss_args", None) or {})
    loss_args.setdefault("ignore_index", self._ignore_index_raw)
    criterion = load_loss(spec, **loss_args)
    logging.info("SegmentationMethod criterion: %s %s", spec, loss_args)
    return criterion

  def _resolve_num_classes(self, args, pipeline):
    explicit = getattr(args, "seg_num_classes", None)
    if explicit:
      logging.info("SegmentationMethod: class count = %d (explicit)", explicit)
      return explicit
    inferred = self._infer_num_classes(pipeline)
    logging.info("SegmentationMethod: class count = %d (inferred)", inferred)
    return inferred

  def _infer_num_classes(self, pipeline):
    """Return the number of classes from the pipeline's mask map.

    The pipeline compacts raw mask ids (sparse / RGB / 1-based) into a
    dense ``[0, C)`` set during a full pre-scan, so ``C = len(map)``
    exactly.  When no map was built (e.g. the segfolder path), fall back
    to scanning a few loader masks for the maximum label id -- ignored
    pixels (``--seg_ignore_index``) are excluded so a ``255`` void label
    cannot inflate the count.
    """
    if getattr(pipeline, "mask_map", None):
      return len(pipeline.mask_map)

    loader = pipeline.train_loader
    if loader is None:
      fatal(
          "SegmentationMethod: pipeline.train_loader is None; call "
          "build_loader before wire_data.", RuntimeError)
    ignore = self._ignore_index_raw
    max_label = -1
    scanned = 0
    for blob in loader:
      masks = blob.meta["mask"]
      valid = masks[masks != ignore] if ignore >= 0 else masks
      if valid.numel():
        max_label = max(max_label, int(valid.max()))
      scanned += masks.shape[0]
      if scanned >= 64:
        break
    if max_label < 0:
      fatal("SegmentationMethod: could not infer classes from empty masks", ValueError)
    return max_label + 1


def _match_mask_size(logits, mask):
  """Bilinearly upsample *logits* to the mask's spatial size when they differ.

  HF segmentation models emit ``(B, C, H/4, W/4)`` logits; the loss and
  metric need full-resolution maps to match the mask.
  """
  if logits.shape[-2:] == mask.shape[-2:]:
    return logits
  return torch.nn.functional.interpolate(logits,
                                         size=mask.shape[-2:],
                                         mode="bilinear",
                                         align_corners=False)
