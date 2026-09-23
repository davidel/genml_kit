"""Supervised classification method: model + loss + macro-F1 metric."""

import logging

import torch
import torch.nn.functional as F

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.losses.focal import CombinedFocalLoss
from genml_kit.training.labels import mixup_data
from genml_kit.training.model_utils import model_mode
from genml_kit.utils.logging import fatal


@register_method
class ClassificationMethod(Method):
  """Supervised image classification (HuggingFace / timm models).

  Owns the criterion (:class:`CombinedFocalLoss`) and the validation
  metric (macro-F1).  ``metric_key`` is ``macro_f1`` (maximize), so the
  best-checkpoint value is stored under ``best_macro_f1``.
  """

  NAME = "classification"
  METRIC_KEY = "macro_f1"
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self._num_labels = None
    self._id2label = None
    self._label2id = None
    self._criterion = None

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("classification method")
    group.add_argument("--mixup_alpha",
                       type=float,
                       default=0.0,
                       help="Mixup interpolation strength (0 disables).")
    group.add_argument("--focal_gamma",
                       type=float,
                       default=0.0,
                       help="Focal loss gamma (0 disables focal modulation).")
    group.add_argument("--label_smoothing",
                       type=float,
                       default=0.0,
                       help="Label smoothing for the focal loss.")
    group.add_argument("--train_augmentation_script",
                       type=str,
                       default=None,
                       help="Path/URL to a script defining create_train_transform().")
    group.add_argument("--tta",
                       type=str,
                       default=None,
                       help="Test-Time Augmentation ('default' or a script path/URL).")
    group.add_argument("--freeze",
                       type=str,
                       default=None,
                       help="Comma-separated regex patterns of parameter names to keep "
                       "trainable; all others freeze.")
    group.add_argument("--sampler_weights",
                       type=str,
                       default="frequency",
                       choices=("frequency", "multipliers", "combined"),
                       help="Weight mode for the classification sampler.")
    group.add_argument("--xgboost_model",
                       type=str,
                       default=None,
                       help="Path to an XGBoost model to train after the NN stage.")
    group.add_argument("--xgb_use_gpu",
                       action="store_true",
                       default=False,
                       help="Use GPU for the XGBoost stage.")
    group.add_argument("--xgb_max_depth", type=int, default=6)
    group.add_argument("--xgb_n_estimators", type=int, default=300)
    group.add_argument("--xgb_learning_rate", type=float, default=0.1)
    group.add_argument("--xgb_subsample", type=float, default=1.0)
    group.add_argument("--xgb_colsample_bytree", type=float, default=1.0)
    group.add_argument("--xgb_min_child_weight", type=int, default=1)
    group.add_argument("--xgb_gamma", type=float, default=0.0)
    group.add_argument("--xgb_reg_alpha", type=float, default=0.0)

  def build_model(self, args, device):
    """Load the classifier via the model registry and finalize it.

    Requires :meth:`wire_data` to have run (the HF head is constructed
    from the pipeline's label space); fails fast with an explicit message
    otherwise instead of a confusing HuggingFace error.
    """
    if self._num_labels is None:
      fatal(
          "ClassificationMethod.build_model requires wire_data(args, "
          "pipeline) to have provided the label space; check the run "
          "order (wire_data before build_model).", RuntimeError)
    from genml_kit.models import load_model
    model = load_model(
        args.model,
        num_labels=self._num_labels,
        id2label=self._id2label,
        label2id=self._label2id,
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        **getattr(args, "model_arg", {}),
    )
    return self._apply_model_extras(args, model, device)

  def set_label_space(self, num_labels, id2label, label2id):
    """Call after the pipeline is built; before model construction."""
    self._num_labels = num_labels
    self._id2label = id2label
    self._label2id = label2id

  def prepare_transforms(self, args, device):
    """Load the model processor and resolve train/val/TTA transforms.

    The HF processor owns the normalization (image_mean/std) the
    classification data path must use, so this runs before the loaders
    are built (lifecycle position 1).
    """
    from genml_kit.models import load_processor
    from genml_kit.training.train_compat import resolve_augmentations
    processor = load_processor(
        args.model,
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
    )
    train_t, val_t, tta_t = resolve_augmentations(args, processor)
    args.train_transforms = train_t
    args.val_transforms = val_t
    args.tta_transform = tta_t

  def wire_data(self, args, pipeline):
    """Read the pipeline's label space and weights; build the criterion.

    Lifecycle position 2: the loaders exist, so `pipeline.num_labels` /
    `pipeline.class_weights` / `pipeline.id2label` are final.  Must run
    before :meth:`build_model` (the HF head is sized from the label
    space).
    """
    if pipeline.num_labels is None:
      fatal("--method classification requires a labeled --dataset.", ValueError)
    self.set_label_space(pipeline.num_labels, pipeline.id2label, pipeline.label2id)
    self.set_mixup_alpha(getattr(args, "mixup_alpha", 0.0))
    class_weights = pipeline.class_weights
    if class_weights is None:
      class_weights = torch.ones(pipeline.num_labels, dtype=torch.float32)
    self.build_criterion(args, class_weights)

  def post_train(self, args, pipeline, device, result):
    """Train the XGBoost head on frozen embeddings after a successful run.

    No-op unless --xgboost_model is set (maybe_train_xgboost owns that
    check).  The driver's interrupt guard runs before this hook.
    """
    from genml_kit.training.train_compat import maybe_train_xgboost
    maybe_train_xgboost(args, pipeline, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    images = blob.data
    targets = blob.meta.get("labels", None)
    if targets is None and labels is not None:
      targets = labels
    if targets is None:
      raise ValueError("ClassificationMethod.train_step requires labels in "
                       "blob.meta (or via the labels keyword).")

    use_mixup = self._mixup_alpha() > 0 and images.size(0) >= 2
    if use_mixup:
      images, targets_a, targets_b, lam = mixup_data(images,
                                                     targets,
                                                     alpha=self._mixup_alpha())
    else:
      targets_a = targets

    out = model(pixel_values=images)
    logits = out.logits

    if use_mixup:
      soft_targets = (lam * F.one_hot(targets_a, logits.size(-1)).float() +
                      (1.0 - lam) * F.one_hot(targets_b, logits.size(-1)).float())
      loss = self._criterion(logits, soft_targets)
      # Dominant label (mixup_data clamps lam>=0.5).
      hard_targets = targets_a
    else:
      loss = self._criterion(logits, targets)
      hard_targets = targets

    top1 = (logits.argmax(dim=1) == hard_targets).float().mean()
    return LossOutput(loss=loss, metrics={"top1": top1.detach(), "loss": loss.detach()})

  def _mixup_alpha(self):
    return getattr(self, "_mixup", 0.0)

  def set_mixup_alpha(self, alpha):
    self._mixup = float(alpha)

  def build_criterion(self, args, class_weights=None):
    """Construct the classifier criterion from parsed args."""
    if args.focal_gamma > 0 and args.label_smoothing > 0:
      logging.warning(
          "Both --focal_gamma (%.1f) and --label_smoothing (%.2f) are > 0. "
          "Focal loss and label smoothing conflict. Proceeding anyway -- "
          "monitor for instability.",
          args.focal_gamma,
          args.label_smoothing,
      )
    self._criterion = CombinedFocalLoss(
        weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    return self._criterion

  def evaluate(self, model, loader, device, to_device):
    """Return dict with macro_f1 (and top1) over *loader* (DataBlob items)."""
    import torch
    from sklearn.metrics import f1_score

    all_preds, all_targets = [], []
    with model_mode(model, "eval"), torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        images = blob.data
        targets = blob.meta["labels"]
        logits = model(pixel_values=images).logits
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    labels = list(range(self._num_labels or 0))
    macro_f1 = f1_score(
        all_targets, all_preds, labels=labels, average="macro", zero_division=0) * 100.0
    top1 = (sum(p == t for p, t in zip(all_preds, all_targets)) /
            max(len(all_targets), 1)) * 100.0
    return {"macro_f1": macro_f1, "top1": top1}

  def has_metric_improved(self, new_metric, best_metric):
    return new_metric > best_metric

  def get_checkpoint_state(self, model, args):
    return {"method": "classification", "num_labels": self._num_labels}
