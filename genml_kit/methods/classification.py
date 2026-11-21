"""Supervised classification method: model + loss + macro-F1 metric."""

import gc
import json
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput, ValidationResult
from genml_kit.losses import load_loss
from genml_kit.training.labels import mixup_data
from genml_kit.training.model_utils import model_mode
from genml_kit.utils.logging import fatal


@METHODS.register
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
    group.add_argument("--train_augmentation_script",
                       type=str,
                       help="Path/URL to a script defining create_train_transform().")
    group.add_argument("--tta",
                       type=str,
                       help="Test-Time Augmentation ('default' or a script path/URL).")
    group.add_argument("--freeze",
                       type=str,
                       help="Comma-separated regex patterns of parameter names to keep "
                       "trainable; all others freeze.")
    group.add_argument("--sampler_weights",
                       type=str,
                       default="frequency",
                       choices=("frequency", "multipliers", "combined"),
                       help="Weight mode for the classification sampler.")
    group.add_argument("--xgboost_model",
                       type=str,
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
        cache_dir=args.cache_dir,
        device=device,
        **args.model_arg,
    )
    return self._apply_model_extras(args, model, device)

  def set_label_space(self, num_labels, id2label, label2id):
    """Call after the pipeline is built; before model construction."""
    self._num_labels = num_labels
    self._id2label = id2label
    self._label2id = label2id

  def prepare_transforms(self, args, device):
    """Resolve train/val/TTA transforms from the model processor.

    The HF processor owns the normalization (image_mean/std) the
    classification data path must use, so this runs before the loaders
    are built (lifecycle position 1).  Transforms are stored on the
    method -- never in ``args`` -- and read back by the pipeline.
    """
    from genml_kit.models import load_processor
    from genml_kit.training.train_compat import resolve_augmentations

    processor = load_processor(
        args.model,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )
    bundle = resolve_augmentations(args, processor)
    self.train_transforms = bundle.train
    self.val_transforms = bundle.val
    self.tta_transform = bundle.tta

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
    self.set_mixup_alpha(args.mixup_alpha)
    class_weights = pipeline.class_weights
    if class_weights is None:
      class_weights = torch.ones(pipeline.num_labels, dtype=torch.float32)
    self.build_criterion(args, class_weights)

  def post_train(self, args, pipeline, device, result):
    """Post-training stages: XGBoost head and/or temperature calibration.

    No-op unless --xgboost_model is set (maybe_train_xgboost owns that
    check) or --fit_temperature is set.  The driver's interrupt guard runs
    before this hook.
    """
    from genml_kit.training.train_compat import maybe_train_xgboost

    maybe_train_xgboost(args, pipeline, device)
    if args.fit_temperature:
      self._fit_and_save_temperature(args, pipeline, device)

  def _fit_and_save_temperature(self, args, pipeline, device):
    """Fit T on the validation set of the *best* checkpoint and persist it.

    Mirrors the XGBoost post-train flow: reload the best checkpoint into a
    fresh model, rebuild the val loader with val transforms, collect logits,
    fit T, then free the model.
    """
    from genml_kit.datasets.hf_proxy import HFDatasetProxy
    from genml_kit.io.checkpointing import select_best_checkpoint
    from genml_kit.training.calibration import fit_temperature
    from genml_kit.training.model_utils import (
        build_val_transform,
        load_model_for_inference,
    )

    checkpoint_dir = args.checkpoint or ""
    ckpt_path = select_best_checkpoint(checkpoint_dir)
    if ckpt_path is None:
      logging.warning("--fit_temperature: no checkpoint found under %s; "
                      "skipping.", checkpoint_dir)
      return
    loaded = load_model_for_inference(
        args.model,
        ckpt_path,
        device=device,
        cache_dir=args.cache_dir,
        num_labels=pipeline.num_labels,
        image_size=args.image_size,
        proc_kwargs=args.proc_arg,
    )
    model = loaded.model
    val_transform = build_val_transform(loaded.processor, args.image_size)
    val_proxy = HFDatasetProxy(pipeline.val_proxy.dataset, transform=val_transform)
    logits, labels = [], []
    with model_mode(model, "eval"), torch.no_grad():
      for pixel_values, batch_labels in DataLoader(val_proxy,
                                                   batch_size=64,
                                                   shuffle=False):
        out = model(pixel_values=pixel_values.to(device))
        logits.append(out.logits.cpu())
        labels.append(batch_labels)
    temperature = fit_temperature(torch.cat(logits), torch.cat(labels))
    out_path = os.path.join(os.path.dirname(checkpoint_dir) or ".",
                            "temperature.json")
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump({"temperature": temperature}, f, indent=2)
    logging.info("Fitted temperature %.4f -> %s", temperature, out_path)
    del model
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
      torch.cuda.empty_cache()

  def train_step(self, model, blob, global_step, *, labels=None):
    images = blob.data
    targets = blob.meta.get("labels", None)
    if targets is None and labels is not None:
      targets = labels
    if targets is None:
      fatal(
          "ClassificationMethod.train_step requires labels in "
          "blob.meta (or via the labels keyword).", ValueError)

    use_mixup = self._mixup_alpha() > 0 and images.size(0) >= 2
    if use_mixup:
      mixup = mixup_data(images, targets, alpha=self._mixup_alpha())
      images, targets_a, targets_b, lam = (mixup.mixed_x, mixup.y_a, mixup.y_b,
                                           mixup.lam)
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
    return LossOutput(loss=loss,
                      metrics={
                          "top1": top1.detach(),
                          "loss": loss.detach()
                      },
                      logits=logits.detach())

  def _mixup_alpha(self):
    return getattr(self, "_mixup", 0.0)

  def set_mixup_alpha(self, alpha):
    self._mixup = float(alpha)

  def build_criterion(self, args, class_weights=None):
    """Construct the classifier criterion from ``--loss`` / ``--loss_args``.

    ``--loss`` may be a registered name (e.g. ``"focal"``) or a path/URL
    to an external ``.py`` script defining ``build_loss(**kwargs)``; when
    ``None`` the built-in ``"focal"`` is used.  ``--loss_args`` is a JSON
    dict forwarded to the builder; the label space weight tensor
    (``class_weights``) is always injected unless the spec already
    carries a ``weights`` key.
    """
    kwargs = dict(args.loss_args or {})
    if class_weights is not None:
      kwargs.setdefault("weights", class_weights)
    spec = args.loss or "focal"
    self._criterion = load_loss(spec, **kwargs)
    logging.info("Criterion: %s", self._criterion)
    return self._criterion

  def evaluate(self, model, loader, device, to_device):
    """Return this rank's ``macro_f1`` / ``top1`` as a ``ValidationResult``.

        macro-F1 is not additive: it cannot be recovered from per-rank means, so
        the raw prediction / target lists ride along in ``accumulators`` and
        ``merge_validation`` recomputes the metric over the concatenation.
        Raw logits also ride along so the merged report can derive
        calibration-aware metrics (ECE, AUROC) without a second pass.
        """
    import torch

    all_preds, all_targets, all_logits = [], [], []
    with model_mode(model, "eval"), torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        images = blob.data
        targets = blob.meta["labels"]
        logits = model(pixel_values=images).logits
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
        all_logits.append(logits.cpu())

    metrics = self._classification_metrics(all_preds, all_targets)
    return ValidationResult(metrics, float(len(all_targets)),
                            (all_preds, all_targets, torch.cat(all_logits)))

  def _classification_metrics(self, all_preds, all_targets):
    """Compute macro-F1 and top1 from flat prediction / target lists."""
    from sklearn.metrics import f1_score

    labels = list(range(self._num_labels or 0))
    macro_f1 = f1_score(
        all_targets, all_preds, labels=labels, average="macro", zero_division=0) * 100.0
    top1 = (sum(p == t for p, t in zip(all_preds, all_targets)) /
            max(len(all_targets), 1)) * 100.0
    return {"macro_f1": macro_f1, "top1": top1}

  def merge_validation(self, results):
    """Recompute macro-F1 / top1 over every rank's predictions.

        When ranks carry logits (recent evaluate), the merged report also
        derives ECE and macro-OvR AUROC; older rank results without logits
        skip those two metrics.
        """
    preds, targets, logits = [], [], []
    for r in results:
      preds.extend(r.accumulators[0])
      targets.extend(r.accumulators[1])
      if len(r.accumulators) > 2:
        logits.append(r.accumulators[2])

    metrics = self._classification_metrics(preds, targets)
    if logits:
      from genml_kit.training.calibration import (
          auroc_macro_ovr,
          expected_calibration_error,
      )

      all_logits = torch.cat(logits)
      metrics["ece"] = expected_calibration_error(all_logits, targets)
      auroc = auroc_macro_ovr(all_logits, targets, num_labels=self._num_labels)
      if auroc is not None:
        metrics["auroc"] = auroc
    return metrics

  def has_metric_improved(self, new_metric, best_metric):
    return new_metric > best_metric

  def get_checkpoint_state(self, model, args):
    return {"method": "classification", "num_labels": self._num_labels}
