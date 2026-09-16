"""Base class for all training methods (supervised + self-supervised)."""

import abc
import collections

import torch


class Method(abc.ABC):
  """Objective side of training: model + loss + metric.

  ``PretrainMethod`` reshapes into this: a method builds the model,
  computes the loss from (model, blob, global_step), and declares the
  metric contract -- for supervised (classification, VO) and self-supervised
  (SimMIM, SupCon, DINO, BYOL, IJEPA, VO photometric) alike.
  """

  NAME = ""  # registry key (matches PretrainMethod.NAME today)

  metric_key = "loss"  # best-checkpoint metric key (best_<metric_key>)

  NEEDS_LABELS = False  # whether this method requires labels in the data blob

  @abc.abstractmethod
  def build_model(self, args, device):
    """Return the model.  Model's forward(data) takes the blob.data shape."""

  @abc.abstractmethod
  def train_step(self, model, blob, global_step, *, labels=None):
    """Compute the loss for one (already device-moved) blob.

    Returns ``LossOutput(loss, metrics)``.  The method receives BOTH the
    model and global_step because existing methods perform per-step
    side effects inside their old ``train_step``:

      BYOL   set_train_mode(model, "train"); model.update_momentum(...)
      DINO   set_train_mode(model, "train"); model.update_momentum(...)
      IJEPA  set_train_mode(model, "train")
      SimMIM builds its mask from blob.data inside train_step

    ``labels`` is passed for backward-compat with today's keyword-call
    convention; the images-pipeline usually delivers labels via blob.meta.
    The loop owns AMP / grad-accum / clipping around this call.
    """

  def get_checkpoint_state(self, model, args):
    """Optional: dict of method-owned state persisted to every checkpoint.

    Carried over verbatim from PretrainMethod (BYOL momentum bounds, DINO
    center/EMA bounds, SimMIM mask/decoder config, IJEPA momentum).
    Called by BaseTrainer while saving; merged into the checkpoint dict.
    """
    return {}

  def load_checkpoint_state(self, model, state, args):  # noqa: B027
    """Optional: restore method-owned state from a checkpoint dict."""

  def evaluate(self, model, loader, device, to_device):
    """Return dict[str, float] of validation metrics (default: loss mean).

    Default implementation iterates *loader*, moves each blob with the
    pipeline's ``to_device`` callback, calls ``train_step`` under
    ``torch.no_grad()`` and returns the mean of each metric key.  Override
    for objective-specific metrics: VO (mce), classification (macro-F1 /
    confusion matrix).
    """
    metrics_acc = collections.defaultdict(float)
    n = 0
    with torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        out = self.train_step(model, blob, 0)
        for k, v in out.metrics.items():
          metrics_acc[k] += float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in metrics_acc.items()}

  def has_metric_improved(self, new_metric, best_metric):
    """Return True when *new_metric* beats *best_metric* (higher is better).

    Direction lives ONLY here; the loop never negates.  Override for
    minimize-direction metrics (e.g. VO mce: ``new < best``).
    """
    return new_metric > best_metric

  def add_args(self, parser):  # noqa: B027
    """Add this method's CLI flags.  Instance method (no-op by default)."""

  def build_transform(self, args, image_size):
    """Return the objective-level augmentation for this method.

    Default: the pipeline's generic image preprocessing (resize, center
    crop, flips, float32 scaling) -- the historic default for methods
    without an objective augmentation.  Objective augmentations (DualView
    / MultiCrop) override this and compose on top.
    """
    from genml_kit.pipelines.images import build_pretrain_transform
    return build_pretrain_transform(image_size)

  def on_epoch_end(self, model, epoch, writer):  # noqa: B027
    """Hook called at the end of each training epoch (optional)."""

  # --- Lifecycle hooks (called by the genml-kit-train driver, in order) ------

  def prepare_transforms(self, args, device):  # noqa: B027
    """Resolve any transforms this method needs BEFORE loaders are built.

    Lifecycle position 1: called by the driver after parsing and seeding,
    before ``pipeline.build_loader``.  The classification data path reads
    ``args.train_transforms`` / ``args.val_transforms`` / ``args.tta_transform``
    while building its loaders, so a method whose preprocessing depends on a
    model processor must set them here.  Default: no-op.
    """

  def wire_data(self, args, pipeline):  # noqa: B027
    """Consume pipeline-built data attributes (labels, weights).

    Lifecycle position 2: called after both loaders are built and before
    ``build_model``.  Override to read ``pipeline.num_labels`` /
    ``pipeline.class_weights`` / ``pipeline.id2label`` and construct
    criteria or label mappings.  Default: no-op.
    """

  def post_train(self, args, pipeline, device, result):  # noqa: B027
    """Optional post-training stage (e.g. shallow-head probing).

    Lifecycle position 4: called by the driver after ``BaseTrainer.run()``
    returns, unless the run was interrupted (the driver owns the interrupt
    guard).  ``result`` is the ``TrainingResult`` of the run.  Default: no-op.
    """

  # --- Model post-construction (called by build_model implementations) -------

  def _apply_model_extras(self, args, model, device):
    """Apply grad checkpointing, LoRA / source weights, freeze patterns.

    Called by ``build_model`` implementations AFTER the raw model is
    constructed and moved to *device*.  Every method gets the same
    treatment, which is what makes ``--lora``, ``--source_checkpoint``,
    ``--freeze`` and ``--grad_checkpoint`` work for every objective, not
    just classification.  Returns the (possibly wrapped) model.

    A method whose outer object is a composite (EMA teacher copies, masked
    wrappers) should apply extras to the module that PEFT/freeze must
    target -- typically the student-side backbone -- and build the composite
    afterwards; see DINO/BYOL/IJEPA for the pattern.
    """
    if getattr(args, "grad_checkpoint", False):
      from genml_kit.training.model_utils import enable_grad_checkpointing
      enable_grad_checkpointing(model)
    if getattr(args, "lora", False):
      from genml_kit.training.model_utils import apply_lora
      if args.lora_target_modules:
        target_modules = args.lora_target_modules.split(",")
      else:
        # PEFT raises when target_modules is None or matches nothing, so the
        # default must intersect the standard transformer projection names
        # with what the model actually has (ViT backbones use q_proj/k_proj/
        # v_proj; timm ViT uses qkv; HF attention uses query/key/value).
        # Models matching none of the standard names (e.g. the fc-only test
        # backbone) fall back to adapting every nn.Linear.
        default_names = {"q_proj", "k_proj", "v_proj", "qkv", "query", "key", "value"}
        present = {name.split(".")[-1]
                   for name, mod in model.named_modules()
                   if isinstance(mod, torch.nn.Linear)}
        matched = sorted(default_names & present)
        target_modules = matched or [n for n, m in model.named_modules()
                                     if isinstance(m, torch.nn.Linear)]
      model = apply_lora(
          model,
          r=args.lora_r,
          alpha=args.lora_alpha,
          dropout=args.lora_dropout,
          target_modules=target_modules,
      )
    elif getattr(args, "source_checkpoint", None):
      from genml_kit.io.checkpointing import load_checkpoint_weights
      load_checkpoint_weights(args.source_checkpoint,
                              model,
                              device=device,
                              param_rename=args.param_rename)
    from genml_kit.training.train_compat import apply_freeze_patterns
    apply_freeze_patterns(args, model)
    return model
