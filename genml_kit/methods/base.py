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
