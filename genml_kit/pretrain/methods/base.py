"""Base class for self-supervised pre-training methods."""

from abc import ABC, abstractmethod


class PretrainMethod(ABC):
  """Interface that every pre-training method must implement.

    A ``PretrainMethod`` encapsulates the model construction, training step,
    loss computation, checkpointing, and optional validation logic for one
    self-supervised pre-training algorithm.  The generic harness in
    ``pretrain.py`` drives the training loop through this interface so that
    new methods can be added without modifying the loop.
    """

  NAME = ""  # CLI name (e.g. "simmim", "ijepa")
  needs_labels = False  # Set True for methods requiring labels.

  @abstractmethod
  def add_args(self, parser):
    """Add method-specific CLI arguments to *parser*."""
    ...

  @abstractmethod
  def build(self, args, encoder, device):
    """Create the pre-training model(s) from CLI arguments.

        Args:
          args: Parsed CLI arguments (including method-specific ones).
          encoder: The raw encoder backbone (e.g. ConvViTEncoder).
          device: Target device.

        Returns:
          A nn.Module whose forward is invoked during training.  The
          returned object may be a plain module or a wrapper that exposes
          additional methods (e.g. ``get_target`` for SimMIM).  The harness
          only calls :meth:`train_step` — it does not invoke ``forward``
          directly.
        """
    ...

  @abstractmethod
  def train_step(self, model, images, global_step, *, labels=None):
    """Compute loss for one batch.

        Args:
          model: Module returned by :meth:`build`.
          images: ``(B, C, H, W)`` batch.
          global_step: Current global training step.
          labels: Optional batch of integer class labels, required
              for label-aware methods (e.g. supervised contrastive).

        Returns:
          ``(loss, info)`` where *info* contains scalars to log.
        """
    ...

  @abstractmethod
  def get_checkpoint_state(self, model, args):
    """Extra state to persist in checkpoints.

        The returned dict is merged into the checkpoint alongside the
        standard model / optimizer / scheduler state.
        """
    ...

  @abstractmethod
  def load_checkpoint_state(self, model, state, args):
    """Restore method-specific state from a checkpoint."""
    ...

  def build_transform(self, image_size):
    """Return the augmentation transform for the training data.

        Override in subclasses that need custom augmentations (e.g.
        dual-view for BYOL, multi-crop for DINO).  The default wraps
        ``build_pretrain_transform`` from ``pretrain.py``.
        """
    from genml_kit.pretrain.cli import build_pretrain_transform
    return build_pretrain_transform(image_size)

  def validate(self, model, images, num_samples):  # noqa: B027
    """Generate validation visualizations.

        Called during validation and image logging.  Return a tensor of
        images ``(N, C, H, W)`` suitable for ``save_image``, or ``None``
        if the method does not produce pixel-space visualizations.

        Intentionally an optional hook (no abstract): the default no-op
        is correct for methods without pixel-space visualizations.
        """

  def on_epoch_end(self, model, epoch, writer):  # noqa: B027
    """Hook called at the end of each training epoch.

        Default is a no-op.  Override for e.g. teacher-momentum ramping.

        Intentionally an optional hook (no abstract): not every method
        needs end-of-epoch behavior.
        """
