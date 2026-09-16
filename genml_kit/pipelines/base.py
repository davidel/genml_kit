"""Base class for all data pipelines."""

import abc


class DataPipeline(abc.ABC):  # noqa: B024
  """Data side of training: loader + blob contract + device transfer.

  Registered via @register_pipeline, constructed via build_pipeline(name).
  Does NOT build models or compute losses (that is Method's job).  The
  concrete pipeline caches its loaders on first build so the loop and the
  trainer can reference ``self.pipeline.train_loader`` / ``val_loader``
  without extra plumbing.
  """

  NAME = ""  # registry key (matches Method.NAME convention)

  def __init__(self, **kwargs):
    self.train_loader = None
    self.val_loader = None

  # --- Arg surface --------------------------------------------------------

  def add_args(self, parser):  # noqa: B027
    """Add this pipeline's CLI flags.  No-op by default."""

  # --- Loading ------------------------------------------------------------

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    """Build and cache the loader for *mode* (train/val).

    Args:
        needs_labels: If True, the pipeline must ensure labels are present in
            the data. If False, labels are stripped. If None (default), the
            pipeline decides based on its own logic (backward compat).

    Raises:
        NotImplementedError: until the concrete pipeline implements it.
    """
    raise NotImplementedError

  # --- Device transfer ----------------------------------------------------

  def to_device(self, blob, device):
    """Move a DataBlob (data AND meta) onto *device*."""
    raise NotImplementedError

  # --- Transforms ---------------------------------------------------------

  def build_transform(self, args, method):
    """Optional generic preprocessing (default: None = no pipeline phase).

    Default returns None so that only the method's own augmentation (or the
    historic fallback) is used.  A pipeline MAY override this to apply its
    generic preprocessing (resize/normalize/crop-flip-jitter); it is composed
    with the METHOD's objective augmentation (DualView / MultiCrop) -- the
    "two-phase contract" in the plan (s 6.1).  Returns a callable applied to
    each raw item before it becomes a DataBlob, or None.
    """
    return None
