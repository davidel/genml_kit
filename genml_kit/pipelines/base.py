"""Base class for all data pipelines."""

import abc


class DataPipeline(abc.ABC):  # noqa: B024
  """Data side of training: loader + blob contract + device transfer.

    Registered via @PIPELINES.register, constructed via build_pipeline(name).
    Does NOT build models or compute losses (that is Method's job).  The
    concrete pipeline caches its loaders on first build so the loop and the
    trainer can reference ``self.pipeline.train_loader`` / ``val_loader``
    without extra plumbing.
    """

  # Registry key (matches Method.NAME convention).
  NAME = ""

  # Throughput label printed by the trainer (e.g. "img" -> "img/s").
  # The pipeline is the authority on what one unit of its data is.
  THROUGHPUT_UNIT = "sample"

  def __init__(self, **kwargs):
    self.train_loader = None
    self.val_loader = None

  @classmethod  # noqa: B027
  def add_args(cls, parser):
    """Add this pipeline's CLI flags.  No-op by default."""

  @classmethod
  def arg_defaults(cls):
    """Return ``{dest: value}`` CLI defaults this pipeline wants.

        Applied between argument registration and parsing, so an explicit
        ``--flag`` on the command line always wins.  A dict value is merged
        per key over the flag's current default.  No-op by default.
        """
    return {}

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

  def to_device(self, blob, device):
    """Move a DataBlob (data AND meta) onto *device*."""
    raise NotImplementedError

  def build_transform(self, args, method):  # noqa: B027
    """Optional generic preprocessing (default: None = no pipeline phase).

        Default returns None so that only the method's own augmentation (or the
        historic fallback) is used.  A pipeline MAY override this to apply its
        generic preprocessing (resize/normalize/crop-flip-jitter); it is composed
        with the METHOD's objective augmentation (DualView / MultiCrop) -- the
        "two-phase contract" in the plan (s 6.1).  Returns a callable applied to
        each raw item before it becomes a DataBlob, or None.
        """

  def init_env(self, args):  # noqa: B027
    """Initialize environment-specific state (e.g., Gymnasium envs for RL).

    Default is a no-op.  RL pipelines override this to create the
    environment and any wrappers/buffers before ``wire_data`` runs.
    """
    pass

  def shard_train_loader(self, args):
    """Shard the training set across ranks (no-op when single-process).

        Pipelines call this after building ``self.train_loader``.  Because
        ``torch`` forbids mutating a ``DataLoader``'s sampler after
        construction, this *rebuilds* the loader with a rank-sharded sampler:
        a loader with a plain index sampler gets a ``DistributedSampler``, and
        one driven by a ``batch_sampler`` (balanced / length-bucketed) gets a
        rank-strided slice of that batch stream.  When ``--dist_arg shard=false``
        is given, or the run is single-process, this does nothing, so the
        single-device loader is bit-for-bit unchanged.

        Args:
            args: Parsed CLI args (uses ``dist_arg["shard"]``).
        """
    from genml_kit.utils.distributed import shard_loader, world_size
    # Default: shard whenever more than one rank is present.  Every rank
    # iterating the full dataset is never what a multi-GPU run wants, and it
    # fails silently (no error, just N times the compute), so the default is
    # derived from the rank count rather than left to an opt-in flag.
    if not args.dist_arg.get("shard", world_size() > 1) or world_size() < 2:
      return
    if getattr(self, "train_loader", None) is not None:
      self.train_loader = shard_loader(self.train_loader, seed=args.seed or 0)

  def shard_val_loader(self, args):
    """Shard the validation set across ranks (no-op when single-process).

        Mirror of :meth:`shard_train_loader`.  The val loader is built with
        ``shuffle=False`` (a ``SequentialSampler``), so ``shard_loader`` installs
        a non-shuffling ``DistributedSampler`` and every rank sees a disjoint,
        order-stable slice.  This is only sound together with a cross-rank merge
        (``BaseTrainer.validate`` gathers the per-rank results and calls
        ``Method.merge_validation``): a single rank now sees just its 1/N slice,
        so its metrics alone would no longer describe the whole val set.

        Args:
            args: Parsed CLI args (uses ``dist_arg["shard"]``).
        """
    from genml_kit.utils.distributed import shard_loader, world_size
    if not args.dist_arg.get("shard", world_size() > 1) or world_size() < 2:
      return
    if getattr(self, "val_loader", None) is not None:
      self.val_loader = shard_loader(self.val_loader, seed=args.seed or 0)

  def set_train_epoch(self, epoch):
    """Advance every per-epoch sampler on the train loader.

        Delegates to :func:`genml_kit.utils.distributed.sampler_epoch`, which
        walks the loader's sampler wrapper chain rather than picking the first
        truthy one.  That matters under DDP: ``shard_loader`` installs a
        ``DistributedSampler`` as ``loader.sampler``, but ``DataLoader`` wraps it
        in a plain ``BatchSampler``, which has no ``set_epoch`` -- so a
        first-truthy lookup silently leaves the epoch pinned at 0.
        """
    loader = self.train_loader
    if loader is None:
      return
    from genml_kit.utils.distributed import sampler_epoch
    sampler_epoch(loader, epoch)
