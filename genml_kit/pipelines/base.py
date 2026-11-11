"""Base class for all data pipelines."""

import abc
import logging
from itertools import islice

from genml_kit.datasets.base import is_random_access


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
    """Tell every epoch-aware layer of the train data path about *epoch*.

        Three places can hold epoch state, and each is told independently:

        1. **The loader wrapper.**  A streaming batch-round-robin wrapper
           reshuffles per epoch, and it is the loader itself rather than a
           sampler.
        2. **The dataset source.**  An ``IterableDataset`` has no sampler to
           call, so epoch control must reach the source; random-access datasets
           that support it are notified the same way.
        3. **The sampler chain.**  Delegates to
           :func:`genml_kit.utils.distributed.sampler_epoch`, which walks the
           loader's sampler wrapper chain rather than picking the first truthy
           one.  That matters under DDP: ``shard_loader`` installs a
           ``DistributedSampler`` as ``loader.sampler``, but ``DataLoader``
           wraps it in a plain ``BatchSampler``, which has no ``set_epoch`` --
           so a first-truthy lookup silently leaves the epoch pinned at 0.
        """
    loader = self.train_loader
    if loader is None:
      return
    # 1. Loader-level epoch state (streaming round-robin wrapper).
    if hasattr(loader, "set_epoch"):
      loader.set_epoch(epoch)
    # 2. Source-level epoch state (streaming sources; no sampler exists).
    seen = set()
    for obj in (getattr(self, "train_dataset", None), getattr(loader, "dataset", None)):
      if obj is None or id(obj) in seen:
        continue
      seen.add(id(obj))
      if hasattr(obj, "set_epoch"):
        obj.set_epoch(epoch)
    # 3. Sampler chain (random access).
    from genml_kit.utils.distributed import sampler_epoch
    sampler_epoch(loader, epoch)

  def seek_batch(self, batch_idx):
    """Position the train loader at batch *batch_idx* of the current epoch.

        One resume code path for both access models.  Rather than re-deriving a
        skip from a length, the trainer asks the pipeline to advance the loader
        itself, which lets each model seek in the cheapest sound way:

        * **Random access** -- an ``O(1)`` offset.  The loader's index sampler is
          wrapped so it starts after ``batch_idx * batch_size`` indices; no
          sample is ever decoded just to be discarded.  ``seek_batch(0)``
          restores the unwrapped sampler.
        * **Iterable** -- batches are consumed and dropped, which is only sound
          if the source can be replayed from a deterministic start.  On a source
          that cannot, a warning is logged and the epoch restarts from batch 0.

        Seeking is per rank: each rank seeks within its own shard, and the
        shuffle seed is a function of ``(seed, epoch)``, so all ranks land on the
        same position.

        Args:
          batch_idx: Batch index to start from (``grad_accum_steps``-aligned by
            the caller, so the first accumulation window after the seek is whole).

        Returns:
          int: The number of batches actually skipped.  Exact for random access;
            ``0`` for a stream that could not be replayed.
        """
    loader = self.train_loader
    if loader is None or batch_idx <= 0:
      self._restore_seek()
      self._stream_seek_offset = 0
      return 0
    # Only one mechanism may be armed at a time, or a previous seek would be
    # applied twice (offset on top of a sampler offset).
    self._stream_seek_offset = 0
    if (is_random_access(getattr(loader, "dataset", None)) and
        self._seek_index_sampler(loader, batch_idx)):
      return batch_idx
    # Iterable source: record the offset the loader iteration will honour.
    return self._seek_by_draining(loader, batch_idx)

  def _seek_index_sampler(self, loader, batch_idx):
    """Wrap the loader's index sampler to start after *batch_idx* batches.

        Returns False (having changed nothing) when the loader has no reachable
        index sampler -- e.g. an explicit ``batch_sampler`` whose ordering is not
        a simple stride -- so the caller can fall back to draining.
        """
    from genml_kit.utils.distributed import _SkippedSampler
    inner = getattr(getattr(loader, "batch_sampler", None), "sampler", None)
    if inner is None:
      return False
    base = getattr(self, "_seek_base_sampler", None)
    if base is None:
      # Remember the pristine sampler once, so repeated/zero seeks are clean.
      self._seek_base_sampler = inner
      base = inner
    skip = batch_idx * (loader.batch_size or 1)
    batch_sampler = loader.batch_sampler
    batch_sampler.sampler = _SkippedSampler(base, skip)
    return True

  def _restore_seek(self):
    """Undo a previous :meth:`_seek_index_sampler` (no-op when never sought)."""
    base = getattr(self, "_seek_base_sampler", None)
    if base is None:
      return
    loader = getattr(self, "train_loader", None)
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
      batch_sampler.sampler = base
    self._seek_base_sampler = None

  def _seek_by_draining(self, loader, batch_idx):
    """Record a batch offset for an iterable loader, or refuse to seek.

        Draining cannot work here: ``iter(loader)`` builds a *fresh* iterator, so
        batches consumed to skip would be discarded while the loop afterwards
        still starts at batch 0.  Instead the offset is recorded and applied by
        :meth:`_iter_train_loader`, which is what the trainer iterates.

        Seeking still requires a source that replays from a deterministic start,
        because the skipped prefix is re-generated.  A source that cannot replay
        is not silently mis-resumed -- the position would be wrong while the logs
        claimed otherwise -- so the epoch restarts from the top and says so.
        """
    dataset = getattr(loader, "dataset", None)
    if not getattr(dataset, "is_replayable", True):
      logging.warning("Resume within a streamed epoch restarts the epoch from the top: "
                      "this source cannot be replayed from a deterministic start.")
      self._stream_seek_offset = 0
      return 0
    self._stream_seek_offset = batch_idx
    return batch_idx

  def iter_train_loader(self):
    """Iterate the train loader honouring any pending streaming seek.

        The single place the trainer pulls batches from, so a recorded seek offset
        is applied to the *same* iterator the loop consumes.  Random access needs
        no help here (see :meth:`_seek_index_sampler`), so the offset stays 0.

        Yields:
          Data blobs, in epoch order, positioned after any pending seek.
        """
    loader = self.train_loader
    offset = getattr(self, "_stream_seek_offset", 0)
    if not offset:
      yield from loader
      return
    yield from islice(iter(loader), offset, None)
