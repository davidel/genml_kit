"""Streaming (iterable) dataset base: iterate once, split across workers.

``IterableDataset`` sources have no index space, so the things a map-style
dataset gets from its sampler have to be done inside ``__iter__`` itself.  The
one that cannot be skipped is the **worker split**: with ``num_workers > 0``
every worker process re-runs ``__iter__`` from the top, so a naive
implementation yields the *whole* corpus in *every* worker -- the same items
``num_workers`` times over.

:class:`StreamingDataset` handles that split once, so subclasses only describe
how to produce items:

```python
class MyStream(StreamingDataset):
  def source_iter(self):
    yield from rows           # no worker logic here

loader = DataLoader(MyStream(), num_workers=4)   # each row seen exactly once
```

The split must happen exactly **once**, at the leaf source.  A wrapper that wraps
another stream (selecting fields, filtering a split bucket, ...) inherits from
:class:`StreamingTransform` instead, which forwards iteration to its source and
never splits again -- splitting an already-split shard would keep only about
``1 / num_workers**2`` of the corpus.

Rank (DDP) sharding is deliberately *not* here: it happens one level up, on the
already-batched stream, so that this class stays composable with any batch size.
"""

from torch.utils.data import IterableDataset, get_worker_info


class StreamingDataset(IterableDataset):
  """Base class for one-pass sources that split themselves across workers.

    Subclasses implement :meth:`source_iter` and nothing else.  ``__iter__``
    applies the strided worker split (``i % num_workers == worker_id``), which
    keeps worker work balanced for any corpus size and never duplicates an item.

    A subclass may pass ``num_samples=`` to the constructor to declare an
    item-count *estimate* for progress output (see :meth:`num_samples`).
    """

  def __init__(self, num_samples=None):
    """Record an optional item-count estimate for progress output only.

        Args:
          num_samples: Optional *estimate* of the total item count.  Never
            load-bearing; ``None`` means unknown.  Kept private because
            ``num_samples`` is also the name of this class' estimate *method*.
        """
    self._num_samples = num_samples

  def num_samples(self):
    """Best-effort estimate of this source's total item count, or ``None``.

        Informational only: it feeds progress output and must never size a loop.
        Subclasses that can compute a better estimate override this.

        Returns:
          int | None: The estimated item count, or ``None`` when unknown.
        """
    return getattr(self, "_num_samples", None)

  def source_iter(self):
    """Yield this source's items in order, ignoring workers.

        Subclasses must override this.  It may be called more than once (once per
        epoch, and once per worker process), so it must not mutate shared state
        that would make a second pass differ.

        Yields:
          One dataset item.
        """
    raise NotImplementedError

  def set_epoch(self, epoch):
    """Record the epoch so a subclass can vary its stream between epochs.

        The base implementation stores the value on ``self.epoch``; sources that
        shuffle or sample per epoch read it.  Called through
        :meth:`genml_kit.pipelines.base.DataPipeline.set_train_epoch`, which
        reaches the dataset because an iterable source has no sampler to advance.

        Args:
          epoch: The epoch about to start.
        """
    self.epoch = epoch

  def worker_slice(self, index):
    """Return True when *index* belongs to the current worker.

        With ``num_workers == 0`` (or outside a worker process) everything
        belongs to the single consumer.

        Args:
          index: Zero-based position in :meth:`source_iter`.

        Returns:
          bool: True when this process should handle the item.
        """
    info = get_worker_info()
    if info is None or info.num_workers <= 1:
      return True
    return index % info.num_workers == info.id

  def __iter__(self):
    for index, item in enumerate(self.source_iter()):
      if self.worker_slice(index):
        yield item


class StreamingTransform(StreamingDataset):
  """Base class for streams that *transform* another stream.

    A transform forwards iteration to its source and applies **no** worker split
    of its own: the wrapped source already shards itself across workers (a
    :class:`StreamingDataset` via :meth:`StreamingDataset.worker_slice`, a
    HuggingFace ``IterableDataset`` via its own ``num_shards`` sharding).  Slicing
    again here would re-split an already-split shard, so each worker would keep
    only about ``1 / num_workers**2`` of the corpus.

    Subclasses implement :meth:`source_iter` exactly like :class:`StreamingDataset`
    and get the single-split behaviour for free.

    Attributes:
      _source: The wrapped stream.  Used by :meth:`set_epoch` to forward the
        epoch; subclasses must set it in ``__init__``.
  """

  @property
  def features(self):
    """The wrapped source's ``datasets.Features``, when it has one.

        A transform yields its source's rows unchanged in shape, so its schema
        is the source's schema.  Forwarding ``features`` lets feature-based
        column detection (image/label/``ClassLabel`` vocabulary) run on the
        transform without touching the corpus.
        """
    return getattr(self._source, "features", None)

  def num_samples(self):
    """Forward the wrapped source's estimate (a transform is count-preserving).

        A transform yields exactly the source's items, one per source item, so
        the source's estimate is the transform's estimate.  Subclasses that
        drop or fan out items (e.g. a positional split bucket) override this.

        Returns:
          int | None: The estimated item count, or ``None`` when unknown.
        """
    from genml_kit.datasets.base import num_samples

    return num_samples(self._source)

  @property
  def column_names(self):
    """The wrapped source's column names, when it declares them."""
    return getattr(self._source, "column_names", None)

  def __iter__(self):
    yield from self.source_iter()

  def set_epoch(self, epoch):
    """Record the epoch and forward it to the wrapped source.

        Parameters
        ----------
        epoch : int
            The epoch about to start.
        """
    super().set_epoch(epoch)
    # Forward so a source that reshuffles per epoch stays consistent.
    if hasattr(self._source, "set_epoch"):
      self._source.set_epoch(epoch)


def is_streaming_source(obj):
  """Return True when *obj* is a torch ``IterableDataset`` (no index space).

    One of three overlapping capability checks; see
    :func:`genml_kit.datasets.base.is_iterable` for how they relate.  This one
    is a strict ``isinstance`` check, so a duck-typed stream that is not a torch
    ``IterableDataset`` (e.g. a HuggingFace stream, or a plain wrapper) returns
    False -- use ``not is_random_access(obj)`` when any non-indexable source
    should count.

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True for a pure streaming source.
    """
  return isinstance(obj, IterableDataset)


def safe_len(loader, default=None):
  """Number of batches in *loader*, or *default* when it has no length.

    The batch-count counterpart to
    :func:`genml_kit.datasets.base.num_samples` (which estimates *samples*).
    It honours the same convention documented there: an iterable-only source
    deliberately raises ``TypeError`` from ``__len__`` to signal "no length",
    so callers must tolerate its absence rather than break on it.

    A plain ``len(loader)`` cannot be used for this.  ``DataLoader.__len__``
    delegates to ``len(self.dataset)``, so an ``IterableDataset``-backed
    loader -- every streaming corpus, and the ``StreamingBatchPipe`` batch
    pipes that wrap one -- raises ``TypeError`` mid-run instead of simply
    reporting an unknown length.  Use this helper anywhere a batch count feeds
    bookkeeping (a step budget, a progress total, a starting counter) that
    must not bring down a streaming run.

    Only ``TypeError`` is swallowed, because that is the one exception the
    "no length" convention raises, so the failure mode is narrow: a ``None``
    loader, or an object whose ``__len__`` raises something else, still
    propagates instead of being mistaken for "unknown".

    Args:
      loader: Any sized-or-unsized object, typically a ``DataLoader``.
      default: Value to return when *loader* has no usable length.  Callers
        that need an integer (a step counter, say) should pass e.g. ``0``;
        callers that want to distinguish "unknown" from a real value should
        leave it ``None``.

    Returns:
      int | None: ``len(loader)`` when available, else *default*.

    Examples:
      >>> safe_len(DataLoader(map_dataset, batch_size=4))       # 25
      >>> safe_len(DataLoader(iterable_dataset, batch_size=4))  # None
      >>> safe_len(DataLoader(iterable_dataset), default=0)     # 0
    """
  try:
    return len(loader)
  except TypeError:
    return default
