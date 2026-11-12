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

    Attributes:
      nsamples: Optional *estimate* of the total item count, for progress output
        only.  Never load-bearing; ``None`` means unknown.
    """

  #: Optional estimate of the total item count.  Informational only.
  nsamples = None

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

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True for a pure streaming source.
    """
  return isinstance(obj, IterableDataset)
