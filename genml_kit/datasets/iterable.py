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


def is_streaming_source(obj):
  """Return True when *obj* is a torch ``IterableDataset`` (no index space).

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True for a pure streaming source.
    """
  return isinstance(obj, IterableDataset)
