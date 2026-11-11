"""Streaming batch pipes: form batches while ingesting a one-pass source.

A random-access dataset can be batched by a ``batch_sampler``: the sampler hands
out *index lists* in the main process, and the DataLoader fetches the rows in
parallel workers.  A streaming source has no index space, so that split is
impossible -- and PyTorch forbids it outright (``DataLoader`` rejects both
``batch_sampler`` and ``shuffle`` for an ``IterableDataset``).

So the batching moves *inside* the stream.  A pipe is a transformer dataset in
the unix sense: it ingests items, holds them in a bounded buffer, and emits a
batch whenever its condition is met -- exactly the policy in
:mod:`genml_kit.datasets.batch_policy`, run incrementally instead of over a
materialised epoch.

The two pipes here mirror the two random-access samplers:

* :class:`StreamingLengthBucketBatch` mirrors ``LengthBucketBatchSampler``.
* :class:`StreamingPackingBatch` mirrors ``PackingBatchSampler``.

Because both feeders call the *same* policy functions, they cannot drift.

The pipe deliberately does **no** worker split of its own.  Every streaming
source in this package already shards itself across workers (``StreamingDataset``
via ``worker_slice``; a HuggingFace stream via its own ``num_shards`` sharding;
``WebDatasetSource`` via the base class).  A pipe that split again would divide
the source's per-worker slice a second time and silently drop items.  So the
pipe just batches whatever its source hands it, and the source owns the split.

A pipe is consumed with ``DataLoader(pipe, batch_size=None, collate_fn=...)``:
the pipe yields whole batches, and ``batch_size=None`` tells the DataLoader to
pass each one through untouched (still applying ``collate_fn``).
"""

import random

from torch.utils.data import IterableDataset

from genml_kit.datasets.batch_policy import (bucket_batches, item_text, packing_batches)
from genml_kit.utils.logging import fatal

#: Buffer capacity as a multiple of ``batch_size``, matching
#: ``LengthBucketBatchSampler``'s default.  A larger buffer gives better
#: length homogeneity (and a stronger shuffle) at the cost of memory.
_DEFAULT_BUFFER_MULTIPLIER = 50


class StreamingBatchPipe(IterableDataset):
  """Buffer items from a one-pass source and emit batches.

    Subclasses implement :meth:`emit_buffer`, which turns one full buffer of
    items into batches using a policy from :mod:`genml_kit.datasets.batch_policy`.

    Attributes:
      source: The item stream being batched.  It owns the worker split; the pipe
        adds none of its own (see the module docstring).
      buffer_size: Items held before a batch is formed.
      drop_last: Whether to drop trailing batches shorter than ``batch_size``.
      seed: Base RNG seed; the per-buffer order is ``seed + epoch + buffer``.
      epoch: Current epoch, advanced through :meth:`set_epoch`.
    """

  def __init__(self, source, buffer_size, *, drop_last=False, seed=None):
    if buffer_size <= 0:
      fatal(f"buffer_size must be positive, got {buffer_size}", ValueError)
    self.source = source
    self.buffer_size = buffer_size
    self.drop_last = drop_last
    self.seed = seed
    self.epoch = 0

  def set_epoch(self, epoch):
    """Advance the epoch so the per-buffer shuffle changes next pass."""
    self.epoch = epoch
    if hasattr(self.source, "set_epoch"):
      self.source.set_epoch(epoch)

  def _buffer_rng(self, buffer_index):
    """A deterministic RNG for one buffer (varies per epoch and buffer)."""
    if self.seed is None:
      return random.Random(None)
    return random.Random(self.seed + self.epoch + buffer_index)

  def __iter__(self):
    # No worker split here: the source owns it (see the module docstring).
    buffer = []
    buffer_index = 0
    for item in self.source:
      buffer.append(item)
      if len(buffer) >= self.buffer_size:
        yield from self.emit_buffer(buffer, buffer_index, final=False)
        buffer = []
        buffer_index += 1
    if buffer:
      yield from self.emit_buffer(buffer, buffer_index, final=True)

  def emit_buffer(self, buffer, buffer_index, final):
    """Turn one buffer of items into batches.

        Args:
          buffer: The buffered items (a list).
          buffer_index: Which buffer this is, for the deterministic shuffle.
          final: True for the last (possibly short) buffer of the pass.

        Yields:
          list: One batch of items.
        """
    raise NotImplementedError


class StreamingLengthBucketBatch(StreamingBatchPipe):
  """Streaming analogue of ``LengthBucketBatchSampler``.

    Holds ``batch_size * buffer_multiplier`` items, then sorts them by length
    and emits length-homogeneous batches -- so the collator pads little, exactly
    as the random-access sampler does over a whole epoch.
    """

  def __init__(self,
               source,
               batch_size,
               *,
               buffer_multiplier=_DEFAULT_BUFFER_MULTIPLIER,
               drop_last=False,
               seed=None):
    super().__init__(source,
                     batch_size * buffer_multiplier,
                     drop_last=drop_last,
                     seed=seed)
    self.batch_size = batch_size

  def emit_buffer(self, buffer, buffer_index, final):
    lengths = [len(item_text(item)) for item in buffer]
    rng = self._buffer_rng(buffer_index)
    # Only the final buffer may drop a short batch.  ``buffer_size`` is a
    # multiple of ``batch_size``, so a short chunk cannot normally arise in an
    # intermediate buffer -- but if the two ever stop being commensurate, this
    # guard keeps items from silently disappearing.
    batches = bucket_batches(lengths,
                             self.batch_size,
                             len(buffer),
                             rng,
                             drop_last=self.drop_last and final)
    for batch in batches:
      yield [buffer[pos] for pos in batch]


class StreamingPackingBatch(StreamingBatchPipe):
  """Streaming analogue of ``PackingBatchSampler``.

    Holds ``batch_size * buffer_multiplier`` items, then greedily packs them into
    token-budgeted batches (``batch_size * max_seq_len`` tokens).  As with the
    random-access sampler, a document that alone exceeds the budget opens its own
    over-full batch, and the collator caps the windows it emits.
    """

  def __init__(self,
               source,
               batch_size,
               max_seq_len,
               tokens_per_unit,
               *,
               buffer_multiplier=_DEFAULT_BUFFER_MULTIPLIER,
               drop_last=False,
               seed=None):
    super().__init__(source,
                     batch_size * buffer_multiplier,
                     drop_last=drop_last,
                     seed=seed)
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.tokens_per_unit = tokens_per_unit

  @property
  def budget(self):
    """Token budget of one batch: ``batch_size * max_seq_len``."""
    return self.batch_size * self.max_seq_len

  def emit_buffer(self, buffer, buffer_index, final):
    lengths = [len(item_text(item)) for item in buffer]
    rng = self._buffer_rng(buffer_index)
    # Only the final buffer may drop a short trailing batch: intermediate
    # buffers are followed by more data, so dropping there would lose items.
    batches = packing_batches(lengths,
                              self.tokens_per_unit,
                              self.budget,
                              rng,
                              drop_last=self.drop_last and final,
                              label="Streaming packing")
    for batch in batches:
      yield [buffer[pos] for pos in batch]
