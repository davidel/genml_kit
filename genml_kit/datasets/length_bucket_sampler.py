"""Length-bucketing batch sampler for padded (non-packed) text training.

Padding waste comes from batching items of very different lengths: every row is
padded up to the longest item in its batch.  :class:`LengthBucketBatchSampler`
attacks that directly -- it groups the epoch's indices into large buckets by a
cheap length proxy, shuffles *within* each bucket, and emits batches of
similar-length items.  Because each batch is length-homogeneous, the collator
pads little, and -- unlike packing -- attention stays exactly document-exact
with a plain 2-D mask: no cross-document leakage and no dense 4-D mask.

The sampler perturbs the global shuffle order (bucketing is inherently a
partial reordering), so it is intentionally exposed as an opt-in knob.
"""

import logging

from torch.utils.data import Sampler

from genml_kit.datasets.batch_policy import bucket_batches, epoch_rng, item_text
from genml_kit.utils.logging import fatal


class LengthBucketBatchSampler(Sampler):
  """Batch indices of similar length to minimise padding.

  Parameters
  ----------
  lengths : sequence of int
      Per-item length proxy (e.g. token count, or a cheap character count).
  batch_size : int
      Items per batch.
  bucket_size_multiplier : int
      Bucket capacity as a multiple of *batch_size*.  Larger buckets give
      better length homogeneity at the cost of a stronger shuffle perturbation
      and more memory for the index lists.
  drop_last : bool
      Drop the final short batch (maps to the DataLoader's ``drop_last``).
  seed : int or None
      Seed for the per-epoch shuffle.  ``None`` draws a fresh seed each epoch.
  """

  def __init__(self,
               lengths,
               batch_size,
               bucket_size_multiplier=50,
               drop_last=False,
               seed=None):
    if batch_size <= 0:
      fatal(f"batch_size must be positive, got {batch_size}", ValueError)
    self.lengths = list(lengths)
    self.batch_size = batch_size
    self.bucket_size = batch_size * bucket_size_multiplier
    self.drop_last = drop_last
    self.seed = seed
    self._epoch = 0

  def __len__(self):
    n = len(self.lengths)
    if self.drop_last:
      return n // self.batch_size
    return (n + self.batch_size - 1) // self.batch_size

  def set_epoch(self, epoch):
    """Vary the shuffle seed per epoch (DataLoader does not call this)."""
    self._epoch = epoch

  def __iter__(self):
    rng = epoch_rng(self.seed, self._epoch)
    yield from bucket_batches(self.lengths,
                              self.batch_size,
                              self.bucket_size,
                              rng,
                              drop_last=self.drop_last)


def text_length_proxy(dataset):
  """Return a per-item length proxy for a ``TextDatasetProxy`` (or similar).

  Uses the raw character length of the item's text (``text``, or
  ``prompt`` + ``response``), read through ``__getitem__`` so it works for any
  wrapper (capped view included).  Character length needs no tokenizer pass and
  is monotonic with token count for BPE/byte tokenizers, so it is a good
  bucketing key at a fraction of the cost.

  This requires a materialisable index space: for a streaming source, measure
  lengths inline instead (see
  :class:`~genml_kit.datasets.stream_batching.StreamingLengthBucketBatch`).
  """
  n = len(dataset)
  lengths = [len(item_text(dataset[i])) for i in range(n)]
  logging.info("Length bucketing: computed %d length proxies.", n)
  return lengths
