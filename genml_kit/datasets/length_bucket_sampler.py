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
import random

from torch.utils.data import Sampler

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
    n = len(self.lengths)
    rng = random.Random(None if self.seed is None else self.seed + self._epoch)
    # Global order sorted by ascending length: [n] indices.
    order = sorted(range(n), key=lambda i: self.lengths[i])
    # Slice the sorted order into contiguous buckets: list of [<=bucket_size].
    # Each bucket therefore holds a narrow length band.
    buckets = [order[i:i + self.bucket_size] for i in range(0, n, self.bucket_size)]
    # Shuffle *within* each bucket so batch membership is not fixed to the sort
    # (length homogeneity is preserved; only intra-band order changes).
    for bucket in buckets:
      rng.shuffle(bucket)
    # Randomize the bucket order too, so lengths still vary across the epoch
    # (avoid training purely short-then-long, which would bias the loss curve).
    rng.shuffle(buckets)
    for bucket in buckets:
      # Emit the bucket in fixed-size chunks; a batch is [<=batch_size] indices
      # of similar length, which is what keeps the collator's pad near zero.
      for i in range(0, len(bucket), self.batch_size):
        batch = bucket[i:i + self.batch_size]
        if self.drop_last and len(batch) < self.batch_size:
          continue
        yield batch


def text_length_proxy(dataset):
  """Return a per-item length proxy for a ``TextDatasetProxy`` (or similar).

  Uses the raw character length of the item's text (``text``, or
  ``prompt`` + ``response``), read through ``__getitem__`` so it works for any
  wrapper (capped view included).  Character length needs no tokenizer pass and
  is monotonic with token count for BPE/byte tokenizers, so it is a good
  bucketing key at a fraction of the cost.
  """
  n = len(dataset)
  lengths = []
  for i in range(n):
    item = dataset[i]
    if "text" in item:
      lengths.append(len(item["text"] or ""))
    else:
      lengths.append(len(item.get("prompt") or "") + len(item.get("response") or ""))
  logging.info("Length bucketing: computed %d length proxies.", n)
  return lengths
