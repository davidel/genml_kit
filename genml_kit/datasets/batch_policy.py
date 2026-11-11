"""Shared batching policy for text training: lengths in, batches out.

Two batching strategies exist for text, and both are needed on **two different
feeders**:

* **Length bucketing** groups similar-length items so the collator pads little.
* **Packing** fills a token budget per batch so ``wrapped`` packing cannot fan
  one long document out into an unbounded number of model rows.

The *policy* -- how a bag of lengths turns into batches -- is identical
whichever feeder supplies the lengths.  Only the feeder differs:

* a random-access dataset has a materialisable index space, so the policy runs
  once over the whole epoch (see :mod:`genml_kit.datasets.length_bucket_sampler`
  and :mod:`genml_kit.datasets.packing_sampler`);
* a streaming source has no index space, so the policy runs incrementally over
  a bounded buffer (see :mod:`genml_kit.datasets.stream_batching`).

Keeping the policy here, in one place, means the two feeders cannot drift: a
change to how buckets or packets are formed changes both at once.

Every function is pure -- no dataset, no indexing, no iteration -- so it can be
unit-tested on plain lists of integers.
"""

import logging
import math
import random


def item_text(item):
  """Return the raw text of one text item, for a length proxy.

    Handles both supported layouts: plain LM (``text``) and instruction
    (``prompt`` + ``response``).

    Args:
      item: A dict text item.

    Returns:
      str: The concatenated raw text (may be empty).
    """
  if "text" in item:
    return item["text"] or ""
  return (item.get("prompt") or "") + (item.get("response") or "")


def bucket_batches(lengths, batch_size, bucket_size, rng, drop_last=False):
  """Group positions into length-homogeneous batches.

    Sorts positions by ascending length, slices them into contiguous buckets of
    *bucket_size*, shuffles within each bucket (so batch membership is not fixed
    to the sort) and shuffles the bucket order (so lengths still vary across the
    epoch).  Each bucket is then emitted in fixed-size chunks.

    Args:
      lengths: Per-position length proxy.
      batch_size: Positions per batch.
      bucket_size: Bucket capacity; larger buckets give better length
        homogeneity at the cost of a stronger shuffle perturbation.
      rng: A ``random.Random`` used for the intra-bucket and bucket shuffles.
      drop_last: Drop a trailing chunk shorter than *batch_size*.

    Yields:
      list[int]: Positions of one batch, ascending within the bucket.
    """
  n = len(lengths)
  order = sorted(range(n), key=lambda i: lengths[i])
  buckets = [order[i:i + bucket_size] for i in range(0, n, bucket_size)]
  for bucket in buckets:
    rng.shuffle(bucket)
  rng.shuffle(buckets)
  for bucket in buckets:
    for i in range(0, len(bucket), batch_size):
      batch = bucket[i:i + batch_size]
      if drop_last and len(batch) < batch_size:
        continue
      yield batch


def packing_batches(lengths,
                    tokens_per_unit,
                    budget,
                    rng,
                    drop_last=False,
                    label="Packing sampler"):
  """Greedily pack positions into token-budgeted batches.

    Positions are shuffled, then accumulated until adding the next one would
    exceed *budget*.  A document that alone exceeds the budget opens its own
    (over-full) batch; the collator caps the windows it emits.

    Args:
      lengths: Per-position length proxy (in *lengths* units, e.g. characters).
      tokens_per_unit: Tokens per unit of *lengths*.
      budget: Token budget of one batch.
      rng: A ``random.Random`` used to shuffle the positions.
      drop_last: Drop a trailing batch that does not fill the budget.
      label: Log prefix used when a short trailing batch is dropped.

    Yields:
      list[int]: Positions of one batch.
    """
  order = list(range(len(lengths)))
  rng.shuffle(order)
  current = []
  used = 0
  for index in order:
    tokens = estimated_tokens(lengths[index], tokens_per_unit)
    if current and used + tokens > budget:
      yield current
      current = []
      used = 0
    current.append(index)
    used += tokens
  if current:
    if drop_last and used < budget:
      logging.debug("%s: dropping a short trailing batch (%d / %d tokens).", label,
                    used, budget)
    else:
      yield current


def estimated_tokens(length, tokens_per_unit):
  """Estimated token count of an item of *length* proxy units (at least one)."""
  return max(1, int(math.ceil(length * tokens_per_unit)))


def epoch_rng(seed, epoch):
  """Return a per-epoch RNG.

    Args:
      seed: Base seed, or ``None`` to keep the system entropy.
      epoch: Epoch index folded into the seed.

    Returns:
      random.Random: A fresh RNG for this epoch.
    """
  return random.Random(None if seed is None else seed + epoch)
