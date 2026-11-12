"""Deterministic train/val/test splitting for streaming sources.

A one-pass source cannot be sliced by index or handed to ``random_split``, and
fanning it out through a shared queue couples the consumers' rates (a slow
validation bucket makes training wait, and buffering what training does not
consume runs the machine out of memory).  So the split is expressed as a pure
*positional* function instead:

    item at position i belongs to bucket pattern[i % P]

``P`` is the smallest period that can express every ratio, and ``pattern`` is a
fixed vector of bucket names of length ``P`` with ``round(ratio * P)`` slots per
bucket.  No RNG is involved, so two runs over the same source order produce
byte-identical splits, and every bucket is an *independent* filter over the
source -- consuming one never advances another and nothing is buffered.

Example:
  >>> splitter = IterableDatasetFilter(source, {"train": 0.8, "val": 0.2})
  >>> train_ds = splitter.get_dataset("train")   # a real iterable dataset
"""

import math

from genml_kit.datasets.iterable import StreamingTransform
from genml_kit.utils.logging import fatal


class _FilteredStream(StreamingTransform):
  """One bucket of an :class:`IterableDatasetFilter` split.

    Filters the shared source by enumerated position, so the bucket is a view
    over the original stream rather than a copy of it.  As a
    :class:`StreamingTransform` it forwards the source's worker split instead of
    applying its own: the source already shards itself across workers, so a
    second split here would keep only about ``1 / num_workers**2`` of the corpus.
    """

  def __init__(self, source, pattern, name):
    self._source = source
    self._pattern = pattern
    self._name = name

  def source_iter(self):
    period = len(self._pattern)
    for index, item in enumerate(self._source):
      if self._pattern[index % period] == self._name:
        yield item


class IterableDatasetFilter:
  """Split one streaming source into named buckets by position.

    This is a *splitter*, not a dataset: it holds no items and iterating it
    yields nothing.  Use :meth:`get_dataset` to obtain a bucket, which is a real
    :class:`~genml_kit.datasets.iterable.StreamingDataset` and can be handed
    straight to a ``DataLoader``.

    The routing pattern is fixed at construction and depends only on the
    ratios, so the same source order always splits the same way.
    """

  def __init__(self, source, ratios):
    """Build the split.

        Args:
          source: The streaming source to split.  May be iterated more than once;
            each bucket iterates it independently.
          ratios: Mapping ``{name: fraction}``.  Fractions must be positive and
            sum to 1 (within floating-point tolerance).

        Raises:
          ValueError: If *ratios* is empty, has a non-positive entry, or does not
            sum to 1.
        """
    if not ratios:
      fatal("IterableDatasetFilter requires at least one bucket.", ValueError)
    for name, ratio in ratios.items():
      if ratio <= 0:
        fatal(
            f"IterableDatasetFilter ratio for {name!r} must be positive, "
            f"got {ratio}.", ValueError)
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
      fatal(
          f"IterableDatasetFilter ratios must sum to 1, got {total} "
          f"({dict(ratios)}).", ValueError)

    self._source = source
    self._ratios = dict(ratios)
    self._pattern = self._build_pattern(self._ratios)

  #: Upper bound on the routing period.  ``P = ceil(1 / min_ratio)``, so a very
  #: small validation fraction (say 0.001) would ask for a 1000-long pattern;
  #: that is fine, but an absurd one is a configuration error worth catching.
  MAX_PERIOD = 10000

  @classmethod
  def _build_pattern(cls, ratios):
    """Return the positional routing pattern for *ratios*.

        The period is ``ceil(1 / min_ratio)``: the shortest cycle that can give
        the rarest bucket at least one slot.  Each bucket then receives
        ``round(ratio * P)`` slots (minimum 1), spread as evenly as possible, and
        any drift from rounding is absorbed by the most common bucket so the
        slots always add up to ``P`` exactly.

        Args:
          ratios: Mapping of bucket name to fraction (already validated).

        Returns:
          list[str]: Length-``P`` vector of bucket names.

        Raises:
          ValueError: If the implied period exceeds :attr:`MAX_PERIOD`.
        """
    period = int(math.ceil(1.0 / min(ratios.values())))
    if period > cls.MAX_PERIOD:
      fatal(
          f"IterableDatasetFilter ratios imply a routing period of {period}, "
          f"above the {cls.MAX_PERIOD} limit. Use a larger minimum ratio.", ValueError)

    # Slots per bucket, at least one each.
    slots = {name: max(1, round(ratio * period)) for name, ratio in ratios.items()}
    drift = period - sum(slots.values())
    if drift:
      # Absorb rounding drift in the largest bucket: it can hide a slot or two
      # without materially changing the split, and it never drops to zero.
      largest = max(slots, key=lambda name: ratios[name])
      slots[largest] += drift
      if slots[largest] <= 0:
        fatal(
            f"IterableDatasetFilter ratios {dict(ratios)} cannot be expressed "
            f"in a period of {period}.", ValueError)

    # Spread each bucket evenly across the period: bucket with n slots goes to
    # the positions floor(k * period / n) for k in range(n).  Smallest-ratio
    # buckets are placed first so they get the most even spread; the pattern is
    # then filled left to right, which keeps the result deterministic.
    pattern = [None] * period
    order = sorted(slots, key=lambda name: (ratios[name], name))
    for name in order:
      count = slots[name]
      for k in range(count):
        position = (k * period) // count
        # Linear probing: positions may collide when several buckets share an
        # offset; the next free slot keeps the spread close to ideal.
        while pattern[position] is not None:
          position = (position + 1) % period
        pattern[position] = name
    return pattern

  @property
  def pattern(self):
    """The routing pattern (one bucket name per position in the period)."""
    return list(self._pattern)

  @property
  def names(self):
    """The bucket names, in the order the ratios were declared."""
    return list(self._ratios)

  @property
  def ratios(self):
    """The declared ``{name: fraction}`` mapping."""
    return dict(self._ratios)

  def get_dataset(self, name):
    """Return the bucket *name* as an iterable dataset.

        Args:
          name: A bucket name from the ratios mapping.

        Returns:
          StreamingDataset: A view over the source yielding only this bucket.

        Raises:
          ValueError: If *name* is not a known bucket.
        """
    if name not in self._ratios:
      fatal(f"Unknown bucket {name!r}; have {sorted(self._ratios)}.", ValueError)
    return _FilteredStream(self._source, self._pattern, name)
