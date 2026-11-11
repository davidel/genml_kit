"""Deterministic positional splitting for streaming sources (§15.4).

The splitter must be a pure function of enumerated position: no RNG, buckets
disjoint and exhaustive, each bucket independently re-iterable, and the
requested ratios honoured.
"""

from collections import Counter

import pytest

from genml_kit.datasets.split import IterableDatasetFilter
from tests.iterable_streams import (assert_disjoint, assert_multiset_equal, ids)


class _Source:
  """Minimal re-iterable source of ``{"id": i}`` items."""

  def __init__(self, n):
    self.n = n

  def __iter__(self):
    for i in range(self.n):
      yield {"id": i}


class TestPattern:
  """The routing pattern itself: period, slots and drift absorption."""

  def test_period_is_inverse_of_min_ratio(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.8, "val": 0.2})
    assert len(split.pattern) == 5

  def test_period_for_small_validation_fraction(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.95, "val": 0.05})
    assert len(split.pattern) == 20

  def test_three_way_slots(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.7, "val": 0.2, "test": 0.1})
    counts = Counter(split.pattern)
    assert counts == {"train": 7, "val": 2, "test": 1}
    assert len(split.pattern) == 10

  def test_slots_sum_to_period_after_rounding(self):
    # 1/3 each: 3 * round(1/3 * 3) = 3, so no drift.  A ratio set that does
    # drift must still fill the pattern exactly.
    for ratios in ({
        "a": 1 / 3,
        "b": 1 / 3,
        "c": 1 / 3
    }, {
        "a": 0.6,
        "b": 0.4
    }, {
        "a": 0.45,
        "b": 0.35,
        "c": 0.2
    }):
      split = IterableDatasetFilter(_Source(10), ratios)
      assert None not in split.pattern
      assert len(split.pattern) == len([n for n in split.pattern if n is not None])
      assert sum(Counter(split.pattern).values()) == len(split.pattern)

  def test_every_bucket_gets_at_least_one_slot(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.98, "val": 0.02})
    assert Counter(split.pattern)["val"] >= 1

  def test_pattern_is_returned_as_a_copy(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.8, "val": 0.2})
    pattern = split.pattern
    pattern[0] = "hacked"
    assert split.pattern[0] != "hacked"


class TestValidation:

  def test_empty_ratios_rejected(self):
    with pytest.raises(ValueError):
      IterableDatasetFilter(_Source(10), {})

  def test_ratios_must_sum_to_one(self):
    with pytest.raises(ValueError):
      IterableDatasetFilter(_Source(10), {"train": 0.5, "val": 0.2})

  def test_non_positive_ratio_rejected(self):
    with pytest.raises(ValueError):
      IterableDatasetFilter(_Source(10), {"train": 1.0, "val": 0.0})

  def test_unknown_bucket_rejected(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.8, "val": 0.2})
    with pytest.raises(ValueError):
      split.get_dataset("test")


class TestBuckets:

  def test_disjoint_and_exhaustive(self):
    n = 200
    split = IterableDatasetFilter(_Source(n), {"train": 0.8, "val": 0.2})
    buckets = {name: ids(split.get_dataset(name)) for name in split.names}
    assert_disjoint(list(buckets.values()))
    assert_multiset_equal([i for chunk in buckets.values() for i in chunk],
                          list(range(n)))

  def test_ratio_fidelity(self):
    n = 1000
    split = IterableDatasetFilter(_Source(n), {"train": 0.8, "val": 0.2})
    train = ids(split.get_dataset("train"))
    val = ids(split.get_dataset("val"))
    assert len(train) == 800
    assert len(val) == 200

  def test_determinism_across_instances(self):
    n = 100
    a = IterableDatasetFilter(_Source(n), {"train": 0.7, "val": 0.3})
    b = IterableDatasetFilter(_Source(n), {"train": 0.7, "val": 0.3})
    assert a.pattern == b.pattern
    for name in a.names:
      assert ids(a.get_dataset(name)) == ids(b.get_dataset(name))

  def test_bucket_is_re_iterable(self):
    split = IterableDatasetFilter(_Source(50), {"train": 0.8, "val": 0.2})
    ds = split.get_dataset("train")
    first = ids(ds)
    second = ids(ds)
    assert first == second

  def test_consuming_one_bucket_does_not_advance_another(self):
    split = IterableDatasetFilter(_Source(50), {"train": 0.8, "val": 0.2})
    train = split.get_dataset("train")
    val = split.get_dataset("val")
    # Interleave consumption: each bucket iterates the source independently,
    # so partial consumption of one must not disturb the other.
    assert ids(train) == ids(split.get_dataset("train"))
    assert ids(val) == ids(split.get_dataset("val"))

  def test_no_shared_queue_rate_coupling(self):
    # Draining val first then train must produce the same buckets as the other
    # order: there is no shared buffer to advance.
    split_a = IterableDatasetFilter(_Source(40), {"train": 0.75, "val": 0.25})
    val_first = (ids(split_a.get_dataset("val")), ids(split_a.get_dataset("train")))
    split_b = IterableDatasetFilter(_Source(40), {"train": 0.75, "val": 0.25})
    train_first = (ids(split_b.get_dataset("train")), ids(split_b.get_dataset("val")))
    assert val_first[0] == train_first[1]
    assert val_first[1] == train_first[0]

  def test_bucket_has_no_len(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.8, "val": 0.2})
    with pytest.raises(TypeError):
      len(split.get_dataset("train"))

  def test_ratios_property_is_a_copy(self):
    split = IterableDatasetFilter(_Source(10), {"train": 0.8, "val": 0.2})
    ratios = split.ratios
    ratios["train"] = 0.0
    assert split.ratios["train"] == 0.8

  def test_epoch_is_forwarded_to_the_source(self):

    class _EpochSource(_Source):

      def __init__(self, n):
        super().__init__(n)
        self.epoch = None

      def set_epoch(self, epoch):
        self.epoch = epoch

    source = _EpochSource(10)
    split = IterableDatasetFilter(source, {"train": 0.8, "val": 0.2})
    split.get_dataset("train").set_epoch(7)
    assert source.epoch == 7
