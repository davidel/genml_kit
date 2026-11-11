"""Multi-worker correctness for streaming sources (§15.5).

``num_workers > 0`` makes every worker process re-run ``__iter__`` from the top.
Without a worker split the same items are emitted ``num_workers`` times over --
the classic ``IterableDataset`` duplication bug.  These tests run a real
``DataLoader`` with real worker processes and assert on exact item multisets.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from genml_kit.datasets.iterable import StreamingDataset
from tests.iterable_streams import (CountingStream, assert_disjoint,
                                    assert_multiset_equal, assert_no_duplicates, ids)


class _Tagged(StreamingDataset):
  """Streaming source that stamps the producing worker onto each item."""

  def __init__(self, n):
    self.n = n

  def source_iter(self):
    from torch.utils.data import get_worker_info
    info = get_worker_info()
    worker = 0 if info is None else info.id
    for i in range(self.n):
      yield {"id": i, "worker": worker}


def _run(dataset, num_workers, batch_size=1):
  loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
  return list(loader)


class TestNoDuplication:
  """The bug these tests exist to catch."""

  def test_workers_do_not_duplicate_items(self):
    loader = DataLoader(CountingStream(300), batch_size=1, num_workers=3)
    got = ids(loader)
    assert len(got) == 300, f"expected 300 items, got {len(got)} (duplication)"
    assert_no_duplicates(got)

  def test_naive_stream_would_duplicate(self):
    # Control: a plain IterableDataset without the worker split yields the
    # whole corpus in *each* worker.  This pins the reason StreamingDataset
    # exists, so a future refactor that drops the split fails here.
    from torch.utils.data import IterableDataset

    class _Naive(IterableDataset):

      def __iter__(self):
        yield from ({"id": i} for i in range(100))

    got = ids(DataLoader(_Naive(), batch_size=1, num_workers=3))
    assert len(got) == 300
    assert len(set(got)) == 100


class TestCoverage:

  @pytest.mark.parametrize("num_workers", [0, 1, 2, 3])
  def test_exact_coverage_for_any_worker_count(self, num_workers):
    got = ids(DataLoader(CountingStream(300), batch_size=1, num_workers=num_workers))
    assert_multiset_equal(got, list(range(300)))

  @pytest.mark.parametrize("num_workers", [2, 3])
  def test_worker_shards_are_disjoint(self, num_workers):
    batches = _run(_Tagged(300), num_workers=num_workers)
    by_worker = {}
    for batch in batches:
      for item_id, worker in zip(batch["id"].tolist(), batch["worker"].tolist()):
        by_worker.setdefault(worker, []).append(item_id)
    assert sorted(by_worker) == list(range(num_workers))
    assert_disjoint(list(by_worker.values()))
    assert_multiset_equal([i for chunk in by_worker.values() for i in chunk],
                          list(range(300)))

  @pytest.mark.parametrize("num_workers", [2, 3])
  def test_each_worker_sees_a_balanced_share(self, num_workers):
    batches = _run(_Tagged(300), num_workers=num_workers)
    counts = {}
    for batch in batches:
      for worker in batch["worker"].tolist():
        counts[worker] = counts.get(worker, 0) + 1
    # Strided split: shares differ by at most one item.
    assert max(counts.values()) - min(counts.values()) <= 1

  def test_worker_count_does_not_change_the_multiset(self):
    two = ids(DataLoader(CountingStream(120), batch_size=1, num_workers=2))
    three = ids(DataLoader(CountingStream(120), batch_size=1, num_workers=3))
    assert_multiset_equal(two, three)


class TestRaggedAndDropLast:

  def test_ragged_tail_is_not_lost(self):
    # 10 items over 3 workers with batch_size 2: uneven shards and a partial
    # final batch.  Nothing may be dropped or double-counted.
    got = ids(DataLoader(CountingStream(10), batch_size=2, num_workers=3))
    assert_multiset_equal(got, list(range(10)))

  def test_inner_drop_last_false_keeps_partial_batches(self):
    # torch drops the last partial batch of *each worker replica* when
    # drop_last=True (pytorch#33413), silently losing items.  The streaming
    # inner loader must not.
    got = ids(
        DataLoader(CountingStream(10), batch_size=3, num_workers=2, drop_last=False))
    assert_multiset_equal(got, list(range(10)))


class TestReiteration:

  def test_second_epoch_yields_the_full_set_again(self):
    dataset = CountingStream(50)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    first = ids(loader)
    second = ids(loader)
    assert_multiset_equal(first, list(range(50)))
    assert_multiset_equal(second, list(range(50)))

  def test_set_epoch_preserves_coverage(self):
    from tests.iterable_streams import ShuffledStream

    dataset = ShuffledStream(60)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)
    dataset.set_epoch(0)
    first = ids(loader)
    dataset.set_epoch(1)
    second = ids(loader)
    assert_multiset_equal(first, list(range(60)))
    assert_multiset_equal(second, list(range(60)))
    # A shuffle changes the order but never the content.
    assert first != second


class TestSingleProcessBaseline:

  def test_no_workers_matches_full_stream(self):
    got = ids(DataLoader(CountingStream(25), batch_size=4, num_workers=0))
    assert got == list(range(25))

  def test_torch_available(self):
    assert torch is not None
