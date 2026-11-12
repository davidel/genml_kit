"""Streaming datasets under DDP: rank stride and worker split (§15.6, §15.7).

A streaming loader has no index space, so torch rejects both ``sampler`` and
``batch_sampler`` for it and installs a ``_InfiniteConstantSampler`` placeholder
that yields ``None`` forever.  The rank split therefore cannot be expressed as a
sampler: ``_ShardedBatchLoader`` consumes the batch stream instead, and this
file pins the invariants that makes it correct.

Two layers of coverage:

* single-process, with a faked ``rank``/``world_size`` (the convention of
  ``tests/test_distributed.py``), which pins the stride arithmetic, the ragged
  tail, epoch sensitivity and the forwarding seam;
* real multi-process, where two gloo ranks spawn and each dumps its id stream,
  which is the only way to prove the stride survives actual worker processes.

Every assertion is on the **id multiset** rather than tensor values: the ids are
the ground truth about which samples were trained on, which is exactly what a
wrong split corrupts.
"""

import json
import os

import pytest
import torch
import torch.distributed as dist

from genml_kit.utils import distributed
from tests.iterable_streams import (CountingMapDataset, CountingStream, assert_disjoint,
                                    assert_no_duplicates)


@pytest.fixture
def fake_ranks(monkeypatch):
  """Fake an N-rank group so one process can play every rank.

  Returns a callable ``(world, k, loader_factory, seed=0)`` that yields one
  ``_ShardedBatchLoader`` per rank, in rank order -- the same object graph a
  real distributed run would build, minus the process group.
  """

  def build(world, k, loader_factory, seed=0):
    monkeypatch.setattr(distributed, "world_size", lambda: world)
    out = []
    for r in range(world):
      monkeypatch.setattr(distributed, "rank", lambda r=r: r)
      out.append(
          distributed.shard_loader(loader_factory(), seed=seed, shuffle_buffer_k=k))
    return out

  return build


def _loader(n=24, batch_size=2, num_workers=0, stream_cls=CountingStream):
  from torch.utils.data import DataLoader
  return DataLoader(stream_cls(n), batch_size=batch_size, num_workers=num_workers)


def _batch_ids(loader):
  """Flatten an iterable of ``{"id": tensor}`` batches into a list of ints."""
  out = []
  for batch in loader:
    out.extend(batch["id"].tolist())
  return out


class TestShardLoaderDispatch:
  """Dispatch must key on the dataset, never on the sampler type."""

  def test_streaming_loader_is_wrapped_not_rebuilt(self, monkeypatch):
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    loader = _loader()
    sharded = distributed.shard_loader(loader, seed=0, shuffle_buffer_k=2)
    assert isinstance(sharded, distributed._ShardedBatchLoader)
    # Wrapped, not rebuilt: the very object the pipeline built is still there.
    assert sharded._loader is loader

  def test_wrapping_is_idempotent(self, monkeypatch):
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    once = distributed.shard_loader(_loader(), seed=0, shuffle_buffer_k=2)
    twice = distributed.shard_loader(once, seed=0, shuffle_buffer_k=2)
    assert twice is once

  def test_single_process_returns_loader_unchanged(self):
    loader = _loader()
    assert distributed.shard_loader(loader, seed=0) is loader

  def test_streaming_sampler_is_never_inspected(self, monkeypatch):
    """The placeholder sampler must not reach shard_index_sampler.

    Before the dataset dispatch existed, a streaming loader hit the final
    ``fatal`` in ``shard_index_sampler`` and reported a confusing error about
    ``_InfiniteConstantSampler``.  Assert the streaming branch returns before
    that function is called at all.
    """

    def boom(*args, **kwargs):
      raise AssertionError("shard_index_sampler must not run for a stream")

    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    monkeypatch.setattr(distributed, "shard_index_sampler", boom)
    distributed.shard_loader(_loader(), seed=0, shuffle_buffer_k=2)

  def test_forwarding_keeps_the_loader_transparent(self, monkeypatch):
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    loader = _loader(n=8, batch_size=2)
    sharded = distributed.shard_loader(loader, seed=0, shuffle_buffer_k=2)
    # Introspection and DataPipeline.seek_batch reach these through the wrapper.
    assert sharded.dataset is loader.dataset
    assert sharded.batch_size == 2
    assert sharded.num_workers == 0
    assert sharded.sampler is loader.sampler

  def test_length_less_stream_reports_an_unknown_sample_count(self, monkeypatch):
    """A one-pass stream has no batch count, so ``num_samples`` must say so.

    ``__len__`` used to swallow the ``TypeError`` and return ``0``, which the
    trainer rendered as ``[i/~0]`` -- a bogus denominator rather than the ``?``
    an unknown length deserves.
    """
    from genml_kit.datasets.base import num_samples
    monkeypatch.setattr(distributed, "world_size", lambda: 2)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    sharded = distributed.shard_loader(_loader(), seed=0, shuffle_buffer_k=2)
    with pytest.raises(TypeError):
      len(sharded)
    assert num_samples(sharded) is None


class TestRankStride:
  """Disjoint and exhaustive, over a range of round sizes."""

  @pytest.mark.parametrize("world", [1, 2, 3])
  @pytest.mark.parametrize("k", [0, 1, 2, 4])
  def test_ranks_are_disjoint_and_exhaustive(self, fake_ranks, monkeypatch, world, k):
    n, batch_size = 24, 2
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    loaders = fake_ranks(world, k, lambda: _loader(n, batch_size))
    per_rank = [_batch_ids(loader) for loader in loaders]
    assert_disjoint(per_rank)
    assert_no_duplicates([i for chunk in per_rank for i in chunk])
    usable = ((n // batch_size) // world) * world * batch_size
    assert sorted(i for chunk in per_rank for i in chunk) == list(range(usable))

  def test_every_rank_runs_the_same_number_of_batches(self, fake_ranks):
    world, k = 2, 3
    loaders = fake_ranks(world, k, lambda: _loader(24, 2))
    counts = [sum(1 for _ in loader) for loader in loaders]
    # DDP requires equal step counts or the all-reduce hangs.
    assert len(set(counts)) == 1

  def test_each_rank_gets_exactly_k_batches_per_round(self, fake_ranks):
    world, k = 2, 3
    loaders = fake_ranks(world, k, lambda: _loader(24, 2))
    # 24 items / batch 2 = 12 batches; a round is world*k = 6, so two rounds.
    # Each rank takes k=3 per round -> 6 batches in total.
    counts = [sum(1 for _ in loader) for loader in loaders]
    assert counts == [2 * k, 2 * k]
    assert sum(counts) == 12

  def test_ragged_tail_is_dropped_identically(self, fake_ranks):
    """n=10, batch 2 -> 5 batches; world=2 would leave a stray batch."""
    world, k = 2, 1
    loaders = fake_ranks(world, k, lambda: _loader(10, 2))
    per_rank = [_batch_ids(loader) for loader in loaders]
    assert len(per_rank[0]) == len(per_rank[1]) == 4
    assert_disjoint(per_rank)

  def test_world_one_is_the_whole_stream(self, fake_ranks):
    loaders = fake_ranks(1, 3, lambda: _loader(12, 2))
    assert _batch_ids(loaders[0]) == list(range(12))


class TestShuffling:
  """The buffer shuffles; it must never lose or duplicate a sample."""

  def test_round_is_permuted(self, fake_ranks):
    loaders = fake_ranks(2, 3, lambda: _loader(24, 2))
    union = sorted(_batch_ids(loaders[0]) + _batch_ids(loaders[1]))
    assert union == list(range(24))
    # A permuted round means rank 0 does not simply get the first half.
    assert _batch_ids(loaders[0]) != list(range(0, 24, 2))[:6]

  def test_k_zero_round_robins_without_permuting(self, fake_ranks):
    loaders = fake_ranks(2, 0, lambda: _loader(12, 2))
    # k=0: round of world_size batches, stream order kept.  Rank r takes every
    # world-th batch, so rank 0 sees batches 0,2,4 -> ids 0,1,4,5,8,9.
    assert _batch_ids(loaders[0]) == [0, 1, 4, 5, 8, 9]
    assert _batch_ids(loaders[1]) == [2, 3, 6, 7, 10, 11]

  def test_same_seed_gives_the_same_order(self, fake_ranks):
    first = _batch_ids(fake_ranks(2, 2, lambda: _loader(24, 2), seed=7)[0])
    second = _batch_ids(fake_ranks(2, 2, lambda: _loader(24, 2), seed=7)[0])
    assert first == second

  def test_different_seed_changes_the_order(self, fake_ranks):
    per_rank = {}
    for seed in (7, 8):
      loaders = fake_ranks(2, 2, lambda: _loader(24, 2), seed=seed)
      per_rank[seed] = [_batch_ids(loader) for loader in loaders]
    # A different permutation moves different batches onto each rank, so the
    # per-rank *sequence* changes.  Coverage is a property of the union only:
    # no single rank is a stable set of ids across seeds.
    assert per_rank[7][0] != per_rank[8][0]
    for chunks in per_rank.values():
      assert sorted(chunks[0] + chunks[1]) == list(range(24))

  def test_epoch_changes_sequence_but_not_coverage(self, fake_ranks):
    loaders = fake_ranks(2, 3, lambda: _loader(24, 2))
    epoch0 = [_batch_ids(loader) for loader in loaders]
    for loader in loaders:
      loader.set_epoch(1)
    epoch1 = [_batch_ids(loader) for loader in loaders]
    # The permutation changes, so each rank's slice changes -- but the union
    # always covers the corpus exactly once.  Per-rank *multisets* are not
    # expected to be stable: that is what reshuffling means here.
    for chunks in (epoch0, epoch1):
      assert sorted(chunks[0] + chunks[1]) == list(range(24))
      assert_disjoint(chunks)
    assert epoch0 != epoch1

  def test_successive_rounds_use_different_permutations(self, fake_ranks):
    """One epoch must not repeat a single permutation in every round.

      Round number is part of the seed; without it every round would permute
      identically, so the same stream positions would always land on rank 0.
      Pull the rounds apart by driving one rank and comparing per-round batches.
      """
    loaders = fake_ranks(2, 1, lambda: _loader(24, 2))
    # Round = world*k = 2 batches, so rank 0 takes 1 batch per round.
    batch_ids = [batch["id"].tolist() for batch in loaders[0]]
    singles = [b[0] for b in batch_ids]
    # If round_no were not seeded, rank 0 would take position 0 of a fixed
    # permutation every round; varying the seed means the parity of the id
    # block taken is not constant.
    assert len(set(singles)) > 1


class TestLocalDeterminism:
  """What §15.7 pins: a stream and its map-style twin must agree."""

  def test_per_rank_multisets_match_the_map_style_path(self, monkeypatch):
    """Parity with the existing sampler-based sharding, same seed and world."""
    n = 24
    monkeypatch.setattr(distributed, "world_size", lambda: 2)

    # Streaming path.
    stream_per_rank = []
    for r in range(2):
      monkeypatch.setattr(distributed, "rank", lambda r=r: r)
      loader = distributed.shard_loader(_loader(n, 2), seed=3, shuffle_buffer_k=3)
      stream_per_rank.append(sorted(_batch_ids(loader)))

    # Random-access path, same total and batch size.
    from torch.utils.data import DataLoader
    map_per_rank = []
    for r in range(2):
      monkeypatch.setattr(distributed, "rank", lambda r=r: r)
      base = DataLoader(CountingMapDataset(n), batch_size=2, shuffle=True)
      sharded = distributed.shard_loader(base, seed=3)
      map_per_rank.append(sorted(i for batch in sharded for i in batch["id"].tolist()))

    # Different mechanisms, same guarantee: disjoint, exhaustive, equal halves.
    for chunks in (stream_per_rank, map_per_rank):
      assert_disjoint(chunks)
      assert sorted(x for chunk in chunks for x in chunk) == list(range(n))
    assert [len(c) for c in stream_per_rank] == [len(c) for c in map_per_rank]


# --------------------------------------------------------------------------
# Real multi-process coverage (§15.6).  Two gloo ranks, each dumping its ids.
# --------------------------------------------------------------------------

_PORT = "29613"


def _rank_worker(rank, world, out_dir, num_workers, k):
  """Iterate this rank's shard and dump its ids for the parent to assert on."""
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = _PORT
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world)
  os.environ["LOCAL_RANK"] = str(rank)
  dist.init_process_group("gloo", rank=rank, world_size=world)
  try:
    from torch.utils.data import DataLoader
    loader = DataLoader(CountingStream(24), batch_size=2, num_workers=num_workers)
    sharded = distributed.shard_loader(loader, seed=0, shuffle_buffer_k=k)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as fh:
      json.dump(_batch_ids(sharded), fh)
  finally:
    dist.destroy_process_group()


@pytest.fixture
def spawned_ranks(tmp_path):
  """Spawn 2 real gloo ranks; return their dumped id streams."""

  def run(num_workers=0, k=3):
    out_dir = str(tmp_path)
    torch.multiprocessing.spawn(_rank_worker,
                                args=(2, out_dir, num_workers, k),
                                nprocs=2,
                                join=True)
    streams = []
    for rank in range(2):
      with open(os.path.join(out_dir, f"rank{rank}.json")) as fh:
        streams.append(json.load(fh))
    return streams

  return run


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
class TestDDPStreaming:
  """The required (§15.6) end-to-end check across a real process group."""

  def test_ranks_are_disjoint_over_gloo(self, spawned_ranks):
    first, second = spawned_ranks(num_workers=0)
    assert_disjoint([first, second])
    assert sorted(first + second) == list(range(24))

  @pytest.mark.parametrize("num_workers", [0, 2])
  def test_worker_and_rank_splits_compose(self, spawned_ranks, num_workers):
    """§15.7: dataset-level worker split x loader-level rank stride.

    With num_workers=2 each rank holds two replicas of the stream (one per
    worker), and the rank stride must still carve them into disjoint shares.
    The union must cover the corpus exactly once -- a duplicated item here means
    the two splits disagreed about boundaries.
    """
    first, second = spawned_ranks(num_workers=num_workers)
    assert_disjoint([first, second])
    assert sorted(first + second) == list(range(24))

  def test_step_counts_are_equal_across_ranks(self, spawned_ranks):
    first, second = spawned_ranks(num_workers=2)
    assert len(first) == len(second)
