"""Tests for the shared text batching policy and the streaming batch pipes.

The policy functions are pure (lengths in, batches out), so they are tested on
plain lists.  The pipes are tested over a synthetic item stream, asserting the
same invariants the random-access samplers guarantee: every item is emitted
exactly once, batches respect the size/budget bound, and the worker split is
left to the source (the pipe must not split again).
"""

import random

from genml_kit.datasets.batch_policy import (bucket_batches, epoch_rng,
                                             estimated_tokens, item_text,
                                             packing_batches)
from genml_kit.datasets.stream_batching import (StreamingLengthBucketBatch,
                                                StreamingPackingBatch)


class _ItemStream:
  """Plain re-iterable item stream (no worker split of its own)."""

  def __init__(self, lengths):
    self.lengths = lengths

  def __iter__(self):
    for i, length in enumerate(self.lengths):
      yield {"id": i, "text": "x" * length}


class _EpochRecordingStream(_ItemStream):
  """An item stream that records the epochs forwarded to it."""

  def __init__(self, n):
    super().__init__([1] * n)
    self.epochs = []

  def set_epoch(self, epoch):
    self.epochs.append(epoch)


class TestItemText:

  def test_plain_lm(self):
    assert item_text({"text": "abc"}) == "abc"

  def test_instruction(self):
    assert item_text({"prompt": "a", "response": "bc"}) == "abc"

  def test_none_becomes_empty(self):
    assert item_text({"text": None}) == ""


class TestEstimatedTokens:

  def test_at_least_one(self):
    assert estimated_tokens(0, 1.0) == 1

  def test_rounds_up(self):
    assert estimated_tokens(3, 0.5) == 2


class TestEpochRng:

  def test_seeded_is_reproducible(self):
    a = epoch_rng(7, 1)
    b = epoch_rng(7, 1)
    assert [a.random() for _ in range(3)] == [b.random() for _ in range(3)]

  def test_epoch_changes_stream(self):
    a = epoch_rng(7, 1)
    b = epoch_rng(7, 2)
    assert [a.random() for _ in range(3)] != [b.random() for _ in range(3)]


class TestBucketBatches:

  def test_covers_every_position_once(self):
    lengths = [1, 10, 3, 8, 5, 2, 9, 4]
    seen = []
    for batch in bucket_batches(lengths, 3, 8, random.Random(0)):
      seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))

  def test_batches_within_size(self):
    lengths = list(range(30))
    for batch in bucket_batches(lengths, 4, 20, random.Random(0)):
      assert len(batch) <= 4

  def test_drop_last_drops_short_tail(self):
    # 5 positions, batch_size 4 -> one full batch, one of 1; drop_last keeps 1.
    lengths = [1] * 5
    batches = list(bucket_batches(lengths, 4, 4, random.Random(0), drop_last=True))
    assert [len(b) for b in batches] == [4]

  def test_seeded_reproducible(self):
    lengths = list(range(50))
    a = list(bucket_batches(lengths, 4, 20, random.Random(1)))
    b = list(bucket_batches(lengths, 4, 20, random.Random(1)))
    assert a == b


class TestPackingBatches:

  def test_covers_every_position_once(self):
    lengths = [1, 10, 3, 8, 5, 2, 9, 4, 7, 6]
    seen = []
    for batch in packing_batches(lengths, 1.0, 32, random.Random(0)):
      seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))

  def test_respects_budget(self):
    lengths = [3] * 40
    for batch in packing_batches(lengths, 1.0, 32, random.Random(0)):
      used = sum(estimated_tokens(lengths[i], 1.0) for i in batch)
      assert used <= 32

  def test_long_document_opens_its_own_batch(self):
    lengths = [100, 2, 2, 2]
    batches = list(packing_batches(lengths, 1.0, 32, random.Random(0)))
    assert any(0 in b for b in batches)


class TestStreamingLengthBucketBatch:

  def test_emits_every_item_once(self):
    pipe = StreamingLengthBucketBatch(_ItemStream([1, 2, 3, 4, 5, 6, 7, 8]),
                                      batch_size=2,
                                      buffer_multiplier=2,
                                      seed=0)
    seen = [item["id"] for batch in pipe for item in batch]
    assert sorted(seen) == list(range(8))

  def test_batches_within_size(self):
    pipe = StreamingLengthBucketBatch(_ItemStream(list(range(30))),
                                      batch_size=3,
                                      buffer_multiplier=3,
                                      seed=0)
    for batch in pipe:
      assert len(batch) <= 3

  def test_no_worker_split_of_its_own(self):
    # The pipe must batch whatever the source yields: with a source that hands
    # out a fixed sequence, no item may be dropped or duplicated.
    pipe = StreamingLengthBucketBatch(_ItemStream(list(range(10))),
                                      batch_size=4,
                                      buffer_multiplier=1,
                                      seed=0)
    seen = [item["id"] for batch in pipe for item in batch]
    assert sorted(seen) == list(range(10))

  def test_seeded_reproducible(self):

    def ids():
      pipe = StreamingLengthBucketBatch(_ItemStream(list(range(20))),
                                        batch_size=3,
                                        buffer_multiplier=2,
                                        seed=5)
      return [[item["id"] for item in batch] for batch in pipe]

    assert ids() == ids()

  def test_drop_last_only_drops_the_tail(self):
    # A short chunk can only be dropped at the very end: an intermediate buffer
    # dropping items would lose data silently.
    for n in (9, 10, 11, 30, 31):
      pipe = StreamingLengthBucketBatch(_ItemStream(list(range(n))),
                                        batch_size=3,
                                        buffer_multiplier=1,
                                        drop_last=True,
                                        seed=0)
      seen = [item["id"] for batch in pipe for item in batch]
      assert len(set(seen)) == len(seen), f"duplicate item at n={n}"
      # Only the final partial batch may be missing, so at most
      # ``batch_size - 1`` items are absent.
      assert n - len(seen) < 3, f"lost {n - len(seen)} items at n={n}"


class TestStreamingPipeEpoch:

  def test_set_epoch_is_stored(self):
    pipe = StreamingLengthBucketBatch(_ItemStream(list(range(6))),
                                      batch_size=2,
                                      buffer_multiplier=1,
                                      seed=0)
    pipe.set_epoch(7)
    assert pipe.epoch == 7

  def test_set_epoch_forwards_to_source(self):
    # The per-epoch seed must change when the source reshuffles, so the pipe has
    # to tell the source about the epoch.
    source = _EpochRecordingStream(6)
    pipe = StreamingLengthBucketBatch(source, batch_size=2, buffer_multiplier=1, seed=0)
    pipe.set_epoch(4)
    assert source.epochs == [4]

  def test_epoch_changes_batch_order(self):

    def order(epoch):
      pipe = StreamingLengthBucketBatch(_ItemStream(list(range(20))),
                                        batch_size=3,
                                        buffer_multiplier=2,
                                        seed=1)
      pipe.set_epoch(epoch)
      return [[item["id"] for item in batch] for batch in pipe]

    assert order(0) != order(1)


class TestStreamingPackingBatch:

  def test_emits_every_item_once(self):
    pipe = StreamingPackingBatch(_ItemStream([3] * 20),
                                 batch_size=4,
                                 max_seq_len=8,
                                 tokens_per_unit=1.0,
                                 buffer_multiplier=1,
                                 seed=0)
    seen = [item["id"] for batch in pipe for item in batch]
    assert sorted(seen) == list(range(20))

  def test_respects_budget(self):
    pipe = StreamingPackingBatch(_ItemStream([3] * 40),
                                 batch_size=4,
                                 max_seq_len=8,
                                 tokens_per_unit=1.0,
                                 buffer_multiplier=2,
                                 seed=0)
    budget = pipe.budget
    for batch in pipe:
      used = sum(estimated_tokens(len(item["text"]), 1.0) for item in batch)
      assert used <= budget

  def test_long_document_opens_its_own_batch(self):
    pipe = StreamingPackingBatch(_ItemStream([100] + [2] * 4),
                                 batch_size=2,
                                 max_seq_len=8,
                                 tokens_per_unit=1.0,
                                 buffer_multiplier=1,
                                 seed=0)
    batches = list(pipe)
    assert any(any(item["id"] == 0 for item in batch) for batch in batches)
