"""Resume positioning: the ``seek_batch`` contract (§15.8).

One resume code path for both access models.  Random access seeks by an O(1)
sampler offset (no sample decoded just to be dropped); a stream replays the
skipped prefix, or refuses and says so when it cannot be replayed.
"""

import pytest
from torch.utils.data import DataLoader

from genml_kit.pipelines.base import DataPipeline
from genml_kit.utils.distributed import _SkippedSampler
from tests.iterable_streams import (CountingMapDataset, CountingStream,
                                    NonResumableStream)


class _Pipeline(DataPipeline):
  """Minimal pipeline around a single training loader."""

  def __init__(self, dataset, batch_size=4):
    self.train_dataset = dataset
    self.val_loader = None
    self.train_loader = DataLoader(dataset, batch_size=batch_size)

  def build_loader(self, *args, **kwargs):
    pass

  def to_device(self, *args, **kwargs):
    pass


def _ids(pipeline):
  return [batch["id"].tolist() for batch in pipeline.iter_train_loader()]


class TestRandomAccessSeek:

  def test_seek_matches_dropping_batches(self):
    pipeline = _Pipeline(CountingMapDataset(20))
    full = _ids(pipeline)
    assert pipeline.seek_batch(2) == 2
    assert _ids(pipeline) == full[2:]

  def test_seek_zero_restores_the_start(self):
    pipeline = _Pipeline(CountingMapDataset(20))
    full = _ids(pipeline)
    pipeline.seek_batch(3)
    assert pipeline.seek_batch(0) == 0
    assert _ids(pipeline) == full

  def test_seek_is_idempotent(self):
    pipeline = _Pipeline(CountingMapDataset(20))
    full = _ids(pipeline)
    pipeline.seek_batch(2)
    first = _ids(pipeline)
    pipeline.seek_batch(2)
    second = _ids(pipeline)
    assert first == second == full[2:]

  def test_seek_beyond_the_epoch_yields_nothing(self):
    pipeline = _Pipeline(CountingMapDataset(8))
    pipeline.seek_batch(10)
    assert _ids(pipeline) == []

  def test_length_reflects_the_remaining_batches(self):
    pipeline = _Pipeline(CountingMapDataset(20))
    assert len(pipeline.train_loader) == 5
    pipeline.seek_batch(2)
    assert len(pipeline.train_loader) == 3

  def test_seek_does_not_decode_skipped_samples(self):
    # The O(1) claim: seeking must not call ``__getitem__`` for the prefix.
    loaded = []

    class _Spy(CountingMapDataset):

      def __getitem__(self, idx):
        loaded.append(idx)
        return super().__getitem__(idx)

    pipeline = _Pipeline(_Spy(20))
    pipeline.seek_batch(2)
    for _ in pipeline.iter_train_loader():
      pass
    assert 0 not in loaded and 7 not in loaded
    assert min(loaded) == 8


class TestStreamSeek:

  def test_seek_matches_dropping_batches(self):
    pipeline = _Pipeline(CountingStream(20))
    full = _ids(pipeline)
    assert pipeline.seek_batch(2) == 2
    assert _ids(pipeline) == full[2:]

  def test_seek_zero_restores_the_start(self):
    pipeline = _Pipeline(CountingStream(20))
    full = _ids(pipeline)
    pipeline.seek_batch(2)
    pipeline.seek_batch(0)
    assert _ids(pipeline) == full

  def test_seek_is_idempotent(self):
    pipeline = _Pipeline(CountingStream(20))
    full = _ids(pipeline)
    pipeline.seek_batch(2)
    assert _ids(pipeline) == full[2:]
    pipeline.seek_batch(2)
    assert _ids(pipeline) == full[2:]

  def test_non_replayable_source_refuses_and_warns(self, caplog):
    pipeline = _Pipeline(NonResumableStream(20))
    pipeline.train_loader.dataset.is_replayable = False
    with caplog.at_level("WARNING"):
      assert pipeline.seek_batch(2) == 0
    assert any(
        "restarts the epoch from the top" in r.getMessage() for r in caplog.records)

  def test_only_one_seek_mechanism_is_armed(self):
    # Seeking a random-access loader must clear any stream offset, or the
    # position would be applied twice.
    pipeline = _Pipeline(CountingMapDataset(20))
    full = _ids(pipeline)
    pipeline._stream_seek_offset = 3
    pipeline.seek_batch(2)
    assert pipeline._stream_seek_offset == 0
    assert _ids(pipeline) == full[2:]


class TestSkippedSampler:

  def test_reports_remaining_length(self):
    sampler = _SkippedSampler(range(10), 4)
    assert len(sampler) == 6
    assert list(sampler) == [4, 5, 6, 7, 8, 9]

  def test_skip_beyond_end_is_empty(self):
    sampler = _SkippedSampler(range(5), 99)
    assert list(sampler) == []
    assert len(sampler) == 0

  def test_negative_skip_is_clamped(self):
    sampler = _SkippedSampler(range(5), -3)
    assert list(sampler) == [0, 1, 2, 3, 4]

  def test_forwards_unknown_attributes(self):

    class _Inner:

      def __iter__(self):
        return iter(())

      def __len__(self):
        return 0

      def set_epoch(self, epoch):
        self.epoch = epoch

    inner = _Inner()
    sampler = _SkippedSampler(inner, 0)
    sampler.set_epoch(7)
    assert inner.epoch == 7


class TestEpochAlignment:

  @pytest.mark.parametrize("skip", [0, 1, 2, 3, 4])
  def test_seek_skip_matches_sequential_drop_for_both_models(self, skip):
    for dataset in (CountingMapDataset(20), CountingStream(20)):
      pipeline = _Pipeline(dataset)
      full = _ids(pipeline)
      pipeline.seek_batch(skip)
      assert _ids(pipeline) == full[skip:]
