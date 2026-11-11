"""Dataset capability contract: iterate first, index only if you can.

Covers plan sections 15.2 (contract and introspection) and 15.3 (loader
construction): the ``IterableMixin`` default iterator, the duck-typed
capability predicates, the estimate-only ``num_samples``, ``__iter__`` on the
wrappers, and the fact that a stream never gets an index-shuffling loader.
"""

import pytest
import torch
from torch.utils.data import DataLoader

from genml_kit.datasets.base import (
    IterableMixin,
    is_iterable,
    is_random_access,
    is_shuffleable,
    num_samples,
)
from genml_kit.datasets.capped import CappedDataset
from genml_kit.datasets.field_dataset import FieldSectorDataset
from genml_kit.datasets.transforms import DictFieldTransform
from tests.iterable_streams import CountingMapDataset, CountingStream, ids


class _NumberedMixin(IterableMixin):

  def __len__(self):
    return 3

  def __getitem__(self, i):
    return i * 10


class TestIterableMixin:

  def test_yields_in_index_order(self):
    assert list(_NumberedMixin()) == [0, 10, 20]

  def test_matches_indexing(self):
    ds = CountingMapDataset(7)
    assert [item["id"] for item in ds] == [ds[i]["id"] for i in range(len(ds))]

  def test_iterator_is_restartable(self):
    ds = _NumberedMixin()
    assert list(ds) == list(ds)


class TestCapabilityPredicates:

  def test_random_access_dataset(self):
    ds = CountingMapDataset(4)
    assert is_random_access(ds) is True
    assert is_iterable(ds) is True
    assert is_shuffleable(ds) is True

  def test_stream(self):
    ds = CountingStream(4)
    assert is_random_access(ds) is False
    assert is_iterable(ds) is True
    assert is_shuffleable(ds) is False

  def test_bare_object(self):
    assert is_random_access(object()) is False
    assert is_iterable(object()) is False
    assert is_shuffleable(object()) is False

  def test_length_without_getitem_is_not_random_access(self):
    # A stream that only declares a count must not be treated as indexable.
    ds = CountingStream(4, nsamples=4)
    assert is_random_access(ds) is False

  def test_class_object_is_not_an_instance(self):
    # ``type`` has ``__getitem__``; passing a class must not look indexable.
    assert is_random_access(CountingMapDataset) is False


class TestNumSamples:

  def test_exact_for_random_access(self):
    assert num_samples(CountingMapDataset(11)) == 11

  def test_none_for_length_less_stream(self):
    assert num_samples(CountingStream(11)) is None

  def test_nsamples_used_when_no_len(self):
    assert num_samples(CountingStream(11, nsamples=42)) == 42

  def test_estimate_wins_over_len_when_declared(self):
    # ``nsamples`` is only consulted when ``len`` is unavailable, so an
    # explicit estimate never contradicts a real length.
    ds = CountingMapDataset(5)
    ds.nsamples = 999
    assert num_samples(ds) == 5

  def test_never_raises_for_stream(self):
    assert num_samples(CountingStream(3)) is None


class TestNoLenSafety:
  """No helper may explode on a source with no ``__len__``."""

  def test_helpers_tolerate_stream(self):
    ds = CountingStream(6)
    assert num_samples(ds) is None
    assert is_random_access(ds) is False
    assert len(ids(ds)) == 6

  def test_step_budget_returns_none_for_stream(self):
    from genml_kit.methods.distillation import _step_budget

    class _Args:
      epochs = 3
      grad_accum_steps = 1

    class _Pipeline:
      train_loader = CountingStream(5)

    assert _step_budget(_Args(), _Pipeline()) is None


class TestWrapperIteration:

  def _source(self):
    return [{"image": i, "label": i % 2, "extra": i * 2} for i in range(5)]

  def test_field_sector_dataset_parity(self):
    ds = FieldSectorDataset(self._source(), fields={"image": "img", "label": "y"})
    assert list(ds) == [ds[i] for i in range(len(ds))]

  def test_dict_field_transform_parity(self):
    transform = torch.nn.Identity()
    ds = DictFieldTransform(self._source(), transform, fields=("image",))
    assert list(ds) == [ds[i] for i in range(len(ds))]

  def test_dict_field_transform_applies_transform(self):

    class _Add(torch.nn.Module):

      def forward(self, value):
        return value + 100

    ds = DictFieldTransform(self._source(), _Add(), fields=("image",))
    assert [item["image"] for item in ds] == [100, 101, 102, 103, 104]

  def test_dict_field_transform_does_not_mutate_source(self):
    source = self._source()
    ds = DictFieldTransform(source, torch.nn.Identity(), fields=("image",))
    list(ds)
    assert source[0]["image"] == 0

  def test_capped_dataset_caps_iteration(self):
    ds = CappedDataset(self._source(), limit=3)
    assert [item["image"] for item in ds] == [0, 1, 2]

  def test_capped_dataset_caps_a_stream(self):
    stream = CountingStream(100)
    ds = CappedDataset(stream, limit=4)
    assert ids(ds) == [0, 1, 2, 3]

  def test_capped_dataset_stops_pulling(self):
    pulled = []

    class _Spy(CountingStream):

      def __iter__(self):
        for i in range(self.n):
          pulled.append(i)
          yield {"id": i}

    ds = CappedDataset(_Spy(100), limit=3)
    assert ids(ds) == [0, 1, 2]
    # ``islice`` must stop the source rather than draining it.
    assert pulled == [0, 1, 2]


class TestLoaderConstruction:

  def test_stream_loader_does_not_shuffle_by_index(self):
    ds = CountingStream(8)
    loader = DataLoader(ds, batch_size=4, shuffle=is_shuffleable(ds))
    assert ids(loader) == list(range(8))

  def test_shuffle_true_is_rejected_for_stream(self):
    # This is *why* ``is_shuffleable`` exists: torch refuses index shuffling
    # for an ``IterableDataset`` outright, so passing ``shuffle=True`` (today's
    # hardcoded value) would break the moment a stream is wired in.
    with pytest.raises(ValueError, match="IterableDataset"):
      DataLoader(CountingStream(4), batch_size=2, shuffle=True)

  def test_shuffle_false_works_on_stream(self):
    loader = DataLoader(CountingStream(4), batch_size=2, shuffle=False)
    assert ids(loader) == [0, 1, 2, 3]

  def test_random_access_loader_yields_everything(self):
    ds = CountingMapDataset(6)
    loader = DataLoader(ds, batch_size=3, shuffle=is_shuffleable(ds))
    assert sorted(ids(loader)) == list(range(6))
