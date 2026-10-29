"""Tests for :class:`genml_kit.datasets.capped.CappedDataset`.

Verifies the read-only view contract: truncated length and column access,
forwarding of every other member, and re-capping of results that are
themselves datasets (``map`` / ``select`` / ``class_encode_column``).
"""

import datasets

from genml_kit.datasets.capped import CappedDataset, maybe_cap


def _ds(n=5):
  return datasets.Dataset.from_dict({
      "text": [f"row{i}" for i in range(n)],
      "label": [i % 2 for i in range(n)],
  })


class TestMaybeCap:

  def test_none_is_a_noop(self):
    ds = _ds()
    assert maybe_cap(ds, None) is ds

  def test_zero_and_negative_are_noops(self):
    ds = _ds()
    assert maybe_cap(ds, 0) is ds
    assert maybe_cap(ds, -3) is ds

  def test_positive_wraps(self):
    capped = maybe_cap(_ds(), 2)
    assert isinstance(capped, CappedDataset)
    assert len(capped) == 2


class TestCappedDataset:

  def test_len_is_truncated(self):
    assert len(CappedDataset(_ds(5), 2)) == 2

  def test_len_never_exceeds_wrapped(self):
    assert len(CappedDataset(_ds(3), 100)) == 3

  def test_int_getitem_delegates(self):
    ds = _ds()
    capped = CappedDataset(ds, 2)
    assert capped[0] == ds[0]
    assert capped[1] == ds[1]

  def test_str_getitem_truncates_column(self):
    capped = CappedDataset(_ds(5), 2)
    assert capped["text"] == ["row0", "row1"]
    assert capped["label"] == [0, 1]

  def test_getattr_forwards_metadata(self):
    ds = _ds()
    capped = CappedDataset(ds, 2)
    assert capped.column_names == ds.column_names
    assert capped.features == ds.features

  def test_map_result_is_recapped(self):
    capped = CappedDataset(_ds(5), 2)
    mapped = capped.map(lambda row: {"text": row["text"].upper()})
    assert isinstance(mapped, CappedDataset)
    assert len(mapped) == 2
    assert mapped["text"] == ["ROW0", "ROW1"]

  def test_select_result_is_recapped(self):
    capped = CappedDataset(_ds(5), 2)
    # select to a range beyond the cap must still be capped.
    selected = capped.select(range(4))
    assert isinstance(selected, CappedDataset)
    assert len(selected) == 2

  def test_class_encode_column_result_is_recapped(self):
    # Regression guard: ``class_encode_column`` is used by
    # HFDatasetProxy.normalize_labels and returns a new Dataset.
    capped = CappedDataset(_ds(6), 2)
    encoded = capped.class_encode_column("label")
    assert len(encoded) == 2

  def test_train_test_split_recaps_each_part(self):
    capped = CappedDataset(_ds(10), 4)
    split = capped.train_test_split(test_size=0.5, seed=0)
    assert isinstance(split, dict)
    assert set(split) == {"train", "test"}
    for part in split.values():
      assert len(part) <= 4

  def test_non_dataset_callable_passes_through(self):
    capped = CappedDataset(_ds(5), 2)
    assert capped.__len__() == 2
