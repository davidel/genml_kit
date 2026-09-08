"""Tests for mixed labeled/unlabeled ensembles and column remapping.

The ensemble protocol: every sub-dataset discovers its own image and
label columns and exposes a ``column_map`` (``{SOURCE: ENSEMBLE}``);
``__getitem__`` remaps names before returning, so ``DatasetEnsemble``
unilaterally emits ``"image"`` / ``"label"`` keys.
"""

import numpy as np
import pytest
from PIL import Image

from genml_kit.datasets.ensemble import DatasetEnsemble, _HFDataset


class _FakeLabeled:
  """Protocol-conformant labeled sub-dataset with non-standard columns."""

  def __init__(self,
               n=4,
               classes=("a", "b"),
               source_image_key="img",
               source_label_key="dx"):
    self.name = "fake_labeled"
    self._n = n
    self._classes = list(classes)
    self._source_image_key = source_image_key
    self._source_label_key = source_label_key
    self._map = {source_image_key: "image", source_label_key: "label"}

  @property
  def has_labels(self):
    return True

  @property
  def num_labels(self):
    return len(self._classes)

  @property
  def label_names(self):
    return list(self._classes)

  @property
  def labels_array(self):
    return np.arange(self._n) % len(self._classes)

  @property
  def column_map(self):
    return dict(self._map)

  def __len__(self):
    return self._n

  def __getitem__(self, idx):
    row = {
        self._source_image_key: Image.new("RGB", (8, 8)),
        self._source_label_key: idx % len(self._classes),
    }
    return {self._map[k]: v for k, v in row.items()}


class _FakeUnlabeled:
  """Protocol-conformant sub-dataset without any label column."""

  def __init__(self, n=2, source_image_key="img"):
    self.name = "fake_unlabeled"
    self._n = n
    self._source_image_key = source_image_key
    self._map = {source_image_key: "image"}

  @property
  def has_labels(self):
    return False

  @property
  def num_labels(self):
    return 0

  @property
  def label_names(self):
    return []

  @property
  def labels_array(self):
    return None

  @property
  def column_map(self):
    return dict(self._map)

  def __len__(self):
    return self._n

  def __getitem__(self, idx):
    row = {self._source_image_key: Image.new("RGB", (8, 8))}
    return {self._map[k]: v for k, v in row.items()}


def _make_ensemble(*datasets):
  """Build a DatasetEnsemble around pre-constructed fake sub-datasets."""
  ens = DatasetEnsemble.__new__(DatasetEnsemble)
  ens._datasets = list(datasets)
  ens._offsets = None
  ens._global_label_names = []
  ens._global_label2id = {}
  ens._build_offsets()
  ens._build_global_label_space()
  return ens


class TestMixedEnsemble:

  def test_mixed_items_have_consistent_keys(self):
    ens = _make_ensemble(_FakeLabeled(n=4), _FakeUnlabeled(n=2))
    assert len(ens) == 6
    for idx in range(4):
      item = ens[idx]
      assert set(item) == {"image", "label"}
      assert item["label"] in (0, 1)
    for idx in range(4, 6):
      item = ens[idx]
      assert set(item) == {"image"}
      assert "label" not in item

  def test_labels_array_requires_all_labeled(self):
    """labels_array refuses mixed ensembles instead of emitting sentinels."""
    ens = _make_ensemble(_FakeLabeled(n=4), _FakeUnlabeled(n=2))
    with pytest.raises(ValueError, match="requires every dataset"):
      _ = ens.labels_array

  def test_labels_array_all_labeled_aligned(self):
    ens = _make_ensemble(_FakeLabeled(n=4), _FakeLabeled(n=2, classes=("c",)))
    arr = ens.labels_array
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64
    assert len(arr) == 6
    # "c" is local id 0 in the second dataset but global id 2.
    np.testing.assert_array_equal(arr, [0, 1, 0, 1, 2, 2])

  def test_canonical_columns(self):
    ens = _make_ensemble(_FakeLabeled(n=4), _FakeUnlabeled(n=2))
    assert ens.image_column == "image"
    assert ens.label_column == "label"
    assert ens.num_labels == 2
    assert ens.label_names == ["a", "b"]

  def test_unlabeled_only_ensemble(self):
    ens = _make_ensemble(_FakeUnlabeled(n=3), _FakeUnlabeled(n=2))
    assert ens.image_column == "image"
    assert ens.label_column is None
    assert ens.num_labels == 0
    assert ens.label_names == []
    for idx in range(len(ens)):
      assert set(ens[idx]) == {"image"}

  def test_ensure_label_space_warns_on_mixed(self, caplog):
    ens = _make_ensemble(_FakeLabeled(n=4), _FakeUnlabeled(n=2))
    ens._global_label_names = []
    ens._global_label2id = {}
    ens.ensure_label_space()
    assert ens.num_labels == 2
    assert any("no labels" in rec.message for rec in caplog.records)

  def test_ensure_label_space_all_unlabeled_is_a_noop(self):
    ens = _make_ensemble(_FakeUnlabeled(n=3), _FakeUnlabeled(n=2))
    ens.ensure_label_space()
    assert ens.num_labels == 0
    assert ens.label_names == []


class TestRemapLabelValidation:

  def test_rejects_non_integer_label(self):
    ens = _make_ensemble(_FakeLabeled(n=4))
    ds = ens._datasets[0]
    with pytest.raises(ValueError, match="expected an integer label id"):
      ens._remap_label(ds, "a")

  def test_rejects_out_of_range_label(self):
    ens = _make_ensemble(_FakeLabeled(n=4))
    ds = ens._datasets[0]
    with pytest.raises(ValueError, match="out of range"):
      ens._remap_label(ds, 99)

  def test_accepts_valid_local_id(self):
    ens = _make_ensemble(_FakeLabeled(n=4, classes=("a", "b")))
    ds = ens._datasets[0]
    assert ens._remap_label(ds, 0) == 0
    assert ens._remap_label(ds, 1) == 1


class TestHFDatasetColumnRemap:

  def _hf_dataset_with_odd_columns(self):
    from datasets import Dataset as HFDataset

    images = [Image.new("RGB", (8, 8)) for _ in range(4)]
    return HFDataset.from_dict({
        "img": images,
        "dx": ["b", "a", "b", "a"],
    })

  def test_getitem_remaps_to_canonical_keys(self, monkeypatch):
    import datasets

    hf_ds = self._hf_dataset_with_odd_columns()
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: hf_ds)
    ds = _HFDataset(name="odd_columns")
    assert ds.column_map == {"img": "image", "dx": "label"}
    item = ds[0]
    assert set(item) == {"image", "label"}
    assert isinstance(item["image"], Image.Image)
    assert item["label"] in (0, 1)

  def test_ensemble_emits_canonical_keys_for_odd_columns(self, monkeypatch):
    import datasets

    hf_ds = self._hf_dataset_with_odd_columns()
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: hf_ds)
    ensemble = DatasetEnsemble([{"name": "odd_columns", "source": "hf"}])
    assert ensemble.image_column == "image"
    assert ensemble.label_column == "label"
    item = ensemble[0]
    assert set(item) == {"image", "label"}

  def test_labels_array_rejects_unknown_string(self):
    hf_ds = self._hf_dataset_with_odd_columns()
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "odd_columns"
    ds._image_column = "img"
    ds._column_map = {}
    ds._ds = hf_ds
    ds._detect_labels(label_column="dx")
    # Corrupt the label space so one column value becomes unknown.
    ds._label_names = ["a", "b", "c"]
    ds._label2id = {"a": 0, "b": 1, "c": 2}
    del ds._label2id["b"]
    with pytest.raises(ValueError, match="unknown label value"):
      _ = ds.labels_array
