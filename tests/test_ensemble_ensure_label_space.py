"""Tests for ``DatasetEnsemble.ensure_label_space`` (#5)."""

from genml_kit.datasets.ensemble import DatasetEnsemble, _HFDataset


def _make_hf_dataset(n=8, num_classes=3):
  from datasets import Dataset as HFDataset
  from PIL import Image

  images = [Image.new("RGB", (16, 16)) for _ in range(n)]
  labels = [f"class_{i % num_classes}" for i in range(n)]
  return HFDataset.from_dict({"image": images, "label": labels})


def _make_patched_init(hf_ds, label_column=None):
  """Return ``(original_init, patched_init)`` bypassing real HF loading."""
  orig = _HFDataset.__init__
  forced_label_column = label_column

  def patched_init(self, name, **kwargs):
    self.name = name
    self._image_column = kwargs.get("image_column") or "image"
    self._label_names = []
    self._label2id = {}
    self._label_col = None
    self._column_map = {}
    self._ds = hf_ds
    if forced_label_column is not None:
      self._detect_labels(label_column=forced_label_column)

  return orig, patched_init


class TestEnsureLabelSpace:

  def test_builds_global_space(self):
    hf_ds = _make_hf_dataset(n=8, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      ensemble = DatasetEnsemble([{"name": "t", "source": "hf"}])
      # Simulate a caller that skipped the init-time build.
      ensemble._global_label_names = []
      ensemble._global_label2id = {}
      ensemble.ensure_label_space()
      assert ensemble.num_labels == 3
      assert ensemble.label_names == ["class_0", "class_1", "class_2"]
    finally:
      _HFDataset.__init__ = orig

  def test_is_idempotent(self):
    hf_ds = _make_hf_dataset(n=8, num_classes=2)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      ensemble = DatasetEnsemble([{"name": "t", "source": "hf"}])
      ensemble.ensure_label_space()
      names_first = ensemble.label_names
      ensemble.ensure_label_space()
      assert ensemble.label_names == names_first
      assert ensemble.num_labels == 2
    finally:
      _HFDataset.__init__ = orig

  def test_no_labels_is_a_noop(self):
    hf_ds = _make_hf_dataset(n=8, num_classes=2)
    orig, patched = _make_patched_init(hf_ds, label_column=None)
    _HFDataset.__init__ = patched
    try:
      ensemble = DatasetEnsemble([{"name": "t", "source": "hf"}])
      ensemble.ensure_label_space()
      assert ensemble.num_labels == 0
    finally:
      _HFDataset.__init__ = orig
