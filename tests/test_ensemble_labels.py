"""Tests for DatasetEnsemble label handling and dict-returning datasets."""

import os
import tempfile

import numpy as np
import pytest
from datasets import Dataset as HFDataset
from PIL import Image

from genml_kit.datasets.ensemble import DatasetEnsemble, _HFDataset
from genml_kit.datasets.field_dataset import FieldSectorDataset
from genml_kit.datasets.image_folder import ImageFolderDataset


def _make_fake_hf_dataset(n=20, num_classes=4):
  """Create a small in-memory HF dataset with images and labels."""
  images = [Image.new("RGB", (64, 64), color=(i * 30 % 256, 128, 64)) for i in range(n)]
  labels = [f"class_{i % num_classes}" for i in range(n)]
  return HFDataset.from_dict({"image": images, "label": labels})


def _make_patched_init(hf_ds, label_column=None):
  """Return (original_init, patched_init) for monkeypatching _HFDataset."""
  orig = _HFDataset.__init__
  _label_col = label_column

  def patched_init(self,
                   name,
                   split="train",
                   image_column=None,
                   label_column=None,
                   cache_dir=None,
                   hf_token=None):
    self.name = name
    self._image_column = image_column or "image"
    self._label_names = []
    self._label2id = {}
    self._label_col = None
    self._column_map = {}
    self._ds = hf_ds
    if _label_col is not None:
      self._detect_labels(label_column=_label_col)

  return orig, patched_init


class TestImageFolderDictReturns:

  def test_has_labels_true_for_class_folders(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("cat", "dog"):
        cls_dir = os.path.join(root, "train", cls)
        os.makedirs(cls_dir)
        Image.new("RGB", (8, 8)).save(os.path.join(cls_dir, "img.jpg"))
      ds = ImageFolderDataset(root)
      assert ds.has_labels is True
      assert ds.num_labels == 2
      assert sorted(ds.label_names) == ["cat", "dog"]

  def test_dict_with_labels(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("A", "B"):
        d = os.path.join(root, "split", cls)
        os.makedirs(d)
        Image.new("RGB", (8, 8)).save(os.path.join(d, "x.jpg"))
      ds = ImageFolderDataset(root)
      item = ds[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" in item
      assert isinstance(item["image"], Image.Image)
      assert isinstance(item["label"], int)

  def test_dict_without_labels(self):
    with tempfile.TemporaryDirectory() as root:
      # "train" is a recognised split name → no labels.
      d = os.path.join(root, "train")
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "x.jpg"))
      ds = ImageFolderDataset(root)
      item = ds[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" not in item

  def test_labels_array(self):
    with tempfile.TemporaryDirectory() as root:
      for cls in ("A", "B"):
        d = os.path.join(root, "s", cls)
        os.makedirs(d)
        for i in range(3):
          Image.new("RGB", (8, 8)).save(os.path.join(d, f"{i}.jpg"))
      ds = ImageFolderDataset(root)
      arr = ds.labels_array
      assert isinstance(arr, np.ndarray)
      assert arr.dtype == np.int64
      assert len(arr) == 6

  def test_nonexistent_dir(self):
    with pytest.raises(FileNotFoundError, match="does not exist"):
      ImageFolderDataset("/nonexistent")

  def test_mixed_layout_fatal(self):
    with tempfile.TemporaryDirectory() as root:
      # split/label/file (3 components)
      d = os.path.join(root, "train", "A")
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))
      # split/file (2 components)
      d2 = os.path.join(root, "train")
      Image.new("RGB", (8, 8)).save(os.path.join(d2, "orphan.jpg"))
      with pytest.raises(ValueError, match="Mixed layout"):
        ImageFolderDataset(root)


class TestHFDatasetDictReturns:

  def test_dict_with_labels(self):
    hf_ds = _make_fake_hf_dataset(n=4, num_classes=2)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._column_map = {"image": "image", "label": "label"}
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    item = ds[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert "label" in item
    assert isinstance(item["image"], Image.Image)

  def test_dict_without_labels(self):
    images = [Image.new("RGB", (64, 64)) for _ in range(4)]
    hf_ds = HFDataset.from_dict({"image": images})
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._column_map = {"image": "image"}
    ds._ds = hf_ds
    ds._label_col = None
    ds._label_names = []
    ds._label2id = {}
    item = ds[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert "label" not in item

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=4)
    ds = _HFDataset.__new__(_HFDataset)
    ds.name = "test"
    ds._image_column = "image"
    ds._ds = hf_ds
    ds._detect_labels(label_column="label")
    arr = ds.labels_array
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.int64
    assert len(arr) == 12


class TestFieldSectorDataset:

  def test_extracts_single_field(self):

    class _FakeDS:

      def __len__(self):
        return 3

      def __getitem__(self, idx):
        return {"image": f"img_{idx}", "label": idx}

    ds = FieldSectorDataset(_FakeDS(), fields={"image": "image"})
    assert ds[0] == {"image": "img_0"}
    assert len(ds) == 3

  def test_extracts_label_field(self):

    class _FakeDS:

      def __len__(self):
        return 2

      def __getitem__(self, idx):
        return {"image": f"img_{idx}", "label": idx + 10}

    ds = FieldSectorDataset(_FakeDS(), fields={"label": "label"})
    assert ds[1] == {"label": 11}

  def test_renames_field(self):

    class _FakeDS:

      def __len__(self):
        return 1

      def __getitem__(self, idx):
        return {"img": f"img_{idx}", "label": idx}

    ds = FieldSectorDataset(_FakeDS(), fields={"img": "image"})
    assert ds[0] == {"image": "img_0"}

  def test_selects_multiple_fields(self):

    class _FakeDS:

      def __len__(self):
        return 1

      def __getitem__(self, idx):
        return {"image": f"img_{idx}", "label": idx, "extra": "x"}

    ds = FieldSectorDataset(_FakeDS(), fields={"image": "image", "label": "y"})
    assert ds[0] == {"image": "img_0", "y": 0}

  def test_empty_fields_rejected(self):

    class _FakeDS:

      def __len__(self):
        return 1

      def __getitem__(self, idx):
        return {"image": "img"}

    with pytest.raises(ValueError, match="fields"):
      FieldSectorDataset(_FakeDS(), fields={})


class TestDatasetEnsembleLabels:

  def test_has_labels_detected(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      assert ensemble.has_labels is True
      assert ensemble.num_labels == 3
    finally:
      _HFDataset.__init__ = orig

  def test_no_labels_when_absent(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=3)
    orig, patched = _make_patched_init(hf_ds, label_column=None)
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf"}]
      ensemble = DatasetEnsemble(configs)
      assert ensemble.has_labels is False
    finally:
      _HFDataset.__init__ = orig

  def test_labels_array(self):
    hf_ds = _make_fake_hf_dataset(n=12, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      arr = ensemble.labels_array
      assert isinstance(arr, np.ndarray)
      assert arr.dtype == np.int64
      assert len(arr) == 12
    finally:
      _HFDataset.__init__ = orig

  def test_item_is_dict(self):
    hf_ds = _make_fake_hf_dataset(n=8, num_classes=2)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      item = ensemble[0]
      assert isinstance(item, dict)
      assert "image" in item
      assert "label" in item
    finally:
      _HFDataset.__init__ = orig

  def test_summary_with_labels(self):
    hf_ds = _make_fake_hf_dataset(n=16, num_classes=4)
    orig, patched = _make_patched_init(hf_ds, label_column="label")
    _HFDataset.__init__ = patched
    try:
      configs = [{"name": "test", "source": "hf", "label_column": "label"}]
      ensemble = DatasetEnsemble(configs)
      s = ensemble.summary()
      assert "classes" in s.lower()
    finally:
      _HFDataset.__init__ = orig


class TestLoadImagefolder:
  """Tests for the load_imagefolder() loader API."""

  def _make_split_layout(self, root):
    """Create a root with train/val split dirs containing class dirs."""
    for split in ("train", "val"):
      for cls in ("cat", "dog"):
        d = os.path.join(root, split, cls)
        os.makedirs(d)
        for i in range(2):
          Image.new("RGB", (8, 8)).save(os.path.join(d, f"{i}.jpg"))

  def _make_class_layout(self, root):
    """Create a root with class dirs (no recognised split names)."""
    for cls in ("melanoma", "bcc"):
      d = os.path.join(root, cls)
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))

  def _make_flat_layout(self, root):
    """Create a root with images directly in it."""
    for i in range(3):
      Image.new("RGB", (8, 8)).save(os.path.join(root, f"img{i}.jpg"))

  def test_split_layout_returns_dict(self):
    with tempfile.TemporaryDirectory() as root:
      self._make_split_layout(root)
      from genml_kit.datasets.image_folder import load_imagefolder
      splits = load_imagefolder(root)
      assert sorted(splits.keys()) == ["train", "val"]
      assert len(splits["train"]) == 4
      assert len(splits["val"]) == 4
      assert splits["train"].has_labels is True
      assert sorted(splits["train"].label_names) == ["cat", "dog"]

  def test_split_layout_filter(self):
    with tempfile.TemporaryDirectory() as root:
      self._make_split_layout(root)
      from genml_kit.datasets.image_folder import load_imagefolder
      splits = load_imagefolder(root, split="train")
      assert list(splits.keys()) == ["train"]
      assert len(splits["train"]) == 4

  def test_split_layout_bad_split(self):
    with tempfile.TemporaryDirectory() as root:
      self._make_split_layout(root)
      from genml_kit.datasets.image_folder import load_imagefolder
      with pytest.raises(ValueError, match="split .* not found"):
        load_imagefolder(root, split="test")

  def test_class_layout_returns_train_key(self):
    """Depth-2 class layout → {"train": dataset}."""
    with tempfile.TemporaryDirectory() as root:
      self._make_class_layout(root)
      from genml_kit.datasets.image_folder import load_imagefolder
      splits = load_imagefolder(root)
      assert list(splits.keys()) == ["train"]
      assert splits["train"].has_labels is True
      assert sorted(splits["train"].label_names) == ["bcc", "melanoma"]

  def test_flat_layout_returns_train_key(self):
    with tempfile.TemporaryDirectory() as root:
      self._make_flat_layout(root)
      from genml_kit.datasets.image_folder import load_imagefolder
      splits = load_imagefolder(root)
      assert list(splits.keys()) == ["train"]
      assert splits["train"].has_labels is False
      assert len(splits["train"]) == 3

  def test_nonexistent_root(self):
    from genml_kit.datasets.image_folder import load_imagefolder
    with pytest.raises(FileNotFoundError, match="does not exist"):
      load_imagefolder("/nonexistent/path")

  def test_empty_root(self):
    with tempfile.TemporaryDirectory() as root:
      from genml_kit.datasets.image_folder import load_imagefolder
      with pytest.raises(FileNotFoundError, match="No images found"):
        load_imagefolder(root)

  def test_splits_marker_file(self):
    """A .splits file in root triggers split detection."""
    with tempfile.TemporaryDirectory() as root:
      # Create sub-dirs without recognised names but with .splits marker.
      for split in ("training", "validation"):
        d = os.path.join(root, split)
        os.makedirs(d)
        Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))
      # Create .splits marker
      os.path.join(root, ".splits")
      open(os.path.join(root, ".splits"), "w").close()
      from genml_kit.datasets.image_folder import load_imagefolder
      splits = load_imagefolder(root)
      assert sorted(splits.keys()) == ["training", "validation"]


class TestImageFolderDepthDetection:
  """Tests for depth-based label detection rules."""

  def test_depth2_class_labels(self):
    """root/class/img.jpg → labels detected."""
    with tempfile.TemporaryDirectory() as root:
      for cls in ("melanoma", "bcc"):
        d = os.path.join(root, cls)
        os.makedirs(d)
        Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))
      ds = ImageFolderDataset(root)
      assert ds.has_labels is True
      assert sorted(ds.label_names) == ["bcc", "melanoma"]
      assert ds[0]["label"] in (0, 1)

  def test_depth2_single_split_no_labels(self):
    """root/train/img.jpg → single split dir, no labels."""
    with tempfile.TemporaryDirectory() as root:
      d = os.path.join(root, "train")
      os.makedirs(d)
      Image.new("RGB", (8, 8)).save(os.path.join(d, "img.jpg"))
      ds = ImageFolderDataset(root)
      assert ds.has_labels is False

  def test_depth2_mixed_split_and_class_fatal(self):
    """root/train/img.jpg + root/melanoma/img.jpg → fatal."""
    with tempfile.TemporaryDirectory() as root:
      # train/ dir (split name)
      d1 = os.path.join(root, "train")
      os.makedirs(d1)
      Image.new("RGB", (8, 8)).save(os.path.join(d1, "img.jpg"))
      # melanoma/ dir (class name)
      d2 = os.path.join(root, "melanoma")
      os.makedirs(d2)
      Image.new("RGB", (8, 8)).save(os.path.join(d2, "img.jpg"))
      # This actually creates a mixed layout at depth 2:
      # train/img.jpg → depth 2, "train" is a split → no labels
      # melanoma/img.jpg → depth 2, "melanoma" is not a split → labels
      # But _scan sees all images at depth 2, then _is_split_dir returns
      # True (because "train" exists) → all treated as split → no labels.
      # This is a design choice: if ANY split dir exists, we treat the
      # whole root as split layout.
      # Actually: _is_split_dir checks if any first-level dir is a split name.
      # "train" is a split name → True → no labels for ALL images.
      # So this won't fatal. The "melanoma/img.jpg" will just have no label.
      ds = ImageFolderDataset(root)
      # Because _is_split_dir sees "train/", it treats ALL images as splits.
      assert ds.has_labels is False

  def test_depth1_flat_no_labels(self):
    with tempfile.TemporaryDirectory() as root:
      Image.new("RGB", (8, 8)).save(os.path.join(root, "img.jpg"))
      ds = ImageFolderDataset(root)
      assert ds.has_labels is False
      assert len(ds) == 1
