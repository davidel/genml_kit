"""Tests for SegFolderDataset and the segfolder/ dataset spec."""

import numpy as np
import pytest
from PIL import Image

from genml_kit.datasets.factory import parse_dataset_specs
from genml_kit.datasets.seg_folder import SegFolderDataset, load_segfolder


def _write_pair(root, stem, h=8, w=8, mask_value=1):
  (root / "images").mkdir(parents=True, exist_ok=True)
  (root / "masks").mkdir(parents=True, exist_ok=True)
  Image.fromarray(np.zeros((h, w, 3),
                           dtype=np.uint8)).save(root / "images" / f"{stem}.png")
  Image.fromarray(np.full((h, w), mask_value,
                          dtype=np.uint8)).save(root / "masks" / f"{stem}.png")


class TestSegFolderDataset:

  def test_basic_loading(self, tmp_path):
    _write_pair(tmp_path, "a", mask_value=2)
    _write_pair(tmp_path, "b", mask_value=3)
    ds = SegFolderDataset(tmp_path)
    assert len(ds) == 2
    item = ds[0]
    assert item["image"].mode == "RGB"
    assert item["mask"].dtype == np.int64
    assert item["mask"].shape == (8, 8)

  def test_missing_mask_dir_raises(self, tmp_path):
    (tmp_path / "images").mkdir()
    Image.fromarray(np.zeros((4, 4, 3),
                             dtype=np.uint8)).save(tmp_path / "images" / "a.png")
    with pytest.raises(FileNotFoundError):
      SegFolderDataset(tmp_path)

  def test_stem_mismatch_raises(self, tmp_path):
    _write_pair(tmp_path, "a")
    # Extra image with no matching mask.
    Image.fromarray(np.zeros((4, 4, 3),
                             dtype=np.uint8)).save(tmp_path / "images" / "orphan.png")
    with pytest.raises(ValueError, match="stems do not match"):
      SegFolderDataset(tmp_path)

  def test_case_insensitive_stem_match(self, tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    Image.fromarray(np.zeros((4, 4, 3),
                             dtype=np.uint8)).save(tmp_path / "images" / "IMG_1.png")
    Image.fromarray(np.zeros((4, 4),
                             dtype=np.uint8)).save(tmp_path / "masks" / "img_1.png")
    ds = SegFolderDataset(tmp_path)
    assert len(ds) == 1


class TestLoadSegFolder:

  def test_flat_layout(self, tmp_path):
    _write_pair(tmp_path, "a")
    splits = load_segfolder(tmp_path)
    assert set(splits) == {"train"}

  def test_split_layout(self, tmp_path):
    _write_pair(tmp_path / "train", "a")
    _write_pair(tmp_path / "val", "b")
    splits = load_segfolder(tmp_path)
    assert set(splits) == {"train", "val"}
    only_val = load_segfolder(tmp_path, split="val")
    assert set(only_val) == {"val"}

  def test_bad_split_raises(self, tmp_path):
    _write_pair(tmp_path / "train", "a")
    with pytest.raises(ValueError, match="not found"):
      load_segfolder(tmp_path, split="test")


class TestParseDatasetSpecs:

  def test_segfolder_prefix(self):
    specs = parse_dataset_specs(["segfolder//data/voc"])
    assert specs[0]["source"] == "segfolder"
    assert specs[0]["name"] == "/data/voc"

  def test_imagefolder_prefix_unaffected(self):
    specs = parse_dataset_specs(["imagefolder//data/mnist"])
    assert specs[0]["source"] == "imagefolder"
