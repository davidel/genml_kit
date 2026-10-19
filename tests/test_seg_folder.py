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

  def test_palette_mask_stays_2d(self, tmp_path):
    """A palette ('P') PNG decodes to (H, W) indices, not a 3-D array."""
    _write_pair(tmp_path, "a")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0, :] = 1
    pal = Image.fromarray(mask, mode="P")
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * (256 * 3 - 6))
    pal.save(tmp_path / "masks" / "a.png")

    item = SegFolderDataset(tmp_path)[0]
    assert item["mask"].ndim == 2
    assert item["mask"].shape == (8, 8)
    assert set(item["mask"].ravel().tolist()) == {0, 1}

  def test_rgb_mask_keeps_channel_axis(self, tmp_path):
    """True RGB masks must not be silently collapsed to channel 0."""
    _write_pair(tmp_path, "a")
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[0, :] = (255, 0, 0)
    rgb[1, :] = (0, 255, 0)
    Image.fromarray(rgb).save(tmp_path / "masks" / "a.png")

    item = SegFolderDataset(tmp_path)[0]
    assert item["mask"].shape == (8, 8, 3)
    # Both colours survive; collapsing to channel 0 would merge them.
    triples = {tuple(px) for px in item["mask"].reshape(-1, 3).tolist()}
    assert (255, 0, 0) in triples
    assert (0, 255, 0) in triples

  def test_rgba_mask_is_rejected(self, tmp_path):
    """An ambiguous 4-channel mask fails loudly instead of guessing."""
    _write_pair(tmp_path, "a")
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(tmp_path / "masks" / "a.png")

    with pytest.raises(ValueError, match="4 channels"):
      SegFolderDataset(tmp_path)[0]


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


class TestSegfolderClassCompaction:
  """B4: the segfolder path gets the same dense class-id compaction as HF."""

  def _pipeline(self, tmp_path, **kwargs):
    from genml_kit.pipelines.segmentation import SegmentationPipeline
    import argparse

    p = SegmentationPipeline()
    args = argparse.Namespace(cache_dir=str(tmp_path / "cache"), **kwargs)
    return p, args

  def test_sparse_1based_ids_compacted(self, tmp_path):
    # Masks carry raw ids {10, 200} (sparse / 1-based-ish); the pipeline
    # must compact them to [0, 2) rather than infer max+1 = 201.
    _write_pair(tmp_path, "a", mask_value=10)
    _write_pair(tmp_path, "b", mask_value=200)
    p, args = self._pipeline(tmp_path, val_split=0.0)
    train, val = p._load_segfolder(str(tmp_path), args)
    assert p.mask_map == {10: 0, 200: 1}
    assert set(train[0]["mask"].ravel()) <= {0, 1}
    assert set(val[0]["mask"].ravel()) <= {0, 1}

  def test_void_255_excluded_and_passed_through(self, tmp_path):

    def write_with_void(stem):
      from PIL import Image as _I

      (tmp_path / "images").mkdir(parents=True, exist_ok=True)
      (tmp_path / "masks").mkdir(parents=True, exist_ok=True)
      _I.fromarray(np.zeros((4, 4, 3),
                            dtype=np.uint8)).save(tmp_path / "images" / f"{stem}.png")
      _I.fromarray(np.array([[0, 255], [1, 255]],
                            dtype=np.uint8)).save(tmp_path / "masks" / f"{stem}.png")

    write_with_void("a")
    write_with_void("b")  # second sample so a derived val split exists
    p, args = self._pipeline(tmp_path, val_split=0.0)
    train, _val = p._load_segfolder(str(tmp_path), args)
    # 0 and 1 compact to [0, 2); 255 stays 255 (void).
    mask = train[0]["mask"]
    assert p.mask_map == {0: 0, 1: 1}
    assert set(mask.ravel()) == {0, 1, 255}

  def test_val_only_id_is_covered(self, tmp_path):
    # A raw id that appears only in validation must be *included* in the
    # map (the scan covers train + val), exactly like the HF path -- it
    # never fatals.
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    _write_pair(train_root, "a", mask_value=1)
    _write_pair(val_root, "b", mask_value=9)
    p, args = self._pipeline(tmp_path, val_split=0.0)
    train, val = p._load_segfolder(str(tmp_path), args)
    assert p.mask_map == {1: 0, 9: 1}
    assert set(train[0]["mask"].ravel()) == {0}
    assert set(val[0]["mask"].ravel()) == {1}

  def test_class_appearing_later_still_counted(self, tmp_path):
    # A class that appears only in a late mask (beyond an old max-batches
    # scan window) must still be present after the full pre-scan (B4).
    for i in range(40):
      _write_pair(tmp_path, f"m{i}", mask_value=1)
    _write_pair(tmp_path, "late", mask_value=7)
    p, args = self._pipeline(tmp_path, val_split=0.0)
    train, _val = p._load_segfolder(str(tmp_path), args)
    assert p.mask_map == {1: 0, 7: 1}
    assert len(p.mask_map) == 2


class TestParseDatasetSpecs:

  def test_segfolder_prefix(self):
    specs = parse_dataset_specs(["segfolder//data/voc"])
    assert specs[0]["source"] == "segfolder"
    assert specs[0]["name"] == "/data/voc"

  def test_imagefolder_prefix_unaffected(self):
    specs = parse_dataset_specs(["imagefolder//data/mnist"])
    assert specs[0]["source"] == "imagefolder"
