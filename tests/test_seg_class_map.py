"""Tests for the segmentation class-map (unique-id compaction) feature.

Covers:

* ``--mask_column`` auto-detection when omitted;
* ``--image_column`` auto-detection (e.g. ``pixel_values``) when the
  dataset does not name the image column ``image``;
* sparse / 1-based / void-255 raw mask ids compacted into ``[0, C)``;
* RGB masks mapped via unique triplets;
* the disk cache (spec-keyed) hit / miss / invalidation behavior;
* ``--seg_num_classes`` still overriding the inferred count.
"""

import os

import numpy as np
import pytest
from PIL import Image

import datasets

from genml_kit.datasets.hf_proxy import HFDatasetProxy


def _make_ds(images=None, masks=None, image_col="image", mask_col="mask"):
  rows = []
  for i in range(len(masks)):
    rows.append({
        image_col:
            images[i] if images is not None else Image.fromarray(
                np.zeros((8, 8, 3), dtype=np.uint8)),
        mask_col:
            masks[i],
    })
  return datasets.Dataset.from_list(rows)


class TestMaskColumnAutoDetect:

  def test_detects_known_mask_name(self):
    ds = _make_ds(masks=[Image.fromarray(np.zeros((8, 8), dtype=np.uint8))])
    assert HFDatasetProxy.detect_mask_column(ds) == "mask"

  def test_detects_label_as_mask(self):
    ds = _make_ds(masks=[Image.fromarray(np.zeros((8, 8), dtype=np.uint8))],
                  mask_col="label")
    assert HFDatasetProxy.detect_mask_column(ds) == "label"

  def test_detects_annotation(self):
    ds = _make_ds(masks=[Image.fromarray(np.zeros((8, 8), dtype=np.uint8))],
                  mask_col="annotation")
    assert HFDatasetProxy.detect_mask_column(ds) == "annotation"

  def test_no_mask_column_returns_none(self):
    ds = datasets.Dataset.from_list([{
        "image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    }])
    assert HFDatasetProxy.detect_mask_column(ds) is None


class TestImageColumnAutoDetect:

  def test_pixel_values_detected(self):
    # No mask column here, so 'mask' cannot shadow the image detection.
    ds = datasets.Dataset.from_list([{
        "pixel_values": np.zeros((8, 8, 3), dtype=np.uint8),
    }])
    assert HFDatasetProxy.detect_image_column(ds) == "pixel_values"

  def test_explicit_image_column_wins(self):
    ds = _make_ds(images=[np.zeros((8, 8, 3), dtype=np.uint8)],
                  masks=[Image.fromarray(np.zeros((8, 8), dtype=np.uint8))],
                  image_col="pixel_values")
    # Explicit column takes precedence regardless of what auto-detect says.
    proxy = HFDatasetProxy(ds, image_column="pixel_values", mask_column="mask")
    assert proxy._image_col == "pixel_values"


class TestMaskRemap:

  def test_sparse_ids_compacted(self):
    """{0, 5, 7} -> {0:0, 5:1, 7:2} (C=3)."""
    mask = np.array([[0, 5], [7, 0]], dtype=np.uint8)
    mask_map = {0: 0, 5: 1, 7: 2}
    out = HFDatasetProxy._apply_mask_map(mask, mask_map)
    assert out.tolist() == [[0, 1], [2, 0]]

  def test_one_based_ids_no_phantom(self):
    """{1,2,3} -> {1:0, 2:1, 3:2}; no phantom class 0."""
    mask = np.array([[1, 2], [3, 1]], dtype=np.uint8)
    mask_map = {1: 0, 2: 1, 3: 2}
    out = HFDatasetProxy._apply_mask_map(mask, mask_map)
    assert out.tolist() == [[0, 1], [2, 0]]

  def test_void_255_excluded_from_map(self):
    # The pipeline scan excludes 255; _apply_mask_map would fatal on it,
    # which is asserted by test_missing_key_fatals.  Here we verify the
    # map-building behavior via the pipeline helper.
    from genml_kit.pipelines.segmentation import SegmentationPipeline
    p = SegmentationPipeline()
    args = type(
        "A", (), {
            "cache_dir": None,
            "image_column": None,
            "mask_column": "mask",
            "seg_ignore_index": 255,
        })()
    ds = _make_ds(masks=[Image.fromarray(np.array([[0, 255]], dtype=np.uint8))])
    m = p._load_or_build_mask_map(args, "ds", "image", "mask", 255, ds)
    assert m == {0: 0}
    assert 255 not in m

  def test_rgb_mask_unique_triplets(self):
    """RGB mask -> tuple-key map compacted to ids."""
    mask = np.array([
        [[255, 0, 0], [0, 255, 0]],
        [[0, 0, 255], [255, 0, 0]],
    ],
                    dtype=np.uint8)
    mask_map = {(255, 0, 0): 0, (0, 255, 0): 1, (0, 0, 255): 2}
    out = HFDatasetProxy._apply_mask_map(mask, mask_map)
    assert out.tolist() == [[0, 1], [2, 0]]

  def test_missing_key_fatals(self):
    mask = np.array([[1]], dtype=np.uint8)
    with pytest.raises(ValueError):
      HFDatasetProxy._apply_mask_map(mask, {0: 0})


class TestMaskMapCache:

  def _pipeline(self, tmp_path, dataset_name="seg/ds", mask_col="label", ignore=255):
    from genml_kit.pipelines.segmentation import SegmentationPipeline
    p = SegmentationPipeline()
    args = type(
        "Args", (), {
            "cache_dir": str(tmp_path),
            "image_column": None,
            "mask_column": mask_col,
            "seg_ignore_index": ignore,
        })()
    return p, args, dataset_name

  def test_write_and_read_back(self, tmp_path):
    p, args, name = self._pipeline(tmp_path)
    ds = _make_ds(masks=[Image.fromarray(np.array([[1, 2], [3, 1]], dtype=np.uint8))],
                  mask_col="label")
    mapping = p._load_or_build_mask_map(args, name, "image", "label", 255, ds)
    assert mapping == {1: 0, 2: 1, 3: 2}
    # Second call hits the cache (same spec) without rescanning.
    mapping2 = p._load_or_build_mask_map(args, name, "image", "label", 255, ds)
    assert mapping2 == mapping

  def test_spec_change_invalidates(self, tmp_path):
    p, args, name = self._pipeline(tmp_path)
    ds = _make_ds(masks=[Image.fromarray(np.array([[1, 2]], dtype=np.uint8))],
                  mask_col="label")
    m1 = p._load_or_build_mask_map(args, name, "image", "label", 255, ds)
    assert m1 == {1: 0, 2: 1}
    # Different mask column -> different cache key -> rebuild.
    args.mask_column = "mask"
    ds2 = _make_ds(masks=[Image.fromarray(np.array([[5]], dtype=np.uint8))],
                   mask_col="mask")
    m2 = p._load_or_build_mask_map(args, name, "image", "mask", 255, ds2)
    assert m2 == {5: 0}
    assert m1 != m2

  def test_ignore_index_in_key(self, tmp_path):
    p, args, name = self._pipeline(tmp_path)
    ds = _make_ds(masks=[Image.fromarray(np.array([[1, 255]], dtype=np.uint8))],
                  mask_col="label")
    m = p._load_or_build_mask_map(args, name, "image", "label", 255, ds)
    assert 255 not in m
    assert m == {1: 0}

  def test_cache_dir_fallback(self, tmp_path, monkeypatch):
    from genml_kit.pipelines.segmentation import SegmentationPipeline
    home = str(tmp_path / "home")
    os.makedirs(home, exist_ok=True)
    monkeypatch.setenv("HOME", home)
    assert os.path.isdir(
        SegmentationPipeline._class_map_cache_dir(type("A", (), {"cache_dir": None})()))


class TestSegNumClassesOverride:

  def test_override_wins_over_map(self):
    from genml_kit.methods.segmentation import SegmentationMethod
    m = SegmentationMethod()
    args = type("Args", (), {"seg_num_classes": 7})()
    pipeline = type("P", (), {"mask_map": {0: 0, 1: 1}})()
    assert m._resolve_num_classes(args, pipeline) == 7

  def test_map_used_when_no_override(self):
    from genml_kit.methods.segmentation import SegmentationMethod
    m = SegmentationMethod()
    args = type("Args", (), {"seg_num_classes": None})()
    pipeline = type("P", (), {"mask_map": {0: 0, 1: 1, 2: 2}})()
    assert m._resolve_num_classes(args, pipeline) == 3


class TestClassificationUnaffected:

  def test_proxy_label_detection_untouched(self):
    """Classification path (no mask_map) still returns (image, label)."""
    ds = datasets.Dataset.from_list([{
        "image": Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)),
        "label": "cat",
    }])
    proxy = HFDatasetProxy(ds, image_column="image", label_column="label")
    img, label = proxy[0]
    assert label == 0  # ClassLabel renumbers to 0 (single class)
    assert proxy.label_names == ["cat"]
    assert proxy.label_column == "label"
