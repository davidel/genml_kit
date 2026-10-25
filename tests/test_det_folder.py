"""Tests for the COCO-JSON detection dataset (DetFolderDataset)."""

import json

import pytest
import torch

from genml_kit.datasets.det_folder import DetFolderDataset, load_detfolder


def _write_coco(root, images, annotations, categories, name="annotations.json"):
  """Write a minimal COCO detection JSON plus placeholder image files."""
  (root / "images").mkdir(parents=True, exist_ok=True)
  from PIL import Image
  import numpy as np

  for entry in images:
    arr = np.zeros((entry.get("height", 32), entry.get("width", 32), 3), dtype="uint8")
    Image.fromarray(arr).save(root / "images" / entry["file_name"])
  payload = {"images": images, "annotations": annotations, "categories": categories}
  (root / name).write_text(json.dumps(payload))


def _images():
  return [
      {
          "id": 1,
          "file_name": "a.png",
          "width": 32,
          "height": 32
      },
      {
          "id": 2,
          "file_name": "b.png",
          "width": 32,
          "height": 32
      },
  ]


def _categories():
  # COCO ids are 1-based and sparse on purpose, to exercise compaction.
  return [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}]


class TestDetFolderDataset:

  def test_bbox_xywh_converted_to_xyxy(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [4, 6, 10, 20]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    ds = DetFolderDataset(tmp_path)
    item = ds[0]
    # [x, y, w, h] = [4, 6, 10, 20] -> [x1, y1, x2, y2] = [4, 6, 14, 26].
    assert torch.allclose(item["boxes"], torch.tensor([[4.0, 6.0, 14.0, 26.0]]))

  def test_categories_compacted_to_dense_range(self, tmp_path):
    annotations = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 5, 5]
        },
        {
            "image_id": 1,
            "category_id": 3,
            "bbox": [10, 10, 5, 5]
        },
    ]
    _write_coco(tmp_path, _images(), annotations, _categories())
    ds = DetFolderDataset(tmp_path)
    item = ds[0]
    # Raw ids {1, 3} compact to {0, 1} in sorted order.
    assert item["labels"].tolist() == [0, 1]
    assert ds.id2label == {0: "cat", 1: "dog"}
    assert ds.num_classes == 2

  def test_image_without_boxes_is_background(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    ds = DetFolderDataset(tmp_path)
    # Image id 2 has no annotations -> empty (0, 4) boxes / (0,) labels.
    assert ds[1]["boxes"].shape == (0, 4)
    assert ds[1]["labels"].shape == (0,)

  def test_degenerate_boxes_dropped(self, tmp_path):
    annotations = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [0, 0, 0, 5]
        },
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [1, 1, 5, 5]
        },
    ]
    _write_coco(tmp_path, _images(), annotations, _categories())
    ds = DetFolderDataset(tmp_path)
    # The zero-width box is dropped; only the valid one remains.
    assert ds[0]["boxes"].shape == (1, 4)

  def test_getitem_returns_independent_tensors(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    ds = DetFolderDataset(tmp_path)
    first = ds[0]
    first["boxes"][0, 0] = 999.0
    # A second read must not see the mutation (cloned records).
    assert ds[0]["boxes"][0, 0].item() == 0.0

  def test_missing_image_file_is_fatal(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    (tmp_path / "images" / "b.png").unlink()
    with pytest.raises(FileNotFoundError):
      DetFolderDataset(tmp_path)

  def test_missing_annotation_file_is_fatal(self, tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError):
      DetFolderDataset(tmp_path)

  def test_malformed_json_is_fatal(self, tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations.json").write_text("{not json")
    with pytest.raises(ValueError):
      DetFolderDataset(tmp_path)


class TestLoadDetFolder:

  def test_flat_layout_returns_train(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    splits = load_detfolder(tmp_path)
    assert list(splits) == ["train"]

  def test_split_layout(self, tmp_path):
    for split in ("train", "val"):
      root = tmp_path / split
      annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
      _write_coco(root, [{
          "id": 1,
          "file_name": f"{split}.png"
      }], annotations, _categories())
    splits = load_detfolder(tmp_path)
    assert set(splits) == {"train", "val"}

  def test_validation_split_normalised_to_val(self, tmp_path):
    for split in ("train", "validation"):
      root = tmp_path / split
      annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
      _write_coco(root, [{
          "id": 1,
          "file_name": f"{split}.png"
      }], annotations, _categories())
    splits = load_detfolder(tmp_path)
    assert set(splits) == {"train", "val"}

  def test_split_selector(self, tmp_path):
    for split in ("train", "val"):
      root = tmp_path / split
      annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
      _write_coco(root, [{
          "id": 1,
          "file_name": f"{split}.png"
      }], annotations, _categories())
    only = load_detfolder(tmp_path, split="val")
    assert list(only) == ["val"]

  def test_unknown_split_is_fatal(self, tmp_path):
    annotations = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]}]
    _write_coco(tmp_path, _images(), annotations, _categories())
    with pytest.raises(ValueError):
      load_detfolder(tmp_path, split="test")
