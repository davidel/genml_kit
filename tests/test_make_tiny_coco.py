"""Tests for scripts/make_tiny_coco.py (tiny COCO-JSON corpus generator)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_tiny_coco.py"
_spec = importlib.util.spec_from_file_location("make_tiny_coco", _SCRIPT)
make_tiny_coco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_tiny_coco)


def _run(root, *argv):
  """Run the generator writing into *root* and return the parsed corpus."""
  make_tiny_coco.main([str(root), *[str(a) for a in argv]])
  return json.loads((root / "annotations.json").read_text())


def test_writes_expected_structure(tmp_path):
  coco = _run(tmp_path / "c", "-n", "3")
  assert (tmp_path / "c" / "images").is_dir()
  assert (tmp_path / "c" / "annotations.json").is_file()
  assert set(coco) == {"images", "annotations", "categories"}


def test_counts_honour_flags(tmp_path):
  root = tmp_path / "c"
  coco = _run(root, "-n", "5", "--boxes_per_image", "3")
  assert len(coco["images"]) == 5
  assert len(coco["annotations"]) == 15
  assert len(list((root / "images").glob("*.png"))) == 5


def test_image_records_are_consistent(tmp_path):
  root = tmp_path / "c"
  coco = _run(root, "-n", "4", "--size", "64")
  names = {entry["file_name"] for entry in coco["images"]}
  assert names == {p.name for p in (root / "images").glob("*.png")}
  for entry in coco["images"]:
    assert entry["width"] == 64
    assert entry["height"] == 64


def test_boxes_are_in_frame_with_area(tmp_path):
  size = 96
  coco = _run(tmp_path / "c", "-n", "6", "--size", str(size))
  for ann in coco["annotations"]:
    x, y, w, h = ann["bbox"]
    assert w > 0 and h > 0
    assert x >= 0 and y >= 0 and x + w <= size and y + h <= size
    assert ann["area"] == w * h
    assert ann["iscrowd"] == 0


def test_categories_are_one_based_and_dense(tmp_path):
  coco = _run(tmp_path / "c", "-n", "2", "--num_classes", "4")
  assert [c["id"] for c in coco["categories"]] == [1, 2, 3, 4]
  ids = {ann["category_id"] for ann in coco["annotations"]}
  assert ids <= {1, 2, 3, 4}


def test_images_are_not_blank(tmp_path):
  root = tmp_path / "c"
  _run(root, "-n", "1", "--size", "64", "--boxes_per_image", "2")
  arr = np.asarray(Image.open(next((root / "images").glob("*.png"))))
  assert arr.shape == (64, 64, 3)
  assert int((arr != 0).any(axis=-1).sum()) > 0


def test_defaults_are_backward_compatible(tmp_path):
  coco = _run(tmp_path / "c")
  assert len(coco["images"]) == 12
  assert len(coco["categories"]) == 2


def test_seed_is_deterministic(tmp_path):
  _run(tmp_path / "a", "-n", "4", "--seed", "7")
  _run(tmp_path / "b", "-n", "4", "--seed", "7")
  assert (tmp_path / "a" / "annotations.json").read_bytes() == \
      (tmp_path / "b" / "annotations.json").read_bytes()
