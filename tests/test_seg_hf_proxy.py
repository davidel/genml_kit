"""Tests for HFDatasetProxy mask_column support and the CLI surface."""

import numpy as np
from PIL import Image

import datasets


def _make_hf_dataset():
  rows = []
  rng = np.random.default_rng(0)
  for _ in range(4):
    rows.append({
        "image": Image.fromarray(rng.integers(0, 255, (12, 12, 3), dtype=np.uint8)),
        "mask": Image.fromarray(rng.integers(0, 3, (12, 12), dtype=np.uint8)),
    })
  return datasets.Dataset.from_list(rows)


class TestHFDatasetProxyMask:

  def test_mask_column_returns_dict(self):
    from genml_kit.datasets.hf_proxy import HFDatasetProxy

    ds = _make_hf_dataset()
    proxy = HFDatasetProxy(ds, image_column="image", mask_column="mask")
    item = proxy[0]
    assert isinstance(item, dict)
    assert item["image"].mode == "RGB"
    assert item["mask"].dtype == np.int64
    assert item["mask"].shape == (12, 12)
    assert proxy.mask_column == "mask"
    assert not proxy.has_labels

  def test_no_mask_column_keeps_tuple_contract(self):
    from genml_kit.datasets.hf_proxy import HFDatasetProxy

    ds = datasets.Dataset.from_list([{
        "image": Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)),
        "label": 1
    }])
    proxy = HFDatasetProxy(ds, image_column="image", label_column="label")
    image, label = proxy[0]
    assert image.mode == "RGB"
    # The label is normalised to a ClassLabel (single class -> id 0).
    assert label == 0
    assert proxy.mask_column is None


class TestSegCLISurface:

  def test_parser_builds_with_seg_owner(self):
    from genml_kit.cli.train import build_parser, register_all_owners

    parser = build_parser()
    register_all_owners(parser)  # must not raise on option collisions
    args = parser.parse_args([
        "--method", "segmentation", "--pipeline", "segmentation", "--mask_column",
        "segmentations"
    ])
    assert args.mask_column == "segmentations"
    assert args.method == "segmentation"

  def test_seg_method_flags(self):
    from genml_kit.cli.train import build_parser, register_all_owners

    parser = build_parser()
    register_all_owners(parser)
    args = parser.parse_args(["--seg_num_classes", "7", "--seg_ignore_index", "0"])
    assert args.seg_num_classes == 7
    assert args.seg_ignore_index == 0
