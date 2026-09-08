"""Tests for the DatasetEnsemble dataset."""

import pytest
from PIL import Image

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.image_folder import ImageFolderDataset


class TestImageFolderDataset:

  def test_loads_images(self, tmp_path):
    """Create a temporary directory with images and verify loading."""
    for i in range(5):
      img = Image.new("RGB", (64, 64), color=(i * 50, 100, 200))
      img.save(tmp_path / f"img_{i}.png")
    ds = ImageFolderDataset(str(tmp_path))
    assert len(ds) == 5
    item = ds[0]
    assert isinstance(item, dict)
    assert "image" in item
    assert isinstance(item["image"], Image.Image)
    assert item["image"].mode == "RGB"

  def test_empty_dir(self, tmp_path):
    with pytest.raises(FileNotFoundError, match="No images found"):
      ImageFolderDataset(str(tmp_path))


class TestDatasetEnsemble:

  def test_imagefolder_ensemble(self, tmp_path):
    """Ensemble of two local image directories."""
    dir_a = tmp_path / "dataset_a"
    dir_b = tmp_path / "dataset_b"
    dir_a.mkdir()
    dir_b.mkdir()
    for i in range(3):
      Image.new("RGB", (64, 64), color=(100, 100, 100)).save(dir_a / f"{i}.png")
    for i in range(4):
      Image.new("RGB", (64, 64), color=(200, 200, 200)).save(dir_b / f"{i}.png")

    configs = [
        {
            "name": str(dir_a),
            "source": "imagefolder"
        },
        {
            "name": str(dir_b),
            "source": "imagefolder"
        },
    ]
    ensemble = DatasetEnsemble(configs)
    assert len(ensemble) == 7

    # Verify flat indexing crosses dataset boundaries
    item_a = ensemble[2]  # last image in dir_a
    item_b = ensemble[3]  # first image in dir_b
    assert isinstance(item_a, dict)
    assert isinstance(item_a["image"], Image.Image)
    assert isinstance(item_b["image"], Image.Image)

  def test_summary(self, tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(2):
      Image.new("RGB", (64, 64)).save(d / f"{i}.png")
    ensemble = DatasetEnsemble([{"name": str(d), "source": "imagefolder"}])
    s = ensemble.summary()
    assert "2 images" in s
    assert "1 dataset" in s

  def test_graceful_failure(self, tmp_path):
    """Error raised at init when every source fails."""
    configs = [{"name": "/nonexistent/path", "source": "imagefolder"}]
    with pytest.raises(RuntimeError, match="No datasets loaded"):
      DatasetEnsemble(configs)

  def test_empty_config(self):
    with pytest.raises(RuntimeError, match="No datasets loaded"):
      DatasetEnsemble([])

  def test_small_image_still_loaded(self, tmp_path):
    """Small images should be loaded without filtering."""
    d = tmp_path / "data"
    d.mkdir()
    Image.new("RGB", (8, 8)).save(d / "small.png")
    Image.new("RGB", (64, 64)).save(d / "large.png")
    ensemble = DatasetEnsemble([{
        "name": str(d),
        "source": "imagefolder",
    }])
    assert len(ensemble) == 2

  def test_getitem_out_of_range(self, tmp_path):
    """Out-of-range indices should raise instead of duplicating data."""
    d = tmp_path / "data"
    d.mkdir()
    Image.new("RGB", (64, 64)).save(d / "img.png")
    ensemble = DatasetEnsemble([{"name": str(d), "source": "imagefolder"}])
    with pytest.raises(IndexError, match="out of range"):
      ensemble[9999]
