"""Integration test for the dataset loading and preprocessing pipeline.

Creates synthetic in-memory datasets to exercise the full
load → detect columns → split → build → sample path without a GPU
and without downloading anything from the Hub.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import datasets
import numpy as np
import pytest
import torch
from PIL import Image

# Helpers


def _import_train():
  """Lazy-import genml_kit.training.train so module-level fixtures resolve cleanly."""
  import importlib

  import genml_kit.training.train

  importlib.reload(genml_kit.training.train)
  return genml_kit.training.train


def _make_synthetic_dataset(label_col="label", image_col="image", n=64, num_classes=3):
  """Create a synthetic HuggingFace Dataset with images and labels."""
  labels = [f"class_{i}" for i in range(num_classes)]

  def _rand_img():
    arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)

  data = {
      image_col: [_rand_img() for _ in range(n)],
      label_col: [labels[i % num_classes] for i in range(n)],
  }

  features = datasets.Features({
      image_col: datasets.Image(),
      label_col: datasets.ClassLabel(names=labels),
  })

  return datasets.Dataset.from_dict(data, features=features)


# Detect image column


class TestDetectImageColumn:

  def test_finds_image_column(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="dx", image_col="image")
    assert train_mod.HFDatasetProxy.detect_image_column(ds) == "image"

  def test_finds_nonstandard_name(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="class", image_col="img")
    assert train_mod.HFDatasetProxy.detect_image_column(ds) == "img"

  def test_returns_none_when_no_image(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict(
        {
            "data": [1.0, 2.0],
            "label": ["a", "b"],
        },
        features=datasets.Features({
            "data": datasets.Value("float32"),
            "label": datasets.ClassLabel(names=["a", "b"]),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_image_column(ds) is None

  def test_prefers_known_name_over_unknown(self):
    train_mod = _import_train()
    """When multiple Image columns exist, prefer the one with a known name."""
    imgs = [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(4)]
    ds = datasets.Dataset.from_dict(
        {
            "thumb": imgs,
            "image": imgs,
            "label": [0, 1, 0, 1]
        },
        features=datasets.Features({
            "thumb": datasets.Image(),
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=["a", "b"]),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_image_column(ds) == "image"

  def test_detects_filepath_string_column(self):
    train_mod = _import_train()
    """Detect a known image name even when stored as file path strings."""
    ds = datasets.Dataset.from_dict(
        {
            "path": ["/a.jpg", "/b.jpg"],
            "label": [0, 1]
        },
        features=datasets.Features({
            "path": datasets.Value("string"),
            "label": datasets.ClassLabel(names=["a", "b"]),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_image_column(ds) == "path"

  def test_filepath_cast_to_pil_image(self):
    train_mod = _import_train()
    """When image column is a string path, load_and_split_dataset should cast
    it to datasets.Image and return actual PIL images."""
    with tempfile.TemporaryDirectory() as tmpdir:
      paths = []
      for i in range(4):
        p = os.path.join(tmpdir, f"{i}.png")
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(p)
        paths.append(p)

      ds = datasets.Dataset.from_dict(
          {
              "image_file": paths,
              "label": ["a", "b", "a", "b"]
          },
          features=datasets.Features({
              "image_file": datasets.Value("string"),
              "label": datasets.ClassLabel(names=["a", "b"]),
          }),
      )
      with patch("genml_kit.training.train.load_dataset", return_value=ds):
        train_p, _val_p = train_mod.load_and_split_dataset("fake_ds")

      img = train_p.dataset[0]["image_file"]
      assert isinstance(img, Image.Image), (f"Expected PIL Image, got {type(img)}")


# Detect label column


class TestDetectLabelColumn:

  def test_prefers_label_classlabel(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="label", image_col="image")
    assert train_mod.HFDatasetProxy.detect_label_column(ds) == "label"

  def test_finds_category_classlabel(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="category", image_col="image")
    assert train_mod.HFDatasetProxy.detect_label_column(ds) == "category"

  def test_finds_breed_classlabel(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="breed", image_col="img")
    assert train_mod.HFDatasetProxy.detect_label_column(ds) == "breed"

  def test_finds_labels_plural(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="labels", image_col="image")
    assert train_mod.HFDatasetProxy.detect_label_column(ds) == "labels"

  def test_finds_species_classlabel(self):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="species", image_col="image")
    assert train_mod.HFDatasetProxy.detect_label_column(ds) == "species"

  def test_finds_string_value_column(self):
    train_mod = _import_train()
    raw = datasets.Dataset.from_dict(
        {
            "img": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(6)
            ],
            "category": ["cat", "dog", "cat", "dog", "cat", "dog"],
        },
        features=datasets.Features({
            "img": datasets.Image(),
            "category": datasets.Value("string"),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_label_column(raw) == "category"

  def test_finds_int_value_column(self):
    train_mod = _import_train()
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(4)
            ],
            "coarse_label": [0, 1, 0, 1],
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "coarse_label": datasets.Value("int64"),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_label_column(raw) == "coarse_label"

  def test_ignores_path_columns(self):
    train_mod = _import_train()
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(4)
            ],
            "file_path": ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg"],
            "target": [0, 1, 0, 1],
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "file_path": datasets.Value("string"),
            "target": datasets.Value("int64"),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_label_column(raw) == "target"

  def test_returns_none_when_no_label(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(2)
            ],
        },
        features=datasets.Features({
            "image": datasets.Image(),
        }),
    )
    assert train_mod.HFDatasetProxy.detect_label_column(ds) is None

  def test_unknown_classlabel_column_warns(self, caplog):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="lesion_type", image_col="image")
    with caplog.at_level("WARNING"):
      assert train_mod.HFDatasetProxy.detect_label_column(ds) == "lesion_type"
    assert any("No known label column name found" in r.message for r in caplog.records)

  def test_known_label_name_does_not_warn(self, caplog):
    train_mod = _import_train()
    ds = _make_synthetic_dataset(label_col="category", image_col="image")
    with caplog.at_level("WARNING"):
      assert train_mod.HFDatasetProxy.detect_label_column(ds) == "category"
    assert not any("No known label" in r.message for r in caplog.records)

  def test_last_resort_fallback_warns(self, caplog):
    """Priority 4: the first non-ignored string/int column wins, loudly."""
    train_mod = _import_train()
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(4)
            ],
            "patient_id": ["P1", "P2", "P3", "P4"],
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "patient_id": datasets.Value("string"),
        }),
    )
    with caplog.at_level("WARNING"):
      assert train_mod.HFDatasetProxy.detect_label_column(raw) == "patient_id"
    assert any("falling back to first string/int64 column 'patient_id'" in r.message
               for r in caplog.records)

  def test_no_label_produces_no_warning(self, caplog):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(2)
            ],
        },
        features=datasets.Features({
            "image": datasets.Image(),
        }),
    )
    with caplog.at_level("WARNING"):
      assert train_mod.HFDatasetProxy.detect_label_column(ds) is None
    assert not any("falling back" in r.message for r in caplog.records)


# Load and split dataset


class TestLoadAndSplit:

  def test_preserves_original_label_column_name(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(label_col="dx", image_col="image")

    with patch("genml_kit.training.train.load_dataset", return_value=raw):
      train_p, val_p = train_mod.load_and_split_dataset("fake_dataset")

    assert isinstance(train_p, train_mod.HFDatasetProxy)
    assert isinstance(val_p, train_mod.HFDatasetProxy)
    assert "dx" in train_p.dataset.column_names
    assert "dx" in val_p.dataset.column_names
    assert train_p.label_column == "dx"
    assert val_p.label_column == "dx"

  def test_preserves_label_column_if_already_named_label(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(label_col="label", image_col="image")

    with patch("genml_kit.training.train.load_dataset", return_value=raw):
      train_p, _val_p = train_mod.load_and_split_dataset("fake_dataset")

    assert "label" in train_p.dataset.column_names

  def test_raises_when_no_image_column(self):
    train_mod = _import_train()
    ds = datasets.Dataset.from_dict(
        {
            "data": [1.0, 2.0],
            "label": ["a", "b"],
        },
        features=datasets.Features({
            "data": datasets.Value("float32"),
            "label": datasets.ClassLabel(names=["a", "b"]),
        }),
    )

    with patch("genml_kit.training.train.load_dataset", return_value=ds), \
         pytest.raises(ValueError, match="No image column"):
      train_mod.load_and_split_dataset("fake_dataset")

  def test_class_encode_non_classlabel_column(self):
    train_mod = _import_train()
    """If the label column is a plain string Value (not ClassLabel),
    it should still work by auto-encoding."""
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                for _ in range(20)
            ],
            "diagnosis": ["benign"] * 10 + ["malignant"] * 10,
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "diagnosis": datasets.Value("string"),
        }),
    )

    with patch("genml_kit.training.train.load_dataset", return_value=raw):
      train_p, _val_p = train_mod.load_and_split_dataset("fake_dataset")
    assert "diagnosis" in train_p.dataset.column_names
    assert train_p.label_column == "diagnosis"


# Compute class weights


class TestComputeClassWeights:

  def test_balanced_classes_get_equal_weights(self):
    train_mod = _import_train()
    # Create a dataset where the TRAIN split has exactly 20 per class.
    # 20 benign + 20 malignant = 40 total train (50 train / 10 test with 20%)
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                for _ in range(50)
            ],
            "label": ["benign"] * 25 + ["malignant"] * 25,
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=["benign", "malignant"]),
        }),
    )
    # Create a dataset dict with exact train counts
    train_ds = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                for _ in range(40)
            ],
            "label": ["benign"] * 20 + ["malignant"] * 20,
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=["benign", "malignant"]),
        }),
    )
    datasets.DatasetDict({
        "train": train_ds,
        "test": raw,
    })

    weights = train_mod.compute_class_weights(train_ds, num_labels=2)
    assert weights.shape == (2,)
    # For balanced classes, all weights should be equal
    assert torch.allclose(weights, weights[0].expand_as(weights), atol=1e-6)

  def test_imbalanced_classes_get_inverse_weights(self):
    train_mod = _import_train()
    labels = ["benign"] * 90 + ["malignant"] * 10
    raw = datasets.Dataset.from_dict(
        {
            "image": [
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                for _ in range(100)
            ],
            "label": labels,
        },
        features=datasets.Features({
            "image": datasets.Image(),
            "label": datasets.ClassLabel(names=["benign", "malignant"]),
        }),
    )
    raw_split = raw.train_test_split(test_size=0.2, seed=42)
    train_ds = raw_split["train"]

    weights = train_mod.compute_class_weights(train_ds, num_labels=2)
    assert weights.shape == (2,)
    # The minority class (malignant) should have higher weight
    assert weights[1] > weights[0]


# HFDatasetProxy


class TestHFDatasetProxy:

  def test_len(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(n=20)
    proxy = train_mod.HFDatasetProxy(raw)
    assert len(proxy) == 20

  def test_getitem_returns_image_and_label(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(n=10)
    proxy = train_mod.HFDatasetProxy(raw)
    image, label = proxy[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)

  def test_getitem_with_transform(self):
    train_mod = _import_train()
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.Resize(size=(32, 32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    raw = _make_synthetic_dataset(n=10)
    proxy = train_mod.HFDatasetProxy(raw, transform=transform)
    image, _label = proxy[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)

  def test_num_labels(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(n=20, num_classes=3)
    proxy = train_mod.HFDatasetProxy(raw)
    assert proxy.num_labels == 3

  def test_num_labels_with_five_classes(self):
    train_mod = _import_train()
    raw = _make_synthetic_dataset(n=30, num_classes=5)
    proxy = train_mod.HFDatasetProxy(raw)
    assert proxy.num_labels == 5


# Build transforms


class TestBuildTransforms:

  def test_returns_train_and_val(self):
    train_mod = _import_train()
    # Mock processor with image_mean/image_std
    mock_processor = MagicMock()
    mock_processor.image_mean = [0.485, 0.456, 0.406]
    mock_processor.image_std = [0.229, 0.224, 0.225]

    train_t, val_t = train_mod.build_transforms(mock_processor, image_size=224)
    assert callable(train_t)
    assert callable(val_t)

  def test_train_transform_output_shape(self):
    train_mod = _import_train()
    mock_processor = MagicMock()
    mock_processor.image_mean = [0.5, 0.5, 0.5]
    mock_processor.image_std = [0.5, 0.5, 0.5]

    train_t, _ = train_mod.build_transforms(mock_processor, image_size=64)
    img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    result = train_t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 64, 64)

  def test_val_transform_output_shape(self):
    train_mod = _import_train()
    mock_processor = MagicMock()
    mock_processor.image_mean = [0.5, 0.5, 0.5]
    mock_processor.image_std = [0.5, 0.5, 0.5]

    _, val_t = train_mod.build_transforms(mock_processor, image_size=64)
    img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    result = val_t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 64, 64)
    assert result.dtype == torch.float32


# Parse args (updated CLI)


class TestLoadAugmentationScript:
  """Tests for load_augmentation_script()."""

  def test_loads_local_script(self, tmp_path):
    train_mod = _import_train()
    script = tmp_path / "my_aug.py"
    script.write_text("from torchvision.transforms import v2\n"
                      "\n"
                      "def create_train_transform(image_size, **kwargs):\n"
                      "    return [\n"
                      "        v2.RandomHorizontalFlip(p=0.5),\n"
                      "    ]\n")
    fn = train_mod.load_augmentation_script(str(script))
    assert callable(fn)
    result = fn(224)
    assert isinstance(result, list)
    assert len(result) == 1

  def test_missing_create_train_transform_raises(self, tmp_path):
    train_mod = _import_train()
    script = tmp_path / "bad_aug.py"
    script.write_text("def wrong_name():\n    pass\n")
    with pytest.raises(ValueError, match="does not define a callable"):
      train_mod.load_augmentation_script(str(script))

  def test_non_callable_raises(self, tmp_path):
    train_mod = _import_train()
    script = tmp_path / "bad_aug2.py"
    script.write_text("create_train_transform = 42\n")
    with pytest.raises(ValueError, match="does not define a callable"):
      train_mod.load_augmentation_script(str(script))

  def test_loads_url(self):
    train_mod = _import_train()
    url_script = ("from torchvision.transforms import v2\n"
                  "\n"
                  "def create_train_transform(image_size, **kwargs):\n"
                  "    return [v2.RandomVerticalFlip(p=1.0)]\n")
    with patch("urllib.request.urlopen") as mock_url:
      mock_resp = MagicMock()
      mock_resp.read.return_value = url_script.encode("utf-8")
      mock_resp.__enter__ = MagicMock(return_value=mock_resp)
      mock_resp.__exit__ = MagicMock(return_value=False)
      mock_url.return_value = mock_resp

      fn = train_mod.load_augmentation_script("https://example.com/aug.py")
      assert callable(fn)
      result = fn(128)
      assert isinstance(result, list)
      assert len(result) == 1


class TestBuildTransformsWithCustomAug:
  """Tests for build_transforms() with a custom train_aug_fn."""

  def _mock_processor(self):
    mock_processor = MagicMock()
    mock_processor.image_mean = [0.485, 0.456, 0.406]
    mock_processor.image_std = [0.229, 0.224, 0.225]
    return mock_processor

  def test_custom_fn_replaces_train_augs(self):
    train_mod = _import_train()
    from torchvision.transforms import v2

    def my_aug(image_size):
      return [v2.RandomHorizontalFlip(p=1.0)]

    train_t, val_t = train_mod.build_transforms(self._mock_processor(),
                                                image_size=64,
                                                train_aug_fn=my_aug)
    assert callable(train_t)
    assert callable(val_t)
    # Train transform should produce output (custom aug + tail).
    # No resize/crop in custom augs, so output matches input size.
    img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    result = train_t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 128, 128)

  def test_custom_fn_empty_list_still_has_tail(self):
    train_mod = _import_train()

    def empty_aug(image_size):
      return []

    train_t, _ = train_mod.build_transforms(self._mock_processor(),
                                            image_size=64,
                                            train_aug_fn=empty_aug)
    img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    result = train_t(img)
    assert isinstance(result, torch.Tensor)
    # No resize/crop, so 128x128 input stays 128x128
    assert result.shape == (3, 128, 128)

  def test_custom_fn_wrong_return_type_raises(self):
    train_mod = _import_train()

    def bad_aug(image_size):
      return "not a list"

    with pytest.raises(TypeError, match="must return a list"):
      train_mod.build_transforms(self._mock_processor(),
                                 image_size=64,
                                 train_aug_fn=bad_aug)

  def test_none_fn_uses_default(self):
    train_mod = _import_train()
    # Passing train_aug_fn=None should behave like the original default
    train_t, _val_t = train_mod.build_transforms(self._mock_processor(),
                                                 image_size=64,
                                                 train_aug_fn=None)
    img = Image.fromarray(np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
    result = train_t(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 64, 64)


class TestArgParsing:

  def test_defaults(self):
    train_mod = _import_train()
    args = train_mod.parse_args(["--dataset", "my-org/my-images"])
    assert args.model == "google/vit-base-patch16-224"
    assert args.dataset == "my-org/my-images"
    assert args.epochs == 5
    assert args.image_size == 448
    assert args.lr == 3e-5
    assert args.batch_size == 32
    assert args.log_every == 20
    assert args.save_every == 500
    assert args.amp_dtype is None

  def test_overrides(self):
    train_mod = _import_train()
    args = train_mod.parse_args([
        "--model",
        "my-model",
        "--dataset",
        "my-dataset",
        "--epochs",
        "3",
        "--image_size",
        "224",
    ])
    assert args.model == "my-model"
    assert args.dataset == "my-dataset"
    assert args.epochs == 3
    assert args.image_size == 224

  def test_amp_dtype_choices(self):
    train_mod = _import_train()
    args = train_mod.parse_args(
        ["--dataset", "my-org/my-images", "--amp_dtype", "bfloat16"])
    assert args.amp_dtype == "bfloat16"
