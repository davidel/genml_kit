"""Unit tests for the v4.2 pipeline + method registries and contracts.

Covers the plan (plans/GENERIC_PIPELINE.md s 11.1) unit-level surface:

- pipeline registry: register/get/build/list + duplicate/unknown fatals;
- method registry: register/get/build/list;
- contracts: DataBlob / LossOutput namedtuple shapes, no ModelOutput
  collision (the name is owned by genml_kit.models.registry);
- ImagesPipeline: label-carrying supervised loader + label-free
  self-supervised loader both yield DataBlob;
- VOPairPipeline: loader yields DataBlob((a, b), meta);
- ClassificationMethod / VOPairMethod: train_step LossOutput + metric_key
  + has_metric_improved direction.
"""

import argparse

import pytest
import torch
from torch.utils.data import DataLoader

from genml_kit.methods import build_method, get_method, list_methods
from genml_kit.methods.registry import register_method
from genml_kit.pipelines import (
    DataBlob,
    LossOutput,
    build_pipeline,
    get_pipeline,
    list_pipelines,
    register_pipeline,
)
from genml_kit.pipelines.contracts import (
    DataBlob as ContractsDataBlob,
    LossOutput as ContractsLossOutput,
)
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.images import ImagesPipeline, build_pretrain_transform
from genml_kit.pipelines.vo_pair import VOPairPipeline

# --- Registries -----------------------------------------------------------


class TestPipelineRegistry:

  def test_list_includes_builtins(self):
    names = list_pipelines()
    assert names == sorted(names)
    assert "images" in names
    assert "vo_pair" in names

  def test_get_returns_class(self):
    cls = get_pipeline("images")
    assert cls is ImagesPipeline
    assert issubclass(cls, DataPipeline)

  def test_build_instantiates(self):
    pipeline = build_pipeline("images")
    assert isinstance(pipeline, ImagesPipeline)

  def test_unknown_raises(self):
    with pytest.raises(ValueError, match="Unknown pipeline"):
      get_pipeline("nope")

  def test_duplicate_registration_raises(self):

    @register_pipeline
    class _Dup(DataPipeline):
      NAME = "dup_pipeline"

    with pytest.raises(RuntimeError, match="Duplicate pipeline name"):

      @register_pipeline
      class _Dup2(DataPipeline):
        NAME = "dup_pipeline"


class TestMethodRegistry:

  def test_list_includes_all_seven(self):
    methods = list_methods()
    assert methods == sorted(methods)
    for name in ("byol", "classification", "dino", "ijepa", "simmim", "supcon",
                 "vo_pair"):
      assert name in methods

  def test_get_and_build(self):
    cls = get_method("simmim")
    assert cls.NAME == "simmim"
    instance = build_method("simmim")
    assert instance.NAME == "simmim"

  def test_unknown_raises(self):
    with pytest.raises(ValueError, match="Unknown method"):
      get_method("nope")


class TestCustomRegistration:
  """Registering a fresh pipeline/method is additive and instantiable."""

  def test_register_custom_pipeline(self):

    @register_pipeline
    class _Custom(DataPipeline):
      NAME = "custom_pipe"

    assert build_pipeline("custom_pipe").NAME == "custom_pipe"

  def test_register_custom_method(self):
    from genml_kit.methods.base import Method

    @register_method
    class _Custom(Method):
      NAME = "custom_method"

      def build_model(self, args, device):
        return None

      def train_step(self, model, blob, global_step, *, labels=None):
        return LossOutput(loss=torch.zeros(()), metrics={})

    assert build_method("custom_method").NAME == "custom_method"


# --- Contracts -------------------------------------------------------------


class TestContracts:

  def test_datablob_shape(self):
    blob = DataBlob(data=torch.zeros(2), meta={"labels": torch.ones(2)})
    assert blob.data.shape == (2,)
    assert blob.meta["labels"].sum() == 2
    # Contracts module exposes the same namedtuples.
    assert ContractsDataBlob is DataBlob

  def test_lossoutput_shape(self):
    out = LossOutput(loss=torch.zeros(()), metrics={"loss": torch.zeros(())})
    assert out.loss.ndim == 0
    assert "loss" in out.metrics
    assert ContractsLossOutput is LossOutput

  def test_no_modeloutput_here(self):
    # The name "ModelOutput" belongs to genml_kit.models.registry and must
    # NOT be re-defined by the pipeline contracts.
    import genml_kit.models.registry as models_registry
    assert hasattr(models_registry, "ModelOutput")
    assert not hasattr(
        __import__("genml_kit.pipelines.contracts", fromlist=["ModelOutput"]),
        "ModelOutput")


# --- Pipelines -------------------------------------------------------------


class _ImageFolderArgs(argparse.Namespace):
  """Minimal args for the images pipeline with an on-disk imagefolder."""

  batch_size = 4
  num_workers = 0
  seed = 42
  cache_dir = None
  hf_token = None
  strict_datasets = False
  image_size = 32
  val_split = 0.2


def _make_imagefolder(root, n=8, classes=("a", "b")):
  """Create an imagefolder fixture (root/<class>/<file>.jpg)."""
  from PIL import Image

  root = root / "images"
  root.mkdir(parents=True, exist_ok=True)
  for cls in classes:
    cls_dir = root / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
      img = Image.fromarray(
          torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8).numpy())
      img.save(cls_dir / f"{cls}_{i}.jpg")
  return root


class TestImagesPipelineLoader:

  def _args(self, tmp_path):
    # The sandbox home has no write access to ~/.cache/huggingface; point
    # the HF datasets cache at the test's tmp dir.
    import os
    os.environ.setdefault("HF_HOME", str(tmp_path / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(tmp_path / "hf_home" / "datasets"))
    args = _ImageFolderArgs()
    args.cache_dir = str(tmp_path / "hf_cache")
    return args

  def test_supervised_labels_in_meta(self, tmp_path):
    data_dir = _make_imagefolder(tmp_path)
    args = self._args(tmp_path)
    args.dataset = f"imagefolder/{data_dir}"
    args.label_column = "label"
    args.image_column = "image"
    args.sampler = "none"
    args.class_multipliers = ""
    args.sampler_weights = "frequency"
    # In production the CLI resolves these from the model processor; here
    # a simple resize->tensor transform stands in for the pipeline test.
    args.train_transforms = build_pretrain_transform(args.image_size)
    args.val_transforms = build_pretrain_transform(args.image_size)
    args.tta_transform = None
    args.needs_labels = True

    pipeline = ImagesPipeline()
    loader = pipeline.build_loader(args, mode="train")
    assert isinstance(loader, DataLoader)
    assert pipeline.num_labels == 2
    assert pipeline.train_loader is loader
    batch = next(iter(loader))
    assert isinstance(batch, DataBlob)
    assert batch.data.ndim == 4  # (B, C, H, W)
    assert batch.meta["labels"] is not None
    assert batch.meta["labels"].shape[0] == batch.data.shape[0]

  def test_balanced_sampler_wired_into_train_loader(self, tmp_path):
    """Regression for #10: --sampler balanced builds a BalancedBatchSampler."""
    from genml_kit.datasets.balanced_sampler import BalancedBatchSampler

    data_dir = _make_imagefolder(tmp_path)
    args = self._args(tmp_path)
    args.dataset = f"imagefolder/{data_dir}"
    args.label_column = "label"
    args.image_column = "image"
    args.sampler = "balanced"
    args.samples_per_class = 2
    args.batch_size = 4
    args.class_multipliers = ""
    args.sampler_weights = "frequency"
    args.train_transforms = build_pretrain_transform(args.image_size)
    args.val_transforms = build_pretrain_transform(args.image_size)
    args.tta_transform = None
    args.needs_labels = True

    pipeline = ImagesPipeline()
    loader = pipeline.build_loader(args, mode="train")
    assert isinstance(loader.sampler, BalancedBatchSampler)

  def test_balanced_sampler_falls_back_on_indivisible_batch(self, tmp_path):
    """batch_size % samples_per_class != 0 => warn and shuffle."""
    data_dir = _make_imagefolder(tmp_path)
    args = self._args(tmp_path)
    args.dataset = f"imagefolder/{data_dir}"
    args.label_column = "label"
    args.image_column = "image"
    args.sampler = "balanced"
    args.samples_per_class = 3
    args.batch_size = 4  # 4 % 3 != 0 -> fallback
    args.class_multipliers = ""
    args.sampler_weights = "frequency"
    args.train_transforms = build_pretrain_transform(args.image_size)
    args.val_transforms = build_pretrain_transform(args.image_size)
    args.tta_transform = None
    args.needs_labels = True

    from torch.utils.data.sampler import RandomSampler

    from genml_kit.datasets.balanced_sampler import BalancedBatchSampler

    pipeline = ImagesPipeline()
    loader = pipeline.build_loader(args, mode="train")
    # shuffle=True -> torch creates a RandomSampler, NOT a balanced one.
    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, BalancedBatchSampler)

  def test_self_supervised_ignores_labels(self, tmp_path):
    data_dir = _make_imagefolder(tmp_path)
    args = self._args(tmp_path)
    args.datasets = [str(data_dir)]
    args.label_column = "label"
    args.image_column = "image"
    args.needs_labels = False
    args.train_transforms = None
    args.val_transforms = None
    args.hf_token = None
    args.strict_datasets = False

    pipeline = ImagesPipeline()
    loader = pipeline.build_loader(args, mode="train")
    batch = next(iter(loader))
    assert isinstance(batch, DataBlob)
    assert batch.data.ndim == 4
    assert batch.meta["labels"] is None
    assert pipeline.ensemble is not None


class TestVOPairPipeline:

  def test_loader_yields_datablob(self):
    args = argparse.Namespace(
        image_size=16,
        batch_size=4,
        num_workers=0,
        vo_length=8,
        vo_val_length=4,
        pitch_deg=45.0,
        terrain="field",
        seeded_pairs=0,
    )
    pipeline = VOPairPipeline()
    loader = pipeline.build_loader(args, mode="train")
    batch = next(iter(loader))
    assert isinstance(batch, DataBlob)
    a, b = batch.data
    assert a.shape == b.shape
    assert a.shape[0] == 4  # batch_size
    assert a.shape[2] == 16
    assert "gt" in batch.meta
    assert "gt_residual" in batch.meta
    assert "terrain" in batch.meta
    assert "range_bin" in batch.meta
    assert batch.meta["gt_residual"].shape[0] == 4

  def test_val_loader_cached(self):
    args = argparse.Namespace(
        image_size=16,
        batch_size=4,
        num_workers=0,
        vo_length=8,
        vo_val_length=4,
        pitch_deg=45.0,
        terrain="field",
        seeded_pairs=0,
    )
    pipeline = VOPairPipeline()
    train_loader = pipeline.build_loader(args, mode="train")
    val_loader = pipeline.build_loader(args, mode="val")
    assert pipeline.train_loader is train_loader
    assert pipeline.val_loader is val_loader


# --- Methods ---------------------------------------------------------------


class TestClassificationMethod:

  def test_metric_key_and_direction(self):
    method = build_method("classification")
    assert method.METRIC_KEY == "macro_f1"
    assert method.has_metric_improved(0.9, 0.8) is True
    assert method.has_metric_improved(0.7, 0.8) is False

  def test_train_step_mixup_disabled(self):
    method = build_method("classification")

    class Cfg:
      focal_gamma = 0.0
      label_smoothing = 0.0

    method.set_mixup_alpha(0.0)
    method.set_label_space(2, {0: "a", 1: "b"}, {"a": 0, "b": 1})
    method.build_criterion(Cfg(), class_weights=torch.ones(2))

    import torch.nn as nn

    class _Model(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

      def forward(self, pixel_values):
        # pixel_values: (B, 3, H, W) -> global pool (B,) -> (B, 2).
        x = pixel_values.mean(dim=[1, 2, 3]).unsqueeze(1)  # (B, 1)
        return type("Out", (), {"logits": self.fc(x.repeat(1, 4))})()

    model = _Model()
    blob = DataBlob(data=torch.randn(4, 3, 8, 8),
                    meta={"labels": torch.tensor([0, 1, 0, 1])})
    out = method.train_step(model, blob, 0)
    assert isinstance(out, LossOutput)
    assert out.loss.ndim == 0
    assert "top1" in out.metrics


class TestVOPairMethod:

  def test_metric_key_and_direction(self):
    method = build_method("vo_pair")
    assert method.METRIC_KEY == "mce"
    # mce is minimized.
    assert method.has_metric_improved(0.5, 0.8) is True
    assert method.has_metric_improved(0.9, 0.8) is False

  def test_stage_default(self):
    method = build_method("vo_pair")
    assert method.NAME == "vo_pair"


class TestMetricDirection:

  def test_loss_keyed_methods_minimize(self):
    for name in ("dino", "byol", "supcon", "simmim", "ijepa"):
      method = build_method(name)
      assert method.METRIC_KEY == "loss"
      assert method.METRIC_MINIMIZE is True
      # Lower loss is better.
      assert method.has_metric_improved(0.5, 0.8) is True
      assert method.has_metric_improved(0.9, 0.8) is False

  def test_default_metric_sentinel_follows_direction(self):
    from genml_kit.training.train import _default_metric
    # Loss-keyed: has_metric_improved(0.0, 1.0) is True (0 < 1) -> -inf.
    dino = build_method("dino")
    assert _default_metric(dino) == float("inf")
    # Classification: maximizes; sentinel is -inf so the first epoch wins.
    cls = build_method("classification")
    assert _default_metric(cls) == float("-inf")
