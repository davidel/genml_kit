"""Tests for the timm model backend."""

import torch
from PIL import Image

from genml_kit.models import (
    ModelOutput,
    is_custom_model,
    load_model,
    load_processor,
)
from genml_kit.models.timm.model import TimmForClassification
from genml_kit.models.timm.processor import TimmProcessor
from genml_kit.training.model_utils import extract_backbone_features

_TINY_MODEL = "resnet18"


def _make_timm_model(num_classes=5):
  """Build a tiny timm model wrapped for the genml_kit protocol."""
  from types import SimpleNamespace

  import timm

  model = timm.create_model(_TINY_MODEL, pretrained=False, num_classes=num_classes)
  config = SimpleNamespace(
      num_labels=num_classes,
      id2label={i: f"cls_{i}" for i in range(num_classes)},
      label2id={f"cls_{i}": i for i in range(num_classes)},
  )
  return TimmForClassification(model, config)


def _make_timm_processor(image_size=224):
  """Build a TimmProcessor for the tiny model."""
  import timm
  import timm.data

  model = timm.create_model(_TINY_MODEL, pretrained=False)
  data_config = timm.data.resolve_data_config(model.pretrained_cfg)
  return TimmProcessor(data_config, image_size=image_size)


class TestRegistry:
  """Verify timm is discoverable via the model registry."""

  def test_is_custom_model(self):
    assert is_custom_model("timm") is True

  def test_load_model_returns_module(self):
    """Verify dispatch through the registry.

    Uses ``pretrained=False`` so no download is needed.
    """
    model = load_model(
        f"timm:{_TINY_MODEL}",
        num_labels=3,
        id2label={
            0: "a",
            1: "b",
            2: "c"
        },
        label2id={
            "a": 0,
            "b": 1,
            "c": 2
        },
        image_size=224,
        device=torch.device("cpu"),
        pretrained=False,
    )
    assert isinstance(model, TimmForClassification)

  def test_load_processor(self):
    proc = load_processor(
        f"timm:{_TINY_MODEL}",
        image_size=224,
    )
    assert isinstance(proc, TimmProcessor)


class TestTimmForClassification:
  """Tests for the wrapper module."""

  def test_forward_returns_model_output(self):
    model = _make_timm_model(num_classes=5)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    out = model(pixel_values=x)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 5)

  def test_single_sample(self):
    model = _make_timm_model(num_classes=3)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    out = model(pixel_values=x)
    assert out.logits.shape == (1, 3)

  def test_config_accessible(self):
    model = _make_timm_model(num_classes=4)
    assert model.config.num_labels == 4
    assert model.config.id2label[2] == "cls_2"
    assert model.config.label2id["cls_1"] == 1

  def test_classifier_property(self):
    model = _make_timm_model()
    # resnet18 uses .fc
    assert model.classifier is not None
    assert isinstance(model.classifier, torch.nn.Module)

  def test_extract_backbone_features(self):
    model = _make_timm_model(num_classes=5)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_backbone_features(x)
    assert features.ndim == 2
    assert features.shape[0] == 2
    # resnet18 has 512-dim features
    assert features.shape[1] == 512


class TestExtractBackboneFeatures:
  """Verify the genml_kit extract_backbone_features hook works."""

  def test_hook_based_extraction(self):
    """extract_backbone_features should find the .fc head and use it."""
    model = _make_timm_model(num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    features = extract_backbone_features(model, x)
    assert features.ndim == 2
    assert features.shape == (1, 512)

  def test_method_and_hook_agree(self):
    """The wrapper's extract_backbone_features and the hook-based
    fallback should return the same pooled features."""
    model = _make_timm_model(num_classes=5)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    method_feats = model.extract_backbone_features(x)
    hook_feats = extract_backbone_features(model, x)
    torch.testing.assert_close(method_feats, hook_feats)


class TestTimmProcessor:
  """Tests for the processor shim."""

  def test_image_mean_std(self):
    proc = _make_timm_processor()
    assert isinstance(proc.image_mean, list)
    assert isinstance(proc.image_std, list)
    assert len(proc.image_mean) == 3
    assert len(proc.image_std) == 3

  def test_size_property(self):
    proc = _make_timm_processor(image_size=384)
    assert proc.size == {"height": 384, "width": 384}

  def test_call_returns_tensor(self):
    proc = _make_timm_processor(image_size=224)
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    result = proc(img)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 224, 224)
    assert result.dtype == torch.float32

  def test_repr(self):
    proc = _make_timm_processor()
    r = repr(proc)
    assert "TimmProcessor" in r
    assert "mean=" in r


class TestEndToEnd:
  """Full pipeline: image → processor → model → logits."""

  def test_pipeline(self):
    proc = _make_timm_processor(image_size=224)
    model = _make_timm_model(num_classes=5)
    model.eval()

    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    pixel_values = proc(img).unsqueeze(0)  # (1, C, H, W)
    out = model(pixel_values=pixel_values)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (1, 5)
