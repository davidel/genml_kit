"""Tests for the custom model registry and ConvViT integration."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from PIL import Image

from genml_kit.models import (
    ModelOutput,
    is_custom_model,
    load_model,
    register_model,
)
from genml_kit.models.convvit.model import CustomPatchTransformer
from genml_kit.models.convvit.processor import ConvViTProcessor
from genml_kit.training.model_utils import extract_backbone_features


class ConvViTConfig:
  """Minimal ConvViT configuration for testing purposes."""

  def __init__(
      self,
      image_size=224,
      patch_size=16,
      in_chans=3,
      depths=None,
      embed_dims=None,
      vit_hidden_dim=768,
      vit_num_heads=12,
      vit_num_layers=12,
      vit_mlp_ratio=4.0,
      drop_rate=0.0,
      drop_path_rate=0.1,
      head_hidden_dim=512,
      num_labels=7,
      id2label=None,
      label2id=None,
  ):
    self.image_size = image_size
    self.patch_size = patch_size
    self.in_chans = in_chans
    self.depths = depths if depths is not None else [2, 2, 6, 2]
    self.embed_dims = embed_dims if embed_dims is not None else [128, 256, 512, 768]
    self.vit_hidden_dim = vit_hidden_dim
    self.vit_num_heads = vit_num_heads
    self.vit_num_layers = vit_num_layers
    self.vit_mlp_ratio = vit_mlp_ratio
    self.drop_rate = drop_rate
    self.drop_path_rate = drop_path_rate
    self.head_hidden_dim = head_hidden_dim
    self.num_labels = num_labels
    self.id2label = id2label if id2label is not None else {}
    self.label2id = label2id if label2id is not None else {}


class TestRegistry:
  """Tests for the model registry dispatch."""

  def test_custom_model_detection(self):
    assert is_custom_model("convvit") is True
    assert is_custom_model("facebook/convnextv2-base-22-22k-384") is False
    assert is_custom_model(None) is False
    assert is_custom_model("") is False

  def test_registry_contains_convvit(self):
    assert is_custom_model("convvit") is True

  def test_register_duplicate_raises(self):
    with pytest.raises(ValueError, match="already registered"):

      @register_model("convvit")
      def _dummy(**kwargs):
        pass

  def test_custom_register_and_lookup(self):

    @register_model("_test_model_xyz")
    def _load(**kwargs):
      return "model", "processor"

    assert is_custom_model("_test_model_xyz") is True
    # cleanup
    from genml_kit.models.registry import _MODEL_REGISTRY
    del _MODEL_REGISTRY["_test_model_xyz"]


class TestModelOutput:
  """Tests for the ModelOutput wrapper."""

  def test_basic(self):
    t = torch.randn(4, 7)
    out = ModelOutput(logits=t)
    assert out.logits is t
    assert out.logits.shape == (4, 7)

  def test_gradient_flows(self):
    t = torch.randn(2, 3, requires_grad=True)
    out = ModelOutput(logits=t)
    loss = out.logits.sum()
    loss.backward()
    assert t.grad is not None


class TestConvViTProcessor:
  """Tests for the ConvViT image processor."""

  @pytest.fixture
  def processor(self):
    return ConvViTProcessor(image_size=224)

  @pytest.fixture
  def sample_image(self):
    return Image.new("RGB", (300, 400), color=(128, 64, 200))

  def test_single_image(self, processor, sample_image):
    out = processor(sample_image)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == torch.float32

  def test_batch_images(self, processor, sample_image):
    batch = [sample_image, sample_image, sample_image]
    out = processor(batch)
    assert out.shape == (3, 3, 224, 224)

  def test_normalization_range(self, processor, sample_image):
    out = processor(sample_image)
    # After ImageNet normalization, values should be roughly in [-2.5, 2.5]
    assert out.min() > -3.0
    assert out.max() < 3.0

  def test_size_property(self, processor):
    assert processor.size == {"height": 224, "width": 224}

  def test_different_image_size(self):
    proc = ConvViTProcessor(image_size=448)
    img = Image.new("RGB", (600, 600))
    out = proc(img)
    assert out.shape == (1, 3, 448, 448)


class TestConvViTConfig:
  """Tests for the ConvViT configuration dataclass."""

  def test_defaults(self):
    cfg = ConvViTConfig()
    assert cfg.image_size == 224
    assert cfg.vit_hidden_dim == 768
    assert cfg.vit_num_layers == 12
    assert cfg.num_labels == 7

  def test_custom_labels(self):
    cfg = ConvViTConfig(
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
    )
    assert cfg.num_labels == 3
    assert cfg.id2label[1] == "b"


class TestConvViTForward:
  """Tests for the CustomPatchTransformer forward pass shape and basic properties."""

  @pytest.fixture
  def model(self):
    return CustomPatchTransformer(
        num_classes=7,
        img_size=224,
        embed_dim=768,
        num_heads=12,
        depth=12,
        dropout=0.0,
        drop_path_rate=0.0,
        num_conv_layers=4,
    )

  def test_forward_shape(self, model):
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)

  def test_single_sample(self, model):
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 7)

  def test_different_num_labels(self):
    model = CustomPatchTransformer(
        num_classes=3,
        img_size=224,
        embed_dim=768,
        num_heads=12,
        depth=12,
        dropout=0.0,
        drop_path_rate=0.0,
        num_conv_layers=4,
    )
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 3)

  def test_gradient_flows(self, model):
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    out.sum().backward()
    # Check at least one parameter got gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

  def test_headless_multi_cls_tokens(self):
    """num_cls_tokens > 1 flattens all CLS tokens in headless mode."""
    model = CustomPatchTransformer(
        num_classes=0,
        img_size=224,
        num_cls_tokens=2,
        embed_dim=768,
        num_heads=12,
        depth=2,
        dropout=0.0,
        drop_path_rate=0.0,
        num_conv_layers=2,
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2 * 768)
    # Headless mode must not build classifier-path parameters.
    assert model.cls_guided_pool is None
    assert model.head is None

  def test_conv_stem_features(self, model):
    """ConvViT's conv stem produces spatial feature maps."""
    x = torch.randn(1, 3, 224, 224)
    # Access the conv stem directly
    features = model.patch_embed.blocks
    assert len(features) == 4  # num_conv_layers=4
    # Forward through the stem manually
    feat_maps = []
    h = x
    for block in features:
      h = block(h)
      feat_maps.append(h)
    assert len(feat_maps) == 4
    # All should be 4-D tensors (B, C, H, W)
    for f in feat_maps:
      assert f.ndim == 4


class TestConvViTAdapter:
  """Tests for the protocol adapter."""

  @pytest.fixture
  def wrapped_model(self):
    from types import SimpleNamespace

    from genml_kit.models.convvit.loader import ConvViTAdapter

    model = CustomPatchTransformer(
        num_classes=5,
        img_size=224,
        embed_dim=768,
        num_heads=12,
        depth=12,
        dropout=0.0,
        drop_path_rate=0.0,
        num_conv_layers=4,
    )
    config = SimpleNamespace(
        id2label={
            0: "melanoma",
            1: "nevus",
            2: "bkl",
            3: "df",
            4: "vasc"
        },
        label2id={
            "melanoma": 0,
            "nevus": 1,
            "bkl": 2,
            "df": 3,
            "vasc": 4
        },
    )
    return ConvViTAdapter(model, config)

  def test_forward_returns_model_output(self, wrapped_model):
    x = torch.randn(2, 3, 224, 224)
    out = wrapped_model(pixel_values=x)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 5)

  def test_config_accessible(self, wrapped_model):
    assert wrapped_model.config.id2label[0] == "melanoma"
    assert wrapped_model.config.label2id["nevus"] == 1

  def test_extract_backbone_features(self, wrapped_model):
    x = torch.randn(1, 3, 224, 224)
    features = wrapped_model.extract_backbone_features(x)
    assert isinstance(features, list)
    assert len(features) == 4


class TestClsModelWrapper:
  """Tests for ClsModelWrapper backbone ownership and feature extraction."""

  def _make_wrapper(
      self,
      num_labels=5,
      classifier="mlp",
      classifier_args=None,
  ):
    """Build a ClsModelWrapper with a tiny ViT backbone (no download)."""
    from unittest.mock import patch

    from transformers import ViTConfig, ViTModel

    from genml_kit.models.cls_model_wrapper.model import ClsModelWrapper

    config = ViTConfig(
        image_size=224,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    tiny_backbone = ViTModel(config)

    with patch(
        "genml_kit.models.cls_model_wrapper.model.AutoModel.from_pretrained",
        return_value=tiny_backbone,
    ):
      return ClsModelWrapper(
          backbone_name="fake-backbone",
          num_labels=num_labels,
          classifier=classifier,
          classifier_args=classifier_args or {},
      )

  def test_forward_returns_model_output(self):
    model = self._make_wrapper()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 5)

  def test_backbone_is_sibling_not_nested(self):
    model = self._make_wrapper()
    # Backbone lives on the wrapper, not inside the classifier
    assert hasattr(model, "backbone")
    assert not hasattr(model.classifier, "backbone")

  def test_extract_backbone_features_mlp(self):
    model = self._make_wrapper(classifier="mlp")
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_backbone_features(x)
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 2
    assert features.ndim == 2  # (B, F)

  def test_extract_backbone_features_cls_attention(self):
    model = self._make_wrapper(
        classifier="cls_attention",
        classifier_args={
            "cls_slice": (0, 1),
            "spc_slice": (1, None),
            "num_heads": 4
        },
    )
    x = torch.randn(2, 3, 224, 224)
    features = model.extract_backbone_features(x)
    assert isinstance(features, torch.Tensor)
    assert features.shape[0] == 2
    assert features.ndim == 2

  def test_extract_features_matches_forward_path(self):
    """extract_features should produce the same result as forward's
    intermediate representation (before the head)."""
    model = self._make_wrapper(classifier="mlp")
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    feats = model.extract_backbone_features(x)
    # Manually run the same path
    with torch.no_grad():
      raw = model.backbone(x)
      hidden = model._extract_hidden_states(raw)
      feats_manual = model.classifier.extract_features(hidden)
    torch.testing.assert_close(feats, feats_manual)

  def test_config_accessible(self):
    model = self._make_wrapper()
    assert hasattr(model.config, "hidden_size")


class TestClassifierSpec:
  """Tests for the inline classifier spec (classifier=mlp:hidden=512,...)."""

  def test_name_only(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    assert parse_classifier_spec("mlp") == ("mlp", {})

  def test_name_with_kwargs(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    name, kwargs = parse_classifier_spec("mlp:hidden=512,dropout=0.3")
    assert name == "mlp"
    assert kwargs == {"hidden": 512, "dropout": 0.3}

  def test_kwarg_values_are_typed(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    _name, kwargs = parse_classifier_spec("m:a=3,b=2.5,c=true")
    assert kwargs == {"a": 3, "b": 2.5, "c": True}

  def test_bracketed_list_values(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    _name, kwargs = parse_classifier_spec("m:cls_slice=(0, 1),n=[1, 2]")
    assert kwargs == {"cls_slice": [0, 1], "n": [1, 2]}

  def test_py_path_spec(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    assert parse_classifier_spec("my_cls.py") == ("my_cls.py", {})

  def test_malformed_spec_raises(self):
    from genml_kit.training.classifiers import parse_classifier_spec

    with pytest.raises(ValueError):
      parse_classifier_spec("mlp:hidden512")

  def test_load_model_consumes_spec(self):
    """load_model(cls_model_wrapper:...) parses classifier=<spec>."""
    from unittest.mock import patch

    from transformers import ViTConfig, ViTModel

    from genml_kit.models import load_model

    config = ViTConfig(
        image_size=224,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    )
    tiny_backbone = ViTModel(config)

    with patch(
        "genml_kit.models.cls_model_wrapper.model.AutoModel.from_pretrained",
        return_value=tiny_backbone,
    ):
      model = load_model(
          "cls_model_wrapper:fake-backbone",
          num_labels=5,
          id2label={
              0: "a",
              1: "b",
              2: "c",
              3: "d",
              4: "e"
          },
          label2id={},
          image_size=224,
          device=torch.device("cpu"),
          classifier="mlp:hidden=256",
      )
    assert model.classifier is not None
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.logits.shape == (2, 5)

  def test_load_model_requires_classifier(self):
    from genml_kit.models import load_model

    with pytest.raises(ValueError):
      load_model(
          "cls_model_wrapper:fake-backbone",
          num_labels=5,
          id2label={0: "a"},
          label2id={},
          image_size=224,
          device=torch.device("cpu"),
      )


class TestLoadCustomModel:
  """Test the full load path through the registry."""

  def test_load_convvit(self):
    id2label = {0: "a", 1: "b", 2: "c"}
    label2id = {"a": 0, "b": 1, "c": 2}
    from genml_kit.models import load_processor

    model = load_model(
        "convvit",
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        image_size=224,
        device=torch.device("cpu"),
    )
    processor = load_processor("convvit", image_size=224)
    assert isinstance(model, torch.nn.Module)
    assert model.config.id2label == id2label
    assert callable(processor)

    # End-to-end: image → processor → model → logits
    img = Image.new("RGB", (256, 256), color=(100, 150, 200))
    pixel_values = processor(img)
    out = model(pixel_values=pixel_values)
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (1, 3)

  def test_model_arg_overrides_convvit(self):
    """--model_arg depth=6 should override the default depth=12."""
    id2label = {0: "a", 1: "b"}
    label2id = {"a": 0, "b": 1}

    model = load_model(
        "convvit",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        image_size=224,
        device=torch.device("cpu"),
        depth=6,
        drop_path_rate=0.2,
    )
    assert model.config.depth == 6
    assert model.config.drop_path_rate == 0.2
    # Default that was NOT overridden should stay.
    assert model.config.num_heads == 12

  def test_model_arg_overrides_uvito(self, monkeypatch):
    """--model_arg should propagate to UVito config and model."""
    id2label = {0: "a", 1: "b"}
    label2id = {"a": 0, "b": 1}

    # Mock UVito.__init__ to avoid downloading ResNet weights.
    from genml_kit.models.uvito import model as uvito_mod

    captured_kwargs = {}

    def _mock_init(self, **kwargs):
      nn.Module.__init__(self)
      captured_kwargs.update(kwargs)

    monkeypatch.setattr(uvito_mod.UVito, "__init__", _mock_init)

    model = load_model(
        "uvito",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        image_size=224,
        device=torch.device("cpu"),
        num_transformer_layers=3,
        drop_path_rate=0.2,
    )
    # Config should carry the overrides.
    assert model.config.num_transformer_layers == 3
    assert model.config.drop_path_rate == 0.2
    # Default that was NOT overridden should stay in config.
    assert model.config.image_size == 224
    # The model constructor should have received the overrides.
    assert captured_kwargs["num_transformer_layers"] == 3
    assert captured_kwargs["drop_path_rate"] == 0.2
    # Only explicitly-passed kwargs appear; constructor defaults are irrelevant
    # because we mocked __init__.
    assert "nhead" not in captured_kwargs  # not passed by test

  def test_unknown_model_raises(self):
    # load_model falls through to HuggingFace for unknown names,
    # which raises OSError for non-existent model ids.
    with pytest.raises((ValueError, OSError)):
      load_model(
          "nonexistent_model",
          num_labels=3,
          id2label={},
          label2id={},
          image_size=224,
          device=torch.device("cpu"),
      )


class TestCheckpointRoundtrip:
  """Verify ConvViT checkpoint save → load preserves weights."""

  def test_roundtrip(self):
    id2label = {0: "melanoma", 1: "nevus"}
    label2id = {"melanoma": 0, "nevus": 1}

    model = load_model(
        "convvit",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        image_size=224,
        device=torch.device("cpu"),
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
      ckpt_path = f.name
      torch.save(
          {
              "model_state_dict": model.state_dict(),
              "epoch": 0,
              "id2label": id2label,
          }, ckpt_path)

    try:
      model2 = load_model(
          "convvit",
          num_labels=2,
          id2label=id2label,
          label2id=label2id,
          image_size=224,
          device=torch.device("cpu"),
          checkpoint_path=ckpt_path,
      )

      # Compare every parameter
      for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                    model2.named_parameters()):
        assert n1 == n2, f"Name mismatch: {n1} vs {n2}"
        assert torch.allclose(p1, p2, atol=1e-6), \
            f"Weight mismatch in {n1}"
    finally:
      os.unlink(ckpt_path)


class TestExtractBackboneFeatures:
  """Test the hook-based feature extraction on both model types."""

  def _make_hf_model(self, num_labels=7):
    """Build an HF ConvNeXtV2 model from config (no download)."""
    from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

    config = ConvNextV2Config(
        image_size=224,
        num_labels=num_labels,
        hidden_size=128,  # small for fast tests
        depths=[2, 2, 6, 2],
        hidden_sizes=[64, 128, 256, 512],
    )
    return ConvNextV2ForImageClassification(config)

  def test_hook_with_hf_convnextv2(self):
    """Hook-based extraction works on an HF ConvNeXtV2 model."""
    model = self._make_hf_model(num_labels=7)
    model.eval()

    pixel_values = torch.randn(2, 3, 224, 224)
    features = extract_backbone_features(model, pixel_values)

    assert isinstance(features, torch.Tensor)
    # ConvNeXtV2 pools to hidden_size, then classifier maps to num_labels.
    # The hook captures the input to the classifier, which is hidden_size.
    expected_dim = model.config.hidden_sizes[-1]
    assert features.shape == (2, expected_dim)
    assert features.dtype == torch.float32

  def test_hook_with_hf_vit(self):
    """Hook-based extraction works on an HF ViT model."""
    from transformers.models.vit import ViTConfig
    from transformers.models.vit.modeling_vit import ViTForImageClassification

    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_labels=7,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
    )
    model = ViTForImageClassification(config)
    model.eval()

    pixel_values = torch.randn(2, 3, 224, 224)
    features = extract_backbone_features(model, pixel_values)

    assert isinstance(features, torch.Tensor)
    # ViT classifier input is hidden_size (CLS token).
    assert features.shape == (2, 128)

  def test_hook_with_custom_convvit(self):
    """extract_backbone_features uses the protocol method for custom models."""
    model = load_model(
        "convvit",
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
    )
    model.eval()

    pixel_values = torch.randn(1, 3, 224, 224)
    features = extract_backbone_features(model, pixel_values)

    assert isinstance(features, torch.Tensor)
    assert features.ndim == 2
    assert features.shape[0] == 1

  def test_hook_no_classifier_raises(self):
    """Raises ValueError if no classifier head is found."""
    model = torch.nn.Linear(10, 5)  # bare Linear, no .classifier attr

    with pytest.raises(ValueError, match="Cannot extract backbone features"):
      extract_backbone_features(model, torch.randn(1, 10))

  def test_hook_cleans_up(self):
    """Verify the hook is removed after extraction (no side effects)."""
    model = self._make_hf_model(num_labels=7)
    model.eval()

    # Count hooks before
    n_hooks_before = len(model.classifier._forward_hooks)

    pixel_values = torch.randn(1, 3, 224, 224)
    extract_backbone_features(model, pixel_values)

    # Count hooks after — should be the same
    n_hooks_after = len(model.classifier._forward_hooks)
    assert n_hooks_after == n_hooks_before
