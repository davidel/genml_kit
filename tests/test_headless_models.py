"""Tests for timm-convention headless models (num_labels == 0).

The self-supervised pre-training paths load encoders with
``num_labels=0``; following timm's ``num_classes=0`` semantics, no
classification head must be created (previously a degenerate
``nn.Linear(d, 0)`` was built, producing zero-element parameters).
"""

import pytest
import torch

from genml_kit.models import load_model
from genml_kit.models.convvit.masked_encoder import ConvViTMaskedImageEncoder
from genml_kit.models.registry import ModelOutput, is_custom_model


def _assert_no_zero_element_params(model):
  zero = [n for n, p in model.named_parameters() if p.numel() == 0]
  assert zero == [], f"zero-element parameters found: {zero}"


class TestConvViTHeadless:

  def test_no_head_created(self):
    model = load_model("convvit", num_labels=0, image_size=64)
    _assert_no_zero_element_params(model)
    assert model.model.head is None  # adapter -> raw model under .model
    # No classifier-specific pooling either: headless mode must leave no
    # learnable parameters outside the pre-training forward path.
    assert model.model.cls_guided_pool is None
    assert model.model.num_cls_tokens == 1

  def test_headless_no_inert_params(self):
    """Every requires_grad param must sit on the pre-training forward path.

    Regression guard for GradMonitor false ``STL`` reports: headless
    encoders must expose no learnable parameters that only the
    classification forward path would exercise.
    """
    model = load_model("convvit", num_labels=0, image_size=64)
    inert = [
        n for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("model.cls_guided_pool.")
    ]
    assert inert == [], f"inert classifier-path parameters found: {inert}"

  def test_forward_returns_cls_features(self):
    """Headless: ModelOutput whose logits ARE the CLS-token features."""
    model = load_model("convvit", num_labels=0, image_size=64)
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert isinstance(out, ModelOutput)
    assert out.logits.shape == (2, 768)  # 1 CLS token x loader embed_dim

  def test_labeled_path_unaffected(self):
    model = load_model("convvit", num_labels=7, image_size=64)
    assert model.model.head is not None
    assert model.model.cls_guided_pool is not None
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert out.logits.shape == (2, 7)

  def test_multi_cls_tokens_headless(self):
    """num_cls_tokens flattens all CLS tokens (uvito convention)."""
    model = load_model("convvit", num_labels=0, image_size=64, num_cls_tokens=3)
    assert model.model.num_cls_tokens == 3
    assert model.model.cls_token.shape == (1, 3, 768)
    num_patches = model.model.patch_embed.num_patches
    assert model.model.pos_embedding.shape == (1, num_patches + 3, 768)
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert out.logits.shape == (2, 3 * 768)

  def test_multi_cls_tokens_labeled(self):
    """Labeled path pools with the first CLS token, head maps to classes."""
    model = load_model("convvit", num_labels=7, image_size=64, num_cls_tokens=2)
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert out.logits.shape == (2, 7)

  def test_config_contract_preserved(self):
    model = load_model("convvit", num_labels=0, image_size=64)
    assert model.config.num_labels == 0

  def test_embed_dim_override_honored(self):
    model = load_model("convvit",
                       num_labels=0,
                       image_size=64,
                       embed_dim=512,
                       num_heads=8)
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert out.logits.shape == (2, 512)

  def test_simmim_encoder_adapter_compatible(self):
    """ConvViTMaskedImageEncoder must work with the raw headless model."""
    model = load_model("convvit", num_labels=0, image_size=64)
    enc = ConvViTMaskedImageEncoder(model)
    assert enc.embed_dim == 768  # loader default
    assert enc.num_patches > 0
    x = torch.randn(2, enc.num_patches, enc.embed_dim)
    assert enc.encode_embeddings(x).shape[0] == 2

  def test_ijepa_patchembedder_compatible(self):
    """I-JEPA's _PatchEmbedder probes `model`/patch attributes."""
    from genml_kit.pretrain.methods.ijepa import _PatchEmbedder

    model = load_model("convvit", num_labels=0, image_size=64)
    wrapped = _PatchEmbedder(model)
    assert wrapped.patch_size > 0
    assert wrapped.embed_dim == 768  # loader default
    assert wrapped.num_patches > 0


class TestUVitoHeadless:

  def test_no_head_created(self):
    model = load_model("uvito", num_labels=0, image_size=64, encoder_weights=None)
    _assert_no_zero_element_params(model)
    assert model.model.mlp_head is None  # adapter -> raw model under .model

  def test_forward_returns_features(self):
    """Headless: ModelOutput whose logits ARE the backbone features."""
    model = load_model("uvito", num_labels=0, image_size=64, encoder_weights=None)
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert isinstance(out, ModelOutput)
    assert out.logits.ndim == 2
    assert out.logits.shape[0] == 2

  def test_labeled_path_unaffected(self):
    model = load_model("uvito", num_labels=3, image_size=64, encoder_weights=None)
    assert model.model.mlp_head is not None
    model.eval()
    with torch.no_grad():
      out = model(pixel_values=torch.randn(2, 3, 64, 64))
    assert out.logits.shape == (2, 3)


class TestClsModelWrapperHeadless:

  def test_rejected(self):
    with pytest.raises(ValueError, match="requires a classification head"):
      load_model("cls_model_wrapper:google/vit-base-patch16-224", num_labels=0)


class TestHuggingFaceHeadless:
  """The registry HF branch must honor timm semantics via AutoModel."""

  def test_auto_model_used_when_num_labels_zero(self, monkeypatch):
    calls = {}

    class _FakeModel(torch.nn.Module):

      def to(self, device):
        calls["to"] = device
        return self

    class _FakeAutoModel:

      @staticmethod
      def from_pretrained(name, **kwargs):
        calls["auto_model"] = (name, kwargs)
        return _FakeModel()

    class _FakeAutoCls:

      @staticmethod
      def from_pretrained(name, **kwargs):
        calls["auto_cls"] = (name, kwargs)
        return _FakeModel()

    import transformers
    monkeypatch.setattr(transformers, "AutoModel", _FakeAutoModel)
    monkeypatch.setattr(transformers, "AutoModelForImageClassification", _FakeAutoCls)

    assert not is_custom_model("google/vit-base-patch16-224")
    load_model("google/vit-base-patch16-224", num_labels=0)
    assert "auto_model" in calls
    assert "auto_cls" not in calls

    load_model("google/vit-base-patch16-224", num_labels=5)
    assert "auto_cls" in calls
