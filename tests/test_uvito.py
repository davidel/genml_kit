"""Tests for the UVito model and its genml_kit integration."""

import torch
import torch.nn as nn

from genml_kit.models.uvito.model import UVito
from genml_kit.training.model_utils import set_train_mode


def _make_uvito(num_classes=3, img_size=64, **kwargs):
  """Build a tiny UVito (small image, no pretrained weights) for tests."""
  return UVito(
      num_classes=num_classes,
      encoder_weights=None,
      img_size=img_size,
      num_transformer_layers=1,
      transformer_dim=32,
      nhead=4,
      dim_feedforward=64,
      **kwargs,
  )


class TestUVitoForward:

  def test_output_shape(self):
    model = _make_uvito(num_classes=5, img_size=64)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 5)

  def test_output_shape_multi_cls(self):
    """Forward works with num_cls_tokens > 1 (crashed before the
    per-token LayerNorm fix: LayerNorm(32) got a (B, 128) input)."""
    model = _make_uvito(num_classes=5, img_size=64, num_cls_tokens=4)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 5)

  def test_headless_shape_multi_cls(self):
    """Headless mode (num_classes=0) passes the flat CLS features through
    regardless of num_cls_tokens."""
    model = _make_uvito(num_classes=0, img_size=64, num_cls_tokens=4)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    # num_cls_tokens=4, transformer_dim=32 → (2, 128)
    assert out.shape == (2, 128)

  def test_backbone_features_shape(self):
    model = _make_uvito(num_classes=5, img_size=64)
    x = torch.randn(2, 3, 64, 64)
    feat = model.backbone_features(x)
    # num_cls_tokens=1, transformer_dim=32 → (2, 32)
    assert feat.shape == (2, 32)

  def test_backbone_features_shape_multi_cls(self):
    model = _make_uvito(num_classes=5, img_size=64, num_cls_tokens=4)
    x = torch.randn(2, 3, 64, 64)
    feat = model.backbone_features(x)
    # num_cls_tokens=4, transformer_dim=32 → (2, 128)
    assert feat.shape == (2, 128)

  def test_backbone_features_differ_from_logits(self):
    """Backbone features are before head_norm/mlp_head."""
    model = _make_uvito(num_classes=5, img_size=64)
    x = torch.randn(2, 3, 64, 64)
    feat = model.backbone_features(x)
    logits = model(x)
    # Different shapes (features ≠ logits)
    assert feat.shape[1] != logits.shape[1] or feat.shape != logits.shape


class TestUVitoExtractFeatures:

  def test_extract_via_wrapper(self):
    """UVitoAdapter.extract_backbone_features returns the
    CLS-flattened tensor, same as backbone_features."""
    from types import SimpleNamespace

    from genml_kit.models.uvito.loader import UVitoAdapter

    raw = _make_uvito(num_classes=3, img_size=64)
    config = SimpleNamespace(num_labels=3, id2label={}, label2id={}, image_size=64)
    wrapped = UVitoAdapter(raw, config)

    x = torch.randn(2, 3, 64, 64)
    feat = wrapped.extract_backbone_features(x)
    assert feat.shape == (2, 32)
    # Must be detached
    assert not feat.requires_grad


class TestUVitoTrainMode:

  def test_train_sets_head_to_train(self):
    model = _make_uvito()
    model.train()
    assert model.head_norm.training is True
    assert model.mlp_head.training is True
    assert model.patch_projection.training is True

  def test_train_keeps_frozen_encoder_in_eval(self):
    model = _make_uvito()
    model.train()
    assert model.frozen_encoder.training is False

  def test_eval_sets_everything_to_eval(self):
    model = _make_uvito()
    model.train()
    model.eval()
    assert model.frozen_encoder.training is False
    assert model.head_norm.training is False
    assert model.mlp_head.training is False


class TestSetTrainModeWithUVito:

  def test_set_train_mode_train(self):
    model = _make_uvito()
    set_train_mode(model, 'train')
    assert model.frozen_encoder.training is False
    assert model.head_norm.training is True
    assert model.mlp_head.training is True

  def test_set_train_mode_eval(self):
    model = _make_uvito()
    set_train_mode(model, 'train')
    set_train_mode(model, 'eval')
    assert model.frozen_encoder.training is False
    assert model.head_norm.training is False

  def test_batchnorm_running_stats_preserved(self):
    """Frozen encoder BatchNorm running stats must not change when
    the parent model is set to train mode."""
    model = _make_uvito()
    set_train_mode(model, 'eval')
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
      model(x)

    # Grab running mean from first BN in the encoder
    bn = None
    for m in model.frozen_encoder.modules():
      if isinstance(m, nn.BatchNorm2d):
        bn = m
        break
    assert bn is not None
    running_mean_before = bn.running_mean.clone()

    set_train_mode(model, 'train')
    with torch.no_grad():
      model(x)

    running_mean_after = bn.running_mean.clone()
    assert torch.equal(
        running_mean_before,
        running_mean_after), ("Frozen encoder BatchNorm running stats were updated!")
