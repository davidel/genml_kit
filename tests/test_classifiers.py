"""Tests for the classifier registry and ClsModelWrapper."""

import pytest
import torch
import torch.nn as nn

from genml_kit.training.classifiers import (
    _CLASSIFIERS,
    build_classifier,
    register_classifier,
)


class TestBuildClassifier:
  """Tests for build_classifier()."""

  def test_mlp_default(self):
    net = build_classifier("mlp", num_labels=7, hidden_size=64)
    x = torch.randn(2, 10, 64)  # (B, N, D)
    out = net(x)
    assert out.shape == (2, 7)

  def test_mlp_custom_hidden(self):
    net = build_classifier("mlp",
                           num_labels=3,
                           hidden_size=128,
                           hidden=256,
                           dropout=0.1)
    x = torch.randn(4, 5, 128)
    out = net(x)
    assert out.shape == (4, 3)

  def test_cls_attention_default(self):
    net = build_classifier("cls_attention", num_labels=7, hidden_size=64, num_heads=4)
    x = torch.randn(2, 10, 64)  # (B, N, D)
    out = net(x)
    assert out.shape == (2, 7)

  def test_cls_attention_with_slices(self):
    net = build_classifier(
        "cls_attention",
        num_labels=5,
        hidden_size=64,
        cls_slice=(0, 1),
        spc_slice=(3, None),
        num_heads=4,
    )
    x = torch.randn(2, 10, 64)
    out = net(x)
    assert out.shape == (2, 5)

  def test_unknown_classifier_raises(self):
    with pytest.raises(ValueError, match="Unknown classifier"):
      build_classifier("nonexistent", num_labels=2, hidden_size=32)

  def test_classifier_kwargs_logged(self, caplog):
    with caplog.at_level("INFO"):
      build_classifier("mlp", num_labels=3, hidden_size=64, hidden=128, dropout=0.5)
    assert "'hidden': 128" in caplog.text

  def test_registered_classifiers(self):
    assert "mlp" in _CLASSIFIERS
    assert "cls_attention" in _CLASSIFIERS


class TestClsAttention:
  """Detailed tests for cls_attention slicing and pooling."""

  def test_cls_token_only(self):
    net = build_classifier(
        "cls_attention",
        num_labels=3,
        hidden_size=32,
        cls_slice=(0, 1),
        spc_slice=(0, 1),
        num_heads=4,
    )
    x = torch.randn(2, 5, 32)
    out = net(x)
    assert out.shape == (2, 3)

  def test_extract_features_shape(self):
    net = build_classifier(
        "cls_attention",
        num_labels=3,
        hidden_size=32,
        cls_slice=(0, 1),
        spc_slice=(1, None),
        num_heads=4,
    )
    x = torch.randn(2, 10, 32)
    feats = net.extract_features(x)
    assert feats.shape == (2, 32)

  def test_mlp_extract_features_shape(self):
    net = build_classifier("mlp", num_labels=3, hidden_size=32)
    x = torch.randn(2, 10, 32)
    feats = net.extract_features(x)
    assert feats.shape == (2, 32)

  def test_mlp_cls_slice_multiple_tokens(self):
    net = build_classifier("mlp", num_labels=5, hidden_size=64, cls_slice=(0, 3))
    x = torch.randn(2, 10, 64)
    feats = net.extract_features(x)
    assert feats.shape == (2, 192)  # 3 * 64
    out = net(x)
    assert out.shape == (2, 5)


class TestRegisterClassifier:
  """Tests for the register_classifier decorator."""

  def test_register_new(self):

    @register_classifier("_test_dummy")
    class _Dummy(nn.Module):

      def __init__(self, num_labels, hidden_size, **kw):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_labels)

      def forward(self, hidden_states):
        return self.head(hidden_states[:, 0])

      def extract_features(self, hidden_states):
        return hidden_states[:, 0]

    assert "_test_dummy" in _CLASSIFIERS
    # Clean up
    del _CLASSIFIERS["_test_dummy"]

  def test_register_duplicate_raises(self):
    with pytest.raises(ValueError, match="already registered"):

      @register_classifier("mlp")
      class _Dummy(nn.Module):
        pass
