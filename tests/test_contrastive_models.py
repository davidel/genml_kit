"""Tests for ContrastiveEncoder and ProjectionHead."""

import torch
import torch.nn as nn

from genml_kit.models.contrastive import ContrastiveEncoder, ProjectionHead


class _FakeBackbone(nn.Module):
  """Minimal backbone that returns a fixed-size feature vector."""

  def __init__(self, out_dim=128):
    super().__init__()
    self.linear = nn.Linear(3, out_dim)

  def forward(self, pixel_values):
    x = pixel_values.mean(dim=[2, 3])
    return self.linear(x)


class _FakeBackboneWithClassifier(nn.Module):
  """Backbone with a classifier that has feat_dim."""

  def __init__(self, out_dim=64):
    super().__init__()
    self.linear = nn.Linear(3, out_dim)
    self.classifier = type("C", (), {"feat_dim": out_dim})()

  def forward(self, pixel_values):
    x = pixel_values.mean(dim=[2, 3])
    return self.linear(x)


class TestProjectionHead:

  def test_output_shape(self):
    head = ProjectionHead(in_dim=128, hidden_dim=256, out_dim=64)
    x = torch.randn(8, 128)
    out = head(x)
    assert out.shape == (8, 64)

  def test_gradient_flows(self):
    head = ProjectionHead(in_dim=32, hidden_dim=64, out_dim=16)
    x = torch.randn(4, 32)
    out = head(x)
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert has_grad


class TestContrastiveEncoder:

  def test_forward_shape(self):
    backbone = _FakeBackbone(out_dim=128)
    model = ContrastiveEncoder(
        backbone,
        proj_dim=32,
        proj_hidden=64,
        backbone_dim=128,
    )
    images = torch.randn(4, 3, 224, 224)
    out = model(images)
    assert out.shape == (4, 32)

  def test_encode_shape(self):
    backbone = _FakeBackbone(out_dim=128)
    model = ContrastiveEncoder(
        backbone,
        proj_dim=32,
        proj_hidden=64,
        backbone_dim=128,
    )
    images = torch.randn(4, 3, 224, 224)
    features = model.encode(images)
    assert features.shape == (4, 128)

  def test_backward(self):
    backbone = _FakeBackbone(out_dim=64)
    model = ContrastiveEncoder(
        backbone,
        proj_dim=16,
        proj_hidden=32,
        backbone_dim=64,
    )
    images = torch.randn(4, 3, 64, 64)
    out = model(images)
    loss = out.sum()
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

  def test_detect_backbone_dim_from_config(self):
    """Auto-detect when config has hidden_size."""

    class _CfgBackbone(nn.Module):

      def __init__(self):
        super().__init__()
        self.config = type("C", (), {"hidden_size": 256})()
        self.linear = nn.Linear(3, 256)

      def forward(self, x):
        return self.linear(x.mean(dim=[2, 3]))

    model = ContrastiveEncoder(
        _CfgBackbone(),
        proj_dim=64,
        proj_hidden=128,
    )
    assert model._backbone_dim == 256

  def test_detect_backbone_dim_from_classifier(self):
    backbone = _FakeBackboneWithClassifier(out_dim=64)
    model = ContrastiveEncoder(backbone, proj_dim=16, proj_hidden=32)
    assert model._backbone_dim == 64

  def test_projection_head_not_in_encode(self):
    """encode() returns raw backbone features, not projected."""
    backbone = _FakeBackbone(out_dim=128)
    model = ContrastiveEncoder(
        backbone,
        proj_dim=32,
        proj_hidden=64,
        backbone_dim=128,
    )
    images = torch.randn(2, 3, 64, 64)
    features = model.encode(images)
    assert features.shape == (2, 128)  # backbone dim, not proj_dim

  def test_explicit_backbone_dim(self):
    backbone = _FakeBackbone(out_dim=99)
    model = ContrastiveEncoder(
        backbone,
        proj_dim=16,
        proj_hidden=32,
        backbone_dim=99,
    )
    assert model._backbone_dim == 99
