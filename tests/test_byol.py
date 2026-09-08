"""Tests for BYOL pre-training method."""

import argparse

import pytest
import torch

from genml_kit.models.byol import BYOL, _PredictorMLP
from genml_kit.pretrain.losses.byol import byol_loss
from genml_kit.pretrain.methods import get_method


class _FakeBackbone(torch.nn.Module):
  """Minimal backbone returning a fixed-size feature vector."""

  def __init__(self, out_dim=128):
    super().__init__()
    self.linear = torch.nn.Linear(3, out_dim)
    self.config = type("C", (), {"hidden_size": out_dim})()

  def forward(self, pixel_values):
    return self.linear(pixel_values.mean(dim=[2, 3]))


class TestBYOLLoss:

  def test_zero_for_identical(self):
    z = torch.randn(8, 64)
    loss = byol_loss(z, z, z, z)
    assert pytest.approx(loss.item(), abs=1e-6) == 0.0

  def test_positive_for_random(self):
    p1 = torch.randn(8, 64)
    z2 = torch.randn(8, 64)
    loss = byol_loss(p1, z2, p1, z2)
    assert loss.item() > 0

  def test_gradient_flows(self):
    p1 = torch.randn(8, 64, requires_grad=True)
    z2 = torch.randn(8, 64)
    loss = byol_loss(p1, z2, p1, z2)
    loss.backward()
    assert p1.grad is not None


class TestBYOLModule:

  def _make_byol(self, proj_dim=32):
    backbone = _FakeBackbone(out_dim=128)
    return BYOL(backbone,
                proj_dim=proj_dim,
                proj_hidden=64,
                predictor_hidden=64,
                backbone_dim=128)

  def test_forward_returns_loss(self):
    model = self._make_byol()
    v1 = torch.randn(4, 3, 32, 32)
    v2 = torch.randn(4, 3, 32, 32)
    loss, info = model((v1, v2))
    assert loss.ndim == 0
    assert "loss" in info

  def test_no_nan(self):
    model = self._make_byol()
    v1 = torch.randn(4, 3, 32, 32)
    v2 = torch.randn(4, 3, 32, 32)
    loss, _ = model((v1, v2))
    assert not torch.isnan(loss)

  def test_backward(self):
    model = self._make_byol()
    v1 = torch.randn(4, 3, 32, 32)
    v2 = torch.randn(4, 3, 32, 32)
    loss, _ = model((v1, v2))
    loss.backward()
    for p in model.parameters():
      if p.requires_grad:
        assert p.grad is not None

  def test_momentum_update(self):
    model = self._make_byol()
    before = {n: p.clone() for n, p in model.target_encoder.named_parameters()}
    model.update_momentum(0.9)
    changed = False
    for n, p in model.target_encoder.named_parameters():
      if not torch.equal(p, before[n]):
        changed = True
        break
    assert changed

  def test_target_no_grad(self):
    model = self._make_byol()
    for p in model.target_encoder.parameters():
      assert not p.requires_grad

  def test_online_has_predictor(self):
    model = self._make_byol()
    assert isinstance(model.online_predictor, _PredictorMLP)


class TestBYOLMethod:

  def test_registered(self):
    cls = get_method("byol")
    assert cls.NAME == "byol"

  def test_needs_labels(self):
    method = get_method("byol")()
    assert method.needs_labels is False

  def test_add_args(self):
    method = get_method("byol")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.byol_proj_dim == 256
    assert args.byol_momentum == 0.996

  def test_build(self):
    method = get_method("byol")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    backbone = _FakeBackbone(out_dim=128)
    device = torch.device("cpu")
    model = method.build(args, backbone, device)
    assert isinstance(model, BYOL)

  def test_train_step(self):
    method = get_method("byol")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    backbone = _FakeBackbone(out_dim=128)
    device = torch.device("cpu")
    model = method.build(args, backbone, device)
    v1 = torch.randn(4, 3, 32, 32)
    v2 = torch.randn(4, 3, 32, 32)
    loss, info = method.train_step(model, (v1, v2), global_step=0)
    assert loss.ndim == 0
    assert "loss" in info

  def test_checkpoint_roundtrip(self):
    method = get_method("byol")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    state = method.get_checkpoint_state(None, args)
    assert state["method"] == "byol"
    method2 = get_method("byol")()
    method2.load_checkpoint_state(None, state, args)
    assert method2._byol_momentum == 0.996

  def test_validate_returns_none(self):
    method = get_method("byol")()
    assert method.validate(None, torch.randn(2, 3, 32, 32), 2) is None
