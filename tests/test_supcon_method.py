"""Tests for SupConMethod pre-training method."""

import argparse

import torch

from genml_kit.pretrain.methods.supcon import SupConMethod


class _FakeBackbone(torch.nn.Module):
  """Minimal backbone returning a fixed-size feature vector."""

  def __init__(self, out_dim=128):
    super().__init__()
    self.linear = torch.nn.Linear(3, out_dim)
    self.config = type("C", (), {"hidden_size": out_dim})()

  def forward(self, pixel_values):
    x = pixel_values.mean(dim=[2, 3])
    return self.linear(x)


class TestSupConMethod:

  def test_registered(self):
    from genml_kit.pretrain.methods import get_method, list_methods
    assert "supcon" in list_methods()
    cls = get_method("supcon")
    assert cls is SupConMethod

  def test_needs_labels(self):
    assert SupConMethod.needs_labels is True

  def test_add_args(self):
    parser = argparse.ArgumentParser()
    method = SupConMethod()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.proj_dim == 256
    assert args.proj_hidden == 2048
    assert args.temperature == 0.07

  def test_build(self):
    parser = argparse.ArgumentParser()
    method = SupConMethod()
    method.add_args(parser)
    args = parser.parse_args([])
    device = torch.device("cpu")
    backbone = _FakeBackbone(out_dim=64)
    model = method.build(args, backbone, device)
    assert hasattr(model, "projection")
    assert hasattr(model, "encoder")

  def test_train_step(self):
    parser = argparse.ArgumentParser()
    method = SupConMethod()
    method.add_args(parser)
    args = parser.parse_args([])
    device = torch.device("cpu")
    backbone = _FakeBackbone(out_dim=64)
    model = method.build(args, backbone, device)

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, info = method.train_step(model, images, 0, labels=labels)
    assert torch.isfinite(loss)
    assert "loss" in info
    assert "temperature" in info

  def test_train_step_backward(self):
    parser = argparse.ArgumentParser()
    method = SupConMethod()
    method.add_args(parser)
    args = parser.parse_args([])
    device = torch.device("cpu")
    backbone = _FakeBackbone(out_dim=64)
    model = method.build(args, backbone, device)

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss, _ = method.train_step(model, images, 0, labels=labels)
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

  def test_validate_returns_none(self):
    method = SupConMethod()
    result = method.validate(None, torch.randn(2, 3, 64, 64), 2)
    assert result is None

  def test_checkpoint_state(self):
    parser = argparse.ArgumentParser()
    method = SupConMethod()
    method.add_args(parser)
    args = parser.parse_args([])
    state = method.get_checkpoint_state(None, args)
    assert state["method"] == "supcon"
    assert state["proj_dim"] == 256
    assert state["temperature"] == 0.07
