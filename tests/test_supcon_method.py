"""Tests for SupConMethod pre-training method (v4.2: genml_kit.methods)."""

import argparse

import torch

from genml_kit.methods import get_method, list_methods
from genml_kit.models.contrastive import ContrastiveEncoder
from genml_kit.pipelines.contracts import DataBlob, LossOutput


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
    assert "supcon" in list_methods()
    cls = get_method("supcon")
    assert cls is not None

  def test_needs_labels(self):
    assert get_method("supcon").NEEDS_LABELS is True

  def test_add_args(self):
    parser = argparse.ArgumentParser()
    method = get_method("supcon")()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.proj_dim == 256
    assert args.proj_hidden == 2048
    assert args.temperature == 0.07

  def test_train_step(self):
    parser = argparse.ArgumentParser()
    method = get_method("supcon")()
    method.add_args(parser)
    args = parser.parse_args([])
    model = ContrastiveEncoder(_FakeBackbone(out_dim=64),
                               proj_dim=args.proj_dim,
                               proj_hidden=64)
    model.temperature = args.temperature

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = method.train_step(model, DataBlob(data=images, meta={"labels": labels}), 0)
    assert isinstance(out, LossOutput)
    assert torch.isfinite(out.loss)
    assert "loss" in out.metrics
    assert "temperature" in out.metrics

  def test_train_step_backward(self):
    parser = argparse.ArgumentParser()
    method = get_method("supcon")()
    method.add_args(parser)
    args = parser.parse_args([])
    model = ContrastiveEncoder(_FakeBackbone(out_dim=64),
                               proj_dim=args.proj_dim,
                               proj_hidden=64)
    model.temperature = args.temperature

    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    out = method.train_step(model, DataBlob(data=images, meta={"labels": labels}), 0)
    out.loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

  def test_train_step_requires_labels(self):
    method = get_method("supcon")()
    model = ContrastiveEncoder(_FakeBackbone(out_dim=64), proj_dim=16, proj_hidden=32)
    model.temperature = 0.07
    import pytest
    with pytest.raises(ValueError, match="requires labels"):
      method.train_step(model, DataBlob(data=torch.randn(2, 3, 64, 64), meta={}), 0)

  def test_log_validation_is_noop(self):
    method = get_method("supcon")()

    class _Writer:

      def __init__(self):
        self.calls = []

      def add_image(self, tag, tensor, step):
        self.calls.append(tag)

    def _to_device(blob, device):
      return blob

    method.log_validation(None, iter([]), _to_device, _Writer(), 0, torch.device("cpu"))
    # Reaching here without raising is the no-vis contract.
    assert True

  def test_checkpoint_state(self):
    parser = argparse.ArgumentParser()
    method = get_method("supcon")()
    method.add_args(parser)
    args = parser.parse_args([])
    state = method.get_checkpoint_state(None, args)
    assert state["method"] == "supcon"
    assert state["proj_dim"] == 256
    assert state["temperature"] == 0.07
