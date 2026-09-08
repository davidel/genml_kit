"""Tests for DINO pre-training method."""

import argparse

import torch

from genml_kit.models.dino import DINO
from genml_kit.pretrain.augmentations.multicrop import MultiCropTransform
from genml_kit.pretrain.losses.dino import DINOLoss
from genml_kit.pretrain.methods import get_method


class _FakeBackbone(torch.nn.Module):
  """Minimal backbone returning a fixed-size feature vector."""

  def __init__(self, out_dim=128):
    super().__init__()
    self.linear = torch.nn.Linear(3, out_dim)
    self.config = type("C", (), {"hidden_size": out_dim})()

  def forward(self, pixel_values):
    return self.linear(pixel_values.mean(dim=[2, 3]))


class TestDINOLoss:

  def test_returns_scalar(self):
    loss_fn = DINOLoss(out_dim=64)
    s = torch.randn(4, 64)
    t = torch.randn(4, 64)
    loss = loss_fn(s, t)
    assert loss.ndim == 0

  def test_no_nan(self):
    loss_fn = DINOLoss(out_dim=64)
    s = torch.randn(4, 64)
    t = torch.randn(4, 64)
    loss = loss_fn(s, t)
    assert not torch.isnan(loss)

  def test_center_update(self):
    loss_fn = DINOLoss(out_dim=64)
    before = loss_fn.center.clone()
    t = torch.randn(8, 64)
    loss_fn.update_center(t)
    assert not torch.equal(loss_fn.center, before)

  def test_center_momentum(self):
    loss_fn = DINOLoss(out_dim=64, center_momentum=0.5)
    t = torch.randn(8, 64)
    expected = loss_fn.center * 0.5 + t.mean(dim=0, keepdim=True) * 0.5
    loss_fn.update_center(t)
    assert torch.allclose(loss_fn.center, expected, atol=1e-6)


class TestMultiCropTransform:

  def test_output_count(self):
    tf = MultiCropTransform(global_size=64, local_size=32, local_num=6)
    image = torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8)
    from PIL import Image
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    crops = tf(pil)
    assert len(crops) == 2 + 6

  def test_split_crops(self):
    tf = MultiCropTransform(global_size=64, local_size=32, local_num=4)
    image = torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8)
    from PIL import Image
    pil = Image.fromarray(image.permute(1, 2, 0).numpy())
    crops = tf(pil)
    g, local = tf.split_crops(crops)
    assert g.shape[0] == 2
    assert g.shape[2:] == (64, 64)
    assert local.shape[0] == 4
    assert local.shape[2:] == (32, 32)


class TestDINOModule:

  def _make_dino(self, proj_dim=32):
    backbone = _FakeBackbone(out_dim=128)
    return DINO(backbone, proj_dim=proj_dim, proj_hidden=64, backbone_dim=128)

  def test_forward_returns_loss(self):
    model = self._make_dino()
    g = torch.randn(2, 3, 32, 32)
    local = torch.randn(4, 3, 32, 32)
    loss, info = model(g, local)
    assert loss.ndim == 0
    assert "loss" in info

  def test_no_nan(self):
    model = self._make_dino()
    g = torch.randn(2, 3, 32, 32)
    local = torch.randn(4, 3, 32, 32)
    loss, _ = model(g, local)
    assert not torch.isnan(loss)

  def test_backward(self):
    model = self._make_dino()
    g = torch.randn(2, 3, 32, 32)
    local = torch.randn(4, 3, 32, 32)
    loss, _ = model(g, local)
    loss.backward()
    for p in model.student.parameters():
      if p.requires_grad:
        assert p.grad is not None

  def test_momentum_update(self):
    model = self._make_dino()
    before = {n: p.clone() for n, p in model.teacher.named_parameters()}
    model.update_momentum(0.9)
    changed = False
    for n, p in model.teacher.named_parameters():
      if not torch.equal(p, before[n]):
        changed = True
        break
    assert changed

  def test_teacher_no_grad(self):
    model = self._make_dino()
    for p in model.teacher.parameters():
      assert not p.requires_grad


class TestDINOMethod:

  def test_registered(self):
    cls = get_method("dino")
    assert cls.NAME == "dino"

  def test_needs_labels(self):
    method = get_method("dino")()
    assert method.needs_labels is False

  def test_add_args(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.dino_proj_dim == 256
    assert args.dino_teacher_temp == 0.04
    assert args.dino_local_num == 8

  def test_build(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    backbone = _FakeBackbone(out_dim=128)
    device = torch.device("cpu")
    model = method.build(args, backbone, device)
    assert isinstance(model, DINO)

  def test_train_step(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    backbone = _FakeBackbone(out_dim=128)
    device = torch.device("cpu")
    model = method.build(args, backbone, device)
    crops = [torch.randn(2, 3, 32, 32) for _ in range(2)]  # 2 global
    crops += [torch.randn(2, 3, 32, 32) for _ in range(4)]  # 4 local
    loss, info = method.train_step(model, crops, global_step=0)
    assert loss.ndim == 0
    assert "loss" in info

  def test_checkpoint_roundtrip(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    backbone = _FakeBackbone(out_dim=128)
    device = torch.device("cpu")
    model = method.build(args, backbone, device)
    state = method.get_checkpoint_state(model, args)
    assert state["method"] == "dino"
    assert "center" in state
    method2 = get_method("dino")()
    method2.load_checkpoint_state(model, state, args)

  def test_validate_returns_none(self):
    method = get_method("dino")()
    assert method.validate(None, torch.randn(2, 3, 32, 32), 2) is None

  def test_build_transform(self):
    method = get_method("dino")()
    method._dino_global_size = 64
    method._dino_local_size = 32
    method._dino_local_num = 4
    tf = method.build_transform(128)
    assert isinstance(tf, MultiCropTransform)
