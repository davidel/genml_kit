"""Tests for DINO pre-training method (v4.2: genml_kit.methods)."""

import argparse

import torch

from genml_kit.methods import get_method
from genml_kit.models.dino import DINO
from genml_kit.pipelines.contracts import DataBlob, LossOutput
from genml_kit.augmentations.multicrop import MultiCropTransform
from genml_kit.losses.dino import DINOLoss


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
    assert method.NEEDS_LABELS is False

  def test_add_args(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.dino_proj_dim == 256
    assert args.dino_teacher_temp == 0.04
    assert args.dino_local_num == 8

  def test_train_step_returns_lossoutput(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=args.dino_proj_dim,
                 proj_hidden=64,
                 backbone_dim=128)
    method._dino_momentum = args.dino_momentum
    method._dino_final_momentum = args.dino_final_momentum
    # Multi-crop collated blob: tuple of stacked global/local crops.
    v1 = torch.randn(2, 3, 32, 32)
    v2 = torch.randn(2, 3, 32, 32)
    local = torch.randn(4, 3, 32, 32)
    out = method.train_step(model,
                            DataBlob(data=(v1, v2, local), meta={}),
                            global_step=0)
    assert isinstance(out, LossOutput)
    assert out.loss.ndim == 0
    assert "loss" in out.metrics

  def test_train_step_unequal_global_local_sizes_end_to_end(self):
    """Regression for #1/#5: multi-crop collate + train_step, unequal sizes.

    MultiCropTransform produces (global1, global2, local1..N); the
    production collate stacks each position; ``_run_loss`` must split the
    tuple back into (2B global, N*B local) tensors.  Before the fix this
    crashed (the old code treated ``images[0]`` as ALL globals) or silently
    mis-paired teacher targets (#5).
    """
    from genml_kit.augmentations.multicrop import MultiCropTransform
    from genml_kit.pipelines.images import ImagesPipeline

    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    method._dino_global_size = 32
    method._dino_local_size = 16
    method._dino_local_num = 3
    method._dino_momentum = args.dino_momentum
    method._dino_final_momentum = args.dino_final_momentum
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=args.dino_proj_dim,
                 proj_hidden=64,
                 backbone_dim=128)

    tf = MultiCropTransform(global_size=32, local_size=16, local_num=3)
    # Apply the real per-item multi-crop transform, then collate: this is
    # the exact production path (#1 crashed before the split fix).
    items = [{"image": tf(torch.randn(3, 64, 64))} for _ in range(2)]
    blob = ImagesPipeline._collate(items)  # DataBlob, tuple of stacked crops
    assert isinstance(blob.data, tuple)
    assert len(blob.data) == 5  # 2 global + 3 local positions
    assert blob.data[0].shape[0] == 2  # each position stacked over B

    out = method.train_step(model, blob, global_step=0)
    assert isinstance(out, LossOutput)
    assert out.loss.ndim == 0
    assert "loss" in out.metrics

  def test_checkpoint_roundtrip(self):
    method = get_method("dino")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=args.dino_proj_dim,
                 proj_hidden=64,
                 backbone_dim=128)
    state = method.get_checkpoint_state(model, args)
    assert state["method"] == "dino"
    assert "center" in state
    method2 = get_method("dino")()
    method2.load_checkpoint_state(model, state, args)
    assert method2._dino_momentum == args.dino_momentum

  def test_log_validation_is_noop(self):
    method = get_method("dino")()

    class _Writer:

      def __init__(self):
        self.calls = []

      def add_image(self, tag, tensor, step):
        self.calls.append(tag)

    def _to_device(blob, device):
      return blob

    method.log_validation(None, iter([]), _to_device, _Writer(), 0, torch.device("cpu"))
    # Nothing was logged; the method has no vis support.
    assert True  # reaching here without raising is the contract

  def test_build_transform(self):
    method = get_method("dino")()
    method._dino_global_size = 64
    method._dino_local_size = 32
    method._dino_local_num = 4
    args = argparse.Namespace(image_size=128)
    tf = method.build_transform(args, 128)
    assert isinstance(tf, MultiCropTransform)

  def test_momentum_ramp_uses_wire_data_budget(self):
    """Regression for #2: the teacher EMA must actually ramp.

    With a step budget wired, step 0 uses the start momentum (< 1.0); the
    old code always returned the END momentum (1.0) because _total_steps
    was never set, freezing the teacher.
    """
    method = get_method("dino")()
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=32,
                 proj_hidden=64,
                 backbone_dim=128)
    method._dino_momentum = 0.996
    method._dino_final_momentum = 1.0
    method._total_steps = 100
    assert method._current_momentum(0, model) < 1.0
    assert method._current_momentum(0, model) == method._momentum_start()
    assert method._current_momentum(100, model) == 1.0

  def test_momentum_missing_budget_falls_back_to_start(self):
    """Without a budget the teacher still MOVES (never pins to 1.0)."""
    method = get_method("dino")()
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=32,
                 proj_hidden=64,
                 backbone_dim=128)
    method._dino_momentum = 0.996
    method._dino_final_momentum = 1.0
    # as if wire_data never ran: no _total_steps attribute at all
    assert not hasattr(method, "_total_steps")
    assert method._current_momentum(0, model) == method._momentum_start()
    assert method._current_momentum(0, model) < 1.0

  def test_teacher_diverges_from_init_within_few_ema_steps(self):
    """A moving teacher must differ from its init after several EMA steps."""
    method = get_method("dino")()
    model = DINO(_FakeBackbone(out_dim=128),
                 proj_dim=32,
                 proj_hidden=64,
                 backbone_dim=128)
    method._dino_momentum = 0.996
    method._dino_final_momentum = 1.0
    method._total_steps = 100
    init = torch.cat([p.detach().reshape(-1) for p in model.teacher.parameters()])
    # Perturb the student a bit, then run several teacher updates.
    with torch.no_grad():
      for p in model.student.parameters():
        p.add_(torch.randn_like(p) * 0.1)
      for step in range(5):
        model.update_momentum(method._current_momentum(step, model))
    after = torch.cat([p.detach().reshape(-1) for p in model.teacher.parameters()])
    assert not torch.equal(init, after)
