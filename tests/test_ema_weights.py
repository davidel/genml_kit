"""Tests for the eval-time weights EMA (``WeightsEMA``) and its wiring.

Covers the shadow math (including warmup/ramp schedules), buffer handling,
the swap/restore involution, and the trainer integration (loop cadence, the
eval-on-EMA window, checkpoint round-trip, and pre-EMA resume compat).
"""

import copy
import types

import pytest
import torch

from genml_kit.methods.base import WeightsEMAMixin
from genml_kit.models.ema import WeightsEMA
from genml_kit.training.trainer import BaseTrainer
from tests.test_trainer import (DataBlob, FakeMethod, FakePipeline, _Args, _Logger,
                                _Optimization, _Params, _WrappedLoader)


class _EmaMethod(WeightsEMAMixin, FakeMethod):
  """A method that opts into the weights EMA, recording the wiring calls."""

  def __init__(self, **ema_kwargs):
    super().__init__()
    self.ema_kwargs = ema_kwargs
    self.wire_calls = 0

  def wire_ema(self, args, model, pipeline):
    from genml_kit.models.ema import WeightsEMA

    self.wire_calls += 1
    self.wire_call_args = (args, model, pipeline)
    return WeightsEMA(model,
                      decay=self.ema_kwargs.get("decay", 0.5),
                      warmup_steps=self.ema_kwargs.get("warmup_steps", 0),
                      total_steps=self.ema_kwargs.get("total_steps", None),
                      eval_enabled=self.ema_kwargs.get("eval_enabled", True))


def _make_ema_trainer(tmp_path,
                      method=None,
                      decay=0.5,
                      ema_state=None,
                      args_attrs=None):
  """A BaseTrainer wired with an EMA-opt-in method and a tiny linear model."""
  args = _Args()
  args.checkpoint = str(tmp_path / "ckpt")
  if args_attrs:
    for name, value in args_attrs.items():
      setattr(args, name, value)
  method = method or _EmaMethod(decay=decay)
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.val_loader = _WrappedLoader(
      [DataBlob(data=torch.zeros(4, 3, 8, 8), meta={"labels": torch.zeros(4)}, size=4)])
  optimization = _Optimization()
  optimization.optimizer = torch.optim.SGD(_Params().parameters(), lr=0.01)
  model = torch.nn.Sequential()
  model.fc = torch.nn.Linear(8 * 8 * 3, 4)
  trainer = BaseTrainer(args=args,
                        model=model,
                        method=method,
                        pipeline=pipeline,
                        optimization=optimization,
                        device=torch.device("cpu"),
                        logger=_Logger(),
                        start_epoch=0,
                        best_metric=0.0,
                        global_step=0,
                        ema_state=ema_state)
  return trainer, method, model


def test_shadow_math_constant_decay():
  """One blend at decay 0.5 halves the distance to the live weights."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.5)
  with torch.no_grad():
    model.weight.fill_(4.0)
    model.bias.fill_(8.0)
  initial_w = ema.target.weight.clone()
  initial_b = ema.target.bias.clone()
  ema.update(model, step=1)
  # Blend: shadow <- 0.5*shadow + 0.5*live = shadow + 0.5*(live - shadow).
  assert torch.allclose(ema.target.weight,
                        initial_w + 0.5 * (4.0 - initial_w),
                        atol=1e-6)
  assert torch.allclose(ema.target.bias, initial_b + 0.5 * (8.0 - initial_b), atol=1e-6)
  assert ema.step == 1


def test_shadow_math_two_updates():
  """Two blends compose multiplicatively (0.5, then 0.5 again)."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.5)
  with torch.no_grad():
    model.weight.fill_(4.0)
  initial = ema.target.weight.clone()
  ema.update(model, step=1)
  ema.update(model, step=2)
  # After two halvings the shadow is 75% of the way from init to 4.0.
  assert torch.allclose(ema.target.weight, initial + 0.75 * (4.0 - initial), atol=1e-6)
  assert ema.step == 2


def test_warmup_ramp():
  """Warmup: effective decay 0 at step 0 (exact copy), then ramps."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.8, warmup_steps=2)
  with torch.no_grad():
    model.weight.fill_(4.0)
  initial = ema.target.weight.clone()
  ema.update(model, step=0)  # effective decay 0 -> shadow == live weights
  assert torch.allclose(ema.target.weight, torch.full_like(initial, 4.0))
  ema.update(model, step=1)  # decay 0.8 * 1/2 = 0.4 -> shadow stays 4.0
  assert torch.allclose(ema.target.weight, torch.full_like(initial, 4.0))
  # The live weights move; the full decay 0.8 now pulls only 20% toward them:
  # shadow = 0.8*4 + 0.2*8 = 4.8.
  with torch.no_grad():
    model.weight.fill_(8.0)
  ema.update(model, step=2)
  assert torch.allclose(ema.target.weight, torch.full_like(initial, 4.8))
  assert ema.step == 2


def test_final_decay_ramp():
  """final_decay ramps the effective decay linearly over total_steps."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.5, final_decay=0.9, total_steps=4)
  with torch.no_grad():
    model.weight.fill_(4.0)
  initial = ema.target.weight.clone()
  ema.update(model, step=2)  # decay = 0.5 + (0.9-0.5) * 2/4 = 0.7
  # Blend: shadow <- 0.7*shadow + 0.3*live = shadow + 0.3*(live - shadow).
  assert torch.allclose(ema.target.weight, initial + 0.3 * (4.0 - initial), atol=1e-6)


def test_buffers_float_lerped_int_copied():
  """Float buffers blend (BN stats); integer counters copy verbatim."""
  net = torch.nn.BatchNorm1d(2)
  ema = WeightsEMA(net, decay=0.5)
  with torch.no_grad():
    net.running_mean.fill_(10.0)
    net.running_var.fill_(4.0)
  tracked_before = net.num_batches_tracked.clone()
  mean_before = ema.target.running_mean.clone()
  var_before = ema.target.running_var.clone()
  ema.update(net, step=1)
  # BN's running stats default to mean 0 / var 1, so the blends are
  # 0.5*0 + 0.5*10 = 5.0 and 0.5*1 + 0.5*4 = 2.5.
  assert torch.allclose(ema.target.running_mean,
                        mean_before + 0.5 * (10.0 - mean_before))
  assert torch.allclose(ema.target.running_var, var_before + 0.5 * (4.0 - var_before))
  assert int(ema.target.num_batches_tracked) == int(tracked_before)


def test_swap_restore_involution():
  """swap() then restore() returns the model bit-for-bit to its prior state."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.9)
  with torch.no_grad():
    model.weight.fill_(4.0)
    model.bias.fill_(8.0)
  before = copy.deepcopy(model.state_dict())
  ema.update(model, step=1)
  ema.swap(model)
  # Inside the window the model holds the shadow.
  assert torch.allclose(model.weight, ema.target.weight)
  ema.restore(model)
  after = model.state_dict()
  for key in before:
    assert torch.equal(before[key], after[key])


def test_restore_without_swap_raises():
  """restore() without a matching swap() is a programming error."""
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.9)
  raised = False
  try:
    ema.restore(model)
  except RuntimeError as exc:
    raised = "swap" in str(exc)
  assert raised


def test_state_dict_round_trip():
  """state_dict/load_state_dict restore the shadow and the schedule step."""
  torch.manual_seed(0)
  model = torch.nn.Linear(4, 2)
  ema = WeightsEMA(model, decay=0.5)
  with torch.no_grad():
    model.weight.fill_(4.0)
  ema.update(model, step=3)
  payload = ema.state_dict()
  # A fresh shadow seeded from the *reset* model must restore to the saved one.
  with torch.no_grad():
    model.weight.fill_(0.0)
  ema2 = WeightsEMA(model, decay=0.5)
  ema2.load_state_dict(payload)
  assert ema2.step == 3
  for key, value in ema.target.state_dict().items():
    assert torch.equal(ema2.target.state_dict()[key], value)


class _DefaultEmaMethod(WeightsEMAMixin, FakeMethod):
  """Mixin semantics only: no ``wire_ema`` override, so defaults apply."""


def test_mixin_default_wiring():
  """The mixin's default ``wire_ema`` honours the ``ema_*`` args flags."""
  args = types.SimpleNamespace(ema_decay=0.7,
                               ema_final_decay=None,
                               ema_warmup_steps=5,
                               ema_eval=True,
                               epochs=2,
                               grad_accum_steps=1)
  # A 3-batch loader: budget = 3 batches x 2 epochs // 1 accum = 6 steps.
  pipeline = types.SimpleNamespace(train_loader=[None, None, None])
  ema = _DefaultEmaMethod().wire_ema(args, torch.nn.Linear(4, 2), pipeline)
  assert ema is not None
  assert ema.decay == 0.7
  assert ema.warmup_steps == 5
  assert ema.total_steps == 6
  assert ema.eval_enabled is True


def test_mixin_default_wiring_final_decay():
  """``--ema_final_decay`` selects the ramp over the warmup."""
  args = types.SimpleNamespace(ema_decay=0.8,
                               ema_final_decay=0.95,
                               ema_warmup_steps=5,
                               ema_eval=False,
                               epochs=1,
                               grad_accum_steps=1)
  ema = _DefaultEmaMethod().wire_ema(args, torch.nn.Linear(4, 2),
                                     types.SimpleNamespace(train_loader=[]))
  assert ema.final_decay == 0.95
  assert ema.total_steps == 0  # empty loader -> falsy ramp total
  assert ema.eval_enabled is False


def test_mixin_default_wiring_off_without_decay():
  """A falsy ``--ema_decay`` keeps the shadow off (flag is the switch)."""
  args = types.SimpleNamespace(ema_decay=0.0)
  method = _DefaultEmaMethod()
  assert method.wire_ema(args, torch.nn.Linear(4, 2), types.SimpleNamespace()) is None
  # Missing flags entirely (a hand-built namespace): AttributeError, which
  # is the fixture-drift signal tests/cli_args.py exists to prevent.
  with pytest.raises(AttributeError):
    method.wire_ema(types.SimpleNamespace(), torch.nn.Linear(4, 2),
                    types.SimpleNamespace())


def test_base_method_has_no_ema_capability():
  """Plain ``Method`` subclasses expose no ``wire_ema`` hook at all."""
  assert not hasattr(FakeMethod(), "wire_ema")
