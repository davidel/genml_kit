"""Tests for the warmup schedulers (epoch-based ramp + shape).

Stepping order matters: PyTorch requires ``optimizer.step()`` before
``scheduler.step()`` (and this project runs pytest with ``filterwarnings =
error``), so every stepping test goes through :func:`_step`.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from genml_kit.training.optim_factory import (
    WarmupScheduler,
    build_optimization,
    create_scheduler,
)
from tests.cli_args import train_args


def _make(model_lr=1.0):
  """A model + optimizer pair (lr=1.0 for readable LR arithmetic)."""
  model = nn.Linear(4, 2)
  return optim.SGD(model.parameters(), lr=model_lr)


def _step(sched, opt, n=1):
  """Advance n epochs in the PyTorch-required order (opt first)."""
  for _ in range(n):
    opt.step()
    sched.step()


class TestDispatch:

  def test_warmup_names_registered(self):
    for name in ("warmup_cosine", "warmup_linear", "warmup_rsqrt"):
      opt = _make()
      sched = create_scheduler(opt, name=name, epochs=10, base_lr=1.0)
      assert isinstance(sched, WarmupScheduler)

  def test_no_warmup_matches_plain_shape(self):
    opt_plain = _make()
    plain = create_scheduler(opt_plain,
                             name="CosineAnnealingLR",
                             epochs=10,
                             base_lr=1.0,
                             T_max=10)
    opt_zero = _make()
    zero = create_scheduler(opt_zero,
                            name="warmup_cosine",
                            epochs=10,
                            base_lr=1.0,
                            warmup_epochs=0.0)
    for _ in range(10):
      opt_plain.step()
      plain.step()
      opt_zero.step()
      zero.step()
    # torch's recursion vs our closed form differ in the last bits, so
    # compare with a tolerance instead of exact equality.
    for a, b in zip(plain.get_last_lr(), zero.get_last_lr(), strict=True):
      assert a == pytest.approx(b, abs=1e-9)


class TestWarmupRamp:
  """Ramp semantics: epoch t gets ``min(1, (t + 1) / w)`` while ``t < w``.

  The multiplier at t=0 is (0+1)/w — a fraction of the peak, never zero —
  so warmup never wastes an entire epoch at LR ~ 0.
  """

  def test_epoch_zero_is_fraction(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=2.0)
    # get_lr() runs once during __init__ with last_epoch=0 -> (0+1)/2.
    assert sched.get_last_lr()[0] == pytest.approx(0.5)

  def test_peak_at_last_warmup_epoch(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=2.0)
    _step(sched, opt)  # last_epoch=1 -> (1+1)/2 == 1.0 (peak)
    assert sched.get_last_lr()[0] == pytest.approx(1.0)

  def test_shape_takes_over_after_warmup(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=2.0)
    _step(sched, opt, 2)  # last_epoch=2 -> shape branch, decaying
    assert sched.get_last_lr()[0] < 1.0

  def test_ramp_is_linear_across_epochs(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=3.0)
    assert sched.get_last_lr()[0] == pytest.approx(1.0 / 3.0)
    _step(sched, opt)
    assert sched.get_last_lr()[0] == pytest.approx(2.0 / 3.0)
    _step(sched, opt)
    assert sched.get_last_lr()[0] == pytest.approx(1.0)

  def test_fractional_warmup_under_one_epoch(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=0.5)
    # (0+1)/0.5 = 2.0 clips to 1.0: a sub-epoch warmup trains epoch 0 at
    # peak LR rather than idling.
    assert sched.get_last_lr()[0] == pytest.approx(1.0)
    _step(sched, opt)
    assert sched.get_last_lr()[0] < 1.0

  def test_warmup_zero_starts_at_peak(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_linear",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=0.0)
    assert sched.get_last_lr()[0] == pytest.approx(1.0)


class TestShapes:
  """Post-warmup shapes, exercised with warmup_epochs=0."""

  def _lr_at(self, shape, epoch, epochs=10, warmup=0.0):
    opt = _make()
    sched = create_scheduler(opt,
                             name=f"warmup_{shape}",
                             epochs=epochs,
                             base_lr=1.0,
                             warmup_epochs=warmup)
    _step(sched, opt, epoch)
    return sched.get_last_lr()[0]

  def test_linear_decays_to_zero(self):
    assert self._lr_at("linear", 10) == pytest.approx(0.0)

  def test_cosine_decays_to_zero(self):
    assert self._lr_at("cosine", 10) == pytest.approx(0.0, abs=1e-9)

  def test_cosine_midpoint_half(self):
    assert self._lr_at("cosine", 5) == pytest.approx(0.5)

  def test_rsqrt_never_reaches_zero(self):
    assert self._lr_at("rsqrt", 10) > 0.0

  def test_warmup_then_decay(self):
    lr_peak = self._lr_at("cosine", 1, warmup=2.0)
    lr_decay = self._lr_at("cosine", 2, warmup=2.0)
    lr_end = self._lr_at("cosine", 10, warmup=2.0)
    assert lr_peak == pytest.approx(1.0)  # peak at ramp end (t=1)
    assert lr_decay < lr_peak  # shape branch has taken over
    assert lr_end < lr_decay  # fully decayed


class TestSchedArgOverride:

  def test_sched_arg_wins_over_flag(self):
    # --sched_arg warmup_epochs=4 must override --warmup_epochs 1.  This is
    # tested through build_optimization, the same precedence path the CLI
    # uses (the override is splatted, not passed as a nested dict).
    args = train_args([
        "--scheduler", "warmup_cosine", "--warmup_epochs", "1", "--sched_arg",
        "warmup_epochs=4"
    ],
                      lr=1.0)
    model = nn.Linear(4, 2)
    optim = build_optimization(args, model, torch.device("cpu"), {}, [])
    _step(optim.scheduler, optim.optimizer, 2)
    # With the override, t=2 is still ramping at 3/4 of the peak; with the
    # flag value (w=1) the shape branch would already control t=2.
    assert optim.scheduler.get_last_lr()[0] == pytest.approx(0.75)

  def test_t_max_override(self):
    opt = _make()
    sched = create_scheduler(opt,
                             name="warmup_cosine",
                             epochs=10,
                             base_lr=1.0,
                             warmup_epochs=0.0,
                             T_max=5)
    _step(sched, opt, 5)
    assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)

  def test_warmup_flag_not_forwarded_to_plain_torch(self):
    # The generic --warmup_epochs flag must NOT reach plain torch schedulers
    # (they would reject the unknown keyword).
    args = train_args([
        "--scheduler", "CosineAnnealingLR", "--sched_arg", "T_max=10",
        "--warmup_epochs", "1"
    ],
                      lr=1.0)
    model = nn.Linear(4, 2)
    optim = build_optimization(args, model, torch.device("cpu"), {}, [])
    assert not isinstance(optim.scheduler, WarmupScheduler)
    _step(optim.scheduler, optim.optimizer)
    # torch's cosine after one step: 0.5 * (1 + cos(pi / T_max)).
    expected = 0.5 * (1 + math.cos(math.pi / 10))
    assert optim.scheduler.get_last_lr()[0] == pytest.approx(expected)
