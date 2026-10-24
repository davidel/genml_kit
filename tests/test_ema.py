"""Tests for the shared EMA / target-network utilities."""

import copy

import torch
import torch.nn as nn

from genml_kit.models.ema import EMAModel, linear_momentum


class _Net(nn.Module):
  """Tiny deterministic network for EMA arithmetic checks."""

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(4, 3)

  def forward(self, x):
    return self.fc(x)


class TestEMAModel:

  def test_freezes_target_parameters(self):
    target = _Net()
    source = _Net()
    EMAModel(target, source)
    assert all(not p.requires_grad for p in target.parameters())

  def test_update_matches_manual_formula(self):
    target = _Net()
    source = copy.deepcopy(target)
    ema = EMAModel(target, source)
    # Perturb the source so the blend is observable.
    with torch.no_grad():
      for p in source.parameters():
        p.add_(torch.randn_like(p))

    before = [p.detach().clone() for p in target.parameters()]
    src = [p.detach().clone() for p in source.parameters()]
    decay = 0.9
    ema.update(decay)
    after = [p.detach() for p in target.parameters()]

    for b, s, a in zip(before, src, after):
      expected = decay * b + (1.0 - decay) * s
      # Tolerance covers the 1-ULP difference between the fused
      # ``mul_().add_(alpha=)`` op and the separately-rounded expectation.
      assert torch.allclose(a, expected, atol=1e-6, rtol=1e-6)

  def test_update_does_not_touch_source(self):
    target = _Net()
    source = copy.deepcopy(target)
    ema = EMAModel(target, source)
    src_before = [p.detach().clone() for p in source.parameters()]
    ema.update(0.5)
    for before, p in zip(src_before, source.parameters()):
      assert torch.equal(before, p.detach())

  def test_copy_hard_is_exact_state_dict_copy(self):
    target = _Net()
    source = _Net()
    with torch.no_grad():
      for p in source.parameters():
        p.add_(torch.randn_like(p))
    ema = EMAModel(target, source)
    ema.copy_hard()
    for st, ss in zip(target.state_dict().values(), source.state_dict().values()):
      assert torch.equal(st, ss)

  def test_polyak_convention_matches_tau(self):
    """EMAModel.update(1 - tau) must equal the SAC Polyak form."""
    target = _Net()
    source = copy.deepcopy(target)
    with torch.no_grad():
      for p in source.parameters():
        p.add_(torch.randn_like(p))

    tau = 0.005
    # Reference: target = (1 - tau) * target + tau * source.
    ref = copy.deepcopy(target)
    with torch.no_grad():
      for pt, ps in zip(ref.parameters(), source.parameters()):
        pt.mul_(1.0 - tau).add_(ps, alpha=tau)

    ema = EMAModel(target, source)
    ema.update(1.0 - tau)
    for a, b in zip(target.parameters(), ref.parameters()):
      assert torch.allclose(a.detach(), b.detach(), atol=1e-6, rtol=1e-6)

  def test_holder_not_registered_as_submodule(self):
    """EMAModel must not become a child module (would rename state keys)."""

    class Parent(nn.Module):

      def __init__(self):
        super().__init__()
        self.online = _Net()
        self.target = copy.deepcopy(self.online)
        self._ema = EMAModel(self.target, self.online)

    parent = Parent()
    # No ``_ema.*`` keys, and target keys keep their plain attribute name.
    keys = set(parent.state_dict())
    assert not any(k.startswith("_ema.") for k in keys)
    assert any(k.startswith("target.") for k in keys)
    assert any(k.startswith("online.") for k in keys)
    assert not any("_ema" in m for m in parent.named_modules())

  def test_state_dict_keys_untouched_vs_plain_hold(self):
    """Key set must match a module that freezes without the utility."""

    class Plain(nn.Module):

      def __init__(self):
        super().__init__()
        self.online = _Net()
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
          p.requires_grad = False

    class WithEMA(nn.Module):

      def __init__(self):
        super().__init__()
        self.online = _Net()
        self.target = copy.deepcopy(self.online)
        self._ema = EMAModel(self.target, self.online)

    assert set(Plain().state_dict()) == set(WithEMA().state_dict())


class TestLinearMomentum:

  def test_zero_total_returns_start(self):
    assert linear_momentum(0, 0, 0.996, 1.0) == 0.996
    assert linear_momentum(5, None, 0.996, 1.0) == 0.996

  def test_endpoints(self):
    assert linear_momentum(0, 100, 0.996, 1.0) == 0.996
    assert linear_momentum(100, 100, 0.996, 1.0) == 1.0
    # Clamps beyond the budget.
    assert linear_momentum(250, 100, 0.996, 1.0) == 1.0

  def test_midpoint(self):
    got = linear_momentum(50, 100, 0.0, 1.0)
    assert abs(got - 0.5) < 1e-12
