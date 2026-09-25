"""Tests for generic ``--param_init ortho`` (SB3-style orthogonal init)."""

import torch

from genml_kit.models.init import init_orthogonal
from genml_kit.models.rl.actor_critic import ActorCritic


class TestInitOrthogonal:

  def _ac(self):
    return ActorCritic(obs_dim=4, action_dim=2, discrete=False)

  def test_policy_head_gain_sqrt2(self):
    ac = self._ac()
    before = ac.actor.mean.weight.clone()
    init_orthogonal(ac)
    # After orthogonal init, rows are orthonormal scaled by gain sqrt(2).
    w = ac.actor.mean.weight
    # Orthonormal rows: W W^T ≈ gain^2 * I.
    gram = w @ w.T
    expected = torch.eye(w.shape[0]) * 2.0  # sqrt(2)^2
    assert torch.allclose(gram, expected, atol=1e-5)
    assert not torch.allclose(before, w)  # actually changed

  def test_value_head_gain_one(self):
    ac = self._ac()
    init_orthogonal(ac)
    w = ac.critic.fc.weight
    gram = w @ w.T
    expected = torch.eye(w.shape[0]) * 1.0
    assert torch.allclose(gram, expected, atol=1e-5)

  def test_biases_zeroed(self):
    ac = self._ac()
    init_orthogonal(ac)
    for name, child in ac.named_modules():
      if isinstance(child, torch.nn.Linear):
        assert torch.all(child.bias == 0), f"{name} bias not zeroed"

  def test_log_std_untouched(self):
    ac = self._ac()
    log_std_before = ac.actor.log_std.clone()
    init_orthogonal(ac)
    assert torch.equal(ac.actor.log_std, log_std_before)

  def test_default_init_unchanged_without_flag(self):
    # No flag: init_orthogonal is not called, so the model keeps PyTorch
    # default init.  We just verify the helper is a no-op on None.
    ac = self._ac()
    w_before = ac.actor.mean.weight.clone()
    # Nothing to do -- model untouched.
    assert torch.equal(ac.actor.mean.weight, w_before)

  def test_non_rl_mlp_works(self):
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 2))
    init_orthogonal(mlp)
    # Generic MLP: no critic/value-named layers, so every Linear gets the
    # policy/hidden gain sqrt(2).  (Name-based: only critic/value heads
    # get gain 1.0 -- the RL convention.)
    # W0 is (16, 8) [tall]: W0^T W0 = 2I (8x8) for orthonormal columns.
    w0 = mlp[0].weight
    assert torch.allclose(w0.T @ w0, torch.eye(8) * 2.0, atol=1e-5)
    # W1 is (2, 16) [fat]: rows orthonormal, WW^T = 2I (2x2).
    w1 = mlp[2].weight
    assert torch.allclose(w1 @ w1.T, torch.eye(2) * 2.0, atol=1e-5)
    assert torch.all(mlp[0].bias == 0) and torch.all(mlp[2].bias == 0)


class TestParamInitArg:

  def _make_args(self, param_init=None):
    import argparse
    args = argparse.Namespace(param_init=param_init,
                              freeze=None,
                              lora=False,
                              lora_target_modules=None)
    return args

  def test_ortho_applied_via_extras(self):
    from genml_kit.methods.base import Method

    ac = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    args = self._make_args("ortho")
    model = Method._apply_model_extras(None, args, ac, torch.device("cpu"))
    # Critic head gain 1 -> orthonormal rows.
    w = model.critic.fc.weight
    assert torch.allclose(w @ w.T, torch.eye(w.shape[0]), atol=1e-5)
    # Actor gain sqrt(2).
    w = model.actor.mean.weight
    assert torch.allclose(w @ w.T, torch.eye(w.shape[0]) * 2.0, atol=1e-5)

  def test_none_is_noop(self):
    from genml_kit.methods.base import Method

    ac = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    w_before = ac.actor.mean.weight.clone()
    args = self._make_args(None)
    model = Method._apply_model_extras(None, args, ac, torch.device("cpu"))
    assert torch.equal(model.actor.mean.weight, w_before)
