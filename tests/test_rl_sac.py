"""Tests for SACMethod (twin critics, reparameterization, auto-alpha)."""

import argparse

import pytest
import torch

from genml_kit.methods import METHODS
from genml_kit.methods.rl_sac import SACMethod
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.datasets.replay_buffer import ReplayBufferDataset


def _make_args(**overrides):
  defaults = dict(
      sac_gamma=0.99,
      sac_tau=0.005,
      sac_alpha=0.2,
      sac_auto_alpha=True,
      sac_target_entropy=None,
      sac_critic_lr=3e-4,
      sac_actor_lr=3e-4,
      sac_alpha_lr=3e-4,
      source_checkpoint=None,
      param_rename=None,
      freeze_patterns=None,
      lora_rank=None,
      lora_alpha=None,
      lora_target_modules=None,
      freeze=None,
      lora=None,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


def _make_pipeline_and_method(obs_dim=4):
  pipeline = RLPipeline()
  env = _ScriptedEnv(obs_dim=obs_dim, max_episode_length=6, continuous=True)
  pipeline.env = env
  pipeline._obs_dim = obs_dim
  pipeline._n_actions = 2
  pipeline.replay_buffer = ReplayBufferDataset(
      obs_dim=obs_dim,
      capacity=50,
  )
  pipeline._action_dim = 2

  method = METHODS.get("sac")()
  args = _make_args()
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  return pipeline, method, model


class TestSACMethod:

  def test_registered(self):
    assert METHODS.get("sac") is SACMethod

  def test_name_and_metric(self):
    assert SACMethod.NAME == "sac"
    assert SACMethod.METRIC_KEY == "eval_return"
    assert SACMethod.METRIC_MINIMIZE is False

  def test_act_returns_valid(self):
    _, method, model = _make_pipeline_and_method()
    obs = torch.randn(4)
    action = method.act(model, obs, deterministic=True)
    assert action.shape == (2,)

  def test_train_step(self):
    _, method, model = _make_pipeline_and_method()
    obs = torch.randn(8, 4)
    batch = {
        "obs": obs,
        "action": torch.randn(8, 2),
        "reward": torch.randn(8),
        "next_obs": torch.randn(8, 4),
        "done": torch.zeros(8),
    }
    loss_out = method.train_step(model, batch, global_step=0)
    assert loss_out.loss.requires_grad
    assert "critic_loss" in loss_out.metrics
    assert "actor_loss" in loss_out.metrics
    assert "alpha" in loss_out.metrics

  def test_update_target_soft(self):
    _, method, model = _make_pipeline_and_method()
    with torch.no_grad():
      for p in model.q1.net.parameters():
        p.add_(1.0)
    old_q1_target = [p.clone() for p in model.q1.target.parameters()]
    method.update_target(model, global_step=1)
    for p_old, p_online, p_target in zip(
        old_q1_target,
        model.q1.net.parameters(),
        model.q1.target.parameters(),
    ):
      expected = (1.0 - method._tau) * p_old + method._tau * p_online
      assert torch.allclose(p_target.data, expected, atol=1e-5)

  def test_evaluate(self):
    pipeline, method, model = _make_pipeline_and_method()
    pipeline.init_env(_make_args())
    result = method.evaluate(model, pipeline, num_episodes=2)
    assert "eval_return" in result
    assert result["eval_return"] >= 0.0

  def test_has_metric_improved(self):
    method = SACMethod()
    # new=5, best=3 -> improved
    assert method.has_metric_improved(5.0, 3.0)
    # new=3, best=5 -> not improved
    assert not method.has_metric_improved(3.0, 5.0)

  def test_checkpoint_round_trip(self):
    _, method, model = _make_pipeline_and_method()
    method._env_steps = 500
    args = _make_args()
    state = method.get_checkpoint_state(model, args)
    assert state["method"] == "sac"
    assert state["env_steps"] == 500
    assert "log_alpha" in state

    method2 = METHODS.get("sac")()
    method2.wire_data(args, _make_pipeline_and_method()[0])
    method2.build_model(args, device=torch.device("cpu"))
    method2.load_checkpoint_state(model, state, args)
    assert method2._env_steps == 500
    assert method2._log_alpha.item() == pytest.approx(state["log_alpha"])

  def test_auto_alpha(self):
    _, method, _ = _make_pipeline_and_method()
    assert method._auto_alpha is True
    assert method._get_alpha() > 0

  def test_fixed_alpha(self):
    pipeline = RLPipeline()
    env = _ScriptedEnv(obs_dim=4, max_episode_length=6, continuous=True)
    pipeline.env = env
    pipeline._obs_dim = 4
    pipeline._action_dim = 2
    pipeline.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=50)

    method = METHODS.get("sac")()
    args = _make_args(sac_auto_alpha=False, sac_alpha=0.5)
    method.wire_data(args, pipeline)
    method.build_model(args, device=torch.device("cpu"))
    assert method._get_alpha().item() == pytest.approx(0.5)

  def test_discrete_env_rejected(self):
    """SAC is continuous-only: a discrete env must raise a clear error."""
    pipeline = RLPipeline()
    env = _ScriptedEnv(obs_dim=4, max_episode_length=6)
    pipeline.env = env
    pipeline._obs_dim = 4
    pipeline._n_actions = env.action_space.n
    pipeline.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=50)
    pipeline._action_dim = 2

    method = METHODS.get("sac")()
    args = _make_args()
    with pytest.raises(ValueError, match="continuous"):
      method.wire_data(args, pipeline)

  def test_add_args_no_collision(self):
    parser = argparse.ArgumentParser()
    SACMethod.add_args(parser)
    args = parser.parse_args([])
    assert args.sac_gamma == 0.99
    assert args.sac_tau == 0.005
    assert args.sac_auto_alpha is True


class TestSACRegressions:
  """Regressions for the SAC actor-gradient / temperature bugs.

  See the corresponding fix: the actor loss must receive dQ/da (not be
  wrapped in ``torch.no_grad``), all backwards must run before any
  optimizer step, and the temperature loss must be parameterised by
  ``log_alpha`` rather than ``alpha``.
  """

  def _batch(self):
    torch.manual_seed(0)
    return {
        "obs": torch.randn(8, 4),
        "action": torch.randn(8, 2),
        "reward": torch.randn(8),
        "next_obs": torch.randn(8, 4),
        "done": torch.zeros(8),
    }

  def test_actor_q_evaluated_with_grad_enabled(self):
    """The actor-update Q evaluation must NOT run under ``torch.no_grad``.

    Regression for the bug where the actor's ``min(Q1, Q2)`` was computed
    inside a ``torch.no_grad`` block, severing ``dQ/da`` and reducing the
    policy update to an entropy-only term.

    We spy on ``q1.get_value`` -- which ``train_step`` calls once during the
    critic update (grad enabled) and once during the actor update -- and
    assert grad is enabled on *every* call.  Deterministic and RNG-free: it
    inspects the real source path rather than re-deriving the loss.
    """
    _, method, model = _make_pipeline_and_method()
    batch = self._batch()

    grad_enabled = []
    orig_get_value = model.q1.get_value

    def spy_get_value(obs, action):
      grad_enabled.append(torch.is_grad_enabled())
      return orig_get_value(obs, action)

    model.q1.get_value = spy_get_value
    method.train_step(model, batch, global_step=0)

    assert grad_enabled, "q1.get_value was never called"
    assert all(grad_enabled), (f"Q evaluated under torch.no_grad during actor update "
                               f"(grad-enabled flags: {grad_enabled})")

  def test_apply_grad_runs_all_backwards_before_steps(self):
    """No in-place autograd error: actor loss graph reuses critic params."""
    _, method, model = _make_pipeline_and_method()
    args = _make_args(grad_clip=1.0)
    method._grad_clip = 1.0
    optimization = method.build_optimization(args, model, torch.device("cpu"), {}, {})
    loss_out = method.train_step(model, self._batch(), global_step=0)
    # Must not raise "a variable needed for gradient computation has been
    # modified by an inplace operation".
    method.apply_grad(loss_out, None, None, optimization)

  def test_alpha_loss_uses_log_alpha(self):
    """Alpha loss must be parameterised by log_alpha, not alpha."""
    from genml_kit.losses.rl import sac_alpha_loss
    log_probs = torch.full((8,), -1.5)
    target_entropy = -2.0
    log_alpha = torch.tensor(-0.5, requires_grad=True)
    loss = sac_alpha_loss(log_probs, target_entropy, log_alpha)
    loss.backward()
    # d/dlog_alpha[log_alpha * base] == base, not alpha * base.
    base = -(log_probs + target_entropy).mean()
    assert torch.allclose(log_alpha.grad, base)
