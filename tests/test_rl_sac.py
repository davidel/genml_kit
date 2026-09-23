"""Tests for SACMethod (twin critics, reparameterization, auto-alpha)."""

import argparse

import pytest
import torch

from genml_kit.methods import get_method
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

  method = get_method("sac")()
  args = _make_args()
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  return pipeline, method, model


class TestSACMethod:

  def test_registered(self):
    assert get_method("sac") is SACMethod

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

    method2 = get_method("sac")()
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

    method = get_method("sac")()
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

    method = get_method("sac")()
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
