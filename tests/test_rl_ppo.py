"""Tests for PPOMethod (GAE, clipped surrogate, train_step)."""

import argparse

import pytest
import torch

from genml_kit.methods import get_method
from genml_kit.methods.rl_ppo import PPOMethod
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.datasets.replay_buffer import ReplayBufferDataset


def _make_args(**overrides):
  defaults = dict(
      ppo_gamma=0.99,
      ppo_lam=0.95,
      ppo_clip_eps=0.2,
      ppo_epochs=2,
      ppo_mini_batch_size=4,
      ppo_entropy_coef=0.01,
      ppo_value_coef=0.5,
      ppo_vf_clip_eps=None,
      ppo_rollout_len=16,
      ppo_discrete=True,
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
  env = _ScriptedEnv(obs_dim=obs_dim, max_episode_length=6)
  pipeline.env = env
  pipeline._obs_dim = obs_dim
  pipeline._n_actions = env.action_space.n
  pipeline.replay_buffer = ReplayBufferDataset(
      obs_dim=obs_dim,
      capacity=50,
  )
  from genml_kit.datasets.rollout_buffer import RolloutBuffer
  pipeline.rollout_buffer = RolloutBuffer(
      obs_dim=obs_dim,
      rollout_len=16,
  )
  # Discrete.
  pipeline._action_dim = None

  method = get_method("ppo")()
  args = _make_args()
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  return pipeline, method, model


class TestPPOMethod:

  def test_registered(self):
    assert get_method("ppo") is PPOMethod

  def test_name_and_metric(self):
    assert PPOMethod.NAME == "ppo"
    assert PPOMethod.METRIC_KEY == "eval_return"
    assert PPOMethod.METRIC_MINIMIZE is False

  def test_act_returns_valid(self):
    _, method, model = _make_pipeline_and_method()
    obs = torch.randn(4)
    action, log_prob, value, raw_action = method.act(model, obs, deterministic=True)
    assert action in (0, 1)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    # Discrete.
    assert raw_action is None

  def test_train_step(self):
    _, method, model = _make_pipeline_and_method()
    batch = {
        "obs": torch.randn(8, 4),
        "action": torch.randint(0, 2, (8,)),
        "log_prob": torch.randn(8) - 1.0,
        "advantage": torch.randn(8),
        "return": torch.randn(8),
        "value": torch.randn(8),
    }
    loss_out = method.train_step(model, batch, global_step=0)
    assert loss_out.loss.requires_grad
    assert "pg_loss" in loss_out.metrics
    assert "value_loss" in loss_out.metrics
    assert "entropy" in loss_out.metrics

  def test_evaluate(self):
    pipeline, method, model = _make_pipeline_and_method()
    pipeline.init_env(_make_args())
    result = method.evaluate(model, pipeline, num_episodes=2)
    assert "eval_return" in result
    assert "eval_steps" in result
    assert result["eval_return"] >= 0.0

  def test_has_metric_improved(self):
    method = PPOMethod()
    # new=5, best=3 -> improved
    assert method.has_metric_improved(5.0, 3.0)
    # new=3, best=5 -> not improved
    assert not method.has_metric_improved(3.0, 5.0)

  def test_update_target_noop(self):
    _, method, model = _make_pipeline_and_method()
    method.update_target(model, global_step=1)

  def test_checkpoint_round_trip(self):
    _, method, model = _make_pipeline_and_method()
    method._env_steps = 999
    args = _make_args()
    state = method.get_checkpoint_state(model, args)
    assert state["method"] == "ppo"
    assert state["env_steps"] == 999

    method2 = get_method("ppo")()
    method2.wire_data(args, _make_pipeline_and_method()[0])
    method2.build_model(args, device=torch.device("cpu"))
    method2.load_checkpoint_state(model, state, args)
    assert method2._env_steps == 999

  def test_add_args_no_collision(self):
    parser = argparse.ArgumentParser()
    PPOMethod.add_args(parser)
    args = parser.parse_args([])
    assert args.ppo_gamma == 0.99
    assert args.ppo_clip_eps == 0.2
    assert args.ppo_epochs == 4

  def test_continuous_mode(self):
    pipeline = RLPipeline()
    env = _ScriptedEnv(obs_dim=4, max_episode_length=6)
    pipeline.env = env
    pipeline._obs_dim = 4
    pipeline._n_actions = 2
    pipeline.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=50)
    pipeline._action_dim = 2

    from genml_kit.datasets.rollout_buffer import RolloutBuffer
    pipeline.rollout_buffer = RolloutBuffer(
        obs_dim=4,
        rollout_len=16,
        action_dim=2,
    )

    method = get_method("ppo")()
    args = _make_args(ppo_discrete=False)
    method.wire_data(args, pipeline)
    method._action_dim = 2
    method._discrete = False
    model = method.build_model(args, device=torch.device("cpu"))

    action, log_prob, value, raw_action = method.act(model, torch.randn(4))
    assert action.shape == (2,)
    assert raw_action.shape == (2,)
    assert isinstance(log_prob, float)

  def test_continuous_log_prob_round_trip(self):
    """log_prob(stored raw action) approx equals the rollout-time log_prob.

    Guards against the classic continuous-PPO bug where the update-time
    re-evaluation disagrees with the log_prob stored during rollout.
    """
    pipeline = RLPipeline()
    env = _ScriptedEnv(obs_dim=4, max_episode_length=6, continuous=True)
    pipeline.env = env
    pipeline._obs_dim = 4
    pipeline._action_dim = 2
    pipeline.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=50)
    from genml_kit.datasets.rollout_buffer import RolloutBuffer
    pipeline.rollout_buffer = RolloutBuffer(
        obs_dim=4,
        rollout_len=16,
        action_dim=2,
    )

    method = get_method("ppo")()
    args = _make_args(ppo_discrete=False)
    method.wire_data(args, pipeline)
    method._discrete = False
    method._action_dim = 2
    model = method.build_model(args, device=torch.device("cpu"))

    # Simulate one rollout step: record log_prob at the raw action.
    obs = torch.randn(4)
    action, log_prob, _, raw_action = method.act(model, obs)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      _, _, re_eval, _, _ = model.get_action_and_value(
          obs_t,
          action=torch.as_tensor(raw_action, dtype=torch.float32).unsqueeze(0),
      )
    assert re_eval.item() == pytest.approx(log_prob, abs=1e-4)
