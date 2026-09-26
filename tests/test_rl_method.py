"""Tests for DQNMethod (epsilon schedule, train_step, checkpoint)."""

import argparse

import pytest
import torch

from genml_kit.methods import METHODS
from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.models.rl.spaces import space_spec
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.datasets.replay_buffer import ReplayBufferDataset


def _make_args(**overrides):
  """Create a minimal argparse.Namespace with DQN defaults."""
  defaults = dict(
      dqn_gamma=0.99,
      dqn_ddqn=True,
      dqn_dueling=False,
      dqn_epsilon_start=1.0,
      dqn_epsilon_end=0.02,
      dqn_epsilon_decay_steps=50_000,
      dqn_tau=1.0,
      dqn_target_update_freq=0,
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


def _make_pipeline_and_method(obs_dim=4, capacity=100):
  """Set up a DQNMethod with a scripted env pipeline."""
  pipeline = RLPipeline()
  env = _ScriptedEnv(obs_dim=obs_dim)
  pipeline.env = env
  pipeline.space = space_spec(env.observation_space, env.action_space)
  pipeline.replay_buffer = ReplayBufferDataset(
      obs_dim=obs_dim,
      capacity=capacity,
  )

  method = METHODS.get("dqn")()
  args = _make_args()
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  return pipeline, method, model


class TestDQNMethod:

  def test_registered(self):
    assert METHODS.get("dqn") is DQNMethod

  def test_name_and_metric(self):
    assert DQNMethod.NAME == "dqn"
    assert DQNMethod.METRIC_KEY == "eval_return"
    assert DQNMethod.METRIC_MINIMIZE is False

  def test_act_returns_valid_action(self):
    _, method, model = _make_pipeline_and_method()
    obs = torch.randn(4)
    action = method.act(model, obs, deterministic=True)
    assert action in (0, 1)

  def test_act_epsilon_greedy(self):
    """With epsilon=1.0, all actions should be random."""
    _, method, model = _make_pipeline_and_method()
    method._epsilon = 1.0
    obs = torch.randn(4)
    # Run many times — with eps=1.0 we should see both actions.
    actions = {method.act(model, obs, deterministic=False) for _ in range(200)}
    assert actions == {0, 1}

  def test_epsilon_decay(self):
    _, method, _ = _make_pipeline_and_method()
    method._eps_start = 1.0
    method._eps_end = 0.02
    method._decay_steps = 100

    for _ in range(100):
      method.step_epsilon()

    assert method._epsilon == pytest.approx(0.02, abs=1e-4)
    assert method._env_steps == 100

  def test_train_step(self):
    pipeline, method, model = _make_pipeline_and_method()
    # Fill buffer so sample() succeeds.
    obs = pipeline.reset_env()
    for _ in range(20):
      action = method.act(model, obs)
      next_obs, reward, done, _ = pipeline.step_env(action)
      pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done))
      obs = next_obs if not done else pipeline.reset_env()
    batch = pipeline.replay_buffer.sample(8)
    loss_out = method.train_step(model, batch, global_step=0)
    assert loss_out.loss.requires_grad
    assert "td_loss" in loss_out.metrics
    assert "q_mean" in loss_out.metrics
    assert "epsilon" in loss_out.metrics

  def test_evaluate(self):
    pipeline, method, model = _make_pipeline_and_method()
    result = method.evaluate(model, pipeline, num_episodes=2)
    assert "eval_return" in result
    assert "eval_steps" in result
    assert result["eval_return"] >= 0.0

  def test_has_metric_improved(self):
    method = DQNMethod()
    # new=5, best=3 -> improved
    assert method.has_metric_improved(5.0, 3.0)
    # new=3, best=5 -> not improved
    assert not method.has_metric_improved(3.0, 5.0)

  def test_checkpoint_round_trip(self):
    pipeline, method, model = _make_pipeline_and_method()
    method._epsilon = 0.5
    method._env_steps = 1234

    args = _make_args()
    state = method.get_checkpoint_state(model, args)
    assert state["method"] == "dqn"
    assert state["epsilon"] == 0.5
    assert state["env_steps"] == 1234

    # Load into a fresh method.
    method2 = METHODS.get("dqn")()
    method2.wire_data(args, pipeline)
    method2.build_model(args, device=torch.device("cpu"))
    method2.load_checkpoint_state(model, state, args)
    assert method2._epsilon == 0.5
    assert method2._env_steps == 1234

  def test_target_update_hard(self):
    _, method, model = _make_pipeline_and_method()
    method._target_update_freq = 0
    method._tau = 1.0
    # Mutate online weights.
    with torch.no_grad():
      for p in model.online.parameters():
        p.add_(1.0)
    method.update_target(model, global_step=1)
    for p1, p2 in zip(model.online.parameters(), model.target.parameters()):
      assert torch.allclose(p1, p2)

  def test_target_update_soft(self):
    pipeline, method, model = _make_pipeline_and_method()
    method._target_update_freq = 0
    method._tau = 0.1
    old_target = [p.clone() for p in model.target.parameters()]
    with torch.no_grad():
      for p in model.online.parameters():
        p.add_(1.0)
    method.update_target(model, global_step=1)
    # Polyak: target = (1-tau)*old_target + tau*online
    for p_old, p_online, p_target in zip(old_target, model.online.parameters(),
                                         model.target.parameters()):
      expected = 0.9 * p_old.data + 0.1 * p_online.data
      assert torch.allclose(p_target.data, expected, atol=1e-5)

  def test_add_args_no_collision(self):
    """DQN args should not collide with other methods."""
    parser = argparse.ArgumentParser()
    DQNMethod.add_args(parser)
    # Just ensure parsing works.
    args = parser.parse_args([])
    assert args.dqn_gamma == 0.99
    assert args.dqn_ddqn is True
