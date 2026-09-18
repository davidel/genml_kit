"""Tests for RLPipeline (env wrapper + replay buffer + eval rollout)."""

import torch

from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.datasets.replay_buffer import Transition


class TestScriptedEnv:
  """Test the scripted env that does not require gymnasium."""

  def test_reset_returns_zero_state(self):
    env = _ScriptedEnv(obs_dim=4)
    obs = env.reset()
    assert obs.shape == (4,)
    assert obs[0] == 1.0
    assert obs.sum() == 1.0

  def test_advance_action(self):
    env = _ScriptedEnv(obs_dim=4)
    env.reset()
    obs, reward, done, _ = env.step(0)  # action 0 = advance
    assert obs[1] == 1.0
    assert obs[0] == 0.0
    assert reward == 0.0
    assert not done

  def test_stay_action(self):
    env = _ScriptedEnv(obs_dim=4)
    env.reset()
    obs, reward, done, _ = env.step(1)  # action 1 = stay
    assert obs[0] == 1.0
    assert not done

  def test_episode_terminates_at_state_3(self):
    env = _ScriptedEnv(obs_dim=4)
    env.reset()
    # Advance 3 times: 0 -> 1 -> 2 -> 3
    for _ in range(3):
      obs, reward, done, _ = env.step(0)
    assert done
    assert reward == 1.0

  def test_action_space(self):
    env = _ScriptedEnv(obs_dim=4)
    assert env.action_space.n == 2


class TestRLPipeline:
  """Test the pipeline with a scripted environment."""

  def _make_pipeline(self, obs_dim=4, capacity=100):
    """Create a pipeline wired up with a scripted env."""
    pipeline = RLPipeline()
    # Directly inject a scripted env instead of going through args.
    env = _ScriptedEnv(obs_dim=obs_dim)
    pipeline.env = env
    pipeline._obs_dim = obs_dim
    pipeline._n_actions = env.action_space.n
    from genml_kit.datasets.replay_buffer import ReplayBufferDataset
    pipeline.replay_buffer = ReplayBufferDataset(
        obs_dim=obs_dim,
        capacity=capacity,
    )
    return pipeline

  def test_build_loader_returns_none(self):
    pipeline = self._make_pipeline()
    assert pipeline.build_loader(None) is None
    assert pipeline.build_val_loader(None) is None

  def test_reset_and_step(self):
    pipeline = self._make_pipeline()
    obs = pipeline.reset_env()
    assert obs.shape == (4,)
    next_obs, reward, done, _ = pipeline.step_env(0)
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)

  def test_env_push(self):
    pipeline = self._make_pipeline()
    obs = pipeline.reset_env()
    next_obs, reward, done, _ = pipeline.step_env(0)
    pipeline.env_push(
        Transition(obs=obs, action=0, reward=reward, next_obs=next_obs, done=done))
    assert len(pipeline.replay_buffer) == 1

  def test_eval_rollout(self):
    pipeline = self._make_pipeline()
    # Deterministic policy: always action 0 (advance).
    result = pipeline.eval_rollout(action_fn=lambda obs: 0, num_episodes=3)
    assert "eval_return" in result
    assert "eval_steps" in result
    # Each episode on the 4-state chain takes exactly 3 steps (return 1.0).
    assert result["eval_return"] == 1.0
    assert result["eval_steps"] == 9  # 3 episodes × 3 steps

  def test_properties(self):
    pipeline = self._make_pipeline(obs_dim=8)
    assert pipeline.obs_dim == 8
    assert pipeline.n_actions == 2

  def test_to_device_dict(self):
    pipeline = self._make_pipeline()
    batch = {
        "obs": torch.randn(4, 4),
        "action": torch.randint(0, 2, (4,)),
        "reward": torch.randn(4),
        "next_obs": torch.randn(4, 4),
        "done": torch.zeros(4),
    }
    result = pipeline.to_device(batch, torch.device("cpu"))
    assert isinstance(result, dict)
    assert result["obs"].device == torch.device("cpu")

  def test_to_device_datablob(self):
    pipeline = self._make_pipeline()
    batch = {
        "obs": torch.randn(4, 4),
        "action": torch.randint(0, 2, (4,)),
        "reward": torch.randn(4),
        "next_obs": torch.randn(4, 4),
        "done": torch.zeros(4),
    }
    blob = DataBlob(data=batch, meta={"step": 42})
    result = pipeline.to_device(blob, torch.device("cpu"))
    assert isinstance(result, DataBlob)
    assert result.meta["step"] == 42
