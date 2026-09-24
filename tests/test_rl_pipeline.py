"""Tests for RLPipeline (env wrapper + replay buffer + eval rollout)."""

import argparse

import numpy as np
import torch

import genml_kit.pipelines.rl as rl
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
    # Action 0 = advance.
    obs, reward, done, _ = env.step(0)
    assert obs[1] == 1.0
    assert obs[0] == 0.0
    assert reward == 0.0
    assert not done

  def test_stay_action(self):
    env = _ScriptedEnv(obs_dim=4)
    env.reset()
    # Action 1 = stay.
    obs, reward, done, _ = env.step(1)
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


class _RenderableScriptedEnv(_ScriptedEnv):
  """Scripted env that pretends to support rgb_array rendering."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._render_calls = 0
    self._render_failure = None

  def can_render(self):
    try:
      self.render()
      return True
    except Exception:  # noqa: BLE001
      return False

  def render(self):
    if self._render_failure is not None:
      raise self._render_failure
    self._render_calls += 1
    return np.zeros((64, 64, 3), dtype=np.uint8)

  def render_frame(self):
    return self.render()


class TestRenderHelpers:
  """RL_VIDEO: render helpers (can_record_video / render_frame)."""

  def _make_pipeline(self):
    pipeline = RLPipeline()
    pipeline.env = _RenderableScriptedEnv(obs_dim=4)
    return pipeline

  def test_can_record_video_true_for_renderable_env(self):
    pipeline = self._make_pipeline()
    assert pipeline.can_record_video() is True

  def test_can_record_video_false_for_plain_scripted_env(self):
    pipeline = RLPipeline()
    pipeline.env = _ScriptedEnv(obs_dim=4)
    assert pipeline.can_record_video() is False

  def test_can_record_video_false_when_render_raises(self):
    pipeline = self._make_pipeline()
    pipeline.env._render_failure = RuntimeError("cannot render")
    assert pipeline.can_record_video() is False

  def test_can_record_video_false_without_env(self):
    pipeline = RLPipeline()
    assert pipeline.can_record_video() is False

  def test_render_frame_returns_frame(self):
    pipeline = self._make_pipeline()
    frame = pipeline.render_frame()
    assert frame is not None
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8

  def test_render_frame_returns_none_when_render_raises(self):
    pipeline = self._make_pipeline()
    pipeline.env._render_failure = RuntimeError("cannot render")
    assert pipeline.render_frame() is None

  def test_render_frame_returns_none_for_plain_scripted_env(self):
    pipeline = RLPipeline()
    pipeline.env = _ScriptedEnv(obs_dim=4)
    assert pipeline.render_frame() is None


class _FakeSpace:
  """Minimal stand-in for a gymnasium space (shape and/or n)."""

  def __init__(self, shape=None, n=None):
    self.shape = shape
    if n is not None:
      self.n = n


class _FakeEnv:
  """Minimal stand-in for a gymnasium Env for init_env testing."""

  def __init__(self, obs_shape, action_space):
    self.observation_space = _FakeSpace(shape=obs_shape)
    self.action_space = action_space


def _fake_env_factory(obs_shape, action_space):
  """Return a callable matching GymnasiumEnvWrapper(env_id, render_mode=...)."""

  def _factory(_env_id, render_mode=None):
    return _FakeEnv(obs_shape, action_space)

  return _factory


class TestInitEnvActionSpace:
  """Regression: init_env must classify a 1-D continuous action correctly.

  The old heuristic (``action_dim > 1`` for continuous) misclassified
  single-element Box spaces as discrete, wiring an int64 scalar replay
  buffer that crashed SAC.  These tests avoid the real gymnasium dependency
  by patching the env factory.
  """

  def _init_pipeline(self, monkeypatch, action_space):
    monkeypatch.setattr(rl, "GymnasiumEnvWrapper", _fake_env_factory((4,),
                                                                     action_space))
    args = argparse.Namespace(env_id="Fake-v0",
                              obs_dim=None,
                              record_eval_video=False,
                              replay_capacity=10,
                              env_seed=0)
    pipeline = RLPipeline()
    pipeline.init_env(args)
    return pipeline

  def test_discrete_space(self, monkeypatch):
    pipeline = self._init_pipeline(monkeypatch, _FakeSpace(n=2))
    assert pipeline.n_actions == 2
    assert pipeline.action_dim is None
    assert pipeline.replay_buffer.discrete is True
    assert pipeline.replay_buffer._action.shape == (10,)

  def test_continuous_single_dim_space(self, monkeypatch):
    pipeline = self._init_pipeline(monkeypatch, _FakeSpace(shape=(1,)))
    assert pipeline.n_actions is None
    assert pipeline.action_dim == 1
    assert pipeline.replay_buffer.discrete is False
    assert pipeline.replay_buffer._action.shape == (10, 1)

  def test_continuous_multi_dim_space(self, monkeypatch):
    pipeline = self._init_pipeline(monkeypatch, _FakeSpace(shape=(3,)))
    assert pipeline.action_dim == 3
    assert pipeline.replay_buffer.discrete is False
    assert pipeline.replay_buffer._action.shape == (10, 3)

  def test_action_dim_property_matches_private_attribute(self):
    pipeline = RLPipeline()
    pipeline._action_dim = 1
    assert pipeline.action_dim == 1
    pipeline._action_dim = None
    assert pipeline.action_dim is None
