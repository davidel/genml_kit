"""Tests for RL evaluation-video recording (RL_VIDEO feature).

Covers:
- ``video_utils.write_video`` (MP4 via imageio when available, GIF via PIL).
- Method ``evaluate(..., record_video=True)`` collecting frames per episode.
- ``RLTrainer.validate()`` writing videos under ``<checkpoint>/videos/``.

Hermetic: logic tests never require gymnasium / pygame.  Only the MP4
writer test uses ``importorskip`` for imageio; the GIF path (PIL) always
runs.
"""

import argparse
import os

import numpy as np
import pytest
import torch

from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.training.video_utils import write_video


def _make_frames(n=5, size=32):
  """Return ``n`` uint8 RGB frames."""
  return [np.zeros((size, size, 3), dtype=np.uint8) + i for i in range(n)]


class _RenderableScriptedEnv(_ScriptedEnv):
  """Scripted env that supports rgb_array rendering (no gymnasium needed)."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._render_calls = 0

  def can_render(self):
    return True

  def render(self):
    self._render_calls += 1
    return np.zeros((16, 16, 3), dtype=np.uint8) + self._render_calls

  def render_frame(self):
    return self.render()


class _RenderableRLPipeline(RLPipeline):
  """``RLPipeline`` whose env is a renderable scripted env."""

  def __init__(self, obs_dim=4, continuous=False):
    super().__init__()
    self.env = _RenderableScriptedEnv(obs_dim=obs_dim, continuous=continuous)
    self._obs_dim = obs_dim
    self._n_actions = 2
    self._action_dim = 2 if continuous else None
    self.action_space = self.env.action_space
    # SAC.path (continuous) needs these to exist on the pipeline.
    self.action_dim = self._action_dim


def _make_args(**overrides):
  defaults = dict(
      obs_dim=4,
      eval_episodes=1,
      record_eval_video=True,
      checkpoint=".",
      ppo_discrete=True,
      gamma=0.99,
      epsilon_start=1.0,
      epsilon_end=0.02,
      epsilon_decay_steps=50_000,
      tau=1.0,
      target_update_freq=0,
      sac_alpha=0.2,
      sac_target_entropy=None,
      sac_auto_alpha=True,
      # SAC policy defaults
      hidden_dim=64,
      lr_actor=1e-3,
      lr_critic=1e-3,
      lr_alpha=1e-3,
      # build_model extras (train_compat)
      freeze=None,
      lora=False,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


class TestWriteVideo:
  """Video writer: GIF (PIL) always; MP4 when imageio is available."""

  def test_write_gif(self, tmp_path):
    path = str(tmp_path / "out.gif")
    result = write_video(_make_frames(), path, fps=5)
    assert result == path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

  def test_write_empty_returns_none(self, tmp_path):
    path = str(tmp_path / "out.mp4")
    assert write_video([], path) is None
    assert write_video([None, None], path) is None
    assert not os.path.exists(path)

  def test_write_skips_none_frames(self, tmp_path):
    path = str(tmp_path / "out.gif")
    frames = [None] + _make_frames(1) + [None]
    result = write_video(frames, path, fps=5)
    assert result == path
    assert os.path.exists(path)

  def test_write_creates_parent_dirs(self, tmp_path):
    path = str(tmp_path / "nested" / "deep" / "out.gif")
    result = write_video(_make_frames(), path, fps=5)
    assert result == path
    assert os.path.exists(path)

  def test_write_mp4_when_ffmpeg_available(self, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    path = str(tmp_path / "out.mp4")
    result = write_video(_make_frames(), path, fps=5)
    assert result == path
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    # Verify it is a real MP4 container (ftyp box).
    with open(path, "rb") as fh:
      assert fh.read(12)[4:8] == b"ftyp"

  def test_write_mp4_falls_back_to_gif(self, tmp_path):
    """When imageio cannot encode MP4, GIF fallback is used, no crash."""
    path = str(tmp_path / "out.mp4")
    result = write_video(_make_frames(2), path, fps=5)
    # Either imageio produced an MP4, or a sibling .gif was written.
    assert result is not None
    assert os.path.exists(result)
    assert result.endswith((".mp4", ".gif"))


class TestEvaluateRecordVideo:
  """Method ``evaluate(..., record_video=True)`` collects frames."""

  def _evaluate(self, method, model, pipeline, num_episodes=2):
    return method.evaluate(model,
                           pipeline,
                           num_episodes=num_episodes,
                           record_video=True)

  def _assert_frames_collected(self, metrics):
    assert "episode_frames" in metrics
    assert len(metrics["episode_frames"]) == 2
    for frames in metrics["episode_frames"]:
      # Reset frame + at least one step frame.
      assert len(frames) >= 2
      for frame in frames:
        assert frame.shape == (16, 16, 3)
        assert frame.dtype == np.uint8

  def test_dqn(self):
    from genml_kit.methods.rl_dqn import DQNMethod
    args = _make_args()
    method = DQNMethod()
    pipeline = _RenderableRLPipeline()
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    self._assert_frames_collected(self._evaluate(method, model, pipeline))

  def test_ppo(self):
    from genml_kit.methods.rl_ppo import PPOMethod
    args = _make_args()
    method = PPOMethod()
    pipeline = _RenderableRLPipeline()
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    self._assert_frames_collected(self._evaluate(method, model, pipeline))

  def test_sac(self):
    from genml_kit.methods.rl_sac import SACMethod
    args = _make_args(sac_gamma=0.99, sac_tau=0.005)
    method = SACMethod()
    pipeline = _RenderableRLPipeline(continuous=True)
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    self._assert_frames_collected(self._evaluate(method, model, pipeline))

  def test_record_video_false_omits_key(self):
    from genml_kit.methods.rl_dqn import DQNMethod
    args = _make_args()
    method = DQNMethod()
    pipeline = _RenderableRLPipeline()
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    metrics = method.evaluate(model, pipeline, num_episodes=1)
    assert "episode_frames" not in metrics

  def test_no_render_support_yields_empty_frames(self):
    """Record requested but env cannot render => empty inner lists, no crash."""
    from genml_kit.methods.rl_dqn import DQNMethod
    args = _make_args()
    method = DQNMethod()
    pipeline = RLPipeline()
    pipeline.env = _ScriptedEnv(obs_dim=4)
    pipeline._obs_dim = 4
    pipeline._n_actions = 2
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    metrics = method.evaluate(model, pipeline, num_episodes=2, record_video=True)
    assert metrics["episode_frames"] == [[], []]


class TestValidatorWritesVideos:
  """RLTrainer.validate() writes videos to <checkpoint>/videos/."""

  def test_validate_writes_video_files(self, tmp_path):
    from genml_kit.training.rl_trainer import RLTrainer
    import numpy as _np

    frames = _make_frames(4, size=16)

    class _FakeMethod:

      def evaluate(self, model, pipeline, num_episodes, record_video=False):
        metrics = {"eval_return": 2.0, "eval_steps": 8}
        if record_video:
          metrics["episode_frames"] = [frames, frames]
        return metrics

    class _FakePipeline:

      def reset_env(self):
        return _np.zeros(4)

      def step_env(self, action):
        return _np.zeros(4), 1.0, False, {}

      def can_record_video(self):
        return True

      def render_frame(self):
        return _np.zeros((16, 16, 3), dtype=_np.uint8)

    trainer = object.__new__(RLTrainer)
    trainer.args = _make_args(checkpoint=str(tmp_path))
    trainer.pipeline = _FakePipeline()
    trainer.method = _FakeMethod()
    trainer.model = object()

    result = trainer.validate()
    assert result == 2.0

    out_dir = os.path.join(str(tmp_path), "videos")
    assert os.path.isdir(out_dir)
    written = sorted(os.listdir(out_dir))
    # imageio may be absent -> GIF fallback; either way a file must exist.
    assert written
    assert any(f.startswith("eval_episode_") for f in written)

  def test_validate_no_frames_is_noop(self, tmp_path):
    from genml_kit.training.rl_trainer import RLTrainer

    class _FakeMethod:

      def evaluate(self, model, pipeline, num_episodes, record_video=False):
        return {"eval_return": 1.0, "eval_steps": 0, "episode_frames": None}

    class _FakePipeline:
      pass

    trainer = object.__new__(RLTrainer)
    trainer.args = _make_args(checkpoint=str(tmp_path))
    trainer.pipeline = _FakePipeline()
    trainer.method = _FakeMethod()
    trainer.model = object()

    result = trainer.validate()
    assert result == 1.0
    assert not os.path.exists(os.path.join(str(tmp_path), "videos"))
