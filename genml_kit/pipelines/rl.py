"""RL data pipeline: environment wrapper + replay buffer + eval rollout.

Phase 1 of ``plans/RL_PLAN.md`` (\u00a76.3).  The pipeline owns the
**data side** of RL training: the Gymnasium environment, the replay
buffer, and the evaluation-rollout helper.  ``build_loader`` returns
``None`` (the ``RLTrainer`` samples directly from the buffer).
"""

import logging

import numpy as np
import torch

from genml_kit.datasets.replay_buffer import ReplayBufferDataset
from genml_kit.datasets.rollout_buffer import RolloutBuffer
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import register_pipeline


class GymnasiumEnvWrapper:
  """Thin adapter around a Gymnasium ``Env`` with numpy-array observations.

  Provides the three methods that ``RLPipeline`` and ``RLTrainer`` need:
  ``reset``, ``step``, ``close``.

  Args:
      env_id:  Gymnasium environment id string (e.g. ``"CartPole-v1"``).
  """

  def __init__(self, env_id):
    try:
      import gymnasium as gym
    except ImportError as exc:
      raise ImportError("gymnasium is required for --pipeline rl.  "
                        "Install it with:  pip install 'genml_kit[rl]'") from exc
    self.env = gym.make(env_id)
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space

  def reset(self):
    """Reset the environment and return the initial observation (numpy)."""
    obs, _info = self.env.reset()
    return obs

  def step(self, action):
    """Execute *action* and return ``(obs, reward, done, info)``."""
    obs, reward, terminated, truncated, info = self.env.step(action)
    done = terminated or truncated
    return obs, reward, done, info

  def close(self):
    self.env.close()


class _ScriptedEnv:
  """Deterministic scripted env for unit testing (no gymnasium required).

  A simple 4-state chain: state 0 → 1 → 2 → 3 → done.
  Action 0 = advance; action 1 = stay.
  Episodes are also forcibly terminated after *max_episode_length* steps
  to prevent infinite loops with untrained policies.
  """

  def __init__(self, obs_dim=4, max_episode_length=10, continuous=False, action_dim=2):
    self.obs_dim = obs_dim
    self._state = 0
    self._step_count = 0
    self._max_episode_length = max_episode_length
    self._continuous = continuous
    self._action_dim = action_dim
    self.observation_space = type("S", (), {
        "shape": (obs_dim,),
    })()
    if continuous:
      import gymnasium as gym
      import numpy as np
      self.action_space = gym.spaces.Box(low=-1.0,
                                         high=1.0,
                                         shape=(action_dim,),
                                         dtype=np.float32)
    else:
      self.action_space = type("A", (), {
          "n": 2,
      })()

  def reset(self):
    self._state = 0
    self._step_count = 0
    return self._obs()

  def step(self, action):
    self._step_count += 1
    import numpy as np
    if self._continuous:
      # Continuous action: accept array, use first element for logic
      if hasattr(action, "__len__"):
        act_val = float(np.asarray(action).flat[0])
      elif hasattr(action, "item"):
        act_val = float(action.item())
      else:
        act_val = float(action)
    else:
      if hasattr(action, "__len__"):
        act_val = int(np.asarray(action).flat[0])
      elif hasattr(action, "item"):
        act_val = int(action.item())
      else:
        act_val = int(action)
      if act_val == 0 and self._state < 3:
        self._state += 1
    reward = 1.0 if self._state == 3 else 0.0
    done = self._state == 3 or self._step_count >= self._max_episode_length
    return self._obs(), reward, done, {}

  def close(self):
    pass

  def _obs(self):
    obs = torch.zeros(self.obs_dim)
    obs[self._state] = 1.0
    return obs.numpy()


@register_pipeline
class RLPipeline(DataPipeline):
  """RL data pipeline: environment + replay buffer + eval rollout."""

  NAME = "rl"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.env = None
    self.replay_buffer = None
    self._obs_dim = None
    self._n_actions = None

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("rl pipeline")
    group.add_argument(
        "--env-id",
        dest="env_id",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment id (default: CartPole-v1).",
    )
    group.add_argument(
        "--obs-dim",
        dest="obs_dim",
        type=int,
        default=None,
        help="Observation dimensionality (inferred from env if omitted).",
    )
    group.add_argument(
        "--env-script",
        dest="env_script",
        type=str,
        default=None,
        help="Python file/URL defining a ``make_env`` factory.",
    )
    group.add_argument(
        "--warmup-steps",
        dest="warmup_steps",
        type=int,
        default=1000,
        help="Random-action steps before learning starts.",
    )
    group.add_argument(
        "--replay-capacity",
        dest="replay_capacity",
        type=int,
        default=100_000,
        help="Maximum transitions in the replay buffer.",
    )
    group.add_argument(
        "--eval-episodes",
        dest="eval_episodes",
        type=int,
        default=5,
        help="Number of episodes for policy evaluation.",
    )
    group.add_argument(
        "--env-seed",
        dest="env_seed",
        type=int,
        default=None,
        help="Environment RNG seed.",
    )
    group.add_argument(
        "--steps-per-epoch",
        dest="steps_per_epoch",
        type=int,
        default=1000,
        help="Environment steps per training epoch (default: 1000).",
    )

  def build_loader(self, args, **kwargs):
    """Return ``None`` — the RLTrainer samples from the buffer directly."""
    return None

  def build_val_loader(self, args, **kwargs):
    """Return ``None`` — validation is done via the method's ``evaluate``."""
    return None

  def init_env(self, args):
    """Build (and cache) the environment and replay buffer.

    Called once at the start of training by ``RLTrainer``.
    """
    from genml_kit.utils.logging import fatal
    from genml_kit.utils.script import extern_call

    if self.env is not None:
      return

    if getattr(args, "env_script", None):
      self.env = extern_call(args.env_script, "make_env")
    else:
      self.env = GymnasiumEnvWrapper(args.env_id)

    obs_dim = getattr(args, "obs_dim", None)
    if obs_dim is None:
      # Try to infer from observation space.
      obs_space = self.env.observation_space
      if hasattr(obs_space, "shape"):
        obs_dim = int(torch.tensor(obs_space.shape).prod())
      else:
        # Fatal: cannot infer obs_dim from this observation space.
        fatal(
            f"Cannot infer obs_dim from observation space {obs_space!r}; "
            "pass --obs_dim explicitly", ValueError)
    self._obs_dim = obs_dim

    # Expose action_space for continuous support (A9)
    self.action_space = self.env.action_space

    # Handle discrete vs continuous action space
    if hasattr(self.env.action_space, 'n'):
      self._n_actions = int(self.env.action_space.n)
      self._action_dim = None
    elif hasattr(self.env.action_space, 'shape'):
      self._n_actions = None
      self._action_dim = int(self.env.action_space.shape[0])
    else:
      # Fallback
      self._n_actions = 2
      self._action_dim = 2

    # Determine action dim and dtype for replay buffer
    if self._action_dim is not None and self._action_dim > 1:
      action_dim = self._action_dim
      action_dtype = np.float32
    else:
      action_dim = 1
      action_dtype = np.int64

    # D5: Use env_seed for reproducible buffer sampling
    buffer_seed = getattr(args, "env_seed", None)

    self.replay_buffer = ReplayBufferDataset(
        obs_dim=obs_dim,
        capacity=getattr(args, "replay_capacity", 100_000),
        action_dim=action_dim,
        action_dtype=action_dtype,
        seed=buffer_seed,
    )

    # On-policy rollout buffer (used by PPO; ignored by DQN/SAC).
    self.rollout_buffer = RolloutBuffer(
        obs_dim=obs_dim,
        rollout_len=getattr(args, "rollout_len", 2048),
        action_dim=self._action_dim,
        device="cpu",
        seed=buffer_seed,
    )

    logging.info(
        "RLPipeline: obs_dim=%d, n_actions=%d, buffer_capacity=%d",
        obs_dim,
        self._n_actions,
        self.replay_buffer.capacity,
    )

  def reset_env(self):
    """Reset the environment and return the initial observation tensor."""
    return self.env.reset()

  def step_env(self, action):
    """Execute *action* in the environment.

    Returns ``(next_obs, reward, done, info)`` as plain Python / numpy.
    """
    return self.env.step(action)

  def env_push(self, transition):
    """Push a single transition into the replay buffer."""
    self.replay_buffer.push(
        transition.obs,
        transition.action,
        transition.reward,
        transition.next_obs,
        transition.done,
    )

  @property
  def obs_dim(self):
    return self._obs_dim

  @property
  def n_actions(self):
    return self._n_actions

  @property
  def buffer(self):
    return self.replay_buffer

  def to_device(self, blob, device):
    """Move a ``TransitionBatch`` dict (or ``DataBlob``) to *device*.

    For RL the ``blob.data`` is typically a dict with keys
    ``obs``, ``action``, ``reward``, ``next_obs``, ``done``.
    """
    if isinstance(blob, DataBlob):
      data = blob.data
      meta = blob.meta
    else:
      data = blob
      meta = {}

    if isinstance(data, dict):
      data = {
          k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
          for k, v in data.items()
      }
    elif hasattr(data, "to"):
      data = data.to(device, non_blocking=True)

    if isinstance(meta, dict):
      meta = {
          k: v.to(device, non_blocking=True) if hasattr(v, "to") else v
          for k, v in meta.items()
      }

    if isinstance(blob, DataBlob):
      return DataBlob(data=data, meta=meta)
    return data

  @classmethod
  def _make_scripted_env(cls, obs_dim=4, continuous=False, action_dim=2):
    """Return a ``_ScriptedEnv`` for unit testing."""
    return _ScriptedEnv(obs_dim=obs_dim, continuous=continuous, action_dim=action_dim)
