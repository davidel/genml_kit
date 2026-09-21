"""RL data pipeline: environment wrapper + replay buffer + eval rollout.

Phase 1 of ``plans/RL_PLAN.md`` (§6.3).  The pipeline owns the
**data side** of RL training: the Gymnasium environment, the replay
buffer, and the evaluation-rollout helper.  ``build_loader`` returns
``None`` (the ``RLTrainer`` samples directly from the buffer).
"""

import logging

import numpy as np
import torch
from torch import nn

from genml_kit.datasets.replay_buffer import ReplayBufferDataset
from genml_kit.datasets.rollout_buffer import RolloutBuffer
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import register_pipeline


class RunningMeanStd(nn.Module):
  """Running mean and standard deviation for observation normalization.

  Uses Welford's online algorithm for numerical stability.
  Based on OpenAI baselines implementation.

  Args:
      shape: Shape of the data (excluding batch dimension).
      epsilon: Small constant for numerical stability.
  """

  def __init__(self, shape, epsilon=1e-4):
    super().__init__()
    self.epsilon = epsilon
    self.register_buffer("mean", torch.zeros(shape, dtype=torch.float64))
    self.register_buffer("var", torch.ones(shape, dtype=torch.float64))
    self.register_buffer("count", torch.tensor(epsilon, dtype=torch.float64))

  def update(self, x):
    """Update running statistics with a batch of data.

    Args:
        x: Batch of observations with shape (batch_size, *shape).
    """
    x = torch.as_tensor(x, dtype=torch.float64)
    batch_mean = torch.mean(x, dim=0)
    batch_var = torch.var(x, dim=0, unbiased=False)
    batch_count = x.shape[0]
    self._update_from_moments(batch_mean, batch_var, batch_count)

  def _update_from_moments(self, batch_mean, batch_var, batch_count):
    """Welford's online update from batch moments."""
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
    new_var = m_2 / tot_count

    self.mean.copy_(new_mean)
    self.var.copy_(new_var)
    self.count.copy_(tot_count)

  def forward(self, x, clip=None):
    """Normalize observations using running statistics.

    Args:
        x: Observations with shape (..., *shape).
        clip: Optional value to clip normalized observations to [-clip, clip].

    Returns:
        Normalized observations (same type as input: torch.Tensor or numpy array).
    """
    is_numpy = isinstance(x, np.ndarray)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    mean = self.mean.to(torch.float32)
    var = self.var.to(torch.float32)
    normalized = (x_t - mean) / torch.sqrt(var + self.epsilon)
    if clip is not None:
      normalized = torch.clamp(normalized, -clip, clip)
    if is_numpy:
      return normalized.numpy()
    return normalized

  def normalize(self, x, clip=None):
    """Alias for forward() for backward compatibility."""
    return self.forward(x, clip=clip)

  def state_dict(self):
    """Return state dict for checkpointing."""
    return {
        "mean": self.mean.clone(),
        "var": self.var.clone(),
        "count": self.count.clone(),
        "epsilon": self.epsilon,
        "shape": self.mean.shape,
    }

  def load_state_dict(self, state):
    """Load state dict from checkpoint."""
    self.mean.copy_(state["mean"].to(torch.float64))
    self.var.copy_(state["var"].to(torch.float64))
    self.count.copy_(state["count"].to(torch.float64))
    self.epsilon = state.get("epsilon", 1e-4)
    # shape is inferred from mean


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
    # Observation normalization (E3) - defaults for test pipelines that bypass init_env
    self._obs_normalize = False
    self._obs_norm_clip = 10.0
    self.obs_rms = None

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
    group.add_argument(
        "--n-step",
        dest="n_step",
        type=int,
        default=1,
        help="Number of steps for n-step returns (default: 1).",
    )
    group.add_argument(
        "--prioritized",
        dest="prioritized",
        action="store_true",
        help="Enable prioritized experience replay (PER).",
    )
    group.add_argument(
        "--per-alpha",
        dest="per_alpha",
        type=float,
        default=0.6,
        help="Priority exponent for PER (default: 0.6).",
    )
    group.add_argument(
        "--per-beta-start",
        dest="per_beta_start",
        type=float,
        default=0.4,
        help="Initial beta for IS weight annealing (default: 0.4).",
    )
    group.add_argument(
        "--per-beta-frames",
        dest="per_beta_frames",
        type=int,
        default=100000,
        help="Frames over which to anneal beta to 1.0 (default: 100000).",
    )
    group.add_argument(
        "--obs-normalize",
        dest="obs_normalize",
        action="store_true",
        help="Enable observation normalization with RunningMeanStd.",
    )
    group.add_argument(
        "--obs-norm-clip",
        dest="obs_norm_clip",
        type=float,
        default=10.0,
        help="Clip normalized observations to [-clip, clip] (default: 10.0).",
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
        n_step=getattr(args, "n_step", 1),
        gamma=getattr(args, "gamma", 0.99),
        prioritized=getattr(args, "prioritized", False),
        alpha=getattr(args, "per_alpha", 0.6),
        beta_start=getattr(args, "per_beta_start", 0.4),
        beta_frames=getattr(args, "per_beta_frames", 100_000),
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

    # Observation normalization (E3)
    self._obs_normalize = getattr(args, "obs_normalize", False)
    self._obs_norm_clip = getattr(args, "obs_norm_clip", 10.0)
    if self._obs_normalize:
      self.obs_rms = RunningMeanStd(shape=(obs_dim,))
    else:
      self.obs_rms = None

    logging.info(
        "RLPipeline: obs_dim=%d, n_actions=%d, buffer_capacity=%d, obs_normalize=%s",
        obs_dim,
        self._n_actions,
        self.replay_buffer.capacity,
        self._obs_normalize,
    )

  def reset_env(self):
    """Reset the environment and return the initial observation tensor."""
    # Handle both old gym API (obs) and new gymnasium API (obs, info)
    reset_result = self.env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    if self._obs_normalize and self.obs_rms is not None:
      self.obs_rms.update(obs[None, ...])  # Add batch dim for update
      obs = self.obs_rms.normalize(obs, clip=self._obs_norm_clip)
    return obs

  def step_env(self, action):
    """Execute *action* in the environment.

    Returns ``(next_obs, reward, done, info)`` as plain Python / numpy.
    """
    # Handle both old gym API (obs, reward, done, info)
    # and new gymnasium API (obs, reward, terminated, truncated, info)
    step_result = self.env.step(action)
    if len(step_result) == 5:
      next_obs, reward, terminated, truncated, info = step_result
      done = terminated or truncated
    else:
      next_obs, reward, done, info = step_result
    if self._obs_normalize and self.obs_rms is not None:
      # Update RMS with the new observation
      self.obs_rms.update(next_obs[None, ...])
      next_obs = self.obs_rms.normalize(next_obs, clip=self._obs_norm_clip)
    return next_obs, reward, float(done), info

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

  def get_checkpoint_state(self):
    """Return observation normalization state for checkpointing."""
    if self._obs_normalize and self.obs_rms is not None:
      return {"obs_rms": self.obs_rms.state_dict()}
    return {}

  def load_checkpoint_state(self, state):
    """Load observation normalization state from checkpoint."""
    if self._obs_normalize and self.obs_rms is not None and "obs_rms" in state:
      self.obs_rms.load_state_dict(state["obs_rms"])

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
