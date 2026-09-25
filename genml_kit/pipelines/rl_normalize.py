"""SB3-style discounted-return reward normalization for RL methods.

Ports the ``norm_reward=True`` behaviour of Stable-Baselines3's
``VecNormalize``: a running mean/std of *discounted returns* is maintained
(Welford's online algorithm) and each reward is divided by the running std
(clipped) before it enters the buffer / GAE computation.

Why normalize returns (not raw rewards): on environments with large
reward scales (Pendulum-v1 returns are O(1e3), MuJoCo similar) the value
loss ``MSE(V, returns)`` is O(1e3-1e4) and dominates the total PPO loss,
swamping the entropy bonus and policy-gradient signal.  Dividing rewards
by the running std of discounted returns brings the value targets to O(1).

Key properties (mirroring SB3 ``VecNormalize``):
- ``training=True`` (default): returns are accumulated with discount
  ``gamma`` and rewards are normalized by the *current* running std.
- ``training=False`` (eval): raw rewards are returned, and the running
  statistics are NOT updated.  Eval returns therefore stay comparable
  across runs and independent of the normalizer state.
- Checkpointable: ``state_dict()`` / ``load_state_dict(state)``.

This is a PPO-only concern for the initial implementation (SAC with
automatic temperature already adapts to the reward scale -- see
``plans/SB3_TRICKS_IMPORT.md``); the module is kept generic so other
methods can adopt it if needed.
"""

import numpy as np
import torch
from torch import nn


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


class ReturnNormalizer(torch.nn.Module):
  """Running discounted-return normalization (SB3 ``VecNormalize``).

  Maintains running statistics of discounted returns and normalizes each
  reward by the running std, clipped to ``clip_reward``.  Disabled at
  eval time (``training=False``) so evaluation sees raw rewards.

  Args:
      gamma: Discount factor used to accumulate the running return.
      clip_reward: Maximum absolute normalized reward.  ``10.0`` matches
          SB3's ``VecNormalize`` default.
      epsilon: Small constant added to the variance for numerical
          stability.
  """

  def __init__(self, gamma=0.99, clip_reward=10.0, epsilon=1e-4):
    super().__init__()
    self.gamma = gamma
    self.clip_reward = clip_reward
    self.epsilon = epsilon
    # Running stats of discounted returns (scalar).  Welford's algorithm.
    self.ret_rms = RunningMeanStd(shape=(), epsilon=epsilon)
    # A single scalar running-return accumulator, reset per episode.  Not
    # a buffer/parameter -- it is transient state managed by the caller
    # via ``reset_per_episode()`` / ``update``.
    self.register_buffer("running_return", torch.zeros(()))

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def reset_per_episode(self):
    """Reset the per-episode running-return accumulator to zero."""

    self.running_return.zero_()

  def normalize_reward(self, reward):
    """Normalize ``reward`` by the current running-return std.

    When ``self.training`` is False (eval), the raw reward is returned
    unchanged and the running statistics are not touched.
    """
    if not self.training:
      return reward
    reward = torch.as_tensor(reward, dtype=torch.float32)
    # ``ret_rms.var`` is kept in float64 (Welford); promote ``reward``
    # so the division is exact, then cast back to float32 for the buffer.
    std = torch.sqrt(self.ret_rms.var + self.epsilon).to(torch.float32)
    normalized = reward / std
    return torch.clamp(normalized, -self.clip_reward, self.clip_reward)

  def update(self, reward):
    """Accumulate ``reward`` into the discounted running return and update
    the return statistics.

    Call once per environment step (during training only), *before*
    ``normalize_reward`` so the statistics reflect the current return.

    ``reward`` may be a Python float, numpy scalar, or 0-dim tensor;
    the running-return accumulator and ``RunningMeanStd`` operate on
    (batch, *shape) tensors, so a batch of size 1 is used internally.
    """
    if not self.training:
      return
    reward = torch.as_tensor(reward, dtype=torch.float32)
    self.running_return = self.gamma * self.running_return + reward
    self.ret_rms.update(self.running_return.unsqueeze(0))

  # ------------------------------------------------------------------
  # Checkpointing
  # ------------------------------------------------------------------

  def state_dict(self):  # noqa: D102 - overrides nn.Module
    return {
        "gamma": self.gamma,
        "clip_reward": self.clip_reward,
        "epsilon": self.epsilon,
        "ret_rms": self.ret_rms.state_dict(),
        "running_return": self.running_return.item(),
    }

  def load_state_dict(self, state):
    """Restore from a checkpoint dict written by ``state_dict``."""

    self.gamma = state.get("gamma", self.gamma)
    self.clip_reward = state.get("clip_reward", self.clip_reward)
    self.epsilon = state.get("epsilon", self.epsilon)
    self.ret_rms.load_state_dict(state["ret_rms"])
    self.running_return.fill_(state.get("running_return", 0.0))
    return self

  def __repr__(self):  # noqa: D105
    return (f"ReturnNormalizer(gamma={self.gamma}, "
            f"clip_reward={self.clip_reward}, "
            f"count={int(self.ret_rms.count.item())})")
