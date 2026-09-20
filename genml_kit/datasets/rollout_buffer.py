"""Rollout buffer for on-policy RL (PPO).

Phase 2 of ``plans/RL_PLAN.md``.  Stores a fixed-length rollout of
transitions with pre-computed advantages and returns.  Converts to a
``torch.utils.data.Dataset`` that yields ``DataBlob``-compatible dicts
for the PPO training epochs.
"""

import torch
from torch.utils.data import Dataset


class RolloutBuffer(Dataset):
  """Fixed-length on-policy rollout storage.

  Data is filled sequentially via :meth:`add` and converted to tensors
  via :meth:`compute`.  After conversion the buffer acts as a
  ``Dataset`` over the stored transitions (for mini-batch SGD epochs).

  Args:
      obs_dim:       Observation dimensionality.
      rollout_len:   Number of time-steps per rollout.
      action_dim:    Action dimensionality (for continuous; ``None``
                     for discrete).
      device:        Target device for tensor conversion.
  """

  def __init__(self, obs_dim, rollout_len, action_dim=None, device="cpu", seed=None):
    self.obs_dim = obs_dim
    self.rollout_len = rollout_len
    self.action_dim = action_dim
    self.device = device

    self.obs = torch.zeros(rollout_len, obs_dim)
    self.actions = (torch.zeros(rollout_len, action_dim) if action_dim is not None else
                    torch.zeros(rollout_len, dtype=torch.long))
    self.raw_actions = (
        torch.zeros(rollout_len, action_dim) if action_dim is not None else None
    )  # Store pre-tanh actions for continuous
    self.log_probs = torch.zeros(rollout_len)
    self.rewards = torch.zeros(rollout_len)
    self.values = torch.zeros(rollout_len)
    self.dones = torch.zeros(rollout_len)
    self.advantages = torch.zeros(rollout_len)
    self.returns = torch.zeros(rollout_len)
    self.next_values = torch.zeros(rollout_len)

    self._ptr = 0
    self._filled = False
    
    # D5: Independent RNG for reproducible sampling
    self._rng = torch.Generator()
    if seed is not None:
      self._rng.manual_seed(seed)

  def add(self, obs, action, log_prob, reward, value, done, raw_action=None):
    """Store a single timestep at the current pointer.

    All arguments are plain Python scalars or numpy values.

    Args:
        raw_action: Optional raw (pre-tanh) action for continuous spaces.
                    Ignored for discrete.
    """
    if self._ptr >= self.rollout_len:
      raise RuntimeError(f"RolloutBuffer overflow: ptr={self._ptr}, "
                         f"capacity={self.rollout_len}")
    self.obs[self._ptr] = torch.as_tensor(obs, dtype=torch.float32)
    self.actions[self._ptr] = torch.as_tensor(action)
    if self.raw_actions is not None and raw_action is not None:
      self.raw_actions[self._ptr] = torch.as_tensor(raw_action)
    self.log_probs[self._ptr] = float(log_prob)
    self.rewards[self._ptr] = float(reward)
    self.values[self._ptr] = float(value)
    self.dones[self._ptr] = float(done)
    self._ptr += 1

  def set_next_values(self, next_values):
    """Set bootstrap value estimates for the rollout's final states.

    ``next_values`` should be a tensor of shape (rollout_len,) containing
    V(s_{t+1}) for each step t in the rollout.

    Args:
        next_values: Tensor of shape (rollout_len,) with per-step bootstrap values.
                     Scalar broadcasting is no longer supported.
    """
    if isinstance(next_values, (int, float)):
      raise ValueError(
          "set_next_values no longer accepts scalars. "
          "Pass a tensor/array of shape (rollout_len,) with per-step bootstrap values.")
    next_values = torch.as_tensor(next_values, dtype=torch.float32).flatten()
    assert next_values.numel() == self.rollout_len, (
        f"next_values must have {self.rollout_len} elements, got {next_values.numel()}")
    self.next_values.copy_(next_values)

  def compute(self, gamma, lam):
    """Compute GAE advantages and discounted returns.

    Must be called after the rollout is full and ``set_next_values``
    has been called.
    """
    from genml_kit.losses.rl import gae
    self.advantages, self.returns = gae(
        self.rewards,
        self.values,
        self.next_values,
        self.dones,
        gamma=gamma,
        lam=lam,
    )
    self._filled = True

  # ------------------------------------------------------------------
  # Dataset protocol
  # ------------------------------------------------------------------

  def __len__(self):
    return self.rollout_len if self._filled else self._ptr

  def __getitem__(self, idx):
    """Return a single timestep as a dict (DataLoader-compatible)."""
    item = {
        "obs": self.obs[idx],
        "action": self.actions[idx],
        "log_prob": self.log_probs[idx],
        "advantage": self.advantages[idx],
        "return": self.returns[idx],
        "value": self.values[idx],
    }
    if self.raw_actions is not None:
      item["raw_action"] = self.raw_actions[idx]
    return item

  def to(self, device):
    """Move all tensors to *device* (in-place)."""
    self.obs = self.obs.to(device)
    self.actions = self.actions.to(device)
    if self.raw_actions is not None:
      self.raw_actions = self.raw_actions.to(device)
    self.log_probs = self.log_probs.to(device)
    self.rewards = self.rewards.to(device)
    self.values = self.values.to(device)
    self.dones = self.dones.to(device)
    self.advantages = self.advantages.to(device)
    self.returns = self.returns.to(device)
    self.next_values = self.next_values.to(device)
    self.device = device
    return self

  def reset(self):
    """Reset the pointer (for the next rollout collection)."""
    self._ptr = 0
    self._filled = False

  def sample(self, batch_size, generator=None):
    """Sample a random mini-batch from the computed rollout.

    Returns a dict of tensors suitable for PPO ``train_step``.
    """
    n = len(self)
    if n == 0:
      raise ValueError("RolloutBuffer is empty")
    if generator is not None:
      idx = torch.randint(n, (batch_size,), generator=generator)
    else:
      # D5: Use independent RNG for reproducible sampling
      idx = torch.randint(n, (batch_size,), generator=self._rng)
    return {
        "obs": self.obs[idx],
        "action": self.actions[idx],
        "log_prob": self.log_probs[idx],
        "advantage": self.advantages[idx],
        "return": self.returns[idx],
        "value": self.values[idx],
    }

  # Alias for backward compatibility
  get_batch = sample

  @property
  def ptr(self):
    return self._ptr

  @property
  def full(self):
    return self._ptr >= self.rollout_len
