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
    self._obs_dim = obs_dim
    self._rollout_len = rollout_len
    self._action_dim = action_dim
    self._device = device

    self._obs = torch.zeros(rollout_len, obs_dim)
    self._actions = (torch.zeros(rollout_len, action_dim) if action_dim is not None else
                     torch.zeros(rollout_len, dtype=torch.long))
    self._raw_actions = (
        torch.zeros(rollout_len, action_dim) if action_dim is not None else None
    )  # Store pre-tanh actions for continuous
    self._log_probs = torch.zeros(rollout_len)
    self._rewards = torch.zeros(rollout_len)
    self._values = torch.zeros(rollout_len)
    self._dones = torch.zeros(rollout_len)
    self._terminated = torch.zeros(rollout_len)
    self._advantages = torch.zeros(rollout_len)
    self._returns = torch.zeros(rollout_len)
    self._next_values = torch.zeros(rollout_len)

    self._ptr = 0
    self._filled = False

    # D5: Independent RNG for reproducible sampling
    self._rng = torch.Generator()
    if seed is not None:
      self._rng.manual_seed(seed)

  def add(self,
          obs,
          action,
          log_prob,
          reward,
          value,
          done,
          raw_action=None,
          terminated=None):
    """Store a single timestep at the current pointer.

        All arguments are plain Python scalars or numpy values.

        Args:
            raw_action: Optional raw (pre-tanh) action for continuous spaces.
                        Ignored for discrete.
            terminated: Optional true MDP-end flag (``False`` for a truncated
                        step).  When omitted, ``terminated = done``.
        """
    if self._ptr >= self._rollout_len:
      raise RuntimeError(f"RolloutBuffer overflow: ptr={self._ptr}, "
                         f"capacity={self._rollout_len}")
    if terminated is None:
      terminated = float(done)
    self._obs[self._ptr] = torch.as_tensor(obs, dtype=torch.float32)
    self._actions[self._ptr] = torch.as_tensor(action)
    if self._raw_actions is not None and raw_action is not None:
      self._raw_actions[self._ptr] = torch.as_tensor(raw_action)
    self._log_probs[self._ptr] = float(log_prob)
    self._rewards[self._ptr] = float(reward)
    self._values[self._ptr] = float(value)
    self._dones[self._ptr] = float(done)
    self._terminated[self._ptr] = float(terminated)
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
      raise ValueError("set_next_values no longer accepts scalars. "
                       "Pass a tensor/array of shape (rollout_len,) with "
                       "per-step bootstrap values.")
    next_values = torch.as_tensor(next_values, dtype=torch.float32).flatten()
    assert next_values.numel() == self._rollout_len, (
        f"next_values must have {self._rollout_len} elements, "
        f"got {next_values.numel()}")
    self._next_values.copy_(next_values)

  def compute(self, gamma, lam):
    """Compute GAE advantages and discounted returns.

        Must be called after the rollout is full and ``set_next_values``
        has been called.
        """
    from genml_kit.losses.rl import gae

    self._advantages, self._returns = gae(
        self._rewards,
        self._values,
        self._next_values,
        self._dones,
        gamma=gamma,
        lam=lam,
        terminated=self._terminated,
    )
    self._filled = True

  # ------------------------------------------------------------------
  # Dataset protocol
  # ------------------------------------------------------------------

  def __len__(self):
    return self._rollout_len if self._filled else self._ptr

  def __getitem__(self, idx):
    """Return a single timestep as a dict (DataLoader-compatible)."""
    item = {
        "obs": self._obs[idx],
        "action": self._actions[idx],
        "log_prob": self._log_probs[idx],
        "advantage": self._advantages[idx],
        "return": self._returns[idx],
        "value": self._values[idx],
    }
    if self._raw_actions is not None:
      item["raw_action"] = self._raw_actions[idx]
    return item

  def to(self, device):
    """Move all tensors to *device* (in-place)."""
    self._obs = self._obs.to(device)
    self._actions = self._actions.to(device)
    if self._raw_actions is not None:
      self._raw_actions = self._raw_actions.to(device)
    self._log_probs = self._log_probs.to(device)
    self._rewards = self._rewards.to(device)
    self._values = self._values.to(device)
    self._dones = self._dones.to(device)
    self._terminated = self._terminated.to(device)
    self._advantages = self._advantages.to(device)
    self._returns = self._returns.to(device)
    self._next_values = self._next_values.to(device)
    self._device = device
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
        "obs": self._obs[idx],
        "action": self._actions[idx],
        "log_prob": self._log_probs[idx],
        "advantage": self._advantages[idx],
        "return": self._returns[idx],
        "value": self._values[idx],
    }

  # Alias for backward compatibility
  get_batch = sample

  @property
  def ptr(self):
    return self._ptr

  @property
  def full(self):
    return self._ptr >= self._rollout_len

  @property
  def rollout_len(self):
    """Public read-only access to rollout length for external consumers."""
    return self._rollout_len

  # Properties for backward compatibility with tests
  @property
  def obs(self):
    return self._obs

  @property
  def actions(self):
    return self._actions

  @property
  def log_probs(self):
    return self._log_probs

  @property
  def rewards(self):
    return self._rewards

  @property
  def values(self):
    return self._values

  @property
  def dones(self):
    return self._dones

  @property
  def terminated(self):
    return self._terminated

  @property
  def advantages(self):
    return self._advantages

  @property
  def returns(self):
    return self._returns

  @property
  def next_values(self):
    return self._next_values

  @property
  def raw_actions(self):
    return self._raw_actions

  @property
  def obs_dim(self):
    return self._obs_dim

  @property
  def action_dim(self):
    return self._action_dim

  @property
  def device(self):
    return self._device

  def state_dict(self):
    """Return state dict for checkpointing."""
    return {
        "obs": self._obs,
        "actions": self._actions,
        "raw_actions": self._raw_actions,
        "log_probs": self._log_probs,
        "rewards": self._rewards,
        "values": self._values,
        "dones": self._dones,
        "terminated": self._terminated,
        "advantages": self._advantages,
        "returns": self._returns,
        "next_values": self._next_values,
        "ptr": self._ptr,
        "filled": self._filled,
        "obs_dim": self._obs_dim,
        "action_dim": self._action_dim,
        "rollout_len": self._rollout_len,
        "device": self._device,
        "rng_state": self._rng.get_state(),
    }

  def load_state_dict(self, state):
    """Load state from checkpoint."""
    self._obs = state["obs"]
    self._actions = state["actions"]
    self._raw_actions = state.get("raw_actions", self._raw_actions)
    self._log_probs = state["log_probs"]
    self._rewards = state["rewards"]
    self._values = state["values"]
    self._dones = state["dones"]
    self._terminated = state.get("terminated", self._terminated)
    self._advantages = state["advantages"]
    self._returns = state["returns"]
    self._next_values = state["next_values"]
    self._ptr = state["ptr"]
    self._filled = state["filled"]
    self._obs_dim = state["obs_dim"]
    self._action_dim = state["action_dim"]
    self._rollout_len = state["rollout_len"]
    self._device = state["device"]
    if "rng_state" in state:
      self._rng.set_state(state["rng_state"])
