"""Fixed-capacity replay buffer for off-policy RL.

Phase 1 of ``plans/RL_PLAN.md`` (§6.2).  A PyTorch ``Dataset`` backed
by fixed-length numpy arrays with a circular write pointer.
"""

import collections

import numpy as np
import torch
from torch.utils.data import Dataset

Transition = collections.namedtuple("Transition",
                                    ["obs", "action", "reward", "next_obs", "done"])


class ReplayBufferDataset(Dataset):
  """Fixed-capacity circular replay buffer compatible with ``DataLoader``.

  Internally stores numpy arrays for memory efficiency.  The
  ``__getitem__`` protocol returns a plain ``dict`` (keys
  ``obs``, ``action``, ``reward``, ``next_obs``, ``done``) so that
  ``default_collate`` stacks them into the ``TransitionBatch`` namedtuple
  consumed by ``DQNMethod.train_step``.

  Args:
      obs_dim:  Dimensionality of the observation vector.
      capacity: Maximum number of transitions stored.
  """

  def __init__(self, obs_dim, capacity=100_000):
    self.capacity = capacity
    self.obs_dim = obs_dim
    self._pos = 0
    self._size = 0

    self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    self.action = np.zeros(capacity, dtype=np.int64)
    self.reward = np.zeros(capacity, dtype=np.float32)
    self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    self.done = np.zeros(capacity, dtype=np.float32)

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def push(self, obs, action, reward, next_obs, done):
    """Add a single transition to the buffer.

    All arguments are plain Python / numpy scalars or arrays.
    """
    self.obs[self._pos] = obs
    self.action[self._pos] = action
    self.reward[self._pos] = reward
    self.next_obs[self._pos] = next_obs
    self.done[self._pos] = float(done)
    self._pos = (self._pos + 1) % self.capacity
    self._size = min(self._size + 1, self.capacity)

  def push_batch(self, transitions):
    """Add a batch of transitions.

    Args:
        transitions: An iterable of ``Transition`` namedtuples (or dicts
            with the right keys).
    """
    for t in transitions:
      if isinstance(t, Transition):
        self.push(t.obs, t.action, t.reward, t.next_obs, t.done)
      else:
        self.push(t["obs"], t["action"], t["reward"], t["next_obs"], t["done"])

  def sample(self, batch_size, generator=None):
    """Sample a batch of transitions.

    Args:
        batch_size:  Number of transitions to sample.
        generator:   Optional ``torch.Generator`` for reproducibility.

    Returns:
        dict with keys ``obs``, ``action``, ``reward``, ``next_obs``,
        ``done`` — each a ``torch.Tensor``.
    """
    if generator is not None:
      indices = torch.randint(0, self._size, (batch_size,), generator=generator).numpy()
    else:
      indices = np.random.randint(0, self._size, size=batch_size)

    return {
        "obs": torch.from_numpy(self.obs[indices]),
        "action": torch.from_numpy(self.action[indices]),
        "reward": torch.from_numpy(self.reward[indices]),
        "next_obs": torch.from_numpy(self.next_obs[indices]),
        "done": torch.from_numpy(self.done[indices]),
    }

  def stats(self):
    """Return diagnostic statistics about the buffer contents."""
    if self._size == 0:
      return {"size": 0, "fill_ratio": 0.0}
    return {
        "size": self._size,
        "capacity": self.capacity,
        "fill_ratio": self._size / self.capacity,
        "mean_reward": float(np.mean(self.reward[:self._size])),
    }

  # ------------------------------------------------------------------
  # Dataset protocol
  # ------------------------------------------------------------------

  def __len__(self):
    return self._size

  def __getitem__(self, idx):
    """Return a single transition as a dict (DataLoader-compatible).

    Note: The returned arrays are numpy views; DataLoader collation
    converts them to tensors via ``default_collate``.
    """
    return {
        "obs": self.obs[idx],
        "action": self.action[idx],
        "reward": self.reward[idx],
        "next_obs": self.next_obs[idx],
        "done": self.done[idx],
    }
