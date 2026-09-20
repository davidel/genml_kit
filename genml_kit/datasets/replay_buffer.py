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
    consumed by ``DQNMethod.train_step`` / ``SACMethod.train_step``.

    Supports n-step returns: when n_step > 1, the buffer stores
    n-step accumulated reward, the observation after n steps, and
    whether the episode terminated within those n steps.

    Args:
        obs_dim:  Dimensionality of the observation vector.
        capacity: Maximum number of transitions stored.
        action_dim: Dimensionality of the action vector (default: 1 for discrete).
        action_dtype: dtype for actions (default: int64 for discrete,
            float32 for continuous).
        n_step: Number of steps for n-step returns (default: 1).
        gamma: Discount factor for n-step return computation
            (default: 0.99).
        seed: Random seed for reproducible sampling.
    """

  def __init__(
      self,
      obs_dim,
      capacity=100_000,
      action_dim=1,
      action_dtype=np.int64,
      n_step=1,
      gamma=0.99,
      seed=None,
  ):
    self.capacity = capacity
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.n_step = n_step
    self.gamma = gamma
    self._pos = 0
    self._size = 0

    self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    if action_dim == 1:
      self.action = np.zeros(capacity, dtype=action_dtype)
    else:
      self.action = np.zeros((capacity, action_dim), dtype=action_dtype)
    self.reward = np.zeros(capacity, dtype=np.float32)
    self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    self.done = np.zeros(capacity, dtype=np.float32)

    # N-step return support
    if n_step > 1:
      self._n_step_buffer = collections.deque(maxlen=n_step)

    # D5: Independent RNG for reproducible sampling
    self._rng = np.random.default_rng(seed)

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def push(self, obs, action, reward, next_obs, done):
    """Add a single transition to the buffer.

        All arguments are plain Python / numpy scalars or arrays.
        When n_step > 1, transitions are first buffered and n-step returns
        are computed when the buffer fills or the episode terminates.
        """
    # Handle n-step returns
    if self.n_step > 1:
      self._n_step_buffer.append((obs, action, reward, next_obs, done))

      # Compute n-step return if buffer is full or episode terminated
      if len(self._n_step_buffer) == self.n_step or done:
        self._push_n_step()
    else:
      self._push_single(obs, action, reward, next_obs, done)

  def _push_single(self, obs, action, reward, next_obs, done):
    """Internal: push a single transition to the main buffer."""
    self.obs[self._pos] = obs
    # Handle both scalar and array actions
    if self.action_dim > 1:
      # Continuous action: ensure it's a 1D array
      action = np.asarray(action, dtype=self.action.dtype).flatten()
      # Ensure correct shape
      assert action.shape == (
          self.action_dim,), f"action shape {action.shape} != ({self.action_dim},)"
    else:
      # Discrete action: ensure scalar
      action = (np.asarray(action, dtype=self.action.dtype).item() if hasattr(
          np.asarray(action), "item") else action)
    self.action[self._pos] = action
    self.reward[self._pos] = reward
    self.next_obs[self._pos] = next_obs
    self.done[self._pos] = float(done)
    self._pos = (self._pos + 1) % self.capacity
    self._size = min(self._size + 1, self.capacity)

  def _push_n_step(self):
    """Compute n-step return from the n-step buffer and push to main buffer."""
    if not self._n_step_buffer:
      return

    # First transition in the n-step window
    obs, action, _, _, _ = self._n_step_buffer[0]

    # Compute n-step accumulated reward
    n_step_reward = 0.0
    gamma_pow = 1.0
    for _i, (_, _, reward, _, done) in enumerate(self._n_step_buffer):
      n_step_reward += gamma_pow * reward
      gamma_pow *= self.gamma
      if done:
        break

    # Last transition in the n-step window
    _, _, _, next_obs, done = self._n_step_buffer[-1]

    # Push the n-step transition
    self._push_single(obs, action, n_step_reward, next_obs, float(done))

    # Clear the n-step buffer after pushing
    self._n_step_buffer.clear()

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
            ``done`` -- each a ``torch.Tensor``.
        """
    if generator is not None:
      indices = torch.randint(0, self._size, (batch_size,), generator=generator).numpy()
    else:
      # D5: Use independent RNG for reproducible sampling
      indices = self._rng.integers(0, self._size, size=batch_size)

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
