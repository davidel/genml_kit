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

  Supports Prioritized Experience Replay (PER): when prioritized=True,
  transitions are sampled proportionally to their priority (TD error)
  with importance-sampling weight correction.

  Args:
      obs_dim:  Dimensionality of the observation vector.
      capacity: Maximum number of transitions stored.
      action_dim: Dimensionality of the action vector (default: 1 for discrete).
      action_dtype: dtype for actions (default: int64 for discrete,
          float32 for continuous).
      n_step: Number of steps for n-step returns (default: 1).
      gamma: Discount factor for n-step return computation
          (default: 0.99).
      prioritized: Enable prioritized experience replay (default: False).
      alpha: Priority exponent for PER (default: 0.6).
      beta_start: Initial beta for IS weight annealing (default: 0.4).
      beta_frames: Frames over which to anneal beta to 1.0 (default: 100000).
      seed: Random seed for reproducible sampling.
  """

  def __init__(self,
               obs_dim,
               capacity=100_000,
               action_dim=1,
               action_dtype=np.int64,
               n_step=1,
               gamma=0.99,
               prioritized=False,
               alpha=0.6,
               beta_start=0.4,
               beta_frames=100_000,
               seed=None):
    self._capacity = capacity
    self._obs_dim = obs_dim
    self._action_dim = action_dim
    self._n_step = n_step
    self._gamma = gamma
    self._prioritized = prioritized
    self._alpha = alpha
    self._beta_start = beta_start
    self._beta_frames = beta_frames
    self._beta = beta_start
    self._pos = 0
    self._size = 0

    self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    if action_dim == 1:
      self._action = np.zeros(capacity, dtype=action_dtype)
    else:
      self._action = np.zeros((capacity, action_dim), dtype=action_dtype)
    self._reward = np.zeros(capacity, dtype=np.float32)
    self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
    self._done = np.zeros(capacity, dtype=np.float32)

    # N-step return support
    if n_step > 1:
      self._n_step_buffer = collections.deque(maxlen=n_step)

    # Prioritized Experience Replay support
    if prioritized:
      self._priorities = np.zeros(capacity, dtype=np.float32)
      self._max_priority = 1.0

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
    if self._n_step > 1:
      self._n_step_buffer.append((obs, action, reward, next_obs, done))

      # Compute n-step return if buffer is full or episode terminated
      if len(self._n_step_buffer) == self._n_step or done:
        self._push_n_step()
    else:
      self._push_single(obs, action, reward, next_obs, done)

  def _push_single(self, obs, action, reward, next_obs, done):
    """Internal: push a single transition to the main buffer."""
    self._obs[self._pos] = obs
    # Handle both scalar and array actions
    if self._action_dim > 1:
      # Continuous action: ensure it's a 1D array
      action = np.asarray(action, dtype=self._action.dtype).flatten()
      # Ensure correct shape
      assert action.shape == (
          self._action_dim,), f"action shape {action.shape} != ({self._action_dim},)"
    else:
      # Discrete action: ensure scalar
      action = (np.asarray(action, dtype=self._action.dtype).item() if hasattr(
          np.asarray(action), "item") else action)
    self._action[self._pos] = action
    self._reward[self._pos] = reward
    self._next_obs[self._pos] = next_obs
    self._done[self._pos] = float(done)

    # Initialize priority for new transition
    if self._prioritized:
      self._priorities[self._pos] = self._max_priority

    self._pos = (self._pos + 1) % self._capacity
    self._size = min(self._size + 1, self._capacity)

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
      gamma_pow *= self._gamma
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

  def update_priorities(self, indices, priorities):
    """Update priorities for sampled transitions (PER).

    Args:
        indices: Array of transition indices to update.
        priorities: New priority values (absolute TD errors + epsilon).
    """
    if not self._prioritized:
      return
    priorities = np.asarray(priorities, dtype=np.float32)
    self._priorities[indices] = priorities
    self._max_priority = max(self._max_priority, float(np.max(priorities)))

  def sample(self, batch_size, generator=None):
    """Sample a batch of transitions.

    Args:
        batch_size:  Number of transitions to sample.
        generator:   Optional ``torch.Generator`` for reproducibility.

    Returns:
        dict with keys ``obs``, ``action``, ``reward``, ``next_obs``,
        ``done`` -- each a ``torch.Tensor``. If prioritized, also includes
        ``indices`` and ``weights`` for importance sampling correction.
    """
    if self._prioritized:
      return self._sample_prioritized(batch_size)

    if generator is not None:
      indices = torch.randint(0, self._size, (batch_size,), generator=generator).numpy()
    else:
      # D5: Use independent RNG for reproducible sampling
      indices = self._rng.integers(0, self._size, size=batch_size)

    return {
        "obs": torch.from_numpy(self._obs[indices]),
        "action": torch.from_numpy(self._action[indices]),
        "reward": torch.from_numpy(self._reward[indices]),
        "next_obs": torch.from_numpy(self._next_obs[indices]),
        "done": torch.from_numpy(self._done[indices]),
    }

  def _sample_prioritized(self, batch_size):
    """Sample transitions proportionally to priority^alpha with IS weights."""
    # Get priorities for valid transitions
    valid_priorities = self._priorities[:self._size]
    probs = valid_priorities**self._alpha
    probs_sum = probs.sum()
    if probs_sum == 0:
      # Fallback to uniform if all priorities are zero
      probs = np.ones(self._size, dtype=np.float32) / self._size
    else:
      probs = probs / probs_sum

    # Sample indices using buffer's RNG (B2: reproducible)
    indices = self._rng.choice(self._size, size=batch_size, p=probs, replace=True)

    # Compute importance-sampling weights: w_i = (N * P(i))^(-beta)
    # Normalized by max weight for stability
    weights = (self._size * probs[indices])**(-self._beta)
    weights = weights / weights.max()

    return {
        "obs": torch.from_numpy(self._obs[indices]),
        "action": torch.from_numpy(self._action[indices]),
        "reward": torch.from_numpy(self._reward[indices]),
        "next_obs": torch.from_numpy(self._next_obs[indices]),
        "done": torch.from_numpy(self._done[indices]),
        "indices": torch.from_numpy(indices.astype(np.int64)),
        "weights": torch.from_numpy(weights.astype(np.float32)),
    }

  def anneal_beta(self, frames):
    """Anneal beta from beta_start to 1.0 over beta_frames.

    Args:
        frames: Number of training frames/steps elapsed.
    """
    if not self._prioritized:
      return
    progress = min(frames / self._beta_frames, 1.0)
    self._beta = self._beta_start + progress * (1.0 - self._beta_start)

  @property
  def beta(self):
    """Current beta value for IS weight computation."""
    return self._beta

  def stats(self):
    """Return diagnostic statistics about the buffer contents."""
    if self._size == 0:
      return {"size": 0, "fill_ratio": 0.0}
    stats = {
        "size": self._size,
        "capacity": self._capacity,
        "fill_ratio": self._size / self._capacity,
        "mean_reward": float(np.mean(self._reward[:self._size])),
    }
    if self._prioritized:
      stats.update({
          "mean_priority": float(np.mean(self._priorities[:self._size])),
          "max_priority": float(np.max(self._priorities[:self._size])),
          "beta": self._beta,
      })
    return stats

  # ------------------------------------------------------------------
  # Dataset protocol
  # ------------------------------------------------------------------

  def __len__(self):
    return self._size

  # Properties for backward compatibility with tests
  @property
  def obs(self):
    return self._obs

  @property
  def action(self):
    return self._action

  @property
  def reward(self):
    return self._reward

  @property
  def next_obs(self):
    return self._next_obs

  @property
  def done(self):
    return self._done

  @property
  def capacity(self):
    return self._capacity

  @property
  def prioritized(self):
    return self._prioritized

  @property
  def alpha(self):
    return self._alpha

  @property
  def gamma(self):
    return self._gamma

  @property
  def n_step(self):
    return self._n_step

  @property
  def obs_dim(self):
    return self._obs_dim

  @property
  def action_dim(self):
    return self._action_dim

  @property
  def priorities(self):
    return self._priorities

  def __getitem__(self, idx):
    """Return a single transition as a dict (DataLoader-compatible).

    Note: The returned arrays are numpy views; DataLoader collation
    converts them to tensors via ``default_collate``.
    """
    return {
        "obs": self._obs[idx],
        "action": self._action[idx],
        "reward": self._reward[idx],
        "next_obs": self._next_obs[idx],
        "done": self._done[idx],
    }
