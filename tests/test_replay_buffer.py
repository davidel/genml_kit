"""Tests for ReplayBufferDataset."""

import pytest
import torch

from genml_kit.datasets.replay_buffer import ReplayBufferDataset, Transition


class TestReplayBufferDataset:

  def test_push_and_len(self):
    buf = ReplayBufferDataset(obs_dim=4, capacity=10)
    assert len(buf) == 0
    buf.push(obs=[1, 2, 3, 4], action=0, reward=1.0, next_obs=[2, 3, 4, 5], done=False)
    assert len(buf) == 1

  def test_capacity_overflow(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=5)
    for i in range(10):
      buf.push(obs=[i, i],
               action=i % 2,
               reward=float(i),
               next_obs=[i + 1, i + 1],
               done=i == 9)
    assert len(buf) == 5
    # Most recent transition should be the last one pushed.
    item = buf[-1]
    assert item["obs"][0] == 9.0

  def test_sample_shape(self):
    buf = ReplayBufferDataset(obs_dim=3, capacity=100)
    for i in range(50):
      buf.push(obs=[i] * 3,
               action=i % 3,
               reward=float(i),
               next_obs=[i + 1] * 3,
               done=False)
    batch = buf.sample(16)
    assert batch["obs"].shape == (16, 3)
    assert batch["action"].shape == (16,)
    assert batch["reward"].shape == (16,)
    assert batch["next_obs"].shape == (16, 3)
    assert batch["done"].shape == (16,)

  def test_sample_reproducible_with_generator(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=100)
    for i in range(50):
      buf.push(obs=[i, 0], action=0, reward=0.0, next_obs=[i, 0], done=False)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    b1 = buf.sample(8, generator=g1)
    b2 = buf.sample(8, generator=g2)
    assert torch.equal(b1["obs"], b2["obs"])

  def test_push_batch(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=100)
    transitions = [
        Transition(obs=[1.0, 2.0],
                   action=0,
                   reward=0.5,
                   next_obs=[3.0, 4.0],
                   done=False) for _ in range(20)
    ]
    buf.push_batch(transitions)
    assert len(buf) == 20

  def test_getitem_dict(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=10)
    buf.push(obs=[1.0, 2.0], action=1, reward=0.5, next_obs=[3.0, 4.0], done=False)
    item = buf[0]
    assert isinstance(item, dict)
    assert set(
        item.keys()) == {"obs", "action", "reward", "next_obs", "done", "terminated"}

  def test_terminated_threaded(self):
    """A truncated step is stored with done=1 but terminated=0."""
    buf = ReplayBufferDataset(obs_dim=2, capacity=10)
    # Truncated: done=1 (episode over), terminated=0 (MDP would continue).
    buf.push(obs=[0, 0],
             action=0,
             reward=0.0,
             next_obs=[1, 1],
             done=True,
             terminated=False)
    # Terminated: both flags set.
    buf.push(obs=[1, 1],
             action=0,
             reward=0.0,
             next_obs=[2, 2],
             done=True,
             terminated=True)
    # sample() draws with replacement, so assert per-index instead.
    item0 = buf[0]
    item1 = buf[1]
    assert item0["done"] == 1.0 and item0["terminated"] == 0.0
    assert item1["done"] == 1.0 and item1["terminated"] == 1.0
    batch = buf.sample(2, generator=None)
    assert batch["done"].tolist() == [1.0, 1.0]

  def test_terminated_defaults_to_done(self):
    """Omitting terminated falls back to done (backward compat)."""
    buf = ReplayBufferDataset(obs_dim=2, capacity=10)
    buf.push(obs=[0, 0], action=0, reward=0.0, next_obs=[1, 1], done=True)
    assert buf[0]["terminated"] == 1.0

  def test_n_step_propagates_terminated(self):
    """N-step window: a true termination anywhere sets terminated=1."""
    buf = ReplayBufferDataset(obs_dim=2, capacity=10, n_step=3, gamma=0.99)
    # episode: 3 steps, the 2nd step truly terminates.
    buf.push(obs=[0, 0],
             action=0,
             reward=1.0,
             next_obs=[1, 1],
             done=False,
             terminated=False)
    buf.push(obs=[1, 1],
             action=0,
             reward=1.0,
             next_obs=[2, 2],
             done=True,
             terminated=True)
    buf.push(obs=[2, 2],
             action=0,
             reward=1.0,
             next_obs=[3, 3],
             done=False,
             terminated=False)
    # After the done step, the window is flushed; only the first window
    # (steps 1-2) is stored before the episode ended.
    assert len(buf) >= 1
    item = buf[0]
    assert item["terminated"] == 1.0
    # Return is discounted over the two steps that happened.
    assert item["reward"] == pytest.approx(1.0 + 0.99 * 1.0)

  def test_stats_empty(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=10)
    s = buf.stats()
    assert s["size"] == 0

  def test_stats_nonempty(self):
    buf = ReplayBufferDataset(obs_dim=2, capacity=10)
    for i in range(5):
      buf.push(obs=[0, 0], action=0, reward=float(i), next_obs=[0, 0], done=False)
    s = buf.stats()
    assert s["size"] == 5
    assert s["fill_ratio"] == pytest.approx(0.5)
    assert s["mean_reward"] == pytest.approx(2.0)

  def test_sample_values_within_buffer(self):
    """Sampled obs should be among those we pushed."""
    buf = ReplayBufferDataset(obs_dim=1, capacity=100)
    for i in range(10):
      buf.push(obs=[float(i)],
               action=0,
               reward=0.0,
               next_obs=[float(i + 1)],
               done=False)
    batch = buf.sample(1000)
    # All sampled first elements should be in [0, 9].
    vals = batch["obs"][:, 0]
    assert vals.min() >= 0.0
    assert vals.max() <= 9.0
