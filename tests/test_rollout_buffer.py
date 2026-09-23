"""Tests for RolloutBuffer (on-policy PPO)."""

import torch

from genml_kit.datasets.rollout_buffer import RolloutBuffer


class TestRolloutBuffer:

  def test_add_and_length(self):
    buf = RolloutBuffer(obs_dim=4, rollout_len=10)
    assert len(buf) == 0
    for _ in range(5):
      buf.add(obs=[1, 2, 3, 4],
              action=0,
              log_prob=-0.5,
              reward=1.0,
              value=0.1,
              done=False)
    assert len(buf) == 5
    assert buf.ptr == 5
    assert not buf.full

  def test_full_detection(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=5)
    for _ in range(5):
      buf.add(obs=[0, 0], action=0, log_prob=-1.0, reward=0.0, value=0.0, done=False)
    assert buf.full
    assert buf.ptr == 5

  def test_overflow_raises(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=3)
    for _ in range(3):
      buf.add(obs=[0, 0], action=0, log_prob=0.0, reward=0.0, value=0.0, done=False)
    import pytest
    with pytest.raises(RuntimeError, match="overflow"):
      buf.add(obs=[0, 0], action=0, log_prob=0.0, reward=0.0, value=0.0, done=False)

  def test_compute_gae(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=4)
    for i in range(4):
      buf.add(obs=[i, 0],
              action=0,
              log_prob=-0.5,
              reward=float(i),
              value=0.0,
              done=False)
    buf.set_next_values([1.0, 1.0, 1.0, 1.0])
    buf.compute(gamma=0.99, lam=0.95)
    assert buf.advantages.shape == (4,)
    assert buf.returns.shape == (4,)
    # With no done flags, advantages should be non-trivial.
    assert buf.advantages.abs().sum() > 0

  def test_getitem_dict(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=4)
    buf.add(obs=[1, 2], action=1, log_prob=-0.3, reward=0.5, value=0.1, done=False)
    # Per-step bootstrap values.
    buf.set_next_values([0.0, 0.0, 0.0, 0.0])
    buf.compute(gamma=0.99, lam=0.95)
    item = buf[0]
    assert isinstance(item, dict)
    assert set(item.keys()) == {
        "obs",
        "action",
        "log_prob",
        "advantage",
        "return",
        "value",
    }

  def test_to_device(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=4)
    buf.add(obs=[0, 0], action=0, log_prob=0.0, reward=0.0, value=0.0, done=False)
    buf.to(torch.device("cpu"))
    assert buf.obs.device == torch.device("cpu")

  def test_reset(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=4)
    for _ in range(4):
      buf.add(obs=[0, 0], action=0, log_prob=0.0, reward=0.0, value=0.0, done=False)
    assert buf.full
    buf.reset()
    assert not buf.full
    assert buf.ptr == 0
    assert len(buf) == 0

  def test_get_batch(self):
    buf = RolloutBuffer(obs_dim=2, rollout_len=8)
    for i in range(8):
      buf.add(obs=[i, 0],
              action=i % 2,
              log_prob=-0.5,
              reward=float(i),
              value=0.0,
              done=False)
    buf.set_next_values([0.0] * 8)
    buf.compute(gamma=0.99, lam=0.95)
    batch = buf.get_batch(4)
    assert batch["obs"].shape == (4, 2)
    assert batch["action"].shape == (4,)
    assert batch["log_prob"].shape == (4,)
    assert batch["advantage"].shape == (4,)
    assert batch["return"].shape == (4,)
