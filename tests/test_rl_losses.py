"""Tests for RL loss functions (td_target + td_loss)."""

import torch
import torch.nn as nn

from genml_kit.losses.rl import td_loss, td_target


class _ToyQNet(nn.Module):
  """Minimal Q-network for testing."""

  def __init__(self, obs_dim=4, n_actions=2):
    super().__init__()
    self.fc = nn.Linear(obs_dim, n_actions)

  def forward(self, obs):
    return self.fc(obs)


class TestTdTarget:
  """Tests for the n-step Double-DQN target computation."""

  def setup_method(self):
    torch.manual_seed(0)
    self.policy = _ToyQNet()
    self.target = _ToyQNet()
    self.target.load_state_dict(self.policy.state_dict())

  def test_shape(self):
    B = 8
    obs = torch.randn(B, 4)
    rewards = torch.randn(B)
    dones = torch.zeros(B)
    result = td_target(rewards, obs, dones, self.policy, self.target, gamma=0.99)
    assert result.shape == (B,)

  def test_done_mask_zeroes_future(self):
    """When done=1 the future value is masked out, leaving only reward."""
    obs = torch.randn(3, 4)
    rewards = torch.tensor([1.0, 2.0, 3.0])
    dones = torch.ones(3)
    target = td_target(rewards, obs, dones, self.policy, self.target, gamma=0.99)
    # With done=1, target == reward for each sample.
    assert torch.allclose(target, rewards, atol=1e-6)

  def test_no_done_value_contributes(self):
    """When done=0 the target includes gamma * Q_target(s', a')."""
    obs = torch.randn(3, 4)
    rewards = torch.zeros(3)
    dones = torch.zeros(3)
    gamma = 0.9
    target = td_target(rewards, obs, dones, self.policy, self.target, gamma=gamma)
    with torch.no_grad():
      next_q = self.target(obs)
      next_q_online = self.policy(obs)
      best_actions = next_q_online.argmax(dim=-1)
      expected = gamma * next_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    assert torch.allclose(target, expected, atol=1e-5)

  def test_double_q_false(self):
    """Vanilla DQN uses target_net for action selection too."""
    obs = torch.randn(3, 4)
    rewards = torch.ones(3)
    dones = torch.zeros(3)
    target = td_target(rewards,
                       obs,
                       dones,
                       self.policy,
                       self.target,
                       gamma=0.99,
                       double_q=False)
    with torch.no_grad():
      next_q_target = self.target(obs)
      expected = rewards + 0.99 * next_q_target.max(dim=-1).values
    assert torch.allclose(target, expected, atol=1e-5)

  def test_n_step(self):
    """n_step=2 applies gamma^2 discount."""
    obs = torch.randn(3, 4)
    rewards = torch.ones(3)
    dones = torch.zeros(3)
    target = td_target(rewards,
                       obs,
                       dones,
                       self.policy,
                       self.target,
                       gamma=0.9,
                       n_step=2)
    with torch.no_grad():
      next_q_target = self.target(obs)
      expected = rewards + (0.9**2) * next_q_target.max(dim=-1).values
    assert torch.allclose(target, expected, atol=1e-5)

  def test_terminated_overrides_dones(self):
    """A truncated step (done=1, terminated=0) still bootstraps."""
    obs = torch.randn(3, 4)
    rewards = torch.zeros(3)
    dones = torch.ones(3)  # episode over from the env's perspective
    terminated = torch.ones(3)
    target_done = td_target(rewards, obs, dones, self.policy, self.target, gamma=0.9)
    # With terminated=1 the bootstrap is masked out: target == reward.
    target_term = td_target(rewards,
                            obs,
                            dones,
                            self.policy,
                            self.target,
                            gamma=0.9,
                            terminated=terminated)
    assert torch.allclose(target_term, rewards, atol=1e-6)
    # Truncation: done=1 but terminated=0 -> gamma * Q is included.
    truncated = torch.zeros(3)
    target_trunc = td_target(rewards,
                             obs,
                             dones,
                             self.policy,
                             self.target,
                             gamma=0.9,
                             terminated=truncated)
    with torch.no_grad():
      next_q_target = self.target(obs)
      expected = 0.9 * next_q_target.max(dim=-1).values
    assert torch.allclose(target_trunc, expected, atol=1e-5)
    assert not torch.allclose(target_trunc, target_done, atol=1e-5)


class TestTdLoss:
  """Tests for the Huber TD loss."""

  def test_reduction_mean(self):
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = td_loss(pred, target, reduction="mean")
    expected = nn.functional.smooth_l1_loss(pred, target, reduction="mean")
    assert torch.allclose(loss, expected)

  def test_reduction_sum(self):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 4.0])
    loss = td_loss(pred, target, reduction="sum")
    expected = nn.functional.smooth_l1_loss(pred, target, reduction="sum")
    assert torch.allclose(loss, expected)

  def test_reduction_none(self):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 2.0])
    loss = td_loss(pred, target, reduction="none")
    assert loss.shape == (2,)

  def test_zero_error(self):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    loss = td_loss(pred, target)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)
