"""Tests for Phase 2 RL loss functions (GAE, PPO, SAC)."""

import pytest
import torch

from genml_kit.losses.rl import (
    clipped_surrogate,
    entropy_bonus,
    gae,
    sac_alpha_loss,
    sac_policy_loss,
    sac_q_loss,
    value_loss,
)


class TestGAE:

  def test_shape(self):
    T = 10
    rewards = torch.randn(T)
    values = torch.randn(T)
    next_values = torch.randn(T)
    dones = torch.zeros(T)
    adv, ret = gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (T,)
    assert ret.shape == (T,)

  def test_advantage_equals_return_minus_value(self):
    T = 5
    rewards = torch.ones(T)
    values = torch.zeros(T)
    next_values = torch.zeros(T)
    dones = torch.zeros(T)
    adv, ret = gae(rewards, values, next_values, dones, gamma=1.0, lam=1.0)
    # With gamma=1, lam=1, no done: advantage = sum of future rewards.
    # returns = advantages + values = advantages (since values=0).
    assert torch.allclose(ret, adv, atol=1e-5)

  def test_done_resets_gae(self):
    """After a done flag, the next advantage should not depend on prior."""
    rewards = torch.tensor([1.0, 10.0])
    values = torch.zeros(2)
    next_values = torch.zeros(2)
    dones = torch.tensor([1.0, 0.0])
    adv, _ = gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    # After done=1 at t=0, advantage at t=1 should be just delta at t=1.
    # delta[1] = 10.0 + 0 - 0 = 10.0.
    assert adv[1] == pytest.approx(10.0, abs=1e-4)

  def test_2d_input(self):
    T = 4
    rewards = torch.randn(T, 1)
    values = torch.randn(T, 1)
    next_values = torch.randn(T, 1)
    dones = torch.zeros(T, 1)
    adv, ret = gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95)
    assert adv.shape == (T,)
    assert ret.shape == (T,)


class TestClippedSurrogate:

  def test_basic(self):
    ratio = torch.ones(4)
    advantages = torch.ones(4)
    loss = clipped_surrogate(ratio, advantages, clip_eps=0.2)
    assert loss.item() == pytest.approx(-1.0, abs=1e-5)

  def test_clipping_effect(self):
    """Clipping changes the loss when ratios are far from 1."""
    ratio = torch.tensor([0.5, 0.5, 2.0, 2.0])
    advantages = torch.ones(4)
    loss_no_clip = clipped_surrogate(ratio, advantages, clip_eps=0.0)
    loss_with_clip = clipped_surrogate(ratio, advantages, clip_eps=0.2)
    # The clipped loss should differ from the unclipped.
    assert loss_no_clip.item() != loss_with_clip.item()

  def test_negative_advantage(self):
    """Negative advantages should still produce valid loss."""
    ratio = torch.ones(4)
    advantages = -torch.ones(4)
    loss = clipped_surrogate(ratio, advantages, clip_eps=0.2)
    assert loss.item() >= 0.0  # loss is negated


class TestValueLoss:

  def test_mse_fallback(self):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 3.0])
    loss = value_loss(pred, target)
    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(loss, expected)

  def test_clipped_value_loss(self):
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 3.0])
    old = torch.tensor([1.0, 2.0])
    loss = value_loss(pred, target, old_values=old, clip_eps=0.1)
    assert loss.item() >= 0.0

  def test_clipped_no_change(self):
    """When pred doesn't change from old, clipping is inactive."""
    pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    old = torch.tensor([1.0, 2.0])
    loss = value_loss(pred, target, old_values=old, clip_eps=0.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestEntropyBonus:

  def test_shape(self):
    log_probs = torch.randn(8)
    ent = entropy_bonus(log_probs)
    assert ent.shape == ()

  def test_positive(self):
    """Entropy bonus should be positive for negative log_probs."""
    log_probs = -torch.ones(8)  # log_prob < 0 is typical
    ent = entropy_bonus(log_probs)
    assert ent.item() > 0

  def test_2d_input(self):
    log_probs = torch.randn(8, 1)
    ent = entropy_bonus(log_probs)
    assert ent.shape == ()


class TestSACQLoss:

  def test_basic(self):
    q_pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([1.0, 2.0])
    loss = sac_q_loss(q_pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

  def test_nonzero(self):
    q_pred = torch.tensor([1.0, 2.0])
    target = torch.tensor([2.0, 3.0])
    loss = sac_q_loss(q_pred, target)
    assert loss.item() > 0


class TestSACPolicyLoss:

  def test_basic(self):
    log_probs = torch.randn(8)
    q_values = torch.randn(8)
    loss = sac_policy_loss(log_probs, q_values, alpha=0.2)
    assert loss.shape == ()

  def test_alpha_effect(self):
    log_probs = torch.ones(8)
    q_values = torch.ones(8)
    loss_low = sac_policy_loss(log_probs, q_values, alpha=0.01)
    loss_high = sac_policy_loss(log_probs, q_values, alpha=1.0)
    assert loss_high.item() > loss_low.item()


class TestSACAlphaLoss:

  def test_basic(self):
    log_probs = torch.randn(8)
    loss = sac_alpha_loss(log_probs, target_entropy=-2.0)
    assert loss.shape == ()

  def test_direction(self):
    """When entropy is high (log_prob very negative), alpha should grow."""
    # loss = -E[log π + H*]. When log π << H*, loss is very negative
    # (large gradient to increase α).
    low_entropy = torch.full((8,), -0.1)   # close to target
    high_entropy = torch.full((8,), -5.0)  # much higher entropy
    loss_low = sac_alpha_loss(low_entropy, target_entropy=-2.0)
    loss_high = sac_alpha_loss(high_entropy, target_entropy=-2.0)
    # -(-5 + -2) = 7 vs -(-0.1 + -2) = 2.1 → loss_high > loss_low
    assert loss_high.item() > loss_low.item()
