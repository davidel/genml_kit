"""Tests for ActorCritic model."""

import torch

from genml_kit.models.registry import load_model
from genml_kit.models.rl.actor_critic import (
    ActorCritic,
    CategoricalActor,
    GaussianActor,
    ValueHead,
    _MLPBackbone,
)


class TestMLPBackbone:

  def test_output_shape(self):
    backbone = _MLPBackbone(obs_dim=8, hidden_dims=[32, 16])
    out = backbone(torch.randn(4, 8))
    assert out.shape == (4, 16)

  def test_default_hidden(self):
    backbone = _MLPBackbone(obs_dim=4)
    assert backbone.out_dim == 256


class TestCategoricalActor:

  def test_shape(self):
    actor = CategoricalActor(64, n_actions=3)
    h = torch.randn(8, 64)
    dist = actor(h)
    assert dist.logits.shape == (8, 3)

  def test_get_action(self):
    actor = CategoricalActor(64, n_actions=3)
    h = torch.randn(4, 64)
    action, log_prob = actor.get_action(h, deterministic=True)
    assert action.shape == (4,)
    assert log_prob.shape == (4,)

  def test_action_range(self):
    actor = CategoricalActor(64, n_actions=3)
    h = torch.randn(10, 64)
    action, _ = actor.get_action(h, deterministic=True)
    assert action.min() >= 0 and action.max() < 3


class TestGaussianActor:

  def test_shape(self):
    actor = GaussianActor(64, action_dim=2)
    h = torch.randn(8, 64)
    dist = actor(h)
    assert dist.mean.shape == (8, 2)
    assert dist.stddev.shape == (8, 2)

  def test_get_action(self):
    actor = GaussianActor(64, action_dim=2)
    h = torch.randn(4, 64)
    action, log_prob = actor.get_action(h, deterministic=False)
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)

  def test_deterministic(self):
    actor = GaussianActor(64, action_dim=2)
    h = torch.randn(4, 64)
    action1, _ = actor.get_action(h, deterministic=True)
    action2, _ = actor.get_action(h, deterministic=True)
    assert torch.equal(action1, action2)


class TestValueHead:

  def test_shape(self):
    head = ValueHead(64)
    h = torch.randn(8, 64)
    v = head(h)
    assert v.shape == (8,)


class TestActorCritic:

  def test_discrete_forward(self):
    ac = ActorCritic(obs_dim=4, n_actions=3, discrete=True)
    obs = torch.randn(8, 4)
    value = ac(obs)
    assert value.shape == (8,)

  def test_discrete_get_action_and_value(self):
    ac = ActorCritic(obs_dim=4, n_actions=3, discrete=True)
    obs = torch.randn(8, 4)
    action, raw_action, log_prob, entropy, value = ac.get_action_and_value(obs)
    assert action.shape == (8,)
    # Discrete.
    assert raw_action is None
    assert log_prob.shape == (8,)
    assert value.shape == (8,)

  def test_discrete_with_stored_action(self):
    ac = ActorCritic(obs_dim=4, n_actions=3, discrete=True)
    obs = torch.randn(8, 4)
    action = torch.randint(0, 3, (8,))
    _, _, log_prob, entropy, value = ac.get_action_and_value(obs, action=action)
    assert log_prob.shape == (8,)

  def test_continuous_forward(self):
    ac = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    obs = torch.randn(8, 4)
    value = ac(obs)
    assert value.shape == (8,)

  def test_continuous_get_action_and_value(self):
    ac = ActorCritic(obs_dim=4, action_dim=2, discrete=False)
    obs = torch.randn(8, 4)
    action, raw_action, log_prob, entropy, value = ac.get_action_and_value(obs)
    assert action.shape == (8, 2)
    # Raw action same shape as action.
    assert raw_action.shape == (8, 2)
    assert log_prob.shape == (8,)
    assert value.shape == (8,)

  def test_get_value(self):
    ac = ActorCritic(obs_dim=4, n_actions=3, discrete=True)
    obs = torch.randn(8, 4)
    v = ac.get_value(obs)
    assert v.shape == (8,)


class TestRegistry:

  def test_discrete_loads(self):
    model = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=10,
        n_actions=4,
        discrete=True,
    )
    assert isinstance(model, ActorCritic)
    assert model.n_actions == 4

  def test_continuous_loads(self):
    model = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=10,
        action_dim=3,
        discrete=False,
    )
    assert isinstance(model, ActorCritic)
    assert model.action_dim == 3

  def test_no_collision(self):
    from genml_kit.models.registry import is_custom_model
    assert is_custom_model("rl/actor_critic")
