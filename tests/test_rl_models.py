"""Tests for Q-network models (QNetwork + DuelingQHead)."""

import torch

from genml_kit.models.registry import load_model
from genml_kit.models.rl.qnetwork import (
    QHead,
    DuelingQHead,
    QNetwork,
    _MLPBackbone,
)
from genml_kit.models.rl.spaces import space_spec


class TestMLPBackbone:

  def test_output_shape(self):
    backbone = _MLPBackbone(obs_dim=10, hidden_dims=[64, 32])
    x = torch.randn(5, 10)
    out = backbone(x)
    assert out.shape == (5, 32)

  def test_default_hidden(self):
    backbone = _MLPBackbone(obs_dim=4)
    assert backbone.out_dim == 128
    out = backbone(torch.randn(2, 4))
    assert out.shape == (2, 128)


class TestQHead:

  def test_shape(self):
    head = QHead(hidden_dim=64, n_actions=3)
    out = head(torch.randn(8, 64))
    assert out.shape == (8, 3)


class TestDuelingQHead:

  def test_shape(self):
    head = DuelingQHead(hidden_dim=64, n_actions=3)
    out = head(torch.randn(8, 64))
    assert out.shape == (8, 3)

  def test_identifiability(self):
    """Q = V + A − mean(A) ⇒ average over actions equals V."""
    head = DuelingQHead(hidden_dim=16, n_actions=4)
    h = torch.randn(2, 16)
    q = head(h)
    v = head.val(h)
    # Mean Q-value across actions should equal the value stream.
    assert torch.allclose(q.mean(dim=-1), v.squeeze(-1), atol=1e-5)


class TestQNetwork:

  def test_forward_shape(self):
    net = QNetwork(obs_dim=4, n_actions=2)
    q = net(torch.randn(8, 4))
    assert q.shape == (8, 2)

  def test_target_is_frozen(self):
    net = QNetwork(obs_dim=4, n_actions=2)
    for p in net.target.parameters():
      assert not p.requires_grad

  def test_target_matches_online_at_init(self):
    net = QNetwork(obs_dim=4, n_actions=2)
    for p1, p2 in zip(net.online.parameters(), net.target.parameters()):
      assert torch.equal(p1, p2)

  def test_hard_update_copies(self):
    net = QNetwork(obs_dim=4, n_actions=2)
    # Mutate online weights.
    with torch.no_grad():
      for p in net.online.parameters():
        p.add_(1.0)
    net.hard_update()
    for p1, p2 in zip(net.online.parameters(), net.target.parameters()):
      assert torch.allclose(p1, p2)

  def test_dueling_variant(self):
    net = QNetwork(obs_dim=4, n_actions=3, dueling=True)
    q = net(torch.randn(8, 4))
    assert q.shape == (8, 3)


class TestRegistry:

  def test_qnet_loads(self):
    model = load_model("rl/qnet", num_labels=0, obs_dim=10, n_actions=4)
    assert isinstance(model, QNetwork)
    assert model.n_actions == 4
    q = model(torch.randn(3, 10))
    assert q.shape == (3, 4)

  def test_qnet_dueling_loads(self):
    model = load_model("rl/qnet_dueling", num_labels=0, obs_dim=8, n_actions=2)
    assert isinstance(model, QNetwork)
    # Verify the internal head is a DuelingQHead.
    assert isinstance(model.online[1], DuelingQHead)

  def test_no_collision_with_existing(self):
    """RL model names use 'rl/' prefix, no collision with existing."""
    from genml_kit.models.registry import is_custom_model
    assert is_custom_model("rl/qnet")
    assert is_custom_model("rl/qnet_dueling")
    # Existing custom models still registered.
    assert is_custom_model("timm")
    assert is_custom_model("convvit")

  def test_qnet_accepts_space(self):
    """The RL factories accept a SpaceSpec instead of scalars."""

    class _S:

      def __init__(self, **kw):
        self.__dict__.update(kw)

    space = space_spec(_S(shape=(10,)), _S(n=4))
    model = load_model("rl/qnet", num_labels=0, space=space)
    assert isinstance(model, QNetwork)
    assert model(torch.randn(3, 10)).shape == (3, 4)

  def test_qnet_dueling_accepts_space(self):

    class _S:

      def __init__(self, **kw):
        self.__dict__.update(kw)

    space = space_spec(_S(shape=(8,)), _S(n=2))
    model = load_model("rl/qnet_dueling", num_labels=0, space=space)
    assert isinstance(model, QNetwork)
    assert isinstance(model.online[1], DuelingQHead)

  def test_implicit_space_override_wins_over_scalars(self):
    """Passing both space and scalars: space is authoritative."""

    class _S:

      def __init__(self, **kw):
        self.__dict__.update(kw)

    space = space_spec(_S(shape=(7,)), _S(n=3))
    model = load_model("rl/qnet", num_labels=0, space=space, obs_dim=99, n_actions=1)
    assert model.online[0].net[0].in_features == 7
    assert model.online[-1].fc.out_features == 3
