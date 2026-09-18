"""Q-network models for DQN / Double-DQN.

Phase 1 of ``plans/RL_PLAN.md`` (§6.4).  Two variants:

- ``QNetwork`` with a flat ``QHead`` — registered as ``rl/qnet``.
- ``QNetwork`` with a ``DuelingQHead`` — registered as ``rl/qnet_dueling``.

Both variants support:

- **Vector observations** (``obs_dim`` specified): a plain MLP backbone.
- **Image observations**: any registered custom backbone via
  ``load_model(args.model, ...)`` (deferred to Phase 2).

The target network lives inside ``QNetwork.target`` (deepcopy, frozen) —
same pattern as ``genml_kit.models.byol.BYOL``.
"""

import copy

import torch
import torch.nn as nn

from genml_kit.models.registry import register_model


class QHead(nn.Module):
  """Single linear layer projecting hidden features to Q-values.

  Args:
      hidden_dim: Input feature dimension.
      n_actions:  Number of discrete actions.
  """

  def __init__(self, hidden_dim, n_actions):
    super().__init__()
    self.fc = nn.Linear(hidden_dim, n_actions)

  def forward(self, h):
    return self.fc(h)  # (B, hidden_dim) -> (B, n_actions)


class DuelingQHead(nn.Module):
  """Dueling Q-network head: Q = V + A − mean(A).

  Implements the identifiability argument from ``rl/README.md`` §3.2.

  Args:
      hidden_dim: Input feature dimension.
      n_actions:  Number of discrete actions.
  """

  def __init__(self, hidden_dim, n_actions):
    super().__init__()
    self.val = nn.Linear(hidden_dim, 1)
    self.adv = nn.Linear(hidden_dim, n_actions)

  def forward(self, h):
    v = self.val(h)  # (B, 1)
    a = self.adv(h)  # (B, n_actions)
    return v + a - a.mean(dim=-1, keepdim=True)  # (B, n_actions)


class _MLPBackbone(nn.Module):
  """Simple MLP backbone for vector observations.

  Architecture: Linear(obs_dim, h1) -> ReLU -> … -> ReLU -> Linear(hN, hidden_dim).
  Hidden dimensions are specified as a list (default ``[128, 128]``).

  Args:
      obs_dim:    Dimensionality of the observation vector.
      hidden_dims: List of hidden-layer widths.
  """

  def __init__(self, obs_dim, hidden_dims=None):
    super().__init__()
    if hidden_dims is None:
      hidden_dims = [128, 128]
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
      layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
      in_dim = h
    # Final projection to the last hidden dim (head reads from here).
    self.net = nn.Sequential(*layers)
    self.out_dim = hidden_dims[-1]

  def forward(self, obs):
    # obs: (B, obs_dim)
    # net: iteratively applies Linear + ReLU
    # returns: (B, hidden_dims[-1])
    return self.net(obs)


class QNetwork(nn.Module):
  """Online + target Q-network with configurable backbone and head.

  The *online* network is trainable; the *target* network is a frozen
  deepcopy used for bootstrapping (updated periodically via hard or
  EMA copy by the method).

  Args:
      obs_dim:      Dimensionality of the observation vector (vector obs).
      n_actions:    Number of discrete actions.
      hidden_dims:  MLP hidden-layer widths (default ``[128, 128]``).
      dueling:      If ``True`` use :class:`DuelingQHead`.
  """

  def __init__(self, obs_dim, n_actions, hidden_dims=None, dueling=False):
    super().__init__()
    backbone = _MLPBackbone(obs_dim, hidden_dims)
    head_cls = DuelingQHead if dueling else QHead
    head = head_cls(backbone.out_dim, n_actions)

    self.online = nn.Sequential(backbone, head)
    self.target = copy.deepcopy(self.online)
    for p in self.target.parameters():
      p.requires_grad = False

    self.n_actions = n_actions

  def forward(self, obs):
    """Forward pass on the **online** network.

    Args:
        obs: (B, obs_dim) observation tensor.

    Returns:
        (B, n_actions) Q-values.
    """
    # obs: (B, obs_dim)
    # -> backbone: (B, hidden_dims[-1])
    # -> head:     (B, n_actions)
    return self.online(obs)

  @torch.no_grad()
  def hard_update(self):
    """Hard-copy online parameters to the target network."""
    self.target.load_state_dict(self.online.state_dict())


@register_model("rl/qnet")
def load_rl_qnet(obs_dim=4, n_actions=2, hidden_dims=None, num_labels=0, **_kwargs):
  """Factory registered as ``rl/qnet`` (standard Q-head).

  Returns the ``QNetwork`` module (no image processor for vector obs).
  """
  return QNetwork(obs_dim=obs_dim,
                  n_actions=n_actions,
                  hidden_dims=hidden_dims,
                  dueling=False)


@register_model("rl/qnet_dueling")
def load_rl_qnet_dueling(obs_dim=4,
                         n_actions=2,
                         hidden_dims=None,
                         num_labels=0,
                         **_kwargs):
  """Factory registered as ``rl/qnet_dueling`` (dueling Q-head).

  Returns the ``QNetwork`` module (no image processor for vector obs).
  """
  return QNetwork(obs_dim=obs_dim,
                  n_actions=n_actions,
                  hidden_dims=hidden_dims,
                  dueling=True)
