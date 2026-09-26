"""SAC critic networks (continuous action space).

SAC uses twin Q-critics that take (obs, action) as input and output
a single Q-value. This is different from DQN's QNetwork which outputs
Q-values for all discrete actions.
"""

import copy

import torch
import torch.nn as nn

from genml_kit.models.registry import MODELS


class SACCritic(nn.Module):
  """Twin Q-critic for SAC (continuous actions).

  Takes (obs, action) concatenated and outputs a scalar Q-value.

  Each critic owns a frozen target network (``self.target``).  Note that
  the target is a plain ``nn.Sequential`` module, *not* an ``SACCritic``:
  call it directly on the concatenated input, e.g.
  ``self.target(torch.cat([obs, action], dim=-1))``.  Do NOT route it
  through ``get_action_and_value`` / ``forward``.
  """

  def __init__(self, obs_dim, action_dim, hidden_dims=None):
    super().__init__()
    if hidden_dims is None:
      hidden_dims = [256, 256]

    in_dim = obs_dim + action_dim
    layers = []
    for h in hidden_dims:
      layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
      in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    self.net = nn.Sequential(*layers)

    # Target network
    self.target = copy.deepcopy(self.net)
    for p in self.target.parameters():
      p.requires_grad = False

  def forward(self, obs, action):
    """Forward pass on the online network.

    Args:
        obs: (B, obs_dim) observation tensor.
        action: (B, action_dim) action tensor.

    Returns:
        (B, 1) Q-values.
    """
    x = torch.cat([obs, action], dim=-1)
    return self.net(x)

  def get_value(self, obs, action):
    """Alias for forward (used by SAC method)."""
    return self.forward(obs, action).squeeze(-1)

  @torch.no_grad()
  def hard_update(self):
    """Hard-copy online parameters to the target network."""
    self.target.load_state_dict(self.net.state_dict())

  def soft_update(self, tau):
    """Soft Polyak update: target = (1-tau)*target + tau*online."""
    with torch.no_grad():
      for p, pt in zip(self.net.parameters(), self.target.parameters()):
        pt.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


@MODELS.register("rl/sac_critic")
def load_rl_sac_critic(
    obs_dim=4,
    action_dim=2,
    hidden_dims=None,
    num_labels=0,
    **_kwargs,
):
  """Factory registered as ``rl/sac_critic``."""
  return SACCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=hidden_dims)
