"""Actor-Critic model for PPO and SAC.

Phase 2 of ``plans/RL_PLAN.md`` (§6.4).  Provides shared-backbone
``ActorCritic`` with separate policy and value heads.

- **Discrete actions:** ``CategoricalActor`` (softmax head).
- **Continuous actions:** ``GaussianActor`` (mean + log-std).
"""

import torch
import torch.nn as nn

from genml_kit.models.registry import register_model

# =====================================================================
# Backbones
# =====================================================================


class _MLPBackbone(nn.Module):
  """Simple MLP backbone for vector observations.

  Args:
      obs_dim:      Observation dimensionality.
      hidden_dims:  List of hidden-layer widths.
  """

  def __init__(self, obs_dim, hidden_dims=None):
    super().__init__()
    if hidden_dims is None:
      hidden_dims = [256, 256]
    layers = []
    in_dim = obs_dim
    for h in hidden_dims:
      layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
      in_dim = h
    self.net = nn.Sequential(*layers)
    self.out_dim = hidden_dims[-1]

  def forward(self, obs):
    return self.net(obs)


# =====================================================================
# Policy heads
# =====================================================================


class CategoricalActor(nn.Module):
  """Categorical policy head for discrete action spaces.

  ``forward(obs)`` returns ``log_probs`` and ``entropy`` for the
  *sampled* actions (for training) or the *mode* (for inference).

  Args:
      hidden_dim: Input feature dimension.
      n_actions:  Number of discrete actions.
  """

  def __init__(self, hidden_dim, n_actions):
    super().__init__()
    self.logits = nn.Linear(hidden_dim, n_actions)

  def forward(self, h):
    """Return (log_probs, entropy) from a Categorical distribution."""
    logits = self.logits(h)
    dist = torch.distributions.Categorical(logits=logits)
    return dist

  @torch.no_grad()
  def get_action(self, h, deterministic=False):
    """Sample or greedily select an action.

    Returns:
        action:  (B,) int64 tensor.
        log_prob: (B,) log π(a|s).
    """
    logits = self.logits(h)
    dist = torch.distributions.Categorical(logits=logits)
    action = logits.argmax(dim=-1) if deterministic else dist.sample()
    return action, dist.log_prob(action)


class GaussianActor(nn.Module):
  """Gaussian policy head for continuous action spaces.

  Uses a squashed Gaussian (tanh) following the SAC convention, or
  an unbounded Gaussian for PPO.

  Args:
      hidden_dim: Input feature dimension.
      action_dim: Dimensionality of the continuous action space.
      log_std_min: Minimum log-standard-deviation (clipping).
      log_std_max: Maximum log-standard-deviation (clipping).
  """

  def __init__(self, hidden_dim, action_dim, log_std_min=-20.0, log_std_max=2.0):
    super().__init__()
    self.mean = nn.Linear(hidden_dim, action_dim)
    self.log_std = nn.Linear(hidden_dim, action_dim)
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.action_dim = action_dim

  def forward(self, h):
    """Return a Normal distribution (before squashing)."""
    mean = self.mean(h)
    log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
    std = log_std.exp()
    return torch.distributions.Normal(mean, std)

  @torch.no_grad()
  def get_action(self, h, deterministic=False):
    """Sample or return the mean action (tanh-squashed).

    Returns:
        action:   (B, action_dim) tensor.
        log_prob: (B,) log π(a|s) accounting for the tanh squashing.
    """
    dist = self.forward(h)
    action = dist.mean if deterministic else dist.rsample()
    # Squash to [-1, 1].
    squashed = torch.tanh(action)
    # Log-prob with tanh squashing correction.
    log_prob = dist.log_prob(action).sum(dim=-1)
    log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
    return squashed, log_prob


# =====================================================================
# Value head
# =====================================================================


class ValueHead(nn.Module):
  """Single linear layer: hidden → scalar value."""

  def __init__(self, hidden_dim):
    super().__init__()
    self.fc = nn.Linear(hidden_dim, 1)

  def forward(self, h):
    return self.fc(h).squeeze(-1)  # (B,)


# =====================================================================
# ActorCritic module
# =====================================================================


class ActorCritic(nn.Module):
  """Shared-backbone Actor-Critic for PPO and SAC.

  Args:
      obs_dim:      Observation dimensionality.
      n_actions:    Number of discrete actions (for ``discrete=True``).
      action_dim:   Action dimensionality (for ``discrete=False``).
      hidden_dims:  MLP hidden-layer widths (default [256, 256]).
      discrete:     If ``True``, use ``CategoricalActor``; else
                    ``GaussianActor``.
  """

  def __init__(self,
               obs_dim,
               n_actions=None,
               action_dim=None,
               hidden_dims=None,
               discrete=True):
    super().__init__()
    self.discrete = discrete
    self.backbone = _MLPBackbone(obs_dim, hidden_dims)
    h = self.backbone.out_dim

    if discrete:
      assert n_actions is not None, "n_actions required for discrete"
      self.actor = CategoricalActor(h, n_actions)
      self.n_actions = n_actions
    else:
      assert action_dim is not None, "action_dim required for continuous"
      self.actor = GaussianActor(h, action_dim)
      self.action_dim = action_dim

    self.critic = ValueHead(h)

  def forward(self, obs):
    """Forward pass through backbone + critic (for value estimates).

    For the policy distribution, call ``self.get_distribution(obs)``.
    """
    h = self.backbone(obs)
    return self.critic(h)

  def get_distribution(self, obs):
    """Return the policy distribution at *obs*."""
    h = self.backbone(obs)
    return self.actor(h)

  def get_value(self, obs):
    """Return scalar value V(s) at *obs*."""
    h = self.backbone(obs)
    return self.critic(h)

  def get_action_and_value(self, obs, action=None, deterministic=False):
    """Full forward: policy sample + value estimate.

    If *action* is provided, evaluates ``log_prob(action)`` instead of
    sampling (used for PPO update epochs on stored actions).

    Returns:
        action:     (B,) or (B, action_dim).
        log_prob:   (B,).
        entropy:    (B,) or scalar.
        value:      (B,).
    """
    h = self.backbone(obs)
    dist = self.actor(h)
    value = self.critic(h)

    if action is not None:
      if self.discrete:
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
      else:
        # For Gaussian: action is tanh-squashed; re-evaluate log_prob.
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.log(1.0 - torch.tanh(action).pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
    else:
      if self.discrete:
        sampled = dist.sample()
        log_prob = dist.log_prob(sampled)
        entropy = dist.entropy()
        action = sampled
      else:
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        action = squashed

    return action, log_prob, entropy, value


# =====================================================================
# Registry
# =====================================================================


@register_model("rl/actor_critic")
def load_actor_critic(obs_dim=4,
                      n_actions=2,
                      action_dim=None,
                      discrete=True,
                      hidden_dims=None,
                      num_labels=0,
                      **_kwargs):
  """Factory registered as ``rl/actor_critic``."""
  return ActorCritic(
      obs_dim=obs_dim,
      n_actions=n_actions if discrete else None,
      action_dim=action_dim if not discrete else None,
      hidden_dims=hidden_dims,
      discrete=discrete,
  )
