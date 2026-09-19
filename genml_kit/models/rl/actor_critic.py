"""Actor-Critic model for PPO and SAC.

Phase 2 of ``plans/RL_PLAN.md`` (§6.4).  Provides shared-backbone
``ActorCritic`` with separate policy and value heads.

- **Discrete actions:** ``CategoricalActor`` (softmax head).
- **Continuous actions:** ``GaussianActor`` (mean + log-std).
"""

import torch
import torch.nn as nn

from genml_kit.models.registry import register_model


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
    # obs: (B, obs_dim)
    # net: iteratively applies Linear + ReLU
    # returns: (B, hidden_dims[-1])
    return self.net(obs)


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
    """Return a Categorical distribution from hidden features.

    Args:
        h: (B, hidden_dim) shared backbone features.

    Returns:
        A ``Categorical`` distribution over *n_actions* classes.
    """
    # h: (B, hidden_dim)
    logits = self.logits(h)  # (B, n_actions)
    dist = torch.distributions.Categorical(logits=logits)
    return dist

  @torch.no_grad()
  def get_action(self, h, deterministic=False):
    """Sample or greedily select an action.

    Args:
        h:            (B, hidden_dim) shared backbone features.
        deterministic: If ``True``, return argmax; otherwise sample.

    Returns:
        action:   (B,) int64 tensor.
        log_prob: (B,) log π(a|s).
    """
    # h: (B, hidden_dim)
    logits = self.logits(h)  # (B, n_actions)
    dist = torch.distributions.Categorical(logits=logits)
    action = logits.argmax(dim=-1) if deterministic else dist.sample()  # (B,) int64
    return action, dist.log_prob(action)  # (B,) int64, (B,) float


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
    """Return a Normal distribution (before squashing).

    Args:
        h: (B, hidden_dim) shared backbone features.

    Returns:
        A ``Normal`` distribution with
        mean (B, action_dim) and std (B, action_dim).
    """
    # h: (B, hidden_dim)
    mean = self.mean(h)  # (B, action_dim)
    log_std = self.log_std(h).clamp(  # (B, action_dim)
        self.log_std_min, self.log_std_max)
    std = log_std.exp()  # (B, action_dim)
    return torch.distributions.Normal(mean, std)  # event_dim = action_dim

  @torch.no_grad()
  def get_action(self, h, deterministic=False):
    """Sample or return the mean action (tanh-squashed).

    Args:
        h:            (B, hidden_dim) shared backbone features.
        deterministic: If ``True``, return the mean; otherwise sample.

    Returns:
        action:   (B, action_dim) tensor in [-1, 1].
        log_prob: (B,) log π(a|s) accounting for the tanh squashing.
    """
    # h: (B, hidden_dim)
    dist = self.forward(h)  # Normal(mean=(B, action_dim), std=(B, action_dim))
    action = dist.mean if deterministic else dist.rsample()  # (B, action_dim) unbounded
    squashed = torch.tanh(action)  # (B, action_dim) in [-1, 1]
    log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
    log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(
        dim=-1)  # (B,) squashing correction
    return squashed, log_prob

  def get_action_and_value(self, obs, backbone=None, action=None, deterministic=False):
    """Full forward: policy sample + value estimate (for SAC).

    Args:
        obs:           (B, obs_dim) observation tensor (if backbone provided)
                       or (B, hidden_dim) pre-computed features (if backbone is None).
        backbone:      Optional backbone module (for computing hidden features).
                       If None, obs is assumed to be pre-computed features.
        action:        Optional (B, action_dim) float (RAW pre-tanh).
                       If None, a new action is sampled.
        deterministic: If True, return the mode instead of a sample.

    Returns:
        action:       (B, action_dim) tanh-squashed action.
        raw_action:   (B, action_dim) raw (pre-tanh) action.
        log_prob:     (B,) log π(a|s) accounting for tanh squashing.
        entropy:      (B,) entropy of the policy.
        value:        (B,) value estimate (None for standalone actor).
    """
    h = backbone(obs) if backbone is not None else obs
    dist = self.forward(h)
    if action is None:
      raw_action = dist.mean if deterministic else dist.rsample()
    else:
      raw_action = action
    squashed = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action).sum(dim=-1)
    log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    # Value is computed by the critic head, not the actor
    value = None
    return squashed, raw_action, log_prob, entropy, value


class ValueHead(nn.Module):
  """Single linear layer: hidden → scalar value."""

  def __init__(self, hidden_dim):
    super().__init__()
    self.fc = nn.Linear(hidden_dim, 1)

  def forward(self, h):
    return self.fc(h).squeeze(-1)  # (B,)


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

    Args:
        obs: (B, obs_dim) observation tensor.

    Returns:
        (B,) scalar value estimates.
    """
    # obs: (B, obs_dim)
    h = self.backbone(obs)  # (B, hidden_dims[-1])
    return self.critic(h)  # (B,)

  def get_distribution(self, obs):
    """Return the policy distribution at *obs*.

    Args:
        obs: (B, obs_dim) observation tensor.

    Returns:
        Categorical (discrete) or Normal (continuous) distribution.
    """
    # obs: (B, obs_dim)
    h = self.backbone(obs)  # (B, hidden_dims[-1])
    return self.actor(h)  # distribution with event_dim = n_actions or action_dim

  def get_value(self, obs):
    """Return scalar value V(s) at *obs*.

    Args:
        obs: (B, obs_dim) observation tensor.

    Returns:
        (B,) value predictions.
    """
    # obs: (B, obs_dim)
    h = self.backbone(obs)  # (B, hidden_dims[-1])
    return self.critic(h)  # (B,)

  def get_action_and_value(self, obs, action=None, deterministic=False):
    """Full forward: policy sample + value estimate.

    If *action* is provided, evaluates ``log_prob(action)`` instead of
    sampling (used for PPO update epochs on stored actions).

    For continuous actions, the *action* argument must be the **raw
    (pre-tanh)** action, because the distribution is defined over raw
    actions. The returned ``action`` is always the squashed action (for
    environment stepping), but during update we re-evaluate on raw.

    Args:
        obs:           (B, obs_dim) observation tensor.
        action:        Optional (B,) int64 (discrete) or
                       (B, action_dim) float (continuous, RAW pre-tanh).
                       If ``None``, a new action is sampled.
        deterministic: If ``True``, return the mode instead of a sample.

    Returns:
        action:       (B,) int64 (discrete) or (B, action_dim) float
                      (continuous, tanh-squashed to [-1, 1]) for env.
        raw_action:   (B, action_dim) float (continuous, RAW pre-tanh).
                      None for discrete. Used for storing in rollout.
        log_prob:     (B,) log π(a|s).
        entropy:      (B,) policy entropy.
        value:        (B,) V(s).
    """
    # obs: (B, obs_dim)
    h = self.backbone(obs)  # (B, hidden_dims[-1])
    dist = self.actor(h)  # Categorical or Normal distribution
    value = self.critic(h)  # (B,)

    raw_action = None
    if action is not None:
      if self.discrete:
        # action: (B,) int64
        log_prob = dist.log_prob(action)  # (B,)
        entropy = dist.entropy()  # (B,)
      else:
        # action is RAW (pre-tanh); re-evaluate log_prob with tanh correction.
        # action: (B, action_dim) -- raw action from rollout buffer
        raw_action = action
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
        log_prob -= torch.log(1.0 - torch.tanh(action).pow(2) + 1e-6).sum(
            dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)
    else:
      if self.discrete:
        sampled = dist.sample()  # (B,) int64
        log_prob = dist.log_prob(sampled)  # (B,)
        entropy = dist.entropy()  # (B,)
        action = sampled
        raw_action = None
      else:
        raw = dist.rsample()  # (B, action_dim) unbounded
        squashed = torch.tanh(raw)  # (B, action_dim) in [-1, 1]
        log_prob = dist.log_prob(raw).sum(dim=-1)  # (B,)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)
        action = squashed
        raw_action = raw

    return action, raw_action, log_prob, entropy, value


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
