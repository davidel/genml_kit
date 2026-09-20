"""Reinforcement-learning loss functions.

Covers DQN (TD target + TD error), PPO (GAE + clipped surrogate +
value loss + entropy bonus), and SAC (soft Q-learning + policy
gradient + alpha loss).  Pure-torch, batched, reduction-aware.
All formulas reference ``rl/README.md``.
"""

import torch
import torch.nn.functional as F


def td_target(
    rewards,
    next_obs,
    dones,
    policy_net,
    target_net,
    gamma,
    double_q=True,
    n_step=1,
):
  """Compute the n-step Bellman TD target for Q-learning.

  Implements (see ``rl/README.md`` §3.3):

    y^(n) = r^(n) + γ^n (1 − d) Q_target(s′, argmax_a Q_policy(s′, a))
                                       ↑ Double DQN

  Falls back to vanilla DQN when *double_q* is ``False``:

    y^(n) = r^(n) + γ^n (1 − d) max_a Q_target(s′, a)

  Args:
    rewards:    (B,) tensor of immediate rewards.
    next_obs:   (B, …) next-state observations.
    dones:      (B,) float tensor (0.0 or 1.0) — episode-terminal flags.
    policy_net: online Q-network (``QNetwork.online``).
    target_net: target Q-network (``QNetwork.target``).
    gamma:      discount factor ∈ (0, 1].
    double_q:   if True, use Double-DQN action selection.
    n_step:     number of lookahead steps for the multi-step return.

  Returns:
    (B,) tensor of TD targets.
  """
  gamma_n = gamma**n_step

  with torch.no_grad():
    next_q_target = target_net(next_obs)  # (B, n_actions)

    if double_q:
      next_q_online = policy_net(next_obs)  # (B, n_actions)
      best_actions = next_q_online.argmax(dim=-1)  # (B,)
      next_q_values = next_q_target.gather(1,
                                           best_actions.unsqueeze(1)).squeeze(1)  # (B,)
    else:
      next_q_values = next_q_target.max(dim=-1).values  # (B,)

    td = rewards + gamma_n * (1.0 - dones) * next_q_values

  return td


def td_loss(pred_q, target, reduction="mean"):
  """Compute the TD error (one-step Bellman residual).

  Uses Huber (SmoothL1) loss by default, matching the standard DQN
  recipe.  Set ``reduction="none"`` to get per-element losses.

  Args:
    pred_q:    (B,) predicted Q-values for the taken actions.
    target:    (B,) TD targets (from :func:`td_target`).
    reduction: ``"mean"`` | ``"sum"`` | ``"none"``.

  Returns:
    Scalar loss (or (B,) tensor with ``reduction="none"``).
  """
  return F.smooth_l1_loss(pred_q, target, reduction=reduction)


def gae(rewards, values, next_values, dones, gamma, lam):
  """Compute Generalized Advantage Estimation (GAE).

  Implements the backward recurrence from ``rl/README.md`` §10.3:

    A_t = δ_t + γ λ (1 − d_t) A_{t+1}

  where δ_t = r_t + γ (1 − d_t) V(s_{t+1}) − V(s_t).

  All inputs are (T,) or (T, 1) tensors over a single rollout
  (T = rollout length).

  Args:
    rewards:     (T,) immediate rewards.
    values:      (T,) value estimates V(s_t).
    next_values: (T,) value estimates V(s_{t+1}) (bootstrapped from
                 the critic at the next state; zero-padded at episode
                 boundaries).
    dones:       (T,) float flags (0.0 / 1.0).
    gamma:       discount factor.
    lam:         GAE lambda (typically 0.95).

  Returns:
    advantages: (T,) GAE advantage estimates.
    returns:    (T,) discounted returns (targets for the value head).
  """
  rewards = rewards.squeeze(-1) if rewards.dim() == 2 else rewards
  values = values.squeeze(-1) if values.dim() == 2 else values
  next_values = (next_values.squeeze(-1) if next_values.dim() == 2 else next_values)
  dones = dones.squeeze(-1) if dones.dim() == 2 else dones

  T = rewards.shape[0]
  advantages = torch.zeros_like(rewards)
  last_gae = 0.0

  for t in reversed(range(T)):
    non_terminal = 1.0 - dones[t]
    delta = rewards[t] + gamma * next_values[t] * non_terminal - values[t]
    last_gae = delta + gamma * lam * non_terminal * last_gae
    advantages[t] = last_gae

  returns = advantages + values
  return advantages, returns


def clipped_surrogate(ratio, advantages, clip_eps):
  """PPO clipped surrogate objective (§11.3 of ``rl/README.md``).

  L_CLIP = E[min(r_t * A_t, clip(r_t, 1−ε, 1+ε) * A_t)]

  Args:
    ratio:      (B,) π_θ(a|s) / π_θ_old(a|s).
    advantages: (B,) GAE advantages.
    clip_eps:   clipping threshold ε (typically 0.2).

  Returns:
    Scalar (negated — we *maximise* the surrogate, so the loss is
    its negative).
  """
  unclipped = ratio * advantages
  clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
  return -torch.min(unclipped, clipped).mean()


def value_loss(pred_v, returns, old_values=None, clip_eps=None):
  """Value function loss for PPO (§11.4 of ``rl/README.md``).

  When *clip_eps* is ``None``, uses simple MSE.  When provided, uses
  the PPO clipped value loss:

    L_V = max((V - R)², (V_clip - R)²)

  Args:
    pred_v:     (B,) predicted value V(s).
    returns:    (B,) discounted returns.
    old_values: (B,) value predictions from the old policy (required
                when *clip_eps* is not ``None``).
    clip_eps:   clipping threshold (or ``None`` for unclipped).

  Returns:
    Scalar loss.
  """
  if clip_eps is None or old_values is None:
    return F.mse_loss(pred_v, returns)

  unclipped = (pred_v - returns)**2
  clipped = (old_values + torch.clamp(pred_v - old_values, -clip_eps, clip_eps) -
             returns)**2
  return torch.max(unclipped, clipped).mean()


def entropy_bonus(log_probs):
  """Entropy bonus for PPO: H(π) = -E_a[log π(a|s)].

  Args:
    log_probs: (B,) or (B, 1) log-probabilities of the taken actions.

  Returns:
    Scalar (to be *maximised* — returned as a positive value).
  """
  if log_probs.dim() == 2:
    log_probs = log_probs.squeeze(-1)
  return -log_probs.mean()


def sac_q_loss(q_pred, soft_target):
  """Soft Q-learning loss for one critic: E[(Q(s,a) − y)²].

  ``soft_target`` = r + γ(1−d)(min_j Q_target_j(s',a') − α log π(a'|s')).

  Args:
    q_pred:      (B,) predicted Q-values from one critic.
    soft_target: (B,) soft Bellman targets.

  Returns:
    Scalar MSE loss.
  """
  return F.mse_loss(q_pred, soft_target)


def sac_policy_loss(log_probs, q_values, alpha):
  """SAC policy gradient loss (§13.2 of ``rl/README.md``).

  L_π = E_{a∼π}[α log π(a|s) − Q(s,a)]

  Minimised by the actor.

  Args:
    log_probs: (B,) log-probabilities of sampled actions.
    q_values:  (B,) Q-values for those actions.
    alpha:     temperature scalar (or 0-dim tensor).

  Returns:
    Scalar loss (to be minimised by the actor).
  """
  return (alpha * log_probs - q_values).mean()


def sac_alpha_loss(log_probs, target_entropy, alpha=None):
  """Auto-tuning temperature loss (§13.4 of ``rl/README.md``).

  L_α = −E_{a∼π}[α (log π(a|s) + H*)]

  ``target_entropy`` is typically −dim(A) for continuous actions.

  Args:
    log_probs:     (B,) log-probabilities.
    target_entropy: scalar target entropy (negative).
    alpha:         temperature parameter (if provided, loss includes alpha factor).

  Returns:
    Scalar loss (to be minimised by the alpha optimizer).
  """
  base_loss = -(log_probs + target_entropy).mean()
  if alpha is not None:
    return alpha * base_loss
  return base_loss
