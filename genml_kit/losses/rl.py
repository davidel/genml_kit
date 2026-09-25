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
    terminated=None,
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
    terminated: optional (B,) float tensor — true MDP end flags.  When
                given, ``(1 − terminated)`` is used as the bootstrap mask
                instead of ``(1 − dones)``.

  Returns:
    (B,) tensor of TD targets.
  """
  gamma_n = gamma**n_step

  # Bootstrap mask: use the true-termination flag when available, so a
  # *truncated* step (dones=1 but terminated=0) still bootstraps.
  mask = 1.0 - dones if terminated is None else 1.0 - terminated

  with torch.no_grad():
    # (B, n_actions)
    next_q_target = target_net(next_obs)

    if double_q:
      # (B, n_actions)
      next_q_online = policy_net(next_obs)
      # (B,)
      best_actions = next_q_online.argmax(dim=-1)
      # (B,)
      next_q_values = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    else:
      # (B,)
      next_q_values = next_q_target.max(dim=-1).values

    td = rewards + gamma_n * mask * next_q_values

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


def gae(rewards, values, next_values, dones, gamma, lam, terminated=None):
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
    terminated:  optional (T,) float tensor — true MDP end flags.  When
                 given, ``(1 − terminated)`` is used as the bootstrap mask
                 instead of ``(1 − dones)``.

  Returns:
    advantages: (T,) GAE advantage estimates.
    returns:    (T,) discounted returns (targets for the value head).
  """
  rewards = rewards.squeeze(-1) if rewards.dim() == 2 else rewards
  values = values.squeeze(-1) if values.dim() == 2 else values
  next_values = (next_values.squeeze(-1) if next_values.dim() == 2 else next_values)
  dones = dones.squeeze(-1) if dones.dim() == 2 else dones
  if terminated is not None:
    terminated = (terminated.squeeze(-1) if terminated.dim() == 2 else terminated)

  T = rewards.shape[0]
  advantages = torch.zeros_like(rewards)
  last_gae = 0.0

  # Bootstrap mask: use the true-termination flag when available, so a
  # *truncated* step (dones=1 but terminated=0) still bootstraps.
  mask = 1.0 - dones if terminated is None else 1.0 - terminated

  for t in reversed(range(T)):
    non_terminal = mask[t]
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


def entropy_bonus(entropy):
  """Mean policy entropy, for the PPO entropy bonus term.

  The combined PPO loss *subtracts* the entropy bonus::

      loss = pg_loss + vf_coef * value_loss - entropy_coef * ent

  so ``ent`` must be the **positive** mean entropy::

      d(loss)/d(entropy) = -entropy_coef  (< 0)

  i.e. maximising the entropy reduces the loss (standard PPO
  convention, Schulman et al. 2017).

  **IMPORTANT:** pass the *distribution entropy* (from ``dist.entropy()``,
  e.g. the ``entropy`` value returned by ``ActorCritic.get_action_and_value``),
  *not* ``-log_prob`` of sampled actions.  For a Gaussian, ``log_prob`` is
  negative and its magnitude grows for low-variance policies, so
  ``-log_prob.mean()`` is *not* the policy entropy and inverts the
  gradient direction -- the historical bug that collapsed ``log_std`` in
  this codebase (logged ``entropy`` went ``-1.3 -> +18.5`` nats on
  Pendulum-v1 while the policy became almost deterministic, ``ratio ->
  0``, ``pg_loss`` stuck at the clip floor).

  Args:
    entropy: (B,) or (B, 1) per-sample policy entropy (>= 0 for
      discrete distributions; can be small or slightly negative for
      low-variance continuous distributions).

  Returns:
    Scalar ``entropy.mean()`` -- subtracted (via ``-entropy_coef * ent``)
    from the combined loss to *maximise* the policy entropy.
  """
  if entropy.dim() == 2:
    entropy = entropy.squeeze(-1)
  return entropy.mean()


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


def sac_alpha_loss(log_probs, target_entropy, coef=None):
  """Auto-tuning temperature loss (§13.4 of ``rl/README.md``).

  L = coef · (−E_{a∼π}[log π(a|s) + H*])

  ``coef`` is the quantity optimised by the temperature optimizer.  SAC
  parameterises the temperature as ``alpha = exp(log_alpha)`` and must
  therefore pass **``log_alpha``** here, *not* ``alpha``: because the
  gradient ``∂L/∂log_alpha`` is then proportional to the base loss rather
  than to ``alpha`` itself, the update step size does not shrink to zero
  as ``alpha → 0`` and the temperature cannot collapse to a spurious
  ``alpha ≈ 0`` fixed point (cf. Stable-Baselines3's ``log_ent_coef``).

  ``target_entropy`` is typically −dim(A) for continuous actions.

  Args:
    log_probs:     (B,) log-probabilities.
    target_entropy: scalar target entropy (negative).
    coef:          coefficient on the base loss (pass ``log_alpha`` for
                   SAC auto-tuning).  When ``None`` the raw base loss is
                   returned.

  Returns:
    Scalar loss (to be minimised by the alpha optimizer).
  """
  base_loss = -(log_probs + target_entropy).mean()
  if coef is not None:
    return coef * base_loss
  return base_loss
