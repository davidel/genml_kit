"""Reinforcement-learning loss functions (TD target + TD error).

Pure-torch, batched, reduction-aware — mirrors the style of
``genml_kit.losses.focal``.  All formulas reference ``rl/README.md``.
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
    rewards:  (B,) tensor of immediate rewards.
    next_obs: (B, …) next-state observations.
    dones:    (B,) float tensor (0.0 or 1.0) — episode-terminal flags.
    policy_net:  online Q-network (``QNetwork.online``).
    target_net:  target Q-network (``QNetwork.target``).
    gamma:    discount factor ∈ (0, 1].
    double_q: if True, use Double-DQN action selection.
    n_step:   number of lookahead steps for the multi-step return.

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
