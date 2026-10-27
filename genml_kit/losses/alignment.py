"""Alignment losses: DPO and the Bradley-Terry reward objective.

Two pure objectives plus the shared log-prob helper they build on:

- :func:`sequence_logps` -- mean per-token log-probability of a sequence under
  a causal LM, over the non-ignored (response) positions.  Used for *both* the
  DPO policy and its frozen reference, so their aggregation is identical by
  construction.
- :func:`dpo_loss` -- the sigmoid DPO objective (Rafailov et al. 2023): the
  policy's implicit reward margin over the reference, pushed to favour the
  chosen response.
- :func:`bradley_terry_loss` -- the pairwise ranking loss for a scalar reward
  model: ``-log sigmoid(score_chosen - score_rejected)``.

Each is a small function, not an ``nn.Module``; the methods bind the
hyperparameters.  Registered in ``LOSSES`` as ``dpo`` / ``bradley_terry`` so an
external script can override either via ``load_loss``.
"""

import torch.nn.functional as F

# Matches training.text_collator.IGNORE_INDEX (CrossEntropyLoss' default).
IGNORE_INDEX = -100


def sequence_logps(logits, labels, ignore_index=IGNORE_INDEX):
  """Mean per-token log-probability of each sequence's response tokens.

    Args:
        logits: ``(B, T, V)`` causal-LM logits.
        labels: ``(B, T)`` target ids; positions equal to *ignore_index*
            (prompt tokens, padding) are excluded.
        ignore_index: label value to skip (default ``-100``).

    Returns:
        ``(B,)`` tensor: the mean log-prob over the counted tokens.  A
        sequence with no counted tokens yields ``0.0`` (it contributes no
        gradient and no score).
    """
  # Shift is done by the caller (logits[:, :-1], labels[:, 1:]) so the same
  # helper serves the standard next-token alignment.
  log_probs = F.log_softmax(logits, dim=-1)
  safe_labels = labels.clamp(min=0)
  gathered = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
  mask = (labels != ignore_index).to(gathered.dtype)
  token_logps = gathered * mask
  counts = mask.sum(dim=-1).clamp(min=1.0)
  return token_logps.sum(dim=-1) / counts


def dpo_loss(policy_chosen_logps,
             policy_rejected_logps,
             ref_chosen_logps,
             ref_rejected_logps,
             beta=0.1):
  """Sigmoid DPO loss and the implicit reward margins.

    The implicit reward of a response is ``beta * (log pi - log pi_ref)``; the
    loss drives the chosen margin above the rejected one.

    Args:
        policy_chosen_logps: ``(B,)`` mean log-prob of the chosen response
            under the policy.
        policy_rejected_logps: ``(B,)`` same, rejected response.
        ref_chosen_logps: ``(B,)`` chosen response under the frozen reference.
        ref_rejected_logps: ``(B,)`` rejected response under the reference.
        beta: KL-strength temperature.

    Returns:
        ``(loss, chosen_rewards, rejected_rewards)`` where *loss* is a scalar
        and the rewards are ``(B,)`` tensors (detached-friendly).
    """
  chosen_logratio = policy_chosen_logps - ref_chosen_logps
  rejected_logratio = policy_rejected_logps - ref_rejected_logps
  logits = beta * (chosen_logratio - rejected_logratio)
  loss = -F.logsigmoid(logits).mean()
  chosen_rewards = beta * chosen_logratio
  rejected_rewards = beta * rejected_logratio
  return loss, chosen_rewards.detach(), rejected_rewards.detach()


def bradley_terry_loss(chosen_scores, rejected_scores):
  """Pairwise ranking loss for a scalar reward model.

    ``-log sigmoid(score_chosen - score_rejected)``, averaged over the batch.
    A margin of ``1.0`` between the two scores gives near-zero loss.

    Args:
        chosen_scores: ``(B,)`` scalar reward for the preferred responses.
        rejected_scores: ``(B,)`` scalar reward for the dispreferred ones.

    Returns:
        A scalar loss tensor.
    """
  return -F.logsigmoid(chosen_scores - rejected_scores).mean()
