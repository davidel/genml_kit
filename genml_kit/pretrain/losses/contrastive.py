"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

Computes NT-Xent loss where same-class pairs are positives and
all other pairs are negatives.
"""

import torch
import torch.nn.functional as F


def supcon_loss(features, labels, temperature=0.07):
  """Supervised contrastive loss.

  Args:
      features: ``(B, D)`` projection features (will be L2-normalized).
      labels: ``(B,)`` integer class labels.
      temperature: Temperature scaling factor.

  Returns:
      Scalar loss tensor.
  """
  device = features.device
  labels = labels.contiguous().view(-1, 1)

  # (B, B) boolean mask: True where two samples share a class.
  mask = torch.eq(labels, labels.T).float().to(device)

  # Pairwise cosine similarity, scaled by temperature.
  features = F.normalize(features, dim=1)
  logits = torch.mm(features, features.T) / temperature

  # Exclude self-pairs by masking the diagonal to a very negative value.
  # logsumexp treats exp(-1e9) ≈ 0, so self-pairs are excluded from the
  # normalization denominator without separate masking.
  diag_mask = torch.eye(logits.shape[0], device=device, dtype=torch.bool)
  logits.masked_fill_(diag_mask, -1e9)

  # log-prob = logit - logsumexp; logsumexp is numerically stable internally.
  log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

  # Mask out self-pairs from the positive mask as well.
  mask = mask * (~diag_mask).float()

  # Mean of log-prob over positive pairs for each anchor.
  n_positives = mask.sum(dim=1)
  # Guard against classes with only one sample (no positives).
  valid = n_positives > 0
  mean_log_prob = (mask * log_prob).sum(dim=1) / n_positives.clamp(min=1)

  if valid.sum() == 0:
    # No positive pairs exist — return zero loss (backward graph intact).
    return (logits * 0.0).sum()
  loss = -mean_log_prob[valid].mean()
  return loss
