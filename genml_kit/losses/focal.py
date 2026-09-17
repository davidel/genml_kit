"""Mathematically rigorous cost-sensitive focal loss for soft targets.

Extracted from ``train.py`` to reduce module size and allow reuse in
other training scripts.
"""

import torch
import torch.nn as nn


class CombinedFocalLoss(nn.Module):
  """Mathematically rigorous cost-sensitive focal loss for soft targets.

  Aligns with Lin et al. (2017) by computing a unified p_t as the
  expected probability under the true distribution vector.  Supports
  both integer labels and continuous soft-target vectors (Mixup).
  """

  def __init__(
      self,
      weights,
      gamma=2.0,
      label_smoothing=0.0,
      reduction="mean",
  ):
    super().__init__()
    self.register_buffer("weights", weights)
    self.gamma = gamma
    self.label_smoothing = label_smoothing
    self.reduction = reduction

  def forward(self, logits, targets):
    """Compute cost-sensitive focal loss.

    Args:
        logits: [B, C] raw model outputs (before softmax).
        targets: [B] integer class indices, or [B, C] soft-target
            probability distributions (e.g. from Mixup).

    Returns:
        Scalar loss (or per-sample losses if reduction='none').
    """
    num_classes = logits.size(-1)

    # 1. Build the target distribution vector.
    # Integer targets (B,) -> one-hot (B, C) with optional label smoothing;
    # continuous soft targets (B, C) are kept as-is.
    if targets.dim() == 1:
      target_dist = torch.zeros_like(logits)
      if self.label_smoothing > 0:
        target_dist.fill_(self.label_smoothing / (num_classes - 1))
      target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
    else:
      # Soft targets already supplied (e.g. from Mixup).
      target_dist = targets

    # 2. Apply per-class cost weights: (B, C) * (1, C) -> (B, C).
    weighted_targets = target_dist * self.weights.unsqueeze(0)

    # 3. Numerically stable log-softmax and its exponential: (B, C) -> (B, C).
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Weighted CE per sample: -sum_k t_k * w_k * log(p_k): (B, C) -> (B,).
    base_ce_loss = -(weighted_targets * log_probs).sum(dim=-1)

    # 4. Expected true probability under the target distribution: (B, C) -> (B,).
    p_t = torch.sum(target_dist * probs, dim=-1)

    # 5. Modulate the per-sample loss once: (B,) * (B,) -> (B,).
    focal_weight = (1.0 - p_t)**self.gamma
    loss = focal_weight * base_ce_loss

    # Reduce per-sample loss (B,) -> scalar (or keep (B,) with 'none').
    if self.reduction == "mean":
      return loss.mean()
    elif self.reduction == "sum":
      return loss.sum()
    return loss
