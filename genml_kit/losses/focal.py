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
    if targets.dim() == 1:
      # Integer targets → one-hot with optional label smoothing.
      target_dist = torch.zeros_like(logits)
      if self.label_smoothing > 0:
        target_dist.fill_(self.label_smoothing / (num_classes - 1))
      target_dist.scatter_(1, targets.unsqueeze(1), 1.0)
    else:
      # Soft targets already supplied (e.g. from Mixup).
      target_dist = targets

    # 2. Apply class weights.
    weighted_targets = target_dist * self.weights.unsqueeze(0)

    # 3. Base cross-entropy via log_softmax (numerically stable).
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Weighted CE: sum_k  -t_k * log(p_k)
    base_ce_loss = -(weighted_targets * log_probs).sum(dim=-1)

    # 4. Compute unified expected true probability p_t.
    p_t = torch.sum(target_dist * probs, dim=-1)

    # 5. Modulate the entire loss once per sample.
    focal_weight = (1.0 - p_t)**self.gamma
    loss = focal_weight * base_ce_loss

    if self.reduction == "mean":
      return loss.mean()
    elif self.reduction == "sum":
      return loss.sum()
    return loss
