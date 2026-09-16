"""BYOL loss (Grill et al., NeurIPS 2020).

Computes the symmetric cosine similarity regression loss between online
predictions and target projections.
"""

import torch.nn.functional as F


def byol_loss(p1, z2, p2, z1):
  """Symmetric BYOL regression loss.

  Args:
      p1: Online predictor output for view 1, ``(B, D)``.
      z2: Target projection for view 2, ``(B, D)`` (stop-gradient).
      p2: Online predictor output for view 2, ``(B, D)``.
      z1: Target projection for view 1, ``(B, D)`` (stop-gradient).

  Returns:
      Scalar loss tensor in [0, 4].  0 means perfect agreement.
  """
  loss = (2.0 - F.cosine_similarity(p1, z2, dim=1).mean() -
          F.cosine_similarity(p2, z1, dim=1).mean())
  return loss
