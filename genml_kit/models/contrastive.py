"""ContrastiveEncoder — backbone + MLP projection head.

Used by supervised contrastive pre-training.  The projection head
is discarded after pre-training; the backbone is kept for downstream
fine-tuning.
"""

import torch.nn as nn

from genml_kit.models.encoder_utils import detect_backbone_dim, encode_with_backbone


class ProjectionHead(nn.Module):
  """Two-layer MLP projection head.

  Architecture: Linear -> BatchNorm -> ReLU -> Linear -> BatchNorm.
  """

  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
        nn.BatchNorm1d(out_dim),
    )

  def forward(self, x):
    return self.net(x)


class ContrastiveEncoder(nn.Module):
  """Backbone encoder with a projection head for contrastive learning.

  During pre-training, ``forward()`` projects backbone features through
  the MLP head.  ``encode()`` returns raw backbone features for
  downstream use.

  Parameters
  ----------
  encoder : nn.Module
      Any registered backbone (ConvViT, HF model, timm, etc.).
  proj_dim : int
      Output dimensionality of the projection head.
  proj_hidden : int
      Hidden dimensionality of the projection MLP.
  backbone_dim : int, optional
      Explicit backbone output dimension.  When ``None``, inferred
      automatically from ``config.hidden_size`` / ``config.d_model``
      / ``classifier.feat_dim``.
  """

  def __init__(self, encoder, proj_dim=256, proj_hidden=2048, backbone_dim=None):
    super().__init__()
    self.encoder = encoder
    self._proj_dim = proj_dim
    self._proj_hidden = proj_hidden
    self._backbone_dim = (backbone_dim
                          if backbone_dim is not None else detect_backbone_dim(encoder))
    self.projection = ProjectionHead(
        self._backbone_dim,
        proj_hidden,
        proj_dim,
    )

  def encode(self, images):
    """Extract raw backbone features ``(B, D)``."""
    return encode_with_backbone(self.encoder, images)

  def forward(self, images):
    """Project backbone features through the MLP head ``(B, proj_dim)``."""
    features = self.encode(images)
    return self.projection(features)
