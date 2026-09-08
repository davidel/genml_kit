"""BYOL module — online + target networks for Bootstrap Your Own Latent."""

import copy

import torch
import torch.nn as nn

from genml_kit.models.contrastive import ProjectionHead
from genml_kit.models.encoder_utils import detect_backbone_dim, encode_with_backbone


class BYOL(nn.Module):
  """BYOL: online network (encoder + projector + predictor) and target
  network (encoder + projector, EMA copy).

  Args:
      encoder: Backbone encoder (e.g. timm, HF model).
      proj_dim: Output dimension of the projection head.
      proj_hidden: Hidden dimension of the projection head.
      predictor_hidden: Hidden dimension of the predictor MLP.
      backbone_dim: Explicit backbone output dimension.  Inferred if None.
  """

  def __init__(self,
               encoder,
               proj_dim=256,
               proj_hidden=2048,
               predictor_hidden=2048,
               backbone_dim=None):
    super().__init__()
    self.online_encoder = _EncoderWithProjection(encoder, proj_dim, proj_hidden,
                                                 backbone_dim)
    self.online_predictor = _PredictorMLP(proj_dim, predictor_hidden, proj_dim)
    self.target_encoder = copy.deepcopy(self.online_encoder)
    for p in self.target_encoder.parameters():
      p.requires_grad = False

  @torch.no_grad()
  def update_momentum(self, momentum):
    """EMA update of target encoder from online encoder."""
    for op, tp in zip(self.online_encoder.parameters(),
                      self.target_encoder.parameters()):
      tp.data.mul_(momentum).add_(op.data, alpha=1.0 - momentum)

  def forward(self, images):
    """Compute BYOL loss from two augmented views.

    Args:
        images: Tuple of two tensors ``(v1, v2)``, each ``(B, C, H, W)``.

    Returns:
        (loss, info_dict)
    """
    v1, v2 = images

    p1 = self.online_predictor(self.online_encoder(v1))
    p2 = self.online_predictor(self.online_encoder(v2))

    with torch.no_grad():
      z1 = self.target_encoder(v1)
      z2 = self.target_encoder(v2)

    from genml_kit.pretrain.losses.byol import byol_loss
    loss = byol_loss(p1, z2, p2, z1)
    return loss, {"loss": loss.item()}


class _EncoderWithProjection(nn.Module):
  """Backbone + projection head."""

  def __init__(self, encoder, proj_dim, proj_hidden, backbone_dim=None):
    super().__init__()
    self.encoder = encoder
    if backbone_dim is None:
      backbone_dim = detect_backbone_dim(encoder)
    self.projection = ProjectionHead(backbone_dim, proj_hidden, proj_dim)

  def encode(self, images):
    return encode_with_backbone(self.encoder, images)

  def forward(self, images):
    return self.projection(self.encode(images))


class _PredictorMLP(nn.Module):
  """Two-layer MLP predictor."""

  def __init__(self, in_dim, hidden_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim),
    )

  def forward(self, x):
    return self.net(x)
