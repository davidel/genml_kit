"""CLS-guided attention pooling over spatial tokens."""

import torch.nn as nn


class CLSGuidedAttentionPooling(nn.Module):
  """Use the final CLS token as a query to attention-weight the spatial tokens.

  Given a sequence of transformer outputs split into a CLS token and
  spatial tokens, this module cross-attends from CLS to spatial to
  produce a single pooled representation.

  Parameters
  ----------
  embed_dim : int
      Dimensionality of each token.
  num_heads : int
      Number of attention heads.
  dropout : float
      Dropout applied inside the multi-head attention.
  """

  def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
    super().__init__()
    self.norm_cls = nn.LayerNorm(embed_dim)
    self.norm_spatial = nn.LayerNorm(embed_dim)
    self.cross_attn = nn.MultiheadAttention(embed_dim,
                                            num_heads,
                                            dropout=dropout,
                                            batch_first=True)

  def forward(self, cls_out, spatial_out):
    """
    cls_out:     [B, 1, D]     — final CLS token from transformer
    spatial_out: [B, N, D]     — final spatial tokens from transformer
    Returns:     [B, D]        — attention-weighted pooling
    """
    attn_out, _ = self.cross_attn(
        query=self.norm_cls(cls_out),
        key=self.norm_spatial(spatial_out),
        value=spatial_out,
    )
    return attn_out.squeeze(1)  # [B, D]
