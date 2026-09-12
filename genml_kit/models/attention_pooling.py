"""CLS-guided attention pooling over spatial tokens."""

import torch.nn as nn


class CLSGuidedAttentionPooling(nn.Module):
  """Use CLS token(s) as queries to attention-weight the spatial tokens.

  Given a sequence of transformer outputs split into CLS token(s) and
  spatial tokens, this module cross-attends from each CLS token to the
  spatial tokens, producing one pooled representation per query token.
  The single-token case collapses back to a ``[B, D]`` vector, keeping
  the historical behavior.

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
    cls_out:     [B, K, D]     — final CLS token(s) from the transformer
    spatial_out: [B, N, D]     — final spatial tokens from the transformer
    Returns:     [B, D]        — single query: attention-weighted pooling
                 [B, K, D]     — K queries: one pooled vector per CLS token
    """
    attn_out, _ = self.cross_attn(
        query=self.norm_cls(cls_out),
        key=self.norm_spatial(spatial_out),
        value=spatial_out,
    )
    if attn_out.shape[1] == 1:
      return attn_out.squeeze(1)  # [B, D] (single-CLS backward compat)
    return attn_out  # [B, K, D]
