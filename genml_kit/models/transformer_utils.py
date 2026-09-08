"""Shared transformer building blocks for custom ViT models.

Provides DropPath (Stochastic Depth), SwiGLU FFN, and a pre-norm
TransformerBlock used by both ConvViT and UVito.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob=0.0, training=False):
  """Stochastic Depth implementation."""
  if drop_prob == 0.0 or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor.floor_()
  output = x.div(keep_prob) * random_tensor
  return output


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample."""

  def __init__(self, drop_prob=0.0):
    super().__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)


class SwiGLUFFN(nn.Module):
  """SwiGLU feed-forward network."""

  def __init__(self, in_features, hidden_features, dropout=0.0):
    super().__init__()
    self.w12 = nn.Linear(in_features, 2 * hidden_features)
    self.w3 = nn.Linear(hidden_features, in_features)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    x12 = self.w12(x)
    x1, x2 = x12.chunk(2, dim=-1)
    return self.dropout(self.w3(F.silu(x1) * x2))


class TransformerBlock(nn.Module):
  """Pre-norm Transformer block with DropPath.

  Structure::

      x = x + drop_path(attn(drop_norm1(x)))
      x = x + drop_path(ffn(drop_norm2(x)))
  """

  def __init__(
      self,
      embed_dim,
      num_heads,
      dropout=0.0,
      drop_path=0.0,
      dim_feedforward=None,
  ):
    super().__init__()
    if dim_feedforward is None:
      dim_feedforward = embed_dim * 4

    self.ln1 = nn.LayerNorm(embed_dim)
    self.self_attn = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self.ln2 = nn.LayerNorm(embed_dim)
    self.ffn = SwiGLUFFN(embed_dim, dim_feedforward, dropout=dropout)
    self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

  def forward(self, x):
    attn_out, _ = self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))
    x = x + self.drop_path(attn_out)
    x = x + self.drop_path(self.ffn(self.ln2(x)))
    return x
