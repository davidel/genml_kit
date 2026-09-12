"""Tests for CLS-guided attention pooling (single- and multi-query)."""

import torch

from genml_kit.models.attention_pooling import CLSGuidedAttentionPooling


def _pool(embed_dim=16, num_heads=4, num_queries=1, num_spatial=8, batch=2):
  pooling = CLSGuidedAttentionPooling(embed_dim=embed_dim,
                                      num_heads=num_heads,
                                      dropout=0.0)
  cls_out = torch.randn(batch, num_queries, embed_dim)
  spatial_out = torch.randn(batch, num_spatial, embed_dim)
  return pooling, cls_out, spatial_out


def test_single_query_returns_2d():
  """Single CLS token keeps the historical [B, D] output."""
  pooling, cls_out, spatial_out = _pool(num_queries=1)
  out = pooling(cls_out, spatial_out)
  assert out.shape == (2, 16)


def test_multi_query_returns_3d():
  """Multiple CLS tokens each produce their own pooled vector."""
  pooling, cls_out, spatial_out = _pool(num_queries=3)
  out = pooling(cls_out, spatial_out)
  assert out.shape == (2, 3, 16)


def test_multi_query_outputs_differ_per_token():
  """Different CLS queries attend the spatial tokens differently."""
  pooling, cls_out, spatial_out = _pool(num_queries=3)
  out = pooling(cls_out, spatial_out)
  # Distinct queries must not collapse to the same pooled representation.
  assert not torch.allclose(out[:, 0, :], out[:, 1, :], atol=1e-6)
  assert not torch.allclose(out[:, 1, :], out[:, 2, :], atol=1e-6)
