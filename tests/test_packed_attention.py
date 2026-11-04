"""Tests for genml_kit/training/packed_attention.py.

Covers the two model-agnostic artefacts that keep documents isolated in a
packed window: the ``cu_seqlens`` offsets for variable-length attention and the
4-D block-diagonal additive mask.  Everything runs on CPU; the flash kernel
itself is optional and untested here.
"""

import pytest
import torch

from genml_kit.training.packed_attention import (build_block_diagonal_mask,
                                                 flash_varlen_available,
                                                 flash_varlen_layout,
                                                 segments_to_cu_seqlens)


class TestCuSeqlens:

  def test_two_documents(self):
    # Flat layout: 3 tokens of doc 0 followed by 2 of doc 1.
    seg = torch.tensor([0, 0, 0, 1, 1])
    cu, max_len = segments_to_cu_seqlens(seg)
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 3, 5]
    assert max_len == 3

  def test_single_document(self):
    cu, max_len = segments_to_cu_seqlens(torch.tensor([0, 0, 0, 0]))
    assert cu.tolist() == [0, 4]
    assert max_len == 4

  def test_requires_1d(self):
    with pytest.raises(ValueError, match="1-D"):
      segments_to_cu_seqlens(torch.zeros(2, 3, dtype=torch.long))


class TestBlockDiagonalMask:

  def test_blocks_across_documents(self):
    # Batch of 1, 4 positions: docs [0, 0, 1, 1].
    seg = torch.tensor([[0, 0, 1, 1]])
    mask = build_block_diagonal_mask(seg)
    assert mask.shape == (1, 1, 4, 4)
    allowed = (mask == 0.0)[0, 0]
    # Position 2 (start of doc 1) must NOT see positions 0..1 (doc 0).
    assert not allowed[2, 0] and not allowed[2, 1]
    # It sees itself but not the later token of its own document (still causal).
    assert allowed[2, 2]
    assert not allowed[2, 3]
    # Causal within doc 0: position 1 sees 0 but not 2 (different doc anyway).
    assert allowed[1, 0]
    assert not allowed[0, 1]
    # Position 3 (doc 1) sees both of doc 1's prefixes but nothing of doc 0.
    assert allowed[3, 2] and not allowed[3, 0] and not allowed[3, 1]

  def test_attention_matches_separate_documents(self):
    """Packed attention output == attending each document in isolation."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 5)  # [B, H, L, D]
    k = torch.randn(1, 2, 4, 5)
    v = torch.randn(1, 2, 4, 5)
    seg = torch.tensor([[0, 0, 1, 1]])  # two 2-token documents in one window

    packed = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=build_block_diagonal_mask(seg), dropout_p=0.0)
    # Reference: attend doc 0 (cols 0:2) and doc 1 (cols 2:4) independently.
    ref = packed.clone()
    for lo, hi in ((0, 2), (2, 4)):
      doc = torch.nn.functional.scaled_dot_product_attention(q[..., lo:hi, :],
                                                             k[..., lo:hi, :],
                                                             v[..., lo:hi, :],
                                                             is_causal=True,
                                                             dropout_p=0.0)
      ref[..., lo:hi, :] = doc
    assert torch.allclose(packed, ref, atol=1e-5)

  def test_padding_row_is_finite(self):
    seg = torch.tensor([[0, 0, -1, -1]])
    mask = build_block_diagonal_mask(seg)
    # No row may be all -inf (softmax would produce NaNs).
    assert torch.isfinite(mask).any(dim=-1).all()

  def test_requires_2d(self):
    with pytest.raises(ValueError, match=r"\[B, L\]"):
      build_block_diagonal_mask(torch.zeros(4, dtype=torch.long))


class TestFlashVarlenLayout:

  def test_flattens_and_counts_documents(self):
    # Batch of 1, two 2-token documents: [0, 0, 1, 1].
    mask = torch.ones(1, 4, dtype=torch.long)
    seg = torch.tensor([[0, 0, 1, 1]])
    valid, cu, max_len = flash_varlen_layout(mask, seg)
    assert valid.tolist() == [[True, True, True, True]]
    assert cu.dtype == torch.int32
    assert cu.tolist() == [0, 2, 4]
    assert max_len == 2

  def test_drops_padding(self):
    # Padding (segment -1 / mask 0) is removed and does not open a document.
    mask = torch.tensor([[1, 1, 0, 1, 1]])
    seg = torch.tensor([[0, 0, -1, 1, 1]])
    valid, cu, max_len = flash_varlen_layout(mask, seg)
    assert valid.tolist() == [[True, True, False, True, True]]
    assert cu.tolist() == [0, 2, 4]
    assert max_len == 2

  def test_batch_offsets_stay_contiguous(self):
    # Two rows, each a single 3-token document: ids must not collide at the row
    # boundary, so the flattened runs total 6 tokens over 2 documents.
    mask = torch.ones(2, 3, dtype=torch.long)
    seg = torch.tensor([[0, 0, 0], [0, 0, 0]])
    valid, cu, max_len = flash_varlen_layout(mask, seg)
    assert valid.all()
    assert cu.tolist() == [0, 3, 6]
    assert max_len == 3

  def test_requires_2d(self):
    with pytest.raises(ValueError, match=r"\[B, L\]"):
      flash_varlen_layout(None, torch.zeros(4, dtype=torch.long))


class TestFlashAvailability:

  def test_probe_is_bool(self):
    # Flash is optional; the probe must never raise regardless of install.
    assert isinstance(flash_varlen_available(), bool)
