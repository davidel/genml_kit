"""Helpers for attention that respects document boundaries in packed batches.

When the text pipeline runs with ``--packing``, several documents share one
``max_seq_len`` window.  A plain causal mask would let tokens of a later
document attend to earlier documents in the same window.  This module builds
the two model-agnostic artefacts a model can use to prevent that:

- :func:`build_block_diagonal_mask` -- a 4-D additive ``[B, 1, L, L]`` mask
  that is causal *within* each document and fully blocked *across* documents.
  Fed to a model as ``attention_mask`` it reproduces padding semantics exactly
  (at the cost of a dense mask, so the flash kernel is not used).
- :func:`segments_to_cu_seqlens` -- the ``cu_seqlens`` cumulative offsets used
  by variable-length (flash) attention kernels, derived from the same
  per-token document index.

Both consume the ``segment_ids`` tensor the collator emits under packing:
``[B, L]`` long, the 0-based document index of every position, with ``-1``
marking padding.
"""

import torch

from genml_kit.utils.logging import fatal


def segments_to_cu_seqlens(segment_ids):
  """Cumulative sequence lengths for variable-length attention.

  Args:
    segment_ids: 1-D ``LongTensor`` of per-token document indices in a
      *flattened* (batch-as-one-stream) layout.  Documents must be contiguous;
      padding is not allowed.

  Returns:
    ``(cu_seqlens, max_seqlen)`` where ``cu_seqlens`` is an ``int32`` tensor of
    ``N + 1`` offsets (``N`` documents) and ``max_seqlen`` is the longest
    document length.
  """
  if segment_ids.dim() != 1:
    fatal(
        f"segments_to_cu_seqlens expects a 1-D tensor, "
        f"got shape {tuple(segment_ids.shape)}", ValueError)
  n = segment_ids.numel()
  if n == 0:
    return torch.zeros(1, dtype=torch.int32, device=segment_ids.device), 0
  # A new sequence starts wherever the document index changes.  Both operands
  # are [n-1]; the comparison is [n-1] bool and nonzero gives one row per start
  # (the +1 shifts a gap-at-i into a start-at-i+1).  Result: [num_changes].
  changes = torch.nonzero(segment_ids[1:] != segment_ids[:-1],
                          as_tuple=False).flatten() + 1
  # [num_changes] -> [num_changes + 2]: a leading 0 and a trailing n, so bounds
  # are the half-open [start, end) offsets of every sequence.
  bounds = torch.cat([
      torch.zeros(1, dtype=torch.long, device=segment_ids.device),
      changes,
      torch.tensor([n], dtype=torch.long, device=segment_ids.device),
  ])
  # [num_changes + 2] -> int32, the dtype flash-attn expects.
  cu_seqlens = bounds.to(torch.int32)
  # Consecutive differences give the [num_docs] per-document lengths.
  lengths = bounds[1:] - bounds[:-1]
  max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
  return cu_seqlens, max_seqlen


def build_block_diagonal_mask(segment_ids):
  """Additive ``[B, 1, L, L]`` mask: causal within a document, blocked across.

  Args:
    segment_ids: ``[B, L]`` long tensor of document indices (``-1`` = padding).

  Returns:
    ``[B, 1, L, L]`` float32 mask, ``0.0`` where attention is allowed and
    ``-inf`` where it is blocked.  Feeding this as ``attention_mask`` to a
    model that accepts a 4-D mask makes packed attention equivalent to padded
    attention: no token attends to a different document, and no key padding is
    attended.
  """
  if segment_ids.dim() != 2:
    fatal(
        f"build_block_diagonal_mask expects [B, L], "
        f"got shape {tuple(segment_ids.shape)}", ValueError)
  b, length = segment_ids.shape
  device = segment_ids.device
  # [L, L] lower-triangular bool: the plain causal (query i may see key j <= i).
  causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
  # Keys that are real tokens (not padding): [B, L] -> [B, 1, L] to broadcast as
  # the key axis of the [B, L, L] attention matrix below.
  valid_key = (segment_ids >= 0)[:, None, :]
  # same_doc[i, j] is True when query i and key j belong to the same document.
  # [B, L] vs [B, L] broadcast to [B, L, L] (query axis x key axis).
  same_doc = segment_ids[:, :, None] == segment_ids[:, None, :]
  # [L, L] -> [1, L, L] broadcasts over the batch: a query may see a key only
  # when it is causal AND in the same document AND the key is not padding.
  allowed = causal[None] & same_doc & valid_key
  # Every query row needs at least one visible key so softmax stays finite (an
  # all -inf row yields NaN).  Real rows already see themselves; this rescues
  # only the padding rows, whose whole row would otherwise be masked.  [L, L].
  eye = torch.eye(length, dtype=torch.bool, device=device)[None]
  allowed = allowed | eye
  # [B, L, L] -> [B, 1, L, L] (the head axis SDPA inserts between batch and L),
  # then fill with -inf where blocked and 0.0 where allowed.
  mask = torch.zeros(b, 1, length, length, dtype=torch.float32, device=device)
  return mask.masked_fill(~allowed[:, None], float("-inf"))


def flash_varlen_available():
  """Return True if the optional ``flash_attn`` package can be imported."""
  try:
    import flash_attn  # noqa: F401
  except Exception:
    return False
  return True


def varlen_attention(q, k, v, cu_seqlens, max_seqlen, dropout_p=0.0, causal=True):
  """Variable-length flash attention over a flattened ``[total, H, D]`` layout.

  Only reached when a model declares ``PACKED_ATTENTION = "varlen"``; the
  import is deferred so CPU-only installs (and CI) never need ``flash_attn``.
  """
  from flash_attn import flash_attn_varlen_func
  return flash_attn_varlen_func(q,
                                k,
                                v,
                                cu_seqlens_q=cu_seqlens,
                                cu_seqlens_k=cu_seqlens,
                                max_seqlen_q=max_seqlen,
                                max_seqlen_k=max_seqlen,
                                dropout_p=dropout_p,
                                causal=causal)
