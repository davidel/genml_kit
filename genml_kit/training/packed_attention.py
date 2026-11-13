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
- :func:`flash_varlen_layout` -- the padding-free flattening plus ``cu_seqlens``
  / ``max_length`` that HuggingFace ``flash_attention_*`` decoders expect (they
  take the boundaries as ``FlashAttentionKwargs`` on a flattened ``[1, total]``
  batch, *not* as the 4-D mask).

All consume the ``segment_ids`` tensor the collator emits under packing:
``[B, L]`` long, the 0-based document index of every position, with ``-1``
marking padding.
"""

import collections

import torch

from genml_kit.utils.logging import fatal

# What ``segments_to_cu_seqlens`` returns: the ``int32`` cumulative offsets and
# the longest document length.
CuSeqlens = collections.namedtuple("CuSeqlens", ["cu_seqlens", "max_seqlen"])

# What ``flash_varlen_layout`` returns: the kept-token mask, the cumulative
# document offsets, and the longest document length.
PackedLayout = collections.namedtuple("PackedLayout",
                                      ["valid", "cu_seqlens", "max_length"])


def segments_to_cu_seqlens(segment_ids):
  """Cumulative sequence lengths for variable-length attention.

  Args:
    segment_ids: 1-D ``LongTensor`` of per-token document indices in a
      *flattened* (batch-as-one-stream) layout.  Documents must be contiguous;
      padding is not allowed.

  Returns:
    CuSeqlens: ``cu_seqlens`` is an ``int32`` tensor of ``N + 1`` offsets
    (``N`` documents) and ``max_seqlen`` is the longest document length.
  """
  if segment_ids.dim() != 1:
    fatal(
        f"segments_to_cu_seqlens expects a 1-D tensor, "
        f"got shape {tuple(segment_ids.shape)}", ValueError)
  n = segment_ids.numel()
  if n == 0:
    return CuSeqlens(cu_seqlens=torch.zeros(1,
                                            dtype=torch.int32,
                                            device=segment_ids.device),
                     max_seqlen=0)
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
  return CuSeqlens(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)


def flash_varlen_layout(attention_mask, segment_ids):
  """Padding-free flattening + boundaries for HF flash decoders.

  HuggingFace ``flash_attention_2``/``3`` do not accept a 4-D additive mask.
  They take the document boundaries as ``FlashAttentionKwargs`` on a flattened
  ``[1, total]`` (batch-as-one-stream) input with ``attention_mask=None``:
  ``cu_seq_lens_q`` / ``cu_seq_lens_k`` (the cumulative document lengths) and
  ``max_length_q`` / ``max_length_k``.  This helper produces exactly that.

  Args:
    attention_mask: ``[B, L]`` tensor (``> 0`` marks a real token) or ``None``.
    segment_ids: ``[B, L]`` long tensor of per-token document indices (``-1``
      marks padding), as emitted by the collator under ``--packing``.

  Returns:
    ``(valid, cu_seqlens, max_length)`` where ``valid`` is a ``[B, L]`` bool
    mask (True for real tokens; index the flattened input with it), and
    ``cu_seqlens`` is the ``int32`` ``[N + 1]`` cumulative offsets of the ``N``
    documents in row-major flattened order.
  """
  if segment_ids.dim() != 2:
    fatal(
        f"flash_varlen_layout expects [B, L] segment_ids, "
        f"got shape {tuple(segment_ids.shape)}", ValueError)
  device = segment_ids.device
  valid = segment_ids >= 0 if attention_mask is None else attention_mask > 0
  # A run starts at the first token of a document: a valid position whose
  # predecessor is padding or belongs to a different document.
  prev_valid = torch.zeros_like(valid)
  prev_valid[:, 1:] = valid[:, :-1]
  prev_seg = torch.zeros_like(segment_ids)
  prev_seg[:, 1:] = segment_ids[:, :-1]
  run_start = valid & (~prev_valid | (segment_ids != prev_seg))
  # Number the runs per row, then offset each row by the runs before it, so the
  # ids are globally unique and strictly increasing across rows (a plain
  # per-window 0-based index would collide at the row boundary).  [B, L] / [B].
  runs_per_row = run_start.sum(dim=1)
  offset = torch.cumsum(runs_per_row, dim=0) - runs_per_row
  row_run_id = run_start.cumsum(dim=1) - 1
  global_run = row_run_id + offset[:, None]
  total_runs = int(runs_per_row.sum().item())
  # Flatten the kept tokens row-major (runs stay contiguous) and count each run.
  flat_run = global_run[valid]
  counts = torch.bincount(flat_run, minlength=total_runs)
  cu_seqlens = torch.zeros(total_runs + 1, dtype=torch.int32, device=device)
  cu_seqlens[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
  max_length = int(counts.max().item()) if total_runs else 0
  return PackedLayout(valid=valid, cu_seqlens=cu_seqlens, max_length=max_length)


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
