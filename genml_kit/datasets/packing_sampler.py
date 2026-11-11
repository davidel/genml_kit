"""Token-budget batch sampler for packed (``--packing wrapped``) text training.

On the padded (default) and ``bfd`` paths ``--batch_size`` counts *documents*.
Under ``wrapped`` packing, however, the collator concatenates the batch and
slices it into fixed ``max_seq_len`` windows, so the number of model rows is
``ceil(total_tokens / max_seq_len)`` -- a quantity the document count does not
bound.  One long document alone fans out into ``len(doc) / max_seq_len`` rows,
and a batch of long documents can allocate far more VRAM than the padded path
ever would (a practical OOM trap on consumer GPUs).

:class:`PackingBatchSampler` fixes the accounting: it batches by a fixed *token
budget* (``batch_size * max_seq_len``) instead of by document count, so every
emitted batch fills at most ``batch_size`` windows.  This is the constant-length
packing recipe large-scale LM training uses (Megatron-LM, HuggingFace packed
datasets): the batch is a fixed amount of tokens, and documents are streamed in
until the budget is met, so a long document's tail is still trained -- it lands
in a later window of the same batch -- rather than being truncated away.

Lengths arrive as a cheap per-item proxy (see ``text_length_proxy``) and are
converted to tokens with ``tokens_per_unit`` (tokens per proxy unit).  The
estimate only steers how many documents fit in a batch; the collator still hard
-caps the emitted windows, so an under-estimate can never break the memory
envelope.
"""

import logging
import math
import random

from genml_kit.utils.logging import fatal


class PackingBatchSampler:
  """Yield document-index batches whose tokens fit a fixed window budget.

    Each batch is a list of dataset indices and is guaranteed to hold at most
    ``batch_size`` ``max_seq_len`` windows worth of (estimated) tokens, so the
    collator's packed batch never exceeds ``batch_size`` model rows.

    Args:
        lengths: Per-item length proxy (e.g. character counts), one entry per
          dataset item.
        batch_size: Number of ``max_seq_len`` windows a packed batch may fill;
          this is the user-facing ``--batch_size``.
        max_seq_len: The packing window length, in tokens per model row.
        tokens_per_unit: Tokens per unit of ``lengths`` (e.g. tokens per
          character).  Defaults to ``1.0`` (the proxy is already in tokens).
        drop_last: Drop a trailing batch that does not fill the token budget.
          Off by default so no tokens are wasted.
        seed: Base RNG seed; the per-epoch order is ``seed + epoch`` so each
          epoch reshuffles deterministically.
    """

  def __init__(self,
               lengths,
               batch_size,
               max_seq_len,
               tokens_per_unit=1.0,
               drop_last=False,
               seed=None):
    if batch_size <= 0:
      fatal(f"batch_size must be positive, got {batch_size}", ValueError)
    if max_seq_len <= 0:
      fatal(f"max_seq_len must be positive, got {max_seq_len}", ValueError)
    if tokens_per_unit <= 0:
      fatal(f"tokens_per_unit must be positive, got {tokens_per_unit}", ValueError)
    self.lengths = list(lengths)
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.tokens_per_unit = tokens_per_unit
    self.drop_last = drop_last
    self.seed = seed
    self.epoch = 0
    self._packets = None

  @property
  def budget(self):
    """Token budget of one batch: ``batch_size * max_seq_len``."""
    return self.batch_size * self.max_seq_len

  def set_epoch(self, epoch):
    """Reshuffle for a new epoch (called by the trainer each epoch)."""
    self.epoch = epoch
    self._packets = None

  def _estimated_tokens(self, index):
    """Estimated token count of item *index* (always at least one)."""
    return max(1, int(math.ceil(self.lengths[index] * self.tokens_per_unit)))

  def _build_packets(self):
    """Greedily pack a shuffled epoch into token-budgeted batches."""
    budget = self.budget
    order = list(range(len(self.lengths)))
    # Deterministic per epoch; unseeded (``None``) keeps the system entropy.
    rng = random.Random(None if self.seed is None else self.seed + self.epoch)
    rng.shuffle(order)
    packets = []
    current = []
    used = 0
    for index in order:
      tokens = self._estimated_tokens(index)
      # A document that alone exceeds the budget opens its own (over-full)
      # packet; the collator caps the windows it emits.
      if current and used + tokens > budget:
        packets.append(current)
        current = []
        used = 0
      current.append(index)
      used += tokens
    if current:
      if self.drop_last and used < budget:
        logging.debug(
            "Packing sampler: dropping a short trailing batch "
            "(%d / %d tokens).", used, budget)
      else:
        packets.append(current)
    return packets

  def _ensure_packets(self):
    if self._packets is None:
      self._packets = self._build_packets()
    return self._packets

  def __iter__(self):
    yield from self._ensure_packets()

  def __len__(self):
    return len(self._ensure_packets())
