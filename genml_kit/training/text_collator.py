"""Collator for the text (causal-LM) pipeline.

Turns raw text items into the token batch the ``causal_lm`` method consumes:
``input_ids`` / ``attention_mask`` tensors plus a ``labels`` tensor where
every position the model must *not* be trained on is set to ``-100``
(``CrossEntropyLoss``'s ``ignore_index``).

Three shapes of raw item are supported:

- **plain LM**: ``{"text": "..."}`` -- whole sequence is a target.
- **instruction**: ``{"prompt": "...", "response": "..."}`` -- with
  ``mask_prompt`` the prompt tokens are masked out of the loss.
- **pre-tokenized**: ``{"input_ids": [...]}`` -- passed through.

``packing`` concatenates the tokenized item stream with EOS separators and
slices it into fixed ``max_seq_len`` windows, which removes padding waste on
short-document corpora.  Two strategies are supported: ``wrapped``
(concatenate and slice, the historical behaviour) and ``bfd`` (best-fit
decreasing bin packing, which yields fewer windows).  Because a window can hold
several documents, packing mode *also* emits two boundary tensors so a model
can keep documents isolated:

- ``position_ids`` -- position index that resets to ``0`` at each document.
- ``segment_ids``  -- 0-based document index per token (``-1`` = padding).

Neither requires any knowledge of the model; consumers that ignore them get
plain cross-document attention, consumers that use them get exact isolation.

The loss at a document boundary is separate from that isolation and is selected
by ``boundary_loss``: ``mask`` (default) drops the label opening each packed
document, so the causal shift never crosses a boundary and packed training
stays equivalent to the per-document padded case.  ``keep`` retains it, which
is the GPT-2/GPT-3 recipe that teaches the model what follows a document.  The
two are compiled together by :mod:`genml_kit.training.sequence_mode`.
"""

import logging

import torch

from genml_kit.pipelines.contracts import DataBlob
from genml_kit.utils.logging import fatal

IGNORE_INDEX = -100


class TextCollator:
  """Collate a batch of text items into a causal-LM token batch.

    Parameters
    ----------
    tokenizer : object
        Anything exposing ``encode(text, add_special_tokens=...)`` and the
        ``pad_token_id`` / ``eos_token_id`` attributes (see
        :mod:`genml_kit.models.tokenizer`).
    max_seq_len : int
        Truncation length (and packing window).
    mask_prompt : bool
        When True, mask the ``prompt`` tokens out of ``labels`` for
        instruction items.
    packing : bool
        When True, pack the batch's tokens into ``max_seq_len`` windows.
    packing_strategy : str
        ``"wrapped"`` (concatenate and slice) or ``"bfd"`` (best-fit
        decreasing).  Only meaningful when ``packing`` is True.  ``bfd``
        reorders the documents, so batch composition is not reproducible with
        respect to ``--seed``; a warning is logged once.
    boundary_loss : str
        ``"mask"`` (default) sets the label opening each packed document to
        ``IGNORE_INDEX`` so the causal shift never predicts across a boundary
        -- packed training stays equivalent to the per-document padded case.
        ``"keep"`` retains that label, so the model learns what follows a
        document (the GPT-2/GPT-3 recipe).  Only valid when ``packing`` is
        True: unpacked, the token after the document's ``EOS`` is padding.
    pad_to_multiple_of : int or None
        If set, round the padded length up to a multiple of this value.
    """

  def __init__(self,
               tokenizer,
               max_seq_len=512,
               mask_prompt=False,
               packing=False,
               packing_strategy="wrapped",
               boundary_loss="mask",
               pad_to_multiple_of=None):
    if max_seq_len < 1:
      fatal(f"max_seq_len must be positive, got {max_seq_len}", ValueError)
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.mask_prompt = mask_prompt
    self.packing = packing
    if packing_strategy not in ("wrapped", "bfd"):
      fatal(
          f"packing_strategy must be 'wrapped' or 'bfd', "
          f"got {packing_strategy!r}", ValueError)
    self.packing_strategy = packing_strategy
    if boundary_loss not in ("mask", "keep"):
      fatal(f"boundary_loss must be 'mask' or 'keep', got {boundary_loss!r}",
            ValueError)
    if boundary_loss == "keep" and not packing:
      fatal(
          "boundary_loss='keep' requires packing (use --sequence_mode open); "
          "unpacked, the tokens after a document's EOS are padding.", ValueError)
    self.boundary_loss = boundary_loss
    if packing and packing_strategy == "bfd":
      logging.warning("packing_strategy='bfd' reorders documents by length; batch "
                      "composition is not reproducible with respect to --seed.")
    self.pad_to_multiple_of = pad_to_multiple_of

  def _encode_item(self, item):
    """Return ``(input_ids, labels)`` for one raw item."""
    if "input_ids" in item:
      ids = list(item["input_ids"])
      return ids, list(ids)

    if "prompt" in item and "response" in item:
      prompt = item["prompt"] or ""
      response = item["response"] or ""
      prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
      response_ids = self.tokenizer.encode(response, add_special_tokens=False)
      if self.tokenizer.eos_token_id is not None:
        response_ids = response_ids + [self.tokenizer.eos_token_id]
      ids = prompt_ids + response_ids
      if self.mask_prompt:
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
      else:
        labels = list(ids)
      return ids, labels

    if "text" in item:
      ids = self.tokenizer.encode(item["text"], add_special_tokens=True)
      return ids, list(ids)

    fatal(
        "TextCollator item has none of 'text', 'prompt'+'response', "
        "or 'input_ids'.", ValueError)

  def _pack(self, sequences):
    """Pack sequences into windows with per-token segment/position ids.

        *sequences* is a list of ``(ids, labels)`` pairs.  Returns a list of
        ``(ids, labels, segment_ids, position_ids)`` tuples, one per window.
        ``segment_ids`` numbers the contiguous *runs* of tokens that belong to
        one document inside that window; ``position_ids`` restarts at ``0`` at
        every run start.  A short document fully inside the window is a single
        run; a document split across a window boundary leaves a run in each
        window (positions restart), and the EOS separator is attributed to the
        document it follows.
        """
    if self.packing_strategy == "bfd":
      raw_windows = self._pack_bfd(sequences)
    else:
      raw_windows = self._pack_wrapped(sequences)
    # Each raw window is [<= max_seq_len] (tok, label, doc_idx) triples; convert
    # every one into the (ids, labels, segment_ids, position_ids) tuple the
    # batch expects.
    return [self._runs_for_window(w) for w in raw_windows]

  def _runs_for_window(self, tokens):
    """Turn ``[(tok, label, doc_idx), ...]`` into the window's four lists.

        Returns ``(ids, labels, segment_ids, position_ids)``.  Runs break
        whenever the source document changes, and position ids reset at every
        run start so each window is self-describing (this matches the
        transformers ``position_ids.min()`` sequence-start convention).
        """
    # Four parallel lists, each of length len(tokens).
    ids, labels, segment_ids, position_ids = [], [], [], []
    # ``run`` is the window-local segment index (0-based, increments at every
    # document change); ``pos`` is the position within the current run and is
    # reset to 0 exactly when a new run starts.
    run = -1
    prev_doc = None
    for tok, label, doc_idx in tokens:
      if doc_idx != prev_doc:
        # New document here (or the very first token): open a fresh run and
        # restart position numbering.  This is what makes a window split across
        # a long document self-describing instead of continuing a stale offset.
        run += 1
        pos = 0
      ids.append(tok)
      labels.append(label)
      segment_ids.append(run)
      position_ids.append(pos)
      pos += 1
      prev_doc = doc_idx
    # Break the next-token target at document boundaries.  The causal shift
    # pairs the logit at position j-1 with the label at position j, so a label
    # opening a new run would otherwise be predicted from the previous document
    # (a cross-document leak).  Masking it makes packed training equivalent to
    # the per-document padded case.  Position 0 has no predecessor, so it is
    # never a shifted target and is left untouched.
    #
    # ``boundary_loss="keep"`` retains the label instead, so the model learns
    # what follows a document (the GPT-2/GPT-3 recipe).  Document isolation is
    # still the model's job (``PACKED_ATTENTION``): keeping the label trains
    # the transition, it does not let the next document attend backwards.
    if self.boundary_loss == "mask":
      for j in range(1, len(labels)):
        if segment_ids[j] != segment_ids[j - 1]:
          labels[j] = IGNORE_INDEX
    return ids, labels, segment_ids, position_ids

  def _pack_wrapped(self, sequences):
    """Concatenate sequences and slice into ``max_seq_len`` windows.

        *sequences* is a list of ``(ids, labels)`` pairs.  Returns a list of
        windows, each a list of ``(token, label, doc_idx)`` triples.
        """
    sep = self.tokenizer.eos_token_id
    # Flatten the batch into one [(tok, label, doc_idx), ...] stream, tagging
    # each token with the source document so boundaries survive the
    # concatenation.  The EOS separator is attributed to the document it
    # terminates (so it never opens a spurious new run) and carries the EOS id
    # as its label (a real target, matching the non-packing contract).
    stream = []
    for doc_idx, (ids, labels) in enumerate(sequences):
      for tok, label in zip(ids, labels):
        stream.append((tok, label, doc_idx))
      if sep is not None:
        stream.append((sep, sep, doc_idx))
    windows = []
    # Slice the flat stream into fixed max_seq_len windows: [flat] -> [n_win,
    # <= max_seq_len].  A document may straddle two windows; the run split is
    # recovered later by _runs_for_window from the doc_idx tags.
    for start in range(0, len(stream), self.max_seq_len):
      if start >= len(stream):
        break
      windows.append(stream[start:start + self.max_seq_len])
    return windows

  def _pack_bfd(self, sequences):
    """Best-fit-decreasing bin packing into ``max_seq_len`` windows.

        *sequences* is a list of ``(ids, labels)`` pairs.  Documents are sorted
        longest-first and each is placed in the open bin with the least
        remaining space that still fits (a new bin otherwise), which yields
        fewer, fuller windows than :meth:`_pack_wrapped`.  Because documents are
        reordered, batch composition is not reproducible with respect to
        ``--seed``.

        Returns a list of windows, each a list of ``(token, label, doc_idx)``
        triples.
        """
    sep = self.tokenizer.eos_token_id
    # Reserve room for the EOS separator so a document plus its separator never
    # exceeds the window (a document of exactly max_seq_len tokens would
    # otherwise become a max_seq_len + 1 window).  Tag each document's tokens
    # with the source document, producing one [(tok, label, doc_idx), ...] list
    # per document.
    keep = self.max_seq_len - 1 if sep is not None else self.max_seq_len
    docs = []
    for doc_idx, (ids, labels) in enumerate(sequences):
      toks = [(tok, label, doc_idx) for tok, label in zip(ids[:keep], labels[:keep])]
      if sep is not None:
        toks.append((sep, sep, doc_idx))
      docs.append(toks)
    # Decreasing-length order: the long items are placed first, so they open
    # bins rather than being left with an unusable remainder at the end.
    order = sorted(range(len(docs)), key=lambda i: len(docs[i]), reverse=True)
    # First-fit-decreasing with a best-fit tie-break: each bin tracks its items
    # and the number of tokens already used (``used <= max_seq_len``).
    bins = []  # each: {"items": [toks], "used": int}
    for i in order:
      toks = docs[i]
      n = len(toks)
      # Pick the open bin with the least remaining space that still fits this
      # document (best fit); ``None`` means "nothing fits, open a new bin".
      best = None
      for b in bins:
        rem = self.max_seq_len - b["used"]
        if n <= rem and (best is None or rem < best[0]):
          best = (rem, b)
      if best is None:
        b = {"items": [], "used": 0}
        bins.append(b)
      else:
        b = best[1]
      b["items"].append(toks)
      b["used"] += n
    # Flatten each bin's document lists into one window of (tok, doc_idx) pairs
    # (window length <= max_seq_len).  Document-to-run segmentation happens in
    # _runs_for_window, exactly as for _pack_wrapped.
    windows = []
    for b in bins:
      flat = []
      for toks in b["items"]:
        flat.extend(toks)
      # Invariant guard: BFD must never emit a window wider than max_seq_len.
      if len(flat) > self.max_seq_len:
        fatal(
            f"BFD produced a window of {len(flat)} tokens, max_seq_len is "
            f"{self.max_seq_len}.", ValueError)
      windows.append(flat)
    return windows

  def __call__(self, batch):
    # Each entry is (ids, labels, doc_ids, pos_ids); the last two are None when
    # not packing (so the emitted blob stays byte-identical to the pre-packing
    # contract and no boundary tensors are produced).
    padded = []
    if self.packing:
      for ids, packed_labels, doc_ids, pos_ids in self._pack(
          [self._encode_item(it) for it in batch]):
        padded.append((ids, packed_labels, doc_ids, pos_ids))
    else:
      for item in batch:
        ids, labels = self._encode_item(item)
        padded.append((ids[:self.max_seq_len], labels[:self.max_seq_len], None, None))

    pad_id = self.tokenizer.pad_token_id
    # Rectangular batch width: the longest row, optionally rounded up (e.g. to 8
    # for tensor-core friendliness).
    max_len = max(len(ids) for ids, _, _, _ in padded)
    if self.pad_to_multiple_of:
      step = self.pad_to_multiple_of
      max_len = ((max_len + step - 1) // step) * step

    # Parallel per-row lists; each becomes one [B, max_len] tensor below.
    input_ids, attention_mask, labels = [], [], []
    segment_ids, position_ids = [], []
    for ids, item_labels, doc_ids, pos_ids in padded:
      # Right-pad every row to max_len.  Pad ids are real tokens in input_ids
      # (masked out downstream), 0 in attention_mask (never attended), IGNORE
      # in labels (never scored), -1 in segment_ids (marked "no document") and
      # 0 in position_ids (inert, the row is masked anyway).
      pad = max_len - len(ids)
      input_ids.append(ids + [pad_id] * pad)
      attention_mask.append([1] * len(ids) + [0] * pad)
      labels.append(item_labels + [IGNORE_INDEX] * pad)
      if doc_ids is not None:
        segment_ids.append(doc_ids + [-1] * pad)
        position_ids.append(pos_ids + [0] * pad)

    # List of [max_len] rows -> [B, max_len] long tensors.
    data = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    # Boundary tensors are added only under packing, so the non-packing blob is
    # byte-identical to the pre-packing contract.
    if self.packing:
      data["segment_ids"] = torch.tensor(segment_ids, dtype=torch.long)
      data["position_ids"] = torch.tensor(position_ids, dtype=torch.long)

    return DataBlob(data=data,
                    meta={"labels": torch.tensor(labels, dtype=torch.long)},
                    size=len(input_ids))
