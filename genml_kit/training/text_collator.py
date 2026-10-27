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
short-document corpora at the cost of cross-document attention.
"""

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
    pad_to_multiple_of : int or None
        If set, round the padded length up to a multiple of this value.
    """

  def __init__(self,
               tokenizer,
               max_seq_len=512,
               mask_prompt=False,
               packing=False,
               pad_to_multiple_of=None):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.mask_prompt = mask_prompt
    self.packing = packing
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
    """Concatenate sequences and slice into ``max_seq_len`` windows."""
    sep = self.tokenizer.eos_token_id
    stream = []
    for ids in sequences:
      stream.extend(ids)
      if sep is not None:
        stream.append(sep)
    windows = []
    for start in range(0, len(stream), self.max_seq_len):
      window = stream[start:start + self.max_seq_len]
      if not window:
        break
      windows.append((list(window), list(window)))
    return windows

  def __call__(self, batch):
    padded = []
    if self.packing:
      for ids, labels in self._pack([self._encode_item(it)[0] for it in batch]):
        padded.append((ids, labels))
    else:
      for item in batch:
        ids, labels = self._encode_item(item)
        padded.append((ids[:self.max_seq_len], labels[:self.max_seq_len]))

    pad_id = self.tokenizer.pad_token_id
    max_len = max(len(ids) for ids, _ in padded)
    if self.pad_to_multiple_of:
      step = self.pad_to_multiple_of
      max_len = ((max_len + step - 1) // step) * step

    input_ids, attention_mask, labels = [], [], []
    for ids, item_labels in padded:
      pad = max_len - len(ids)
      input_ids.append(ids + [pad_id] * pad)
      attention_mask.append([1] * len(ids) + [0] * pad)
      labels.append(item_labels + [IGNORE_INDEX] * pad)

    return DataBlob(
        data={
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        },
        meta={"labels": torch.tensor(labels, dtype=torch.long)},
    )
