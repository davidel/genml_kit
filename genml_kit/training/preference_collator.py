"""Collator for the preference (alignment) pipeline.

Turns chosen/rejected preference pairs into the token batch the ``dpo`` and
``reward_model`` methods consume.  Both members of a pair are tokenized and
padded independently, giving four tensors in ``data`` and two label tensors
in ``meta``::

    data = {chosen_input_ids, chosen_attention_mask,
            rejected_input_ids, rejected_attention_mask}
    meta = {chosen_labels, rejected_labels}

When a ``prompt`` is present it is prepended to both responses and masked out
of the labels (``-100``), so the sequence score covers only the response
tokens -- the DPO/reward convention.  When there is no prompt, the chosen /
rejected strings are treated as complete sequences and every token is scored.
"""

import torch

from genml_kit.pipelines.contracts import DataBlob
from genml_kit.utils.logging import fatal

IGNORE_INDEX = -100


class PreferenceCollator:
  """Collate chosen/rejected pairs into an alignment token batch.

    Parameters
    ----------
    tokenizer : object
        Anything exposing ``encode(text, add_special_tokens=...)`` and the
        ``pad_token_id`` / ``eos_token_id`` attributes (see
        :mod:`genml_kit.models.tokenizer`).
    max_seq_len : int
        Truncation length for each sequence.
    mask_prompt : bool
        When True (and a prompt is present), mask the prompt tokens out of
        the labels so only the response is scored.
    pad_to_multiple_of : int or None
        If set, round each padded length up to a multiple of this value.
    """

  def __init__(self,
               tokenizer,
               max_seq_len=512,
               mask_prompt=True,
               pad_to_multiple_of=None):
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    self.mask_prompt = mask_prompt
    self.pad_to_multiple_of = pad_to_multiple_of

  def _encode(self, prompt, response):
    """Return ``(input_ids, labels)`` for one response (prompt prepended)."""
    prompt_ids = []
    if prompt:
      prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = self.tokenizer.encode(response, add_special_tokens=False)
    if self.tokenizer.eos_token_id is not None:
      response_ids = response_ids + [self.tokenizer.eos_token_id]
    ids = prompt_ids + response_ids
    if self.mask_prompt and prompt_ids:
      labels = [IGNORE_INDEX] * len(prompt_ids) + list(response_ids)
    else:
      labels = list(ids)
    return ids[:self.max_seq_len], labels[:self.max_seq_len]

  def _pad(self, sequences):
    """Right-pad ``(ids, labels)`` pairs to a shared length -> tensor dict."""
    pad_id = self.tokenizer.pad_token_id
    max_len = max(len(ids) for ids, _ in sequences)
    if self.pad_to_multiple_of:
      step = self.pad_to_multiple_of
      max_len = ((max_len + step - 1) // step) * step
    input_ids, attention_mask, labels = [], [], []
    for ids, item_labels in sequences:
      pad = max_len - len(ids)
      input_ids.append(ids + [pad_id] * pad)
      attention_mask.append([1] * len(ids) + [0] * pad)
      labels.append(item_labels + [IGNORE_INDEX] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask,
                         dtype=torch.long), torch.tensor(labels, dtype=torch.long))

  def __call__(self, batch):
    chosen, rejected = [], []
    for item in batch:
      if "chosen" not in item or "rejected" not in item:
        fatal("PreferenceCollator items require 'chosen' and 'rejected'.", ValueError)
      prompt = item.get("prompt", "")
      chosen.append(self._encode(prompt, item["chosen"]))
      rejected.append(self._encode(prompt, item["rejected"]))

    chosen_ids, chosen_mask, chosen_labels = self._pad(chosen)
    rejected_ids, rejected_mask, rejected_labels = self._pad(rejected)
    return DataBlob(
        data={
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": chosen_mask,
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": rejected_mask,
        },
        meta={
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
        },
    )
