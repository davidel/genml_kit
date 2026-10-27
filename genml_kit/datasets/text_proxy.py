"""Bridge HuggingFace text datasets to the text pipeline.

The text analogue of :class:`genml_kit.datasets.hf_proxy.HFDatasetProxy`:
where the image proxy yields decoded images, this yields raw text items
(dicts) that :class:`genml_kit.training.text_collator.TextCollator` tokenizes.

Two layouts are supported:

- **plain LM**: one text column (``--text_column``) -> ``{"text": ...}``.
- **instruction**: a prompt column plus a response column ->
  ``{"prompt": ..., "response": ...}``.
"""

import logging
import os

import datasets

from genml_kit.utils.logging import fatal


class TextDatasetProxy:
  """Adapt a HuggingFace dataset split to dict text items.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        A split returned by ``datasets.load_dataset``.
    text_column : str or None
        Column holding a raw LM sequence.
    prompt_column, response_column : str or None
        Columns holding an instruction and its response.
    max_items : int or None
        Optional cap (useful for smoke tests on tiny corpora).
    """

  def __init__(self,
               hf_dataset,
               text_column=None,
               prompt_column=None,
               response_column=None,
               max_items=None):
    self.dataset = hf_dataset
    self.text_column = text_column
    self.prompt_column = prompt_column
    self.response_column = response_column
    self.max_items = max_items
    self._validate()

  def _validate(self):
    columns = set(self.dataset.column_names)
    if self.text_column is not None:
      if self.text_column not in columns:
        fatal(
            f"Text column {self.text_column!r} not in dataset columns "
            f"{sorted(columns)}.", ValueError)
      return
    if self.prompt_column is not None and self.response_column is not None:
      if not {self.prompt_column, self.response_column} <= columns:
        fatal(
            f"Instruction columns {self.prompt_column!r} / "
            f"{self.response_column!r} not in dataset columns "
            f"{sorted(columns)}.", ValueError)
      return
    fatal(
        "TextDatasetProxy requires --text_column, or both "
        "--instruction_column and --response_column.  "
        f"Dataset columns: {sorted(columns)}.", ValueError)

  @property
  def is_instruction(self):
    """Return True when the proxy yields prompt/response pairs."""
    return self.text_column is None

  def __len__(self):
    n = len(self.dataset)
    return min(n, self.max_items) if self.max_items else n

  def __getitem__(self, idx):
    row = self.dataset[idx]
    if self.is_instruction:
      return {
          "prompt": row[self.prompt_column],
          "response": row[self.response_column],
      }
    return {"text": row[self.text_column]}


def load_text_dataset(name,
                      split="train",
                      cache_dir=None,
                      token=None,
                      max_items=None,
                      text_column=None,
                      prompt_column=None,
                      response_column=None):
  """Load a HuggingFace text dataset split wrapped in a TextDatasetProxy.

    Args:
        name: HuggingFace dataset identifier.
        split: Dataset split (``"train"``, ``"validation"``, ...).
        cache_dir: HuggingFace cache directory.
        token: Optional HuggingFace token.
        max_items: Optional cap on the number of items.
        text_column, prompt_column, response_column: Column selection,
            forwarded to :class:`TextDatasetProxy`.

    Returns:
        A :class:`TextDatasetProxy`.
    """
  logging.info("Loading text dataset %s (split=%s).", name, split)
  hf = load_raw_split(name, split, cache_dir=cache_dir, token=token)
  return TextDatasetProxy(hf,
                          text_column=text_column,
                          prompt_column=prompt_column,
                          response_column=response_column,
                          max_items=max_items)


def load_raw_split(name, split, *, cache_dir=None, token=None):
  """Load *split* from a HF hub dataset or a ``save_to_disk`` directory.

    Returns the bare ``datasets.Dataset`` (no proxy wrapper), so the
    pipeline can select a held-out slice or wrap it once.
    """
  if os.path.isdir(name):
    loaded = datasets.load_from_disk(name)
    if isinstance(loaded, datasets.DatasetDict):
      if split not in loaded:
        fatal(
            f"Split {split!r} not in on-disk dataset {name!r}. "
            f"Available: {sorted(loaded.keys())}.", ValueError)
      return loaded[split]
    return loaded
  return datasets.load_dataset(name, split=split, cache_dir=cache_dir, token=token)
