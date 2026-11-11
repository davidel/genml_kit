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

from genml_kit.datasets.capped import maybe_cap
from genml_kit.datasets.iterable import is_streaming_source
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
    """

  def __init__(self,
               hf_dataset,
               text_column=None,
               prompt_column=None,
               response_column=None):
    self.dataset = hf_dataset
    self.text_column = text_column
    self.prompt_column = prompt_column
    self.response_column = response_column
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
    # A streaming HF source has no length; surface that as TypeError rather
    # than AttributeError so ``num_samples`` returns its estimate/None.
    if is_streaming_source(self.dataset):
      raise TypeError("streaming text dataset has no length")
    return len(self.dataset)

  def __iter__(self):
    # A streaming source has no index, so build each row from the iterator.
    # Same per-item shape as ``__getitem__``.
    for row in self.dataset:
      yield self._convert(row)

  def __getitem__(self, idx):
    return self._convert(self.dataset[idx])

  def _convert(self, row):
    """Build the text item for one raw row (plain-LM or instruction layout)."""
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
                      response_column=None,
                      dataset_arg=None):
  """Load a HuggingFace text dataset split wrapped in a TextDatasetProxy.

    Args:
        name: HuggingFace dataset identifier.
        split: Dataset split (``"train"``, ``"validation"``, ...).
        cache_dir: HuggingFace cache directory.
        token: Optional HuggingFace token.
        max_items: Optional cap on the number of items.
        text_column, prompt_column, response_column: Column selection,
            forwarded to :class:`TextDatasetProxy`.
        dataset_arg: Extra kwargs forwarded verbatim to
            ``datasets.load_dataset`` (e.g. the HF subset/config ``name``).

    Returns:
        A :class:`TextDatasetProxy`.
    """
  logging.info("Loading text dataset %s (split=%s).", name, split)
  hf = load_raw_split(name,
                      split,
                      cache_dir=cache_dir,
                      token=token,
                      dataset_arg=dataset_arg)
  hf = maybe_cap(hf, max_items)
  return TextDatasetProxy(hf,
                          text_column=text_column,
                          prompt_column=prompt_column,
                          response_column=response_column)


def load_raw_split(name, split, *, cache_dir=None, token=None, dataset_arg=None):
  """Load *split* from a HF hub dataset or a ``save_to_disk`` directory.

    Returns the bare ``datasets.Dataset`` (no proxy wrapper), so the
    pipeline can select a held-out slice or wrap it once.

    *dataset_arg* (a ``KEY=VALUE`` dict from ``--dataset_arg``) is forwarded
    verbatim to ``datasets.load_dataset`` -- e.g. ``name`` selects an HF
    subset/config.  It is ignored for on-disk ``save_to_disk`` directories.
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
  return datasets.load_dataset(name,
                               split=split,
                               cache_dir=cache_dir,
                               token=token,
                               **(dataset_arg or {}))
