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
from torch.utils.data import IterableDataset

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
    # A streaming source with unknown features (e.g. a hub stream, or anything
    # after ``.map()``) reports ``column_names is None`` until its first row is
    # pulled, and a pure streaming wrapper may not expose the attribute at all.
    # The column check is deferred to that first row; the layout choice is still
    # validated now, since it does not need the schema.
    self._columns_pending = getattr(self.dataset, "column_names", None) is None
    self._validate()

  def _validate(self):
    if self._columns_pending:
      if self.text_column is None and not (self.prompt_column and self.response_column):
        fatal(
            "TextDatasetProxy requires --text_column, or both "
            "--instruction_column and --response_column.  "
            "Dataset columns: unknown (streaming source).", ValueError)
      return
    self._check_columns(set(self.dataset.column_names))

  def _check_columns(self, columns):
    """Validate the column selection against a known set of *columns*."""
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
    first = True
    for row in self.dataset:
      if first:
        # First row is the earliest point the schema is known for a source
        # that reported ``column_names is None``; validate there, once.
        self._ensure_columns(row)
        first = False
      yield self._convert(row)

  def __getitem__(self, idx):
    row = self.dataset[idx]
    self._ensure_columns(row)
    return self._convert(row)

  def _ensure_columns(self, row):
    """Run the deferred column check against the first row, once."""
    if self._columns_pending:
      self._check_columns(set(row))
      self._columns_pending = False

  def _convert(self, row):
    """Build the text item for one raw row (plain-LM or instruction layout)."""
    if self.is_instruction:
      return {
          "prompt": row[self.prompt_column],
          "response": row[self.response_column],
      }
    return {"text": row[self.text_column]}


class StreamingTextDataset(IterableDataset):
  """Adapt a :class:`TextDatasetProxy` over a stream to a torch Dataset.

    :class:`TextDatasetProxy` is deliberately *not* a torch ``Dataset`` -- it is a
    plain wrapper that also serves random-access callers.  A ``DataLoader``,
    however, picks its iteration strategy purely from ``isinstance(dataset,
    IterableDataset)``, so a stream has to be presented as one.  This adapter is
    that thin shell: it defers to the proxy's own iteration and imposes no worker
    split of its own (the underlying HuggingFace stream shards itself).

    ``set_epoch`` is accepted for the pipeline's epoch hook; a text stream has no
    per-epoch state to advance, so it is a no-op.
    """

  def __init__(self, proxy):
    self.proxy = proxy

  def __iter__(self):
    yield from self.proxy

  def set_epoch(self, epoch):
    """No-op: a text stream carries no per-epoch shuffle state."""


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
