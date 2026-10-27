"""Bridge HuggingFace preference datasets to the preference pipeline.

The alignment analogue of :class:`genml_kit.datasets.text_proxy.TextDatasetProxy`:
where the text proxy yields plain/instruction text items, this yields
chosen/rejected preference pairs that
:class:`genml_kit.training.preference_collator.PreferenceCollator` tokenizes.

Layout: a ``chosen`` column and a ``rejected`` column (names configurable),
plus an optional ``prompt`` column.  When a prompt is present the collator
masks it out of the loss (only the response tokens are scored); when it is
absent the chosen/rejected columns are assumed to be full sequences.
"""

import logging
import os

import datasets

from genml_kit.utils.logging import fatal


class PreferenceDatasetProxy:
  """Adapt a HuggingFace dataset split to chosen/rejected pairs.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        A split returned by ``datasets.load_dataset``.
    prompt_column : str or None
        Column holding the shared prompt (masked out of the loss).
    chosen_column, rejected_column : str
        Columns holding the preferred and dispreferred responses.
    max_items : int or None
        Optional cap (useful for smoke tests on tiny corpora).
    """

  def __init__(self,
               hf_dataset,
               chosen_column="chosen",
               rejected_column="rejected",
               prompt_column=None,
               max_items=None):
    self.dataset = hf_dataset
    self.chosen_column = chosen_column
    self.rejected_column = rejected_column
    self.prompt_column = prompt_column
    self.max_items = max_items
    self._validate()

  def _validate(self):
    columns = set(self.dataset.column_names)
    required = {self.chosen_column, self.rejected_column}
    if self.prompt_column is not None:
      required.add(self.prompt_column)
    missing = required - columns
    if missing:
      fatal(
          f"Preference columns {sorted(missing)} not in dataset columns "
          f"{sorted(columns)}.", ValueError)

  @property
  def has_prompt(self):
    """Return True when a shared prompt column is present."""
    return self.prompt_column is not None

  def __len__(self):
    n = len(self.dataset)
    return min(n, self.max_items) if self.max_items else n

  def __getitem__(self, idx):
    row = self.dataset[idx]
    item = {
        "chosen": row[self.chosen_column],
        "rejected": row[self.rejected_column],
    }
    if self.has_prompt:
      item["prompt"] = row[self.prompt_column]
    return item


def load_preference_dataset(name,
                            split="train",
                            cache_dir=None,
                            token=None,
                            max_items=None,
                            chosen_column="chosen",
                            rejected_column="rejected",
                            prompt_column=None):
  """Load a preference dataset split wrapped in a PreferenceDatasetProxy.

    Args:
        name: HuggingFace dataset identifier, or a ``save_to_disk`` directory.
        split: Dataset split (``"train"``, ``"validation"``, ...).
        cache_dir: HuggingFace cache directory.
        token: Optional HuggingFace token.
        max_items: Optional cap on the number of items.
        chosen_column, rejected_column, prompt_column: Column selection,
            forwarded to :class:`PreferenceDatasetProxy`.

    Returns:
        A :class:`PreferenceDatasetProxy`.
    """
  logging.info("Loading preference dataset %s (split=%s).", name, split)
  hf = load_raw_split(name, split, cache_dir=cache_dir, token=token)
  return PreferenceDatasetProxy(hf,
                                chosen_column=chosen_column,
                                rejected_column=rejected_column,
                                prompt_column=prompt_column,
                                max_items=max_items)


def load_raw_split(name, split, *, cache_dir=None, token=None):
  """Load *split* from a HF hub dataset or a ``save_to_disk`` directory.

    Returns the bare ``datasets.Dataset`` (no proxy wrapper), so the pipeline
    can select a held-out slice or wrap it once.
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
