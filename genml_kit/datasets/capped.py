"""Cap a HuggingFace dataset to a maximum number of rows.

:class:`CappedDataset` is a thin, read-only *view* over a
``datasets.Dataset``: it reports a truncated length and truncates column
access, forwarding everything else to the wrapped object.  It exists so
``--max_items`` can be honoured in one place at the load boundary, instead
of every dataset wrapper re-implementing the truncation.

Forwarding is generic: any attribute lookup that resolves to a callable is
proxied so that, if it returns a ``Dataset`` or ``DatasetDict``, the result
is re-capped.  This keeps the cap intact across arbitrary dataset-deriving
operations (``map``, ``select``, ``cast_column``, ``class_encode_column``,
...) without enumerating them.
"""

import itertools
import logging

import datasets


def maybe_cap(dataset, limit):
  """Return *dataset* limited to *limit* rows, or unchanged when no cap.

  A falsy or non-positive *limit* is a no-op and returns *dataset* as-is
  (no wrapper, zero overhead).
  """
  if not limit or limit <= 0:
    return dataset
  try:
    total = len(dataset)
  except TypeError:
    # A streaming source has no length; cap it anyway -- ``CappedDataset``
    # iterates through ``islice``, so the cap needs no count.
    logging.info("Capping streaming dataset at %d items.", limit)
    return CappedDataset(dataset, limit)
  if total > limit:
    logging.info("Capping dataset at %d items (from %d).", limit, total)
  return CappedDataset(dataset, limit)


def _recap(result, limit):
  """Re-apply *limit* to a ``Dataset`` / ``DatasetDict`` *result*."""
  if isinstance(result, datasets.DatasetDict):
    return datasets.DatasetDict({
        name: maybe_cap(part, limit) for name, part in result.items()
    })
  if isinstance(result, datasets.Dataset):
    return maybe_cap(result, limit)
  return result


class CappedDataset:
  """Read-only view exposing at most ``limit`` rows of *dataset*.

  Column access (``ds["label"]``) is truncated to ``limit`` so callers
  that derive per-sample arrays (samplers, class weights) stay consistent
  with ``len(ds)``.  Every other attribute is forwarded to the wrapped
  dataset; methods whose result is itself a ``Dataset``/``DatasetDict``
  (``map``, ``select``, ``cast_column``, ``class_encode_column``, ...)
  re-wrap it so the cap survives.
  """

  def __init__(self, dataset, limit):
    self._dataset = dataset
    self._limit = limit

  @property
  def dataset(self):
    """The wrapped (uncapped) dataset."""
    return self._dataset

  @property
  def limit(self):
    """The maximum number of visible rows."""
    return self._limit

  def __len__(self):
    return min(len(self._dataset), self._limit)

  def __getitem__(self, key):
    value = self._dataset[key]
    if isinstance(key, str):
      # Column access: truncate so length-derived logic (samplers,
      # class weights) agrees with ``len(self)``.
      return value[:self._limit]
    return value

  def __iter__(self):
    # A cap is a natural stream operation: take at most ``_limit`` items and
    # stop pulling from the source.  ``islice`` never asks the source for a
    # length, so this works for random-access and streaming sources alike.
    return itertools.islice(iter(self._dataset), self._limit)

  def __getattr__(self, name):
    # Reached only for attributes not defined on the view itself.  The
    # underscore guard prevents infinite recursion when ``_dataset`` is not
    # yet set (e.g. mid-unpickle).
    if name.startswith("_"):
      raise AttributeError(name)
    attr = getattr(self._dataset, name)
    if not callable(attr):
      return attr

    def _forwarded(*args, **kwargs):
      return _recap(attr(*args, **kwargs), self._limit)

    return _forwarded
