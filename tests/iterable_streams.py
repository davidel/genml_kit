"""Synthetic sources and helpers shared by the iterable-dataset tests.

Every item is tagged with its own integer id so a test can compare the exact
multiset of items that reaches the model, independently of tensor values.  The
streaming source deliberately exposes **no** ``__len__``: any accidental length
dependency in the code under test then fails loudly with ``TypeError`` instead
of being papered over by a plausible number.
"""

import random

import torch
from torch.utils.data import IterableDataset


class CountingMapDataset:
  """Random-access twin of :class:`CountingStream`: ``len`` + ``getitem``."""

  def __init__(self, n):
    self.n = n

  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.n:
      raise IndexError(idx)
    return {"id": idx}

  def __iter__(self):
    for i in range(self.n):
      yield {"id": i}


class CountingStream(IterableDataset):
  """One-pass streaming source with no ``__len__`` and no ``__getitem__``."""

  def __init__(self, n, nsamples=None, epoch=None):
    self.n = n
    if nsamples is not None:
      self.nsamples = nsamples
    self._epoch = epoch

  def set_epoch(self, epoch):
    self._epoch = epoch

  def __iter__(self):
    for i in range(self.n):
      yield {"id": i}


class ShuffledStream(CountingStream):
  """A stream that permutes per epoch but never loses or repeats an item."""

  def __iter__(self):
    ids = list(range(self.n))
    if self._epoch is not None:
      random.Random(self._epoch).shuffle(ids)
    yield from ({"id": i} for i in ids)


class NonResumableStream(CountingStream):
  """A stream that yields different items on every iteration.

  Used to prove the resume path warns instead of silently continuing from an
  unreproducible position.
  """

  def __init__(self, n):
    super().__init__(n)
    self._iteration = 0

  def __iter__(self):
    self._iteration += 1
    offset = (self._iteration - 1) * self.n
    yield from ({"id": offset + i} for i in range(self.n))


def ids(batches):
  """Flatten a batch iterable to the flat list of ``id`` values."""
  out = []
  for batch in batches:
    values = batch["id"] if isinstance(batch, dict) else batch
    if isinstance(values, torch.Tensor):
      values = values.tolist()
    if isinstance(values, (list, tuple)):
      out.extend(int(v) for v in values)
    else:
      out.append(int(values))
  return out


def assert_multiset_equal(got, want):
  """Assert *got* is exactly *want* as a multiset (order-insensitive)."""
  assert sorted(got) == sorted(want), (
      f"multiset mismatch: missing={sorted(set(want) - set(got))[:10]} "
      f"extra={sorted(set(got) - set(want))[:10]} "
      f"len(got)={len(got)} len(want)={len(want)}")


def assert_no_duplicates(values):
  """Assert every value in *values* occurs exactly once."""
  assert len(values) == len(set(values)), (
      f"{len(values) - len(set(values))} duplicated id(s), "
      f"e.g. {sorted(set(v for v in values if values.count(v) > 1))[:10]}")


def assert_disjoint(chunks):
  """Assert the given id collections are pairwise disjoint."""
  seen = set()
  for i, chunk in enumerate(chunks):
    overlap = seen & set(chunk)
    assert not overlap, f"chunk {i} overlaps earlier chunks on {sorted(overlap)[:10]}"
    seen |= set(chunk)
