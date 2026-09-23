"""BalancedBatchSampler — ensures uniform class representation per batch."""

import logging

import numpy as np
from torch.utils.data import Sampler

from genml_kit.utils.logging import fatal


class BalancedBatchSampler(Sampler):
  """Yields mini-batches with a guaranteed number of samples per class.

  Each batch is split into ``ceil(batch_size / samples_per_class)`` groups,
  where every group holds examples from a single class.  When *batch_size*
  is not a multiple of *samples_per_class* the shortfall is spread evenly
  (round-robin) across the groups, so group sizes differ by at most one
  and the batch always contains exactly *batch_size* samples.  A batch
  smaller than *samples_per_class* is rejected: it cannot hold a full
  class group.

  Classes are sampled uniformly at random so that rare classes appear
  with the same frequency as common ones.

  Parameters
  ----------
  labels : array-like of int
      Integer class label for every sample in the dataset.
  batch_size : int
      Total batch size.  Must be at least *samples_per_class*; need not be
      divisible by it (remainder samples are spread evenly across groups).
  samples_per_class : int
      Number of examples drawn from each class within a batch.  When the
      batch is not divisible, groups hold either this many or one fewer
      sample.
  seed : int, optional
      Seed for the class/index RNG.  When None (default) each epoch's
      batch composition is drawn from fresh OS entropy; pass a seed for
      reproducible batches (a fresh generator is created per ``__iter__``,
      so epoch-to-epoch variation is preserved while runs remain
      repeatable).
  """

  def __init__(self, labels, batch_size, samples_per_class, seed=None):
    labels = np.asarray(labels)
    if batch_size < samples_per_class:
      fatal(
          f"batch_size ({batch_size}) must be at least samples_per_class "
          f"({samples_per_class})", ValueError)
    if batch_size % samples_per_class != 0:
      logging.warning(
          f"BalancedBatchSampler: batch_size ({batch_size}) is not divisible "
          f"by samples_per_class ({samples_per_class}); the remainder will be "
          f"spread evenly across groups so group sizes differ by at most one.")

    self._batch_size = batch_size
    self._samples_per_class = samples_per_class
    # Ceil division.
    self._n_groups = -(-batch_size // samples_per_class)
    self._group_sizes = self._even_group_sizes(batch_size, self._n_groups)

    # Build per-class index lists.
    self._class_indices = {}
    for cls in np.unique(labels):
      self._class_indices[int(cls)] = np.where(labels == cls)[0]

    n_classes = len(self._class_indices)
    if n_classes < self._n_groups:
      logging.warning(f"BalancedBatchSampler: only {n_classes} classes available "
                      f"but batch needs {self._n_groups} groups.  Some classes "
                      f"will be oversampled within each batch.")

    self._all_classes = np.array(list(self._class_indices.keys()))
    self._n_batches = len(labels) // batch_size
    self._seed = seed
    logging.info(f"BalancedBatchSampler: {len(labels):,} samples, "
                 f"{n_classes} classes, batch_size={batch_size}, "
                 f"samples_per_class={samples_per_class}, "
                 f"group_sizes={self._group_sizes}, "
                 f"batches/epoch={self._n_batches}")

  @staticmethod
  def _even_group_sizes(batch_size, n_groups):
    """Distribution of samples across groups.

    Groups hold ``batch_size // n_groups`` samples each, except the first
    ``batch_size % n_groups`` groups which hold one extra -- the sizes
    differ by at most one and sum to *batch_size*.
    """
    base = batch_size // n_groups
    extra = batch_size % n_groups
    return [base + 1] * extra + [base] * (n_groups - extra)

  def __len__(self):
    return self._n_batches

  def __iter__(self):
    rng = np.random.default_rng(self._seed)
    for _ in range(self._n_batches):
      # Pick which classes appear in this batch.
      chosen = rng.choice(self._all_classes, size=self._n_groups, replace=True)
      batch = []
      for cls, group_size in zip(chosen, self._group_sizes):
        pool = self._class_indices[int(cls)]
        chosen_idx = rng.choice(
            pool,
            size=group_size,
            replace=len(pool) < group_size,
        )
        batch.extend(chosen_idx.tolist())
      yield batch
