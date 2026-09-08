"""BalancedBatchSampler — ensures uniform class representation per batch."""

import logging

import numpy as np
from torch.utils.data import Sampler

from genml_kit.utils.logging import fatal


class BalancedBatchSampler(Sampler):
  """Yields mini-batches with a guaranteed number of samples per class.

  Each batch contains ``batch_size // samples_per_class`` groups, where
  every group holds ``samples_per_class`` examples from the same class.
  Classes are sampled uniformly at random so that rare classes appear
  with the same frequency as common ones.

  Parameters
  ----------
  labels : array-like of int
      Integer class label for every sample in the dataset.
  batch_size : int
      Total batch size (must be divisible by *samples_per_class*).
  samples_per_class : int
      Number of examples drawn from each class within a batch.
  seed : int, optional
      Seed for the class/index RNG.  When None (default) each epoch's
      batch composition is drawn from fresh OS entropy; pass a seed for
      reproducible batches (a fresh generator is created per ``__iter__``,
      so epoch-to-epoch variation is preserved while runs remain
      repeatable).
  """

  def __init__(self, labels, batch_size, samples_per_class, seed=None):
    labels = np.asarray(labels)
    if batch_size % samples_per_class != 0:
      fatal(
          f"batch_size ({batch_size}) must be divisible by "
          f"samples_per_class ({samples_per_class})", ValueError)

    self._batch_size = batch_size
    self._samples_per_class = samples_per_class
    self._n_groups = batch_size // samples_per_class

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
                 f"batches/epoch={self._n_batches}")

  def __len__(self):
    return self._n_batches

  def __iter__(self):
    rng = np.random.default_rng(self._seed)
    for _ in range(self._n_batches):
      # Pick which classes appear in this batch.
      chosen = rng.choice(self._all_classes, size=self._n_groups, replace=True)
      batch = []
      for cls in chosen:
        pool = self._class_indices[int(cls)]
        chosen_idx = rng.choice(
            pool,
            size=self._samples_per_class,
            replace=len(pool) < self._samples_per_class,
        )
        batch.extend(chosen_idx.tolist())
      yield batch
