"""Build a WeightedRandomSampler for class-imbalanced training."""
import logging

import torch
from torch.utils.data import WeightedRandomSampler

from genml_kit.utils.logging import fatal


def build_weighted_sampler(dataset,
                           num_labels,
                           label_column,
                           weight_mode,
                           multipliers=None):
  """Compute per-sample weights and return a WeightedRandomSampler.

  Uses ``replacement=True`` so that minority-class samples are duplicated
  throughout the epoch, keeping every batch balanced.  With
  ``replacement=False`` minority samples are exhausted early, causing the
  model to collapse to the majority class by end-of-epoch.

  Args:
    dataset: Raw HuggingFace ``Dataset`` (before ``set_transform``).
    num_labels: Number of classes.
    label_column: Name of the label column in the dataset.
    weight_mode: ``"frequency"`` (inverse-freq), ``"multipliers"``
      (per-class priority), or ``"combined"`` (freq × multipliers).
    multipliers: Per-class multiplier tensor. Required when *weight_mode*
      is ``"multipliers"`` or ``"combined"``.

  Returns:
    A ``torch.utils.data.WeightedRandomSampler`` that yields indices with
    probabilities proportional to the computed per-sample weights.
  """
  labels = dataset[label_column]

  class_counts = torch.zeros(num_labels)
  for label in labels:
    class_counts[label] += 1

  # Inverse-frequency weights (normalized so sum == num_labels).
  inv_freq = num_labels / class_counts.clamp(min=1)

  if weight_mode == "frequency":
    class_weights = inv_freq
  elif weight_mode == "multipliers":
    if multipliers is None:
      fatal("multipliers must be provided when weight_mode='multipliers'", ValueError)
    class_weights = multipliers.float()
  elif weight_mode == "combined":
    if multipliers is None:
      fatal("multipliers must be provided when weight_mode='combined'", ValueError)
    class_weights = inv_freq * multipliers.float()
  else:
    fatal(f"Unknown weight_mode: {weight_mode!r}", ValueError)

  sample_weights = [class_weights[label].item() for label in labels]

  sampler = WeightedRandomSampler(
      weights=sample_weights,
      num_samples=len(sample_weights),
      replacement=True,
  )

  logging.info(f"WeightedRandomSampler: {len(sample_weights):,} samples, "
               f"weight_mode={weight_mode!r}")
  for i, w in enumerate(class_weights):
    logging.info(f"  class {i}: count={int(class_counts[i])}, "
                 f"weight={w:.4f}")
  return sampler
