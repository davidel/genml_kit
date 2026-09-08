"""Tests for BalancedBatchSampler."""

import numpy as np
import pytest

from genml_kit.datasets.balanced_sampler import BalancedBatchSampler


class TestBalancedBatchSampler:

  def test_batch_size_divisible(self):
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    sampler = BalancedBatchSampler(labels, batch_size=6, samples_per_class=2)
    batches = list(sampler)
    for batch in batches:
      assert len(batch) == 6

  def test_raises_on_indivisible_batch(self):
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="divisible"):
      BalancedBatchSampler(labels, batch_size=5, samples_per_class=2)

  def test_each_group_has_same_class(self):
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 5, size=200)
    sampler = BalancedBatchSampler(labels, batch_size=6, samples_per_class=3)
    for batch in sampler:
      for i in range(0, len(batch), 3):
        group_classes = labels[batch[i:i + 3]]
        assert len(set(group_classes.tolist())) == 1, (
            f"Group {batch[i:i + 3]} has mixed classes: {group_classes}")

  def test_length(self):
    labels = np.arange(100)
    sampler = BalancedBatchSampler(labels, batch_size=10, samples_per_class=2)
    assert len(sampler) == 10  # 100 // 10

  def test_small_class_replaced(self):
    """Class with fewer samples than samples_per_class uses replacement."""
    labels = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    sampler = BalancedBatchSampler(labels, batch_size=4, samples_per_class=2)
    batches = list(sampler)
    assert len(batches) > 0
    for batch in batches:
      assert len(batch) == 4

  def test_fewer_classes_than_groups(self):
    """When there are fewer classes than groups, classes are repeated."""
    labels = np.array([0, 0, 1, 1])
    sampler = BalancedBatchSampler(labels, batch_size=4, samples_per_class=2)
    batches = list(sampler)
    for batch in batches:
      assert len(batch) == 4
      # Each pair must be same class.
      assert labels[batch[0]] == labels[batch[1]]
      assert labels[batch[2]] == labels[batch[3]]

  def test_indices_are_valid(self):
    """All yielded indices must be within [0, len(labels))."""
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    sampler = BalancedBatchSampler(labels, batch_size=6, samples_per_class=2)
    for batch in sampler:
      assert all(0 <= i < len(labels) for i in batch)
