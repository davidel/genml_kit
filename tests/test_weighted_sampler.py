"""Tests for genml_kit.datasets.weighted_sampler."""
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from genml_kit.datasets.weighted_sampler import build_weighted_sampler


class _FakeDataset:
  """Minimal dataset backed by a list of label dicts."""

  def __init__(self, labels):
    self._data = [{"label": int(lbl)} for lbl in labels]

  def __len__(self):
    return len(self._data)

  def __getitem__(self, key):
    if isinstance(key, str):
      if key == "label":
        return [d["label"] for d in self._data]
      raise KeyError(key)
    return self._data[key]


class TestBuildWeightedSampler:

  def test_frequency_weights_inversely_proportional(self):
    """Rare classes should get higher weights than common classes."""
    # 90 samples of class 0, 10 samples of class 1.
    labels = [0] * 90 + [1] * 10
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     label_column="label",
                                     weight_mode="frequency")
    assert isinstance(sampler, WeightedRandomSampler)
    weights = list(sampler.weights)
    # Class 1 weight should be ~9x class 0 weight.
    assert weights[90] > weights[0] * 5

  def test_frequency_weights_num_samples(self):
    labels = [0, 0, 1, 1, 2]
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=3,
                                     label_column="label",
                                     weight_mode="frequency")
    assert sampler.num_samples == 5

  def test_combined_weights_match_loss_weights(self):
    """Combined mode should multiply inv_freq by multipliers."""
    labels = [0] * 80 + [1] * 20
    ds = _FakeDataset(labels)
    multipliers = torch.tensor([2.0, 1.0])
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     label_column="label",
                                     weight_mode="combined",
                                     multipliers=multipliers)
    weights = list(sampler.weights)
    # Class 0: inv_freq=2/80=0.025, weight=0.025*2.0=0.05
    # Class 1: inv_freq=2/20=0.1,   weight=0.1*1.0=0.1
    # So class 1 should still be higher (freq dominates).
    assert weights[80] > weights[0]

  def test_multipliers_mode(self):
    """Multipliers mode should use the multiplier values directly."""
    labels = [0] * 50 + [1] * 50
    ds = _FakeDataset(labels)
    multipliers = torch.tensor([3.0, 1.0])
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     label_column="label",
                                     weight_mode="multipliers",
                                     multipliers=multipliers)
    weights = list(sampler.weights)
    # Equal counts, so weights = multipliers.
    assert abs(weights[0] - 3.0) < 1e-5
    assert abs(weights[50] - 1.0) < 1e-5

  def test_balanced_batches_throughout_epoch(self):
    """With replacement=True, minority classes appear in late batches."""
    labels = [0] * 90 + [1] * 10
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=2,
                                     label_column="label",
                                     weight_mode="frequency")
    indices = list(sampler)
    assert len(indices) == 100
    # Minority class indices (90-99) should appear in the last half.
    last_half = indices[50:]
    has_minority = any(90 <= i <= 99 for i in last_half)
    assert has_minority, "Minority class missing from late batches"

  def test_missing_multipliers_raises(self):
    ds = _FakeDataset([0, 1])
    with pytest.raises(ValueError):
      build_weighted_sampler(ds,
                             num_labels=2,
                             label_column="label",
                             weight_mode="combined")

  def test_invalid_weight_mode_raises(self):
    ds = _FakeDataset([0, 1])
    with pytest.raises(ValueError):
      build_weighted_sampler(ds,
                             num_labels=2,
                             label_column="label",
                             weight_mode="bogus")

  def test_single_class(self):
    labels = [0] * 100
    ds = _FakeDataset(labels)
    sampler = build_weighted_sampler(ds,
                                     num_labels=1,
                                     label_column="label",
                                     weight_mode="frequency")
    indices = list(sampler)
    assert len(indices) == 100
    # Single class: all indices 0-99 valid; duplicates OK with replacement.
    assert all(0 <= i < 100 for i in indices)
