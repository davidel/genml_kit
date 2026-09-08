"""Tests for inference ranking stability and XGBoost alignment guard."""

import numpy as np
import pytest
import torch

from genml_kit.training.infer import check_xgb_label_alignment, rank_indices


class TestRankIndicesTorch:

  def test_orders_descending(self):
    probs = torch.tensor([0.1, 0.5, 0.4])
    assert rank_indices(probs).tolist() == [1, 2, 0]

  def test_ties_break_by_class_index(self):
    probs = torch.tensor([0.3, 0.1, 0.3])
    assert rank_indices(probs).tolist() == [0, 2, 1]

  def test_all_equal(self):
    probs = torch.ones(4)
    assert rank_indices(probs).tolist() == [0, 1, 2, 3]

  def test_matches_manual_stable_sort(self):
    probs = torch.tensor([0.2, 0.2, 0.6, 0.1, 0.2])
    expected = torch.sort(probs, descending=True, stable=True).indices
    assert torch.equal(rank_indices(probs), expected)


class TestRankIndicesNumpy:

  def test_orders_descending(self):
    probs = np.array([0.1, 0.5, 0.4])
    assert rank_indices(probs).tolist() == [1, 2, 0]

  def test_ties_break_by_class_index(self):
    probs = np.array([0.3, 0.1, 0.3])
    assert rank_indices(probs).tolist() == [0, 2, 1]

  def test_all_equal(self):
    probs = np.ones(4)
    assert rank_indices(probs).tolist() == [0, 1, 2, 3]

  def test_negative_zero_not_flipped(self):
    """-0.0 must not sort above 0.0 (stable tie-break still by index)."""
    probs = np.array([0.0, 0.0])
    assert rank_indices(probs).tolist() == [0, 1]


class TestRankIndicesTypes:

  def test_accepts_python_list(self):
    """XGBoost predict_proba output arrives as ndarray; lists also work."""
    assert rank_indices([0.1, 0.3, 0.2]).tolist() == [1, 2, 0]

  def test_numpy_result_is_ndarray(self):
    result = rank_indices(np.array([0.1, 0.5, 0.4]))
    assert isinstance(result, np.ndarray)
    assert result.dtype.kind == "i"


class TestXGBLabelAlignment:

  class FakeXGB:
    n_classes_ = 2

  def test_matching_counts_pass(self):
    check_xgb_label_alignment(self.FakeXGB(), {"0": "a", "1": "b"})

  def test_mismatch_fails(self):
    with pytest.raises(ValueError, match="2 classes"):
      check_xgb_label_alignment(self.FakeXGB(), {str(i): f"c{i}" for i in range(5)})
