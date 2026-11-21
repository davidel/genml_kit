"""Tests for calibration: ECE, temperature scaling, AUROC."""

import math

import numpy as np
import pytest
import torch

from genml_kit.training.calibration import (
    auroc_macro_ovr,
    expected_calibration_error,
    fit_temperature,
    _softmax,
)


class TestEce:

  def test_perfectly_calibrated_has_zero_ece(self):
    # All samples have confidence 1/C and accuracy 1/C: single bin,
    # |acc - conf| = 0 -> ECE = 0.
    n_classes = 4
    n_per_class = 100
    labels = np.repeat(np.arange(n_classes), n_per_class)
    # Uniform logits -> confidence 1/C for every sample.
    logits = np.zeros((n_classes * n_per_class, n_classes))
    ece = expected_calibration_error(logits, labels, n_bins=10)
    # Predictions are argmax=0, so accuracy = 1/4 == confidence -> zero ECE.
    assert ece == pytest.approx(0.0, abs=1e-6)

  def test_overconfident_model_has_high_ece(self):
    # Confidence 1.0 everywhere, but wrong half the time.
    n = 100
    labels = np.array([i % 2 for i in range(n)])
    logits = np.full((n, 2), -20.0)
    logits[np.arange(n), 0] = 20.0  # always predicts class 0
    ece = expected_calibration_error(logits, labels, n_bins=10)
    # Single bin: |acc - conf| = |0.5 - 1.0| = 0.5.
    assert ece == pytest.approx(0.5, abs=1e-6)

  def test_perfect_predictions_zero_ece(self):
    n = 64
    labels = np.arange(n) % 3
    logits = np.full((n, 3), -10.0)
    logits[np.arange(n), labels] = 10.0
    assert expected_calibration_error(logits, labels) == pytest.approx(0.0,
                                                                       abs=1e-6)

  def test_ece_bounded_by_one(self):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 5))
    labels = rng.integers(0, 5, size=200)
    ece = expected_calibration_error(logits, labels)
    assert 0.0 <= ece <= 1.0

  def test_accepts_torch_inputs(self):
    # Both predictions correct, confidence sigmoid(2) ~ 0.881: a single
    # bin with |acc - conf| = 1 - 0.881.
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    conf = 1.0 / (1.0 + math.exp(-2.0))
    assert expected_calibration_error(logits, labels) == pytest.approx(
        1.0 - conf, abs=1e-6)

  def test_confidence_one_samples_are_binned(self):
    # confidence == 1.0 must land in the last bin (right-edge inclusive).
    n = 10
    logits = np.full((n, 2), -20.0)
    logits[np.arange(n), 0] = 20.0
    labels = np.zeros(n, dtype=int)
    ece = expected_calibration_error(logits, labels, n_bins=5)
    assert ece == pytest.approx(0.0, abs=1e-6)


class TestTemperature:

  def test_identity_when_already_calibrated(self):
    # A model with zero logits is perfectly calibrated (uniform softmax);
    # the fitted temperature should be ~1.
    logits = torch.zeros(50, 4)
    labels = torch.arange(50) % 4
    t = fit_temperature(logits, labels)
    assert t == pytest.approx(1.0, abs=0.05)

  def test_overconfident_model_gets_t_gt_one(self):
    # Sharpened logits with random labels -> the model is overconfident;
    # a fitted T must exceed 1 to soften it.
    torch.manual_seed(0)
    logits = torch.randn(300, 6) * 8.0
    labels = torch.randint(0, 6, (300,))
    t = fit_temperature(logits, labels)
    assert t > 1.0

  def test_temperature_preserves_argmax(self):
    # Temperature scaling must not change top-1 predictions.
    torch.manual_seed(1)
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    t = fit_temperature(logits, labels)
    before = logits.argmax(1)
    after = (logits / t).argmax(1)
    assert torch.equal(before, after)

  def test_fitted_t_reduces_nll(self):
    # The fitted temperature should not increase NLL relative to T=1.
    torch.manual_seed(2)
    logits = torch.randn(200, 4) * 5.0
    labels = torch.randint(0, 4, (200,))
    t = fit_temperature(logits, labels)
    nll_at_one = torch.nn.functional.cross_entropy(logits, labels).item()
    nll_at_t = torch.nn.functional.cross_entropy(logits / t, labels).item()
    assert nll_at_t <= nll_at_one + 1e-6

  def test_perfect_logits_stay_at_one(self):
    # A separable problem with confident correct predictions: the optimum
    # is T -> 1 (softmax already saturated).
    logits = torch.full((20, 2), -10.0)
    labels = torch.arange(20) % 2
    logits[torch.arange(20), labels] = 10.0
    t = fit_temperature(logits, labels)
    assert t == pytest.approx(1.0, abs=0.1)


class TestAuroc:

  def test_perfect_separation_is_one(self):
    n = 30
    labels = np.arange(n) % 3
    logits = np.full((n, 3), -10.0)
    logits[np.arange(n), labels] = 10.0
    assert auroc_macro_ovr(logits, labels, num_labels=3) == pytest.approx(1.0)

  def test_random_logits_give_midpoint(self):
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(300, 3))
    labels = rng.integers(0, 3, size=300)
    auroc = auroc_macro_ovr(logits, labels, num_labels=3)
    assert 0.3 < auroc < 0.7

  def test_single_class_returns_none(self):
    logits = np.zeros((10, 3))
    labels = np.zeros(10, dtype=int)
    assert auroc_macro_ovr(logits, labels) is None

  def test_accepts_torch_inputs(self):
    logits = torch.randn(60, 2)
    labels = torch.randint(0, 2, (60,))
    auroc = auroc_macro_ovr(logits, labels)
    assert 0.0 < auroc < 1.0


class TestSoftmax:

  def test_rows_sum_to_one(self):
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(16, 7)) * 50  # large values: stability check
    probs = _softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()

  def test_argmax_matches_logits_argmax(self):
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(20, 4))
    assert (_softmax(logits).argmax(1) == logits.argmax(1)).all()
