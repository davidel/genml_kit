"""Tests for xgb_utils.py train_xgboost and eval_xgboost."""

import numpy as np

from genml_kit.training.xgb_utils import eval_xgboost, train_xgboost


class TestTrainXGBoost:
  """Tests for train_xgboost()."""

  def test_returns_fitted_classifier(self):
    """Should return a fitted XGBClassifier."""
    rng = np.random.RandomState(42)
    features = rng.randn(50, 32).astype(np.float32)
    labels = rng.randint(0, 3, size=50)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)

    # XGBoost 2.x predict() with multi:softprob returns probabilities (2D).
    proba = clf.predict_proba(features)
    assert proba.shape == (50, 3)
    preds = np.argmax(proba, axis=1)
    assert set(preds).issubset({0, 1, 2})

  def test_predict_proba_shape(self):
    """predict_proba should return [N, num_classes]."""
    rng = np.random.RandomState(42)
    features = rng.randn(40, 16).astype(np.float32)
    labels = rng.randint(0, 4, size=40)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    proba = clf.predict_proba(features)

    assert proba.shape == (40, 4)
    # Each row should sum to ~1.0
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

  def test_overfits_tiny_data(self):
    """With enough trees, should perfectly fit a tiny dataset."""
    rng = np.random.RandomState(0)
    features = rng.randn(20, 8).astype(np.float32)
    labels = (features[:, 0] > 0).astype(int)

    clf = train_xgboost(features, labels, max_depth=4, n_estimators=50)
    proba = clf.predict_proba(features)
    preds = np.argmax(proba, axis=1)

    # Should get high accuracy on training data
    accuracy = np.mean(preds == labels)
    assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy:.1%}"

  def test_deterministic(self):
    """Two runs with same data should produce identical models."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf1 = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    clf2 = train_xgboost(features, labels, max_depth=3, n_estimators=10)

    proba1 = clf1.predict_proba(features)
    proba2 = clf2.predict_proba(features)
    np.testing.assert_allclose(proba1, proba2, atol=1e-6)


class TestEvalXGBoost:
  """Tests for eval_xgboost()."""

  def test_returns_accuracy_and_per_class(self):
    """Should return accuracy, per_class_accuracy, report and confusion matrix."""
    rng = np.random.RandomState(42)
    features = rng.randn(40, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=40)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    result = eval_xgboost(clf, features, labels)

    assert "accuracy" in result
    assert "per_class_accuracy" in result
    assert "classification_report" in result
    assert "confusion_matrix" in result
    assert 0.0 <= result["accuracy"] <= 1.0
    assert isinstance(result["classification_report"], str)
    assert result["confusion_matrix"].shape == (3, 3)

  def test_per_class_uses_id2label(self):
    """When id2label is provided, per_class keys should be label names."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 2, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    id2label = {"0": "benign", "1": "malignant"}
    result = eval_xgboost(clf, features, labels, id2label=id2label)

    assert "benign" in result["per_class_accuracy"]
    assert "malignant" in result["per_class_accuracy"]

  def test_perfect_prediction_gives_100_percent(self):
    """If model predicts perfectly, accuracy should be 1.0."""
    rng = np.random.RandomState(42)
    features = rng.randn(20, 8).astype(np.float32)
    labels = np.array([0, 1] * 10)

    clf = train_xgboost(features, labels, max_depth=4, n_estimators=100)
    result = eval_xgboost(clf, features, labels)

    # With enough trees and a simple pattern, should get high accuracy.
    assert result["accuracy"] >= 0.95, f"Expected >= 95%, got {result['accuracy']:.1%}"

  def test_without_id2label(self):
    """Should work without id2label, using CLASS_N fallback."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    result = eval_xgboost(clf, features, labels)

    # Should have CLASS_0, CLASS_1, CLASS_2
    for key in result["per_class_accuracy"]:
      assert key.startswith("CLASS_")

  def test_single_class_evaluation_has_no_class_warning(self):
    """A single-class evaluation uses an explicit confusion-matrix label."""
    rng = np.random.RandomState(42)
    features = rng.randn(20, 8).astype(np.float32)
    labels = np.zeros(20, dtype=np.int64)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    result = eval_xgboost(clf, features, labels, id2label={"0": "only"})

    assert result["confusion_matrix"].shape == (1, 1)
    assert result["confusion_matrix"].tolist() == [[20]]

  def test_confusion_matrix_sums_to_samples(self):
    """Confusion matrix row sums should equal true label counts."""
    rng = np.random.RandomState(42)
    features = rng.randn(50, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=50)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    result = eval_xgboost(clf, features, labels)

    cm = result["confusion_matrix"]
    np.testing.assert_array_equal(cm.sum(axis=1), np.bincount(labels, minlength=3))


class TestRandomStatePassthrough:

  def test_default_seed_is_42(self):
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    assert clf.random_state == 42

  def test_explicit_seed_propagates(self):
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10, random_state=7)
    assert clf.random_state == 7


class TestEvalLabelSpace:

  def test_id2label_larger_than_seen_labels(self):
    """id2label covering classes absent from the eval split must not crash."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 2, size=30)  # only classes 0 and 1 present

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    id2label = {"0": "a", "1": "b", "2": "c", "3": "d"}
    result = eval_xgboost(clf, features, labels, id2label=id2label)

    # Present classes get per-class accuracy; absent ones still appear
    # in the 4x4 confusion matrix and the classification report.
    assert result["confusion_matrix"].shape == (4, 4)
    assert "a" in result["per_class_accuracy"]
    assert "b" in result["per_class_accuracy"]
    assert "CLASS_2" not in result["per_class_accuracy"]
    assert "d" in result["classification_report"]

  def test_id2label_integer_keys(self):
    """Integer-keyed id2label (custom models) resolves via get_label."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    id2label = {0: "x", 1: "y", 2: "z"}
    result = eval_xgboost(clf, features, labels, id2label=id2label)

    assert "x" in result["per_class_accuracy"]
    assert "y" in result["per_class_accuracy"]
    assert "z" in result["per_class_accuracy"]

  def test_missing_id2label_entry_uses_placeholder(self, caplog):
    """A gap in id2label falls back to CLASS_N and warns instead of raising."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 3, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    id2label = {"0": "a", "1": "b", "3": "d"}  # class 2 missing
    with caplog.at_level("WARNING"):
      result = eval_xgboost(clf, features, labels, id2label=id2label)

    assert "CLASS_2" in result["per_class_accuracy"]
    assert "a" in result["per_class_accuracy"]
    assert any("no entry for class 2" in r.message for r in caplog.records)

  def test_n_classes_mismatch_warns(self, caplog):
    """Model trained on a different class count than id2label warns."""
    rng = np.random.RandomState(42)
    features = rng.randn(30, 16).astype(np.float32)
    labels = rng.randint(0, 2, size=30)

    clf = train_xgboost(features, labels, max_depth=3, n_estimators=10)
    id2label = {str(i): f"c{i}" for i in range(5)}  # 5 vs 2 classes
    with caplog.at_level("WARNING"):
      eval_xgboost(clf, features, labels, id2label=id2label)

    assert any(
        "trained on 2 classes but id2label has 5" in r.message for r in caplog.records)
