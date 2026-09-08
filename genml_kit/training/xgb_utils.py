"""XGBoost training and evaluation helpers."""

import logging

import numpy as np
from xgboost import XGBClassifier

from genml_kit.utils.label import get_label


def train_xgboost(train_features,
                  train_labels,
                  max_depth=6,
                  n_estimators=200,
                  learning_rate=0.1,
                  subsample=0.8,
                  colsample_bytree=0.8,
                  min_child_weight=1,
                  gamma=0.0,
                  reg_alpha=0.0,
                  use_gpu=False,
                  random_state=42):
  """Train an XGBoost classifier on backbone features.

  Args:
      train_features: np.ndarray of shape [N, hidden_size].
      train_labels: np.ndarray of shape [N].
      max_depth: Maximum tree depth.
      n_estimators: Number of trees.
      learning_rate: Boosting learning rate.
      subsample: Row subsampling ratio per tree.
      colsample_bytree: Column subsampling ratio per tree.
      min_child_weight: Minimum sum of instance weight in a child.
      gamma: Minimum loss reduction for a split.
      reg_alpha: L1 regularization term.
      use_gpu: If True, train on GPU with tree_method='hist' and device='cuda'.
      random_state: Seed for XGBoost's row/column sampling.

  Returns:
      Fitted XGBClassifier.
  """
  num_class = len(set(train_labels))
  logging.info(f"Training XGBoost: {len(train_labels)} samples, {num_class} classes")

  xgb_kwargs = {
      "objective": "multi:softprob",
      "num_class": num_class,
      "max_depth": max_depth,
      "n_estimators": n_estimators,
      "learning_rate": learning_rate,
      "subsample": subsample,
      "colsample_bytree": colsample_bytree,
      "min_child_weight": min_child_weight,
      "gamma": gamma,
      "reg_alpha": reg_alpha,
      "random_state": random_state,
      "verbosity": 0,
  }
  if use_gpu:
    xgb_kwargs["tree_method"] = "hist"
    xgb_kwargs["device"] = "cuda"
    logging.info("XGBoost: using GPU (device=cuda)")

  xgb_model = XGBClassifier(**xgb_kwargs)

  xgb_model.fit(train_features, train_labels)
  logging.info("XGBoost training complete")
  return xgb_model


def eval_xgboost(xgb_model, features, labels, id2label=None):
  """Evaluate an XGBoost classifier.

  Args:
      xgb_model: Fitted XGBClassifier.
      features: np.ndarray of shape [N, hidden_size].
      labels: np.ndarray of shape [N].
      id2label: Optional dict mapping class indices to names.

  Returns:
      {"accuracy": float, "per_class_accuracy": dict[str, float],
       "classification_report": str, "confusion_matrix": np.ndarray}
  """
  from sklearn.metrics import classification_report, confusion_matrix

  if id2label is not None and hasattr(xgb_model, "n_classes_") and \
      xgb_model.n_classes_ != len(id2label):
    logging.warning(
        "XGBoost model was trained on %d classes but id2label has %d "
        "entries; report rows may not match the label space.", xgb_model.n_classes_,
        len(id2label))

  # XGBoost 2.x predict() with multi:softprob returns probabilities (2D),
  # not class labels. Use predict_proba + argmax to get class predictions.
  proba = xgb_model.predict_proba(features)
  predictions = np.argmax(proba, axis=1)
  accuracy = np.mean(predictions == labels)

  per_class = {}
  for cls in sorted(set(labels)):
    mask = labels == cls
    cls_acc = np.mean(predictions[mask] == labels[mask])
    try:
      name = get_label(id2label, cls) if id2label else f"CLASS_{cls}"
    except KeyError:
      name = f"CLASS_{cls}"
    per_class[name] = float(cls_acc)

  # Use a stable, explicit class order for reports and confusion matrices.
  if id2label:
    metric_labels = list(range(len(id2label)))
    # Keyed per-class lookup with the same fallback as per_class above,
    # instead of a positional construction that raises a confusing
    # sklearn length error when id2label has no entry for a class in
    # the label space.
    target_names = []
    for cls in metric_labels:
      try:
        target_names.append(get_label(id2label, cls))
      except KeyError:
        target_names.append(f"CLASS_{cls}")
        logging.warning(
            "id2label has no entry for class %d; using placeholder "
            "'CLASS_%d' in the classification report.", cls, cls)
  else:
    metric_labels = sorted(set(labels) | set(predictions))
    target_names = None

  report = classification_report(labels,
                                 predictions,
                                 labels=metric_labels,
                                 target_names=target_names,
                                 zero_division=0)
  if len(metric_labels) == 1:
    cm = np.array([[len(labels)]], dtype=np.int64)
  else:
    cm = confusion_matrix(labels, predictions, labels=metric_labels)

  return {
      "accuracy": float(accuracy),
      "per_class_accuracy": per_class,
      "classification_report": report,
      "confusion_matrix": cm,
  }
