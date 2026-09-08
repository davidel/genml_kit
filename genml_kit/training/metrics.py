"""Human-readable confusion-matrix helpers."""

import numpy as np


def confusion_row_strings(cm, id2label=None, top_n=5, min_prob=0.05):
  """Return one compact line per class for logging a confusion matrix.

  Each line shows the true-class recall (i.e. what fraction of the
  class was predicted correctly) plus the top *top_n* confused classes
  whose probability exceeds *min_prob*.

  Example output for a single class::

      cat: 82.1% | confused with: dog (8.3%), fox (5.2%)

  Parameters
  ----------
  cm : numpy.ndarray
      A 2-D confusion matrix ``(num_classes, num_classes)`` where rows
      are the true labels and columns are the predicted labels — as
      returned by ``sklearn.metrics.confusion_matrix``.
  id2label : dict or None
      Optional mapping from integer class indices to human-readable
      label strings.  If *None*, class indices are formatted as
      ``Class 0``, ``Class 1``, etc.
  top_n : int
      Maximum number of confused classes to display per row.
  min_prob : float
      Minimum fraction (0–1) a confused class must have to be shown.

  Returns
  -------
  list[str]
      One formatted string per class (in order of the confusion-matrix
      rows).
  """
  n_classes = cm.shape[0]
  rows = []

  for true_idx in range(n_classes):
    row = cm[true_idx].astype(np.float64)
    total = row.sum()
    if total == 0:
      continue

    label = _label(id2label, true_idx)
    recall = row[true_idx] / total * 100.0

    # Build list of confused classes (excluding self).
    confused = []
    for pred_idx in range(n_classes):
      if pred_idx == true_idx:
        continue
      prob = row[pred_idx] / total * 100.0
      if prob / 100.0 >= min_prob:
        confused.append((_label(id2label, pred_idx), prob))

    # Sort by probability descending, take top_n.
    confused.sort(key=lambda x: x[1], reverse=True)
    confused = confused[:top_n]

    if confused:
      parts = ", ".join(f"{name} ({p:.1f}%)" for name, p in confused)
      rows.append(f"{label}: {recall:.1f}% | confused with: {parts}")
    else:
      rows.append(f"{label}: {recall:.1f}%")

  return rows


def _label(id2label, idx):
  """Resolve a class index to a human-readable name."""
  if id2label is None:
    return f"Class {idx}"
  return id2label.get(str(idx), id2label.get(idx, f"Class {idx}"))
