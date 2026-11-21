"""Post-hoc calibration and reliability metrics for classification.

The scdiag workflow (and any imbalanced clinical setting) needs more than
top-1: a model whose confidence is badly miscalibrated produces probabilities
that cannot be trusted as risk scores.  This module adds the standard tools:

* :func:`expected_calibration_error` — ECE over equal-width confidence bins
  (the classic reliability-diagram statistic, Guo et al. 2017).
* :func:`fit_temperature` — the simplest post-hoc fix: a single scalar ``T``
  dividing the logits before the softmax, fitted on held-out data by
  minimizing NLL.  Temperature scaling does not change the argmax, so top-1
  and all threshold-free metrics are untouched; only the *confidences*
  change.
* :func:`auroc_macro_ovr` — macro one-vs-rest AUROC over the confusion
  matrix's class list, complementing the existing balanced-accuracy /
  macro-F1 block in ``training/eval.py``.

Fitted temperatures are persisted next to the checkpoint as
``temperature.json`` (see ``cli/train.py``) and applied at inference via
``cli/infer.py --temperature``.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def expected_calibration_error(logits, labels, n_bins=15):
  """Expected calibration error over equal-width confidence bins.

  ECE = sum over bins of (bin_count / N) * |accuracy(bin) - confidence(bin)|.

  Args:
      logits: ``(N, C)`` float tensor or ndarray of unnormalized logits.
      labels: ``(N,)`` integer labels.
      n_bins: Number of equal-width bins over the confidence range [0, 1].

  Returns:
      ECE in [0, 1] (0.0 for a perfectly calibrated model).
  """
  if isinstance(logits, torch.Tensor):
    logits = logits.detach().cpu().numpy()
  if isinstance(labels, torch.Tensor):
    labels = labels.detach().cpu().numpy()
  probs = _softmax(logits)
  confidences = probs.max(axis=1)
  predictions = probs.argmax(axis=1)
  accuracies = (predictions == labels).astype(np.float64)

  n = len(labels)
  bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
  ece = 0.0
  for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    in_bin = (confidences > lo) & (confidences <= hi)
    if not in_bin.any():
      continue
    # Equal-width bins; the last edge is inclusive so max-confidence
    # samples (confidence == 1.0) are not dropped.
    prop = in_bin.sum() / n
    ece += prop * abs(accuracies[in_bin].mean() - confidences[in_bin].mean())
  return float(ece)


def fit_temperature(logits, labels, max_iter=100, lr=0.05):
  """Fit the temperature that minimizes NLL on held-out data.

  Args:
      logits: ``(N, C)`` float tensor of *unnormalized* logits.
      labels: ``(N,)`` integer labels.

  Returns:
      A float >= 1e-6.  ``T > 1`` means the model is overconfident (logits
      are too sharp); ``T < 1`` means underconfident.  ``T == 1`` leaves the
      softmax unchanged.
  """
  if not isinstance(logits, torch.Tensor):
    logits = torch.as_tensor(logits, dtype=torch.float32)
  if not isinstance(labels, torch.Tensor):
    labels = torch.as_tensor(labels, dtype=torch.long)
  log_T = torch.zeros(1, requires_grad=True)
  opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

  def _nll():
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(logits / log_T.exp(), labels)
    loss.backward()
    return loss

  opt.step(_nll)
  with torch.no_grad():
    temperature = float(log_T.exp())
  return max(temperature, 1e-6)


def auroc_macro_ovr(logits, labels, num_labels=None):
  """Macro one-vs-rest AUROC; ``None`` when a class is absent.

  Computed with scikit-learn (already a dependency of ``training/eval.py``).
  Binary problems use the positive-class score (sklearn rejects
  ``multi_class="ovr"`` for 2-column inputs); multiclass problems use the
  macro average of the per-class one-vs-rest AUROCs.
  """
  if isinstance(logits, torch.Tensor):
    logits = logits.detach().cpu().numpy()
  if isinstance(labels, torch.Tensor):
    labels = labels.detach().cpu().numpy()
  if num_labels is None:
    num_labels = int(labels.max()) + 1
  if len(np.unique(labels)) < 2:
    return None
  probs = _softmax(logits)
  if num_labels <= 2:
    return float(roc_auc_score(labels, probs[:, 1]))
  return float(
      roc_auc_score(labels, probs, multi_class="ovr",
                    average="macro", labels=list(range(num_labels))))


def _softmax(logits):
  """Row-wise softmax for ``(N, C)`` numpy arrays (numerically stable)."""
  shifted = logits - logits.max(axis=1, keepdims=True)
  exp = np.exp(shifted)
  return exp / exp.sum(axis=1, keepdims=True)
