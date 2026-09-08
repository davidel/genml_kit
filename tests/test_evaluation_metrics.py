"""Tests for validation metric reporting."""

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from genml_kit.training.train import evaluate_performance


class _FixedPredictionModel(torch.nn.Module):
  """Return logits whose predicted class is encoded in each input value."""

  def __init__(self, predictions, num_labels):
    super().__init__()
    self.predictions = predictions
    self.config = SimpleNamespace(num_labels=num_labels)

  def forward(self, pixel_values):
    logits = torch.full((pixel_values.size(0), self.config.num_labels), -10.0)
    for row, prediction in enumerate(pixel_values[:, 0].long().tolist()):
      logits[row, prediction] = 10.0
    return SimpleNamespace(logits=logits)


def test_evaluate_performance_preserves_metrics_for_absent_class():
  """Metrics retain model class ordering when a class is absent."""
  images = torch.tensor([[0.0], [1.0], [1.0]])
  targets = torch.tensor([0, 0, 1])
  loader = DataLoader(TensorDataset(images, targets), batch_size=3)
  model = _FixedPredictionModel(predictions=[0, 1, 1], num_labels=3)

  result = evaluate_performance(
      model,
      loader,
      torch.nn.CrossEntropyLoss(),
      torch.device("cpu"),
      amp_dtype=None,
      id2label={
          0: "zero",
          1: "one",
          2: "two"
      },
  )

  (_, top1, balanced_accuracy, macro_f1, weighted_f1, per_class, cm,
   original_metrics) = result
  assert top1 == 66.66666666666666
  assert balanced_accuracy == 50.0
  assert macro_f1 == (4.0 / 9.0) * 100.0
  assert weighted_f1 == (2.0 / 3.0) * 100.0
  assert per_class["zero"] == {
      "precision": 100.0,
      "recall": 50.0,
      "f1": (2.0 / 3.0) * 100.0,
      "support": 2,
  }
  assert per_class["two"] == {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "support": 0,
  }
  assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
  assert original_metrics is None
