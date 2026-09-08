"""Tests for model_utils.py extract_features and collect_features."""

from collections import namedtuple

import numpy as np
import torch

from genml_kit.training.model_utils import collect_features, extract_features


class TestExtractFeatures:
  """Tests for extract_features()."""

  def test_uses_pooler_output_when_present(self):
    """Should return pooler_output when it is not None."""
    OutputWithPooler = namedtuple("Output", ["pooler_output", "last_hidden_state"])
    pooler = torch.randn(2, 1024)
    last_hidden = torch.randn(2, 1024, 7, 7)
    outputs = OutputWithPooler(pooler_output=pooler, last_hidden_state=last_hidden)
    result = extract_features(outputs)
    assert result is pooler
    assert result.shape == (2, 1024)

  def test_falls_back_to_gap_when_pooler_none(self):
    """Should apply global average pooling when pooler_output is None."""
    OutputNoPooler = namedtuple("Output", ["pooler_output", "last_hidden_state"])
    last_hidden = torch.randn(2, 512, 14, 14)
    outputs = OutputNoPooler(pooler_output=None, last_hidden_state=last_hidden)
    result = extract_features(outputs)
    # Should be mean over spatial dims [-2, -1]
    expected = last_hidden.mean([-2, -1])
    assert result.shape == (2, 512)
    assert torch.allclose(result, expected)

  def test_batch_size_preserved(self):
    """Output batch dimension should match input."""
    OutputWithPooler = namedtuple("Output", ["pooler_output", "last_hidden_state"])
    outputs = OutputWithPooler(
        pooler_output=torch.randn(5, 768),
        last_hidden_state=torch.randn(5, 768, 16, 16),
    )
    result = extract_features(outputs)
    assert result.shape[0] == 5


class _TinyBackbone(torch.nn.Module):
  """Mock backbone that returns a dict-like with pooler_output."""

  def __init__(self, hidden_size=64):
    super().__init__()
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    self.proj = torch.nn.Linear(3, hidden_size)  # project channels → hidden_size
    self.hidden_size = hidden_size

  def forward(self, pixel_values):
    # pixel_values: [B, 3, H, W]
    pooled = self.pool(pixel_values).flatten(1)  # [B, 3]
    projected = self.proj(pooled)  # [B, hidden_size]
    Output = namedtuple("Output", ["pooler_output", "last_hidden_state"])
    return Output(pooler_output=projected, last_hidden_state=pixel_values)


class _TinyClassifier(torch.nn.Module):
  """Mock ConvNextV2ForImageClassification-like model."""

  def __init__(self, hidden_size=64):
    super().__init__()
    self.convnextv2 = _TinyBackbone(hidden_size)
    self.classifier = torch.nn.Linear(hidden_size, 3)

  def forward(self, pixel_values, **kwargs):
    out = self.convnextv2(pixel_values)
    logits = self.classifier(out.pooler_output)
    LogitsOutput = namedtuple("LogitsOutput", ["logits"])
    return LogitsOutput(logits=logits)


class _SimpleDataset:
  """Minimal dataset returning (image_tensor, label) pairs."""

  def __init__(self, n=20, channels=3, height=32, width=32, num_classes=3):
    self.n = n
    self.channels = channels
    self.height = height
    self.width = width
    rng = np.random.RandomState(0)
    self.labels = rng.randint(0, num_classes, size=n)
    # Pre-generate images for determinism.
    self.images = [torch.randn(channels, height, width) for _ in range(n)]

  def __len__(self):
    return self.n

  def __getitem__(self, idx):
    return self.images[idx], self.labels[idx]


class TestCollectFeatures:
  """Tests for collect_features()."""

  def test_output_shapes(self):
    """Features should be [N, hidden_size], labels should be [N]."""
    model = _TinyClassifier(hidden_size=64)
    dataset = _SimpleDataset(n=20, channels=3, height=32, width=32)
    device = torch.device("cpu")

    features, labels = collect_features(model, dataset, device, batch_size=8)

    assert features.shape == (20, 64)
    assert labels.shape == (20,)
    assert features.dtype == np.float32

  def test_labels_match_dataset(self):
    """Labels returned should match the dataset labels."""
    dataset = _SimpleDataset(n=15, channels=3, height=32, width=32)
    model = _TinyClassifier(hidden_size=64)
    device = torch.device("cpu")

    _, labels = collect_features(model, dataset, device, batch_size=5)
    np.testing.assert_array_equal(labels, dataset.labels)

  def test_deterministic(self):
    """Running collect_features twice should give the same output."""
    model = _TinyClassifier(hidden_size=64)
    dataset = _SimpleDataset(n=16, channels=3, height=32, width=32)
    device = torch.device("cpu")

    feat1, lab1 = collect_features(model, dataset, device, batch_size=4)
    feat2, lab2 = collect_features(model, dataset, device, batch_size=4)

    np.testing.assert_array_equal(feat1, feat2)
    np.testing.assert_array_equal(lab1, lab2)

  def test_batch_size_effect(self):
    """Different batch sizes should produce the same result."""
    model = _TinyClassifier(hidden_size=64)
    dataset = _SimpleDataset(n=20, channels=3, height=32, width=32)
    device = torch.device("cpu")

    feat_a, _ = collect_features(model, dataset, device, batch_size=3)
    feat_b, _ = collect_features(model, dataset, device, batch_size=7)
    feat_c, _ = collect_features(model, dataset, device, batch_size=20)

    np.testing.assert_allclose(feat_a, feat_b, atol=1e-6)
    np.testing.assert_allclose(feat_a, feat_c, atol=1e-6)

  def test_mode_restored(self):
    """collect_features must not leave the model in a different mode."""
    model = _TinyClassifier(hidden_size=64)
    dataset = _SimpleDataset(n=8, channels=3, height=32, width=32)
    device = torch.device("cpu")

    model.train()
    collect_features(model, dataset, device, batch_size=4)
    assert model.training

    model.eval()
    collect_features(model, dataset, device, batch_size=4)
    assert not model.training

  def test_eval_during_extraction(self):
    """Model must be in eval mode while features are extracted."""
    model = _TinyClassifier(hidden_size=64)
    dataset = _SimpleDataset(n=8, channels=3, height=32, width=32)
    device = torch.device("cpu")

    seen = []
    handle = model.classifier.register_forward_hook(
        lambda m, i, o: seen.append(model.training))
    collect_features(model, dataset, device, batch_size=4)
    handle.remove()

    assert seen and not any(seen)
