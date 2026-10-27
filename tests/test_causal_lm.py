"""Tests for the causal_lm method (methods/causal_lm.py).

Uses a tiny hand-written model so the shifted cross-entropy can be checked
against a directly computed reference, plus the registry/metric contract.
"""

import pytest
import torch

from genml_kit.methods import METHODS, build_method
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.training.text_collator import IGNORE_INDEX


class _FakeOutput:

  def __init__(self, logits):
    self.logits = logits


class _FakeLM(torch.nn.Module):
  """Deterministic LM: logits are a fixed function of the input ids."""

  def __init__(self, vocab_size=10, hidden=None):
    super().__init__()
    self.vocab_size = vocab_size

  def forward(self, input_ids, attention_mask=None):
    del attention_mask
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
    # Put a large logit on id 0 for every position.
    logits[..., 0] = 5.0
    return _FakeOutput(logits)

  def train(self, mode=True):
    return self

  def eval(self):
    return self


class TestRegistryAndContract:

  def test_causal_lm_registered(self):
    assert "causal_lm" in METHODS.list_names()
    assert METHODS.get("causal_lm").NAME == "causal_lm"

  def test_metric_key_and_direction(self):
    method = build_method("causal_lm")
    assert method.METRIC_KEY == "ppl"
    assert method.METRIC_MINIMIZE is True
    # Lower perplexity is better.
    assert method.has_metric_improved(1.0, 10.0) is True
    assert method.has_metric_improved(10.0, 1.0) is False


class TestTrainStep:

  def test_shift_is_left_to_right(self):
    """The loss must compare logits[t] with labels[t+1]."""
    method = build_method("causal_lm")
    model = _FakeLM(vocab_size=3)
    # Labels: positions 0..2 -> [1, 0, 0].  After shifting, the model
    # predicts labels at positions 1..2 == [0, 0], both of which the model
    # nails (logit 0 is largest), so the loss is the small "correct" value.
    input_ids = torch.tensor([[1, 0, 0]])
    labels = torch.tensor([[1, 0, 0]])
    blob = DataBlob(data={
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    },
                    meta={"labels": labels})
    out = method.train_step(model, blob, 0)
    # Cross-entropy of a one-hot target against logits [5, 0, 0].
    expected = torch.nn.functional.cross_entropy(torch.tensor([[5.0, 0.0, 0.0]]),
                                                 torch.tensor([0]))
    assert torch.allclose(out.loss, expected, atol=1e-6)

  def test_ignore_index_positions_excluded(self):
    method = build_method("causal_lm")
    model = _FakeLM(vocab_size=3)
    input_ids = torch.tensor([[1, 0, 0, 0]])
    # Only the last label is real; the rest are masked.
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 0]])
    blob = DataBlob(data={
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    },
                    meta={"labels": labels})
    out = method.train_step(model, blob, 0)
    assert torch.isfinite(out.loss)

  def test_missing_labels_is_fatal(self):
    method = build_method("causal_lm")
    model = _FakeLM()
    blob = DataBlob(data={
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]])
    },
                    meta={"labels": None})
    with pytest.raises(ValueError, match="requires meta"):
      method.train_step(model, blob, 0)

  def test_ppl_is_a_derived_metric(self):
    """``ppl`` is a DERIVED metric (exp of the reported loss), not an
    averaged per-step value."""
    import math
    from genml_kit.training.train_reporting import MetricKind, derived_metric
    method = build_method("causal_lm")
    model = _FakeLM(vocab_size=3)
    input_ids = torch.tensor([[1, 0, 0]])
    labels = torch.tensor([[1, 0, 0]])
    blob = DataBlob(data={
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids)
    },
                    meta={"labels": labels})
    out = method.train_step(model, blob, 0)
    ppl = out.metrics["ppl"]
    assert isinstance(ppl, type(derived_metric("x", math.exp)))
    assert ppl.kind is MetricKind.DERIVED
    assert ppl.derive({"loss": 2.0}) == pytest.approx(math.exp(2.0))
    # The raw per-step exp(loss) is not carried as an averaged tensor.
    assert not torch.is_tensor(ppl.value)


class TestEvaluate:

  def test_evaluate_reports_loss_and_ppl(self):
    method = build_method("causal_lm")
    model = _FakeLM(vocab_size=3)
    blob = DataBlob(data={
        "input_ids": torch.tensor([[1, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    },
                    meta={"labels": torch.tensor([[1, 0, 0]])})
    report = method.evaluate(model, [blob],
                             torch.device("cpu"),
                             to_device=lambda b, d: b)
    assert "loss" in report and "ppl" in report
    assert report["ppl"] == pytest.approx(float(torch.exp(torch.tensor(
        report["loss"]))),
                                          rel=1e-5)
