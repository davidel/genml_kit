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
                             to_device=lambda b, d: b).metrics
    assert "loss" in report and "ppl" in report
    assert report["ppl"] == pytest.approx(float(torch.exp(torch.tensor(
        report["loss"]))),
                                          rel=1e-5)


class _RecordingLM(torch.nn.Module):
  """LM that records the kwargs its forward was called with."""

  PACKED_ATTENTION = "none"

  def __init__(self, vocab_size=3):
    super().__init__()
    self.vocab_size = vocab_size
    self.calls = []

  def forward(self, input_ids, attention_mask=None, **kwargs):
    self.calls.append({"attention_mask": attention_mask, "kwargs": kwargs})
    logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.vocab_size)
    logits[..., 0] = 5.0
    return _FakeOutput(logits)

  def train(self, mode=True):
    return self

  def eval(self):
    return self


def _packed_blob():
  return DataBlob(data={
      "input_ids": torch.tensor([[1, 2, 3, 4]]),
      "attention_mask": torch.tensor([[1, 1, 1, 1]]),
      "segment_ids": torch.tensor([[0, 0, 1, 1]]),
      "position_ids": torch.tensor([[0, 1, 0, 1]]),
  },
                  meta={"labels": torch.tensor([[2, 3, 4, 0]])})


class TestPackedDispatch:

  def test_plain_batch_uses_2d_mask(self):
    method = build_method("causal_lm")
    model = _RecordingLM()
    blob = DataBlob(data={
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    },
                    meta={"labels": torch.tensor([[2, 3, 0]])})
    method._forward(model, blob)
    assert model.calls[-1]["attention_mask"].dim() == 2

  def test_block_diagonal_model_gets_4d_mask(self):
    method = build_method("causal_lm")
    method._packed_capability = "block_diagonal"
    model = _RecordingLM()
    method._forward(model, _packed_blob())
    mask = model.calls[-1]["attention_mask"]
    assert mask.dim() == 4 and mask.shape == (1, 1, 4, 4)
    assert "position_ids" in model.calls[-1]["kwargs"]

  def test_segment_ids_model_gets_raw_boundaries(self):
    method = build_method("causal_lm")
    method._packed_capability = "segment_ids"
    model = _RecordingLM()
    method._forward(model, _packed_blob())
    call = model.calls[-1]
    # The raw document runs are handed through as a kwarg...
    assert "segment_ids" in call["kwargs"]
    assert call["kwargs"]["segment_ids"].tolist() == [[0, 0, 1, 1]]
    assert "position_ids" in call["kwargs"]
    # ... alongside the *ordinary* 2-D key-padding mask (not a 4-D one), so a
    # model that never materializes the L x L scores still gets padding info.
    assert call["attention_mask"].dim() == 2
    assert call["attention_mask"].tolist() == [[1, 1, 1, 1]]

  def test_varlen_model_gets_flash_boundaries(self):
    method = build_method("causal_lm")
    method._packed_capability = "varlen"
    model = _RecordingLM()
    out = method._forward(model, _packed_blob())
    kwargs = model.calls[-1]["kwargs"]
    # The batch is flattened to [1, total] and the document boundaries are the
    # HF FlashAttentionKwargs, not a raw segment_ids kwarg.
    assert model.calls[-1]["attention_mask"] is None
    assert "segment_ids" not in kwargs
    assert kwargs["cu_seq_lens_q"].tolist() == [0, 2, 4]
    assert kwargs["cu_seq_lens_k"].tolist() == [0, 2, 4]
    assert kwargs["max_length_q"] == kwargs["max_length_k"] == 2
    # Logits are scattered back to the full [B, L, V] shape so the shifted
    # cross-entropy keeps lining up with labels.
    assert out.logits.shape == (1, 4, 3)

  def test_capability_resolved_from_model_attribute(self):
    from genml_kit.methods.causal_lm import _resolve_packed_capability
    model = _RecordingLM()
    model.PACKED_ATTENTION = "block_diagonal"
    assert _resolve_packed_capability(model) == "block_diagonal"
    # The segment_ids mode is a declared capability too.
    model.PACKED_ATTENTION = "segment_ids"
    assert _resolve_packed_capability(model) == "segment_ids"

  def test_capability_reads_hf_private_attn_attr(self):
    from genml_kit.methods.causal_lm import _resolve_packed_capability
    from transformers import GPT2Config, GPT2LMHeadModel
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=64,
                   n_embd=16,
                   n_layer=1,
                   n_head=2,
                   attn_implementation="sdpa"))
    # ``PACKED_ATTENTION`` is absent on HF models -> falls through to the
    # ``config._attn_implementation`` property (the public ``attn_implementation``
    # name does not exist as an attribute).
    assert getattr(type(model), "PACKED_ATTENTION", None) is None
    assert _resolve_packed_capability(model) == "block_diagonal"
    # A flash backend resolves to the varlen path.
    model.config._attn_implementation_internal = "flash_attention_2"
    assert _resolve_packed_capability(model) == "varlen"
    # Any flash *variant* (not just the literal "flash_attention_2") does too.
    model.config._attn_implementation_internal = "flash_attention_3"
    assert _resolve_packed_capability(model) == "varlen"

  def test_none_capability_is_fatal_on_packed_batch(self):
    method = build_method("causal_lm")
    method._packed_capability = "none"
    with pytest.raises(ValueError, match="cannot isolate documents"):
      method._forward(_RecordingLM(), _packed_blob())
