"""Tests for training/llm_eval.py metrics (pure functions + perplexity).

Perplexity is checked against a directly computed shifted cross-entropy over
a deterministic fake LM; exact-match / token-F1 are pure and hand-checked;
the pass@k hook is exercised with an in-process scorer.
"""

import pytest
import torch

from genml_kit.models.tokenizer import ByteTokenizer
from genml_kit.training.llm_eval import (compute_perplexity, exact_match, token_f1)


class _FakeOutput:

  def __init__(self, logits):
    self.logits = logits


class _UniformLM(torch.nn.Module):
  """Emits all-zero logits, i.e. a uniform distribution over the vocab."""

  def __init__(self, vocab_size=260):
    super().__init__()
    self.vocab_size = vocab_size

  def forward(self, input_ids, attention_mask=None):
    del attention_mask
    batch, length = input_ids.shape
    return _FakeOutput(torch.zeros(batch, length, self.vocab_size))


class TestPerplexity:

  def test_uniform_lm_ppl_equals_vocab_size(self):
    # A uniform distribution has cross-entropy ln(V) per token, so perplexity
    # is exactly V.
    model = _UniformLM(vocab_size=260)
    tokenizer = ByteTokenizer()
    ppl = compute_perplexity(model, tokenizer, ["hello world"], "cpu")
    assert ppl == pytest.approx(260.0, rel=1e-4)

  def test_empty_text_list_is_finite(self):
    model = _UniformLM()
    tokenizer = ByteTokenizer()
    ppl = compute_perplexity(model, tokenizer, [], "cpu")
    assert ppl == pytest.approx(1.0, rel=1e-6)


class TestExactMatch:

  def test_identical_after_strip(self):
    assert exact_match("  hello ", "hello") == 1.0

  def test_different(self):
    assert exact_match("hello", "world") == 0.0


class TestTokenF1:

  def test_perfect_overlap(self):
    assert token_f1("the cat", "the cat") == pytest.approx(1.0)

  def test_partial_overlap(self):
    # pred = {the, cat}; ref = {the, dog}: overlap 1.
    # precision = 1/2, recall = 1/2 -> F1 = 0.5.
    assert token_f1("the cat", "the dog") == pytest.approx(0.5)

  def test_no_overlap(self):
    assert token_f1("a b", "c d") == 0.0

  def test_both_empty(self):
    assert token_f1("", "") == 1.0

  def test_empty_prediction(self):
    assert token_f1("", "a b") == 0.0


class TestPassAtK:

  def test_pass_at_k_with_in_process_scorer(self):
    from genml_kit.training import llm_eval

    class _ConstantLM(torch.nn.Module):

      def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 260)
        logits[..., 1] = 5.0
        return _FakeOutput(logits)

    def scorer(prompt, completion, reference):
      del prompt, completion, reference
      return 1.0

    args = type(
        "A", (), {
            "scorer": "x.py",
            "pass_at_k": 2,
            "max_new_tokens": 2,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "seed": 0,
            "batch_size": 2,
            "no_chat_template": True
        })()

    original = llm_eval.load_scorer
    llm_eval.load_scorer = lambda spec: scorer
    try:
      rate = llm_eval._pass_at_k(_ConstantLM(), ByteTokenizer(), ["a", "b"], ["a", "b"],
                                 args, "cpu")
    finally:
      llm_eval.load_scorer = original
    assert rate == 1.0
