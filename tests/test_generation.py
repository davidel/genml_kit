"""Tests for utils/generation.py and the GenerativeLM contract.

The samplers are pure functions over logits, so they are checked against
hand-computed expectations.  The portable loop and the dispatcher are driven
with a hand-written ``.logits`` model (no network, no transformers), plus a
``GenerativeLM`` subclass to exercise the contract and its override path.
"""

import torch

from genml_kit.models.generative import GenerativeLM
from genml_kit.models.tokenizer import ByteTokenizer
from genml_kit.utils.generation import (SamplingConfig, apply_chat_template,
                                        apply_repetition_penalty, apply_temperature,
                                        autoregressive_sample, filter_top_k,
                                        filter_top_p, generate, generate_from_prompts,
                                        has_chat_template, has_native_generate,
                                        native_generate_callable, sample_next_token)


class _FakeOutput:

  def __init__(self, logits):
    self.logits = logits


class _ConstantLM(torch.nn.Module):
  """Always predicts the same fixed logits vector at every position."""

  def __init__(self, logits):
    super().__init__()
    self.register_buffer("_logits", torch.tensor([logits], dtype=torch.float))

  def forward(self, input_ids, attention_mask=None):
    del attention_mask
    batch, length = input_ids.shape
    logits = self._logits.expand(batch, length, -1).clone()
    return _FakeOutput(logits)


class TestSamplers:

  def test_temperature_scales_logits(self):
    logits = torch.tensor([[2.0, 0.0]])
    assert torch.allclose(apply_temperature(logits, 2.0), torch.tensor([[1.0, 0.0]]))
    # Neutral temperature is a no-op.
    assert torch.equal(apply_temperature(logits, 1.0), logits)

  def test_top_k_keeps_k_highest(self):
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    filtered = filter_top_k(logits, 2)
    assert filtered[0, 1].item() == 3.0
    assert filtered[0, 2].item() == 2.0
    assert filtered[0, 0].item() == float("-inf")

  def test_top_k_zero_is_noop(self):
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert torch.equal(filter_top_k(logits, 0), logits)

  def test_top_p_keeps_until_cumulative(self):
    # A dominant token: nucleus of 0.5 must keep only that token.
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    filtered = filter_top_p(logits, 0.5)
    assert filtered[0, 0].item() == 10.0
    assert filtered[0, 1].item() == float("-inf")
    assert filtered[0, 2].item() == float("-inf")

  def test_top_p_one_is_noop(self):
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert torch.equal(filter_top_p(logits, 1.0), logits)

  def test_repetition_penalty_downweights_seen(self):
    logits = torch.tensor([[2.0, -2.0, 0.0]])
    generated = torch.tensor([[0, 1]])
    penalized = apply_repetition_penalty(logits, generated, 2.0)
    assert penalized[0, 0].item() == 1.0  # positive -> divided
    assert penalized[0, 1].item() == -4.0  # negative -> multiplied
    assert penalized[0, 2].item() == 0.0  # untouched

  def test_greedy_picks_argmax(self):
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    token = sample_next_token(logits, False, 1.0, 0, 1.0)
    assert token.item() == 1

  def test_sampling_is_deterministic_with_generator(self):
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    a = sample_next_token(logits, True, 1.0, 0, 1.0, gen_a)
    b = sample_next_token(logits, True, 1.0, 0, 1.0, gen_b)
    assert a.item() == b.item()


class TestPortableLoop:

  def test_greedy_generates_expected_tokens(self):
    model = _ConstantLM([0.0, 5.0, 1.0])
    input_ids = torch.tensor([[1, 2]])
    config = SamplingConfig(max_new_tokens=3, do_sample=False)
    out = autoregressive_sample(model, input_ids, config=config)
    # Two prompt tokens + three generated, all equal to argmax (id 1).
    assert out.tolist() == [[1, 2, 1, 1, 1]]

  def test_eos_stops_generation(self):
    model = _ConstantLM([5.0, 0.0])  # argmax is id 0 == eos
    input_ids = torch.tensor([[1]])
    config = SamplingConfig(max_new_tokens=10,
                            do_sample=False,
                            eos_token_id=0,
                            pad_token_id=0)
    out = autoregressive_sample(model, input_ids, config=config)
    assert out.shape[1] == 2  # prompt + one EOS, then stopped

  def test_num_return_sequences_expands_batch(self):
    model = _ConstantLM([0.0, 5.0])
    input_ids = torch.tensor([[1], [2]])
    config = SamplingConfig(max_new_tokens=2, num_return_sequences=2)
    out = autoregressive_sample(model, input_ids, config=config)
    assert out.shape[0] == 4  # 2 prompts * 2 returns

  def test_beam_search_returns_best_per_prompt(self):
    model = _ConstantLM([0.0, 5.0, 1.0])
    input_ids = torch.tensor([[1], [2]])
    config = SamplingConfig(max_new_tokens=2, num_beams=2)
    out = autoregressive_sample(model, input_ids, config=config)
    assert out.shape[0] == 2  # one best beam per prompt
    assert out.tolist() == [[1, 1, 1], [2, 1, 1]]


class TestGenerativeLMContract:

  def test_default_generate_uses_portable_loop(self):
    model = _ConstantLM([0.0, 5.0, 1.0])
    contract = GenerativeLM()
    contract.forward = model.forward  # borrow the fake forward
    config = SamplingConfig(max_new_tokens=2)
    out = contract.generate(torch.tensor([[1]]), config=config)
    assert out.tolist() == [[1, 1, 1]]

  def test_override_routes_to_custom_backend(self):

    class _Custom(GenerativeLM):

      def generate(self, input_ids, attention_mask=None, config=None):
        del attention_mask, config
        return torch.full((input_ids.size(0), input_ids.size(1) + 1), 9)

    out = _Custom().generate(torch.tensor([[1, 2]]))
    assert out.tolist() == [[9, 9, 9]]

  def test_native_detection_ignores_inherited_generate(self):
    # A GenerativeLM subclass has no can_generate, so it must not be seen as
    # native -- otherwise the default body would recurse into itself.
    assert native_generate_callable(GenerativeLM()) is None
    assert has_native_generate(GenerativeLM()) is False


class TestDispatcher:

  def test_contract_model_uses_its_generate(self):

    class _Model(GenerativeLM):

      def generate(self, input_ids, attention_mask=None, config=None):
        del attention_mask, config
        return input_ids + 1

    out = generate(_Model(), torch.tensor([[1, 2]]))
    assert out.tolist() == [[2, 3]]

  def test_plain_module_falls_back_to_portable_loop(self):
    model = _ConstantLM([0.0, 5.0])
    out = generate(model, torch.tensor([[1]]), config=SamplingConfig(max_new_tokens=2))
    assert out.tolist() == [[1, 1, 1]]

  def test_hf_model_uses_native_generate(self):

    class _HF(torch.nn.Module):

      def can_generate(self):
        return True

      def generate(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask
        assert kwargs["max_new_tokens"] == 2
        return torch.cat([input_ids, input_ids], dim=1)

    out = generate(_HF(),
                   torch.tensor([[1, 2]]),
                   config=SamplingConfig(max_new_tokens=2))
    assert out.tolist() == [[1, 2, 1, 2]]


class TestChatTemplate:

  def test_byte_tokenizer_has_no_template(self):
    assert has_chat_template(ByteTokenizer()) is False

  def test_apply_chat_template_passthrough_when_disabled(self):
    prompts = ["hello"]
    assert apply_chat_template(ByteTokenizer(), prompts, enabled=True) == prompts
    assert apply_chat_template(ByteTokenizer(), prompts, enabled=False) == prompts

  def test_apply_chat_template_uses_tokenizer_template(self):

    class _Templated:

      chat_template = "{{ messages }}"

      def apply_chat_template(self,
                              conversation,
                              tokenize=True,
                              add_generation_prompt=False,
                              **kwargs):
        del tokenize, add_generation_prompt, kwargs
        return f"<user>{conversation[0]['content']}</user>"

    out = apply_chat_template(_Templated(), ["hi"], enabled=True)
    assert out == ["<user>hi</user>"]


class TestGenerateFromPrompts:

  def test_returns_one_completion_per_prompt(self):
    model = _ConstantLM([0.0, 5.0, 1.0])
    tokenizer = ByteTokenizer()
    completions = generate_from_prompts(model,
                                        tokenizer, ["ab", "cd"],
                                        config=SamplingConfig(max_new_tokens=3),
                                        device="cpu",
                                        batch_size=2)
    assert len(completions) == 2
    # Byte id 1 decodes to chr(1); three greedy steps all pick id 1.
    assert completions[0] == chr(1) * 3

  def test_left_padding_keeps_prompts_aligned(self):
    model = _ConstantLM([0.0, 5.0])
    tokenizer = ByteTokenizer()
    # Unequal-length prompts exercise the left-pad path.
    completions = generate_from_prompts(model,
                                        tokenizer, ["a", "longer"],
                                        config=SamplingConfig(max_new_tokens=1),
                                        device="cpu",
                                        batch_size=2)
    assert len(completions) == 2
    assert all(c == chr(1) for c in completions)

  def test_fills_pad_and_eos_from_tokenizer(self):
    model = _ConstantLM([0.0, 5.0])
    tokenizer = ByteTokenizer()
    config = SamplingConfig(max_new_tokens=1)
    generate_from_prompts(model, tokenizer, ["a"], config=config, device="cpu")
    assert config.pad_token_id == tokenizer.pad_token_id
    assert config.eos_token_id == tokenizer.eos_token_id
