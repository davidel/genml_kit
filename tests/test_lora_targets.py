"""Tests for structural LoRA target resolution.

``resolve_lora_targets`` derives PEFT ``target_modules`` from the model's own
module graph rather than hardcoding names, so it must work across
architectures (GPT-2 Conv1D, Llama, timm, BERT) and stay collision-safe (a
short suffix like ``c_proj`` must not also match a non-attention projection).

These tests build tiny configs locally (no downloads).
"""

import torch

import pytest

from genml_kit.training.model_utils import (
    _is_attention_container,
    _is_projection,
    resolve_lora_targets,
)

peft = pytest.importorskip("peft", reason="LoRA tests need the peft package")
transformers = pytest.importorskip("transformers")


class _MlpOnly(torch.nn.Module):
  """A backbone with no attention structure at all."""

  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(4, 8)
    self.fc2 = torch.nn.Linear(8, 2)


class _FakeAttn(torch.nn.Module):
  """Hand-rolled attention container using ``attn`` as the leaf name."""

  def __init__(self):
    super().__init__()
    self.q = torch.nn.Linear(4, 4)
    self.k = torch.nn.Linear(4, 4)


class _FakeBlock(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.attn = _FakeAttn()
    self.mlp = torch.nn.Linear(4, 4)


class _FakeModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleList([_FakeBlock(), _FakeBlock()])


def _gpt2():
  from transformers import GPT2Config, GPT2LMHeadModel

  return GPT2LMHeadModel(
      GPT2Config(vocab_size=260, n_positions=64, n_embd=32, n_layer=2, n_head=2))


def _llama():
  from transformers import LlamaConfig, LlamaForCausalLM

  return LlamaForCausalLM(
      LlamaConfig(vocab_size=260,
                  hidden_size=32,
                  intermediate_size=64,
                  num_hidden_layers=2,
                  num_attention_heads=2,
                  num_key_value_heads=2))


class TestProjectionPredicate:

  def test_linear_is_projection(self):
    assert _is_projection(torch.nn.Linear(4, 4))

  def test_embedding_is_not_projection(self):
    assert not _is_projection(torch.nn.Embedding(10, 4))

  def test_container_is_not_projection(self):
    assert not _is_projection(torch.nn.Sequential(torch.nn.Linear(4, 4)))


class _FakeAttention(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.q = torch.nn.Linear(4, 4)


class TestAttentionDetection:

  def test_class_name_match(self):
    # Class name contains "attention" -> detected regardless of path name.
    assert _is_attention_container("blocks.0.inner", _FakeAttention())

  def test_leaf_name_match(self):
    assert _is_attention_container("blocks.0.attn", torch.nn.Identity())

  def test_non_attention(self):
    assert not _is_attention_container("blocks.0.mlp", torch.nn.Linear(4, 4))


class TestResolverFallback:

  def test_no_attention_falls_back_to_linear(self):
    targets = resolve_lora_targets(_MlpOnly())
    assert targets == ["fc1", "fc2"]

  def test_hand_rolled_attention(self):
    targets = resolve_lora_targets(_FakeModel())
    # `attn` is the container; its q/k projections are the only ones with
    # those leaf names, so the minimal (shortest unique) suffixes are used.
    assert set(targets) == {"q", "k"}
    # The sibling mlp projection must not be picked up.
    assert "mlp" not in targets


class TestResolverRealModels:

  def test_gpt2_targets_attention_not_lm_head(self):
    targets = resolve_lora_targets(_gpt2())
    # The historical bug: GPT-2 attention is Conv1D, so the old nn.Linear
    # based resolver selected only `lm_head`.
    assert "lm_head" not in targets
    assert "c_attn" in targets
    assert "attn.c_proj" in targets
    # c_proj must be namespaced so it does not also hit mlp.c_proj.
    assert "c_proj" not in targets

  def test_llama_targets_include_output_projection(self):
    targets = resolve_lora_targets(_llama())
    assert set(targets) == {"q_proj", "k_proj", "v_proj", "o_proj"}

  def test_targets_do_not_touch_mlp(self):
    for model in (_gpt2(), _llama()):
      targets = resolve_lora_targets(model)
      assert not any("mlp" in t or "feed_forward" in t for t in targets)


class TestResolverAppliesToPeft:

  @pytest.mark.filterwarnings("ignore:fan_in_fan_out is set to False:UserWarning")
  def test_gpt2_resolved_targets_adapt_attention(self):
    from genml_kit.training.model_utils import apply_lora

    model = _gpt2()
    targets = resolve_lora_targets(model)
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=targets)
    adapted = {
        n.rsplit(".lora_A", 1)[0].replace("base_model.model.", "")
        for n in dict(wrapped.named_modules())
        if n.endswith("lora_A.default")
    }
    assert adapted
    assert all("mlp" not in a for a in adapted)
    assert any(a.endswith("attn.c_attn") for a in adapted)
    assert any(a.endswith("attn.c_proj") for a in adapted)
