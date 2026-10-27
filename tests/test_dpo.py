"""Offline tests for the DPO and reward_model methods (Phase 4).

Builds a tiny GPT-2 (saved to a temp dir, no download), a preference corpus
saved with ``save_to_disk``, and the offline ``byte`` tokenizer, then drives
the real loader/method lifecycle: ``prepare_transforms`` -> ``build_loader``
-> ``build_model`` -> ``train_step`` -> ``evaluate``.
"""

import argparse

import datasets
import pytest
import torch

from genml_kit.methods import METHODS, build_method
from genml_kit.methods.reward_model import (attach_reward_head, gather_last_token,
                                            resolve_hidden_size, sequence_reward)
from genml_kit.pipelines import build_pipeline


def _write_pref_corpus(root, n=8):
  hf = datasets.Dataset.from_dict({
      "prompt": [f"question {i}" for i in range(n)],
      "chosen": [f"good answer {i}" for i in range(n)],
      "rejected": [f"bad answer {i}" for i in range(n)],
  })
  hf.save_to_disk(str(root))


def _tiny_model_dir(root):
  from transformers import GPT2Config, GPT2LMHeadModel

  config = GPT2Config(vocab_size=260, n_positions=64, n_embd=16, n_layer=2, n_head=2)
  GPT2LMHeadModel(config).save_pretrained(str(root))
  return root


def _args(corpus_dir, model_dir, method="dpo", **overrides):
  base = dict(
      pipeline="preference",
      method=method,
      dataset=str(corpus_dir),
      split="train",
      val_split=0.25,
      chosen_column="chosen",
      rejected_column="rejected",
      prompt_column="prompt",
      cache_dir=None,
      max_items=None,
      max_seq_len=32,
      mask_prompt=True,
      pad_to_multiple_of=None,
      model=str(model_dir),
      tokenizer="byte",
      reference_model=None,
      reference_checkpoint=None,
      beta=0.1,
      reward_head_hidden=None,
      trust_remote_code=False,
      attn_implementation=None,
      batch_size=2,
      num_workers=0,
      seed=0,
      device="cpu",
      grad_checkpoint=False,
      lora=False,
      freeze=None,
      source_checkpoint=None,
      param_init=None,
      loss=None,
  )
  base.update(overrides)
  return argparse.Namespace(**base)


def _build(method_name, tmp_path):
  _write_pref_corpus(tmp_path / "corpus")
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model", method=method_name)
  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  method.prepare_transforms(args, args.device)
  pipeline.build_loader(args, mode="train", method=method)
  model = method.build_model(args, args.device)
  return args, pipeline, method, model


class TestRegistration:

  def test_methods_registered(self):
    assert "dpo" in METHODS.list_names()
    assert "reward_model" in METHODS.list_names()

  def test_metrics_maximized(self):
    for name in ("dpo", "reward_model"):
      cls = METHODS.get(name)
      assert cls.METRIC_KEY == "accuracy"
      assert cls.METRIC_MINIMIZE is False


class TestRewardHelpers:

  def test_resolve_hidden_size_from_config(self, tmp_path):
    _tiny_model_dir(tmp_path / "model")
    from genml_kit.models import load_model
    model = load_model(str(tmp_path / "model"),
                       task="causal_lm",
                       num_labels=0,
                       device=torch.device("cpu"))
    assert resolve_hidden_size(model) == 16

  def test_gather_last_token_skips_padding(self):
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = gather_last_token(hidden, mask)
    assert torch.equal(pooled[0], hidden[0, 1])
    assert torch.equal(pooled[1], hidden[1, 0])

  def test_attach_head_adds_score(self, tmp_path):
    _tiny_model_dir(tmp_path / "model")
    from genml_kit.models import load_model
    model = load_model(str(tmp_path / "model"),
                       task="causal_lm",
                       num_labels=0,
                       device=torch.device("cpu"))
    attach_reward_head(model)
    assert "score.weight" in model.state_dict()
    # Backbone keys stay unprefixed (no wrapper).
    assert any(k.startswith("transformer.") for k in model.state_dict())

  def test_sequence_reward_shape(self, tmp_path):
    _tiny_model_dir(tmp_path / "model")
    from genml_kit.models import load_model
    model = load_model(str(tmp_path / "model"),
                       task="causal_lm",
                       num_labels=0,
                       device=torch.device("cpu"))
    attach_reward_head(model)
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    reward = sequence_reward(model, ids, mask)
    assert reward.shape == (2,)


class TestDpoE2E:

  def test_train_step_and_evaluate(self, tmp_path):
    args, pipeline, method, model = _build("dpo", tmp_path)
    assert method.reference_model is not None
    # The reference is not a submodule of the policy -> no 'reference' keys.
    assert not any(k.startswith("reference") for k in model.state_dict())
    blob = next(iter(pipeline.train_loader))
    blob = pipeline.to_device(blob, torch.device("cpu"))
    out = method.train_step(model, blob, 0)
    assert out.loss.item() == pytest.approx(0.0, abs=5.0)  # finite
    assert "accuracy" in out.metrics
    report = method.evaluate(model, pipeline.val_loader, torch.device("cpu"),
                             pipeline.to_device)
    assert "loss" in report and "accuracy" in report

  def test_reference_reconstructed_from_args(self, tmp_path):
    args, _, method, _ = _build("dpo", tmp_path)
    state = method.get_checkpoint_state(method.reference_model, args)
    assert state["reference_model"] == args.model
    assert state["beta"] == args.beta

  def test_training_reduces_loss_on_tiny_overfit(self, tmp_path):
    args, pipeline, method, model = _build("dpo", tmp_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    blob = pipeline.to_device(next(iter(pipeline.train_loader)), torch.device("cpu"))
    first = float(method.train_step(model, blob, 0).loss.detach())
    for _ in range(5):
      optimizer.zero_grad()
      method.train_step(model, blob, 0).loss.backward()
      optimizer.step()
    last = float(method.train_step(model, blob, 0).loss.detach())
    assert last <= first + 1e-4


class TestRewardModelE2E:

  def test_train_step_and_evaluate(self, tmp_path):
    args, pipeline, method, model = _build("reward_model", tmp_path)
    assert "score.weight" in model.state_dict()
    blob = pipeline.to_device(next(iter(pipeline.train_loader)), torch.device("cpu"))
    out = method.train_step(model, blob, 0)
    assert torch.isfinite(out.loss)
    report = method.evaluate(model, pipeline.val_loader, torch.device("cpu"),
                             pipeline.to_device)
    assert "loss" in report and "accuracy" in report

  def test_training_reduces_loss_on_tiny_overfit(self, tmp_path):
    args, pipeline, method, model = _build("reward_model", tmp_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    blob = pipeline.to_device(next(iter(pipeline.train_loader)), torch.device("cpu"))
    first = float(method.train_step(model, blob, 0).loss.detach())
    for _ in range(5):
      optimizer.zero_grad()
      method.train_step(model, blob, 0).loss.backward()
      optimizer.step()
    last = float(method.train_step(model, blob, 0).loss.detach())
    assert last <= first + 1e-4
