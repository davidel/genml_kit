"""Offline end-to-end test: text pipeline -> causal_lm -> loss/eval.

Builds a tiny GPT-2 from a config (saved to a temp dir, no download), a
``save_to_disk`` text corpus, and the offline ``byte`` tokenizer, then runs
the real loader/method lifecycle: ``prepare_transforms`` -> ``build_loader``
-> ``build_model`` -> ``train_step`` -> ``evaluate``.
"""

import argparse

import datasets
import pytest
import torch

from genml_kit.methods import build_method
from genml_kit.pipelines import build_pipeline


def _write_corpus(root, n=8):
  hf = datasets.Dataset.from_dict({
      "text": [f"the cat sat on mat number {i}" for i in range(n)],
  })
  hf.save_to_disk(str(root))


def _tiny_model_dir(root):
  """Save a tiny randomly-initialised GPT-2 so from_pretrained is offline."""
  from transformers import GPT2Config, GPT2LMHeadModel

  config = GPT2Config(vocab_size=260, n_positions=64, n_embd=16, n_layer=2, n_head=2)
  model = GPT2LMHeadModel(config)
  model.save_pretrained(str(root))
  return root


def _args(corpus_dir, model_dir, **overrides):
  base = dict(
      pipeline="text",
      method="causal_lm",
      dataset=str(corpus_dir),
      split="train",
      val_split=0.25,
      text_column="text",
      instruction_column=None,
      response_column=None,
      cache_dir=None,
      max_items=None,
      max_seq_len=32,
      mask_prompt=False,
      packing=False,
      pad_to_multiple_of=None,
      model=str(model_dir),
      tokenizer="byte",
      model_arg={},
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
  )
  base.update(overrides)
  return argparse.Namespace(**base)


def test_metrics_declared(tmp_path):
  _write_corpus(tmp_path / "corpus")
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model")
  method = build_method(args.method)
  assert method.METRIC_KEY == "ppl"
  assert method.METRIC_MINIMIZE is True


def test_full_lifecycle_train_and_eval(tmp_path):
  _write_corpus(tmp_path / "corpus")
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model")

  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  # Lifecycle position 1: tokenizer resolved before the loaders read it.
  method.prepare_transforms(args, args.device)
  assert method.tokenizer is not None

  train_loader = pipeline.build_loader(args, mode="train", method=method)
  val_loader = pipeline.build_loader(args, mode="val", method=method)
  assert pipeline.train_loader is train_loader
  assert pipeline.val_loader is val_loader

  model = method.build_model(args, args.device)
  model.to(args.device)

  blob = next(iter(train_loader))
  blob = pipeline.to_device(blob, args.device)
  loss_out = method.train_step(model, blob, 0)
  assert torch.isfinite(loss_out.loss)
  loss_out.loss.backward()  # gradients flow

  report = method.evaluate(model, val_loader, args.device, pipeline.to_device)
  assert report["ppl"] == pytest.approx(float(torch.exp(torch.tensor(report["loss"]))),
                                        rel=1e-4)


def test_val_split_sizes(tmp_path):
  _write_corpus(tmp_path / "corpus", n=8)
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model", val_split=0.25)
  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  method.prepare_transforms(args, args.device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)
  assert len(pipeline.train_dataset) == 6
  assert len(pipeline.val_dataset) == 2
