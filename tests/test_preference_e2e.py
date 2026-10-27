"""Offline end-to-end test: preference pipeline -> dpo/reward_model.

Builds a tiny GPT-2 (saved to a temp dir, no download), a preference corpus,
and the offline ``byte`` tokenizer, then runs the real pipeline lifecycle:
``prepare_transforms`` -> ``build_loader`` (train/val split) -> ``build_model``
-> ``train_step`` -> ``evaluate``.
"""

import argparse

import datasets
import pytest
import torch

from genml_kit.methods import build_method
from genml_kit.pipelines import build_pipeline


def _write_pref_corpus(root, n=8):
  hf = datasets.Dataset.from_dict({
      "prompt": [f"q{i}" for i in range(n)],
      "chosen": [f"good {i}" for i in range(n)],
      "rejected": [f"bad {i}" for i in range(n)],
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


@pytest.mark.parametrize("method_name", ["dpo", "reward_model"])
def test_preference_pipeline_lifecycle(tmp_path, method_name):
  _write_pref_corpus(tmp_path / "corpus")
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model", method=method_name)

  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  method.prepare_transforms(args, args.device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)

  # 8 items, 25% held out -> 6 train / 2 val.
  assert len(pipeline.train_dataset) == 6
  assert len(pipeline.val_dataset) == 2

  model = method.build_model(args, args.device)
  blob = next(iter(pipeline.train_loader))
  blob = pipeline.to_device(blob, torch.device("cpu"))
  out = method.train_step(model, blob, 0)
  assert torch.isfinite(out.loss)
  report = method.evaluate(model, pipeline.val_loader, torch.device("cpu"),
                           pipeline.to_device)
  assert "accuracy" in report


def test_pipeline_requires_tokenizer(tmp_path):
  _write_pref_corpus(tmp_path / "corpus")
  _tiny_model_dir(tmp_path / "model")
  args = _args(tmp_path / "corpus", tmp_path / "model")
  pipeline = build_pipeline(args.pipeline)
  with pytest.raises(ValueError):
    pipeline.build_loader(args, mode="train", method=None)
