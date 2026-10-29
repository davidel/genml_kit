"""Offline end-to-end tests for the ``genml-kit-llm`` CLI subcommands.

Builds a tiny GPT-2 from a config (saved to a temp dir, no download), a
``save_to_disk`` corpus, and drives the real ``run_generate`` / ``run_eval``
paths -- including the shared registry model loader -- so the JSONL/JSON
outputs and the metric plumbing are exercised without a network.
"""

import argparse
import json

import datasets
import pytest
import torch

from genml_kit.cli.llm import build_parser
from genml_kit.training.generate import run_generate
from genml_kit.training.llm_eval import run_eval


def _write_corpus(root, n=6):
  hf = datasets.Dataset.from_dict({
      "text": [f"the cat sat on mat number {i}" for i in range(n)],
      "prompt": [f"count to {i}" for i in range(n)],
      "reference": [f"answer {i}" for i in range(n)],
  })
  hf.save_to_disk(str(root))


def _tiny_model_dir(root):
  from transformers import GPT2Config, GPT2LMHeadModel

  config = GPT2Config(vocab_size=260, n_positions=64, n_embd=16, n_layer=2, n_head=2)
  model = GPT2LMHeadModel(config)
  model.save_pretrained(str(root))
  return root


def _common(model_dir, **overrides):
  base = dict(
      model=str(model_dir),
      tokenizer="byte",
      checkpoint=None,
      lora_adapter=None,
      device="cpu",
      cache_dir=None,
      trust_remote_code=False,
      attn_implementation=None,
      log_level="INFO",
      log_targets="STDERR",
      batch_size=2,
      no_chat_template=True,
  )
  base.update(overrides)
  return base


class TestParser:

  def test_generate_subcommand_parses(self):
    args = build_parser().parse_args(["generate", "--model", "gpt2"])
    assert args.command == "generate"
    assert args.max_new_tokens == 64

  def test_eval_subcommand_parses(self):
    args = build_parser().parse_args(["eval", "--model", "gpt2"])
    assert args.command == "eval"
    assert args.metrics == ["ppl", "exact_match", "token_f1"]

  def test_subcommand_required(self):
    with pytest.raises(SystemExit):
      build_parser().parse_args([])


class TestRunGenerate:

  def test_generate_writes_jsonl(self, tmp_path):
    _tiny_model_dir(tmp_path / "model")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("hello\nworld\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"
    args = argparse.Namespace(**_common(tmp_path / "model",
                                        prompts=str(prompts),
                                        dataset=None,
                                        split="test",
                                        prompt_column="prompt",
                                        text_column=None,
                                        max_items=None,
                                        max_new_tokens=4,
                                        do_sample=False,
                                        temperature=1.0,
                                        top_k=0,
                                        top_p=1.0,
                                        repetition_penalty=1.0,
                                        num_beams=1,
                                        num_return_sequences=1,
                                        seed=None,
                                        output=str(output)))
    run_generate(args)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["prompt"] == "hello"
    assert "completion" in records[0]

  def test_generate_from_dataset_column(self, tmp_path):
    _write_corpus(tmp_path / "corpus")
    _tiny_model_dir(tmp_path / "model")
    output = tmp_path / "out.jsonl"
    args = argparse.Namespace(**_common(tmp_path / "model",
                                        prompts=None,
                                        dataset=str(tmp_path / "corpus"),
                                        split="train",
                                        prompt_column="prompt",
                                        text_column=None,
                                        max_items=3,
                                        max_new_tokens=2,
                                        do_sample=False,
                                        temperature=1.0,
                                        top_k=0,
                                        top_p=1.0,
                                        repetition_penalty=1.0,
                                        num_beams=1,
                                        num_return_sequences=1,
                                        seed=None,
                                        output=str(output)))
    run_generate(args)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 3


class TestRunEval:

  def test_eval_reports_metrics(self, tmp_path):
    _write_corpus(tmp_path / "corpus")
    _tiny_model_dir(tmp_path / "model")
    output = tmp_path / "report.json"
    args = argparse.Namespace(**_common(tmp_path / "model",
                                        dataset=str(tmp_path / "corpus"),
                                        split="train",
                                        text_column="text",
                                        prompt_column="prompt",
                                        reference_column="reference",
                                        max_items=4,
                                        max_seq_len=32,
                                        metrics=["ppl", "exact_match", "token_f1"],
                                        max_new_tokens=4,
                                        do_sample=False,
                                        temperature=1.0,
                                        top_k=0,
                                        top_p=1.0,
                                        seed=None,
                                        pass_at_k=0,
                                        scorer=None,
                                        output=str(output)))
    run_eval(args)
    report = json.loads(output.read_text())
    assert report["n_items"] == 4
    assert torch.isfinite(torch.tensor(report["ppl"]))
    assert 0.0 <= report["exact_match"] <= 1.0
    assert 0.0 <= report["token_f1"] <= 1.0

  def test_pass_at_k_requires_scorer(self, tmp_path):
    _write_corpus(tmp_path / "corpus")
    _tiny_model_dir(tmp_path / "model")
    args = argparse.Namespace(**_common(tmp_path / "model",
                                        dataset=str(tmp_path / "corpus"),
                                        split="train",
                                        text_column="text",
                                        prompt_column="prompt",
                                        reference_column="reference",
                                        max_items=2,
                                        max_seq_len=32,
                                        metrics=["ppl"],
                                        max_new_tokens=2,
                                        do_sample=False,
                                        temperature=1.0,
                                        top_k=0,
                                        top_p=1.0,
                                        seed=None,
                                        pass_at_k=2,
                                        scorer=None,
                                        output=None))
    with pytest.raises(ValueError):
      run_eval(args)
