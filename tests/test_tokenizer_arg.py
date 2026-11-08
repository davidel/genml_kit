"""Tests for the ``--tokenizer_arg`` escape hatch.

``--tokenizer_arg`` forwards arbitrary ``KEY=VALUE`` pairs to an external
tokenizer script's ``create_tokenizer(**kwargs)`` so knobs like ``vocab_size``,
the training corpus and the cache basename are driven from the CLI instead of
being hardcoded in the script.  The flag reuses the shared ``KVPairAction``
mechanism (mirrors ``--model_arg`` / ``--dataset_arg``).
"""

import argparse

from genml_kit.models import tokenizer as tok_mod
from genml_kit.utils.shared_args import ARG_TOKENIZER_ARG, add_shared_arg


def _parser():
  parser = argparse.ArgumentParser()
  add_shared_arg(parser, ARG_TOKENIZER_ARG)
  return parser


class TestTokenizerArgParsing:

  def test_default_is_empty_dict(self):
    # KVPairAction normalizes the default so ``**args.tokenizer_arg`` always
    # unpacks (mirrors --model_arg / --dataset_arg).
    args = _parser().parse_args([])
    assert args.tokenizer_arg == {}

  def test_single_pair(self):
    args = _parser().parse_args(["--tokenizer_arg", "vocab_size=16384"])
    assert args.tokenizer_arg == {"vocab_size": 16384}

  def test_multiple_pairs(self):
    args = _parser().parse_args(
        ["--tokenizer_arg", "vocab_size=16384", "dataset=HuggingFaceFW/fineweb-edu"])
    assert args.tokenizer_arg == {
        "vocab_size": 16384,
        "dataset": "HuggingFaceFW/fineweb-edu",
    }

  def test_quoted_value_stays_string(self):
    args = _parser().parse_args(["--tokenizer_arg", 'cache_basename="16k.json"'])
    assert args.tokenizer_arg == {"cache_basename": "16k.json"}


class TestLoadTokenizerForwards:

  def test_kwargs_reach_external_create_tokenizer(self, monkeypatch):
    captured = {}

    def fake_load_extern(name, func):
      del func

      def create_tokenizer(**kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

      return create_tokenizer

    monkeypatch.setattr(tok_mod, "is_script_spec", lambda _: True)
    monkeypatch.setattr(tok_mod, "load_extern", fake_load_extern)

    tok_mod.load_tokenizer("tokenizer.py",
                           cache_dir="/tmp/c",
                           vocab_size=16384,
                           dataset="HuggingFaceFW/fineweb-edu")

    # cache_dir is a keyword-only parameter of load_tokenizer and is not part
    # of ``**kwargs``; only the extra tokenizer kwargs reach the script factory.
    assert captured["name"] == "tokenizer.py"
    assert captured["kwargs"] == {
        "vocab_size": 16384,
        "dataset": "HuggingFaceFW/fineweb-edu",
    }

  def test_no_extra_kwargs_reaches_factory(self, monkeypatch):
    captured = {}

    def fake_load_extern(name, func):
      del name, func

      def create_tokenizer(**kwargs):
        captured["kwargs"] = kwargs
        return object()

      return create_tokenizer

    monkeypatch.setattr(tok_mod, "is_script_spec", lambda _: True)
    monkeypatch.setattr(tok_mod, "load_extern", fake_load_extern)

    tok_mod.load_tokenizer("tokenizer.py", cache_dir="/tmp/c")

    assert captured["kwargs"] == {}
