"""Tests for the ``--dataset_arg`` escape hatch.

``--dataset_arg`` forwards arbitrary ``KEY=VALUE`` pairs verbatim to
``datasets.load_dataset`` so HF subset/configs (e.g. FineWeb-Edu's
``name=sample-10BT``), ``data_files`` and ``revision`` become reachable from
the CLI.  The flag reuses the shared ``KVPairAction`` mechanism.
"""

import argparse

from genml_kit.datasets.text_proxy import load_raw_split
from genml_kit.utils.shared_args import ARG_DATASET_ARG, add_shared_arg


def _parser():
  parser = argparse.ArgumentParser()
  add_shared_arg(parser, ARG_DATASET_ARG)
  return parser


class TestDatasetArgParsing:

  def test_default_is_empty_dict(self):
    # KVPairAction normalizes the default so ``**args.dataset_arg`` always
    # unpacks (mirrors --model_arg).
    args = _parser().parse_args([])
    assert args.dataset_arg == {}

  def test_single_pair(self):
    args = _parser().parse_args(["--dataset_arg", "name=sample-10BT"])
    assert args.dataset_arg == {"name": "sample-10BT"}

  def test_multiple_pairs(self):
    args = _parser().parse_args(["--dataset_arg", "name=sample-10BT", "revision=main"])
    assert args.dataset_arg == {"name": "sample-10BT", "revision": "main"}

  def test_quoted_numeric_value_stays_string(self):
    # The escape hatch for numeric-looking config names.
    args = _parser().parse_args(["--dataset_arg", 'name="10"'])
    assert args.dataset_arg == {"name": "10"}


class TestLoadRawSplitForwards:

  def test_kwargs_reach_load_dataset(self, monkeypatch):
    import genml_kit.datasets.text_proxy as tp

    captured = {}

    # The first positional parameter of datasets.load_dataset is ``path``
    # (the repo id); ``name`` is the optional HF subset/config kwarg.
    def fake_load_dataset(path, **kwargs):
      captured["path"] = path
      captured["kwargs"] = kwargs
      return object()

    monkeypatch.setattr(tp.datasets, "load_dataset", fake_load_dataset)
    load_raw_split("HuggingFaceFW/fineweb-edu",
                   "train",
                   cache_dir="/tmp/cache",
                   dataset_arg={
                       "name": "sample-10BT",
                       "revision": "main"
                   })

    assert captured["path"] == "HuggingFaceFW/fineweb-edu"
    assert captured["kwargs"]["name"] == "sample-10BT"
    assert captured["kwargs"]["revision"] == "main"
    assert captured["kwargs"]["split"] == "train"
    assert captured["kwargs"]["cache_dir"] == "/tmp/cache"

  def test_no_dataset_arg_preserves_signature(self, monkeypatch):
    import genml_kit.datasets.text_proxy as tp

    captured = {}

    def fake_load_dataset(path, **kwargs):
      captured["path"] = path
      captured["kwargs"] = kwargs
      return object()

    monkeypatch.setattr(tp.datasets, "load_dataset", fake_load_dataset)
    load_raw_split("roneneldan/TinyStories", "train", cache_dir="/tmp/c", token=None)

    assert captured["path"] == "roneneldan/TinyStories"
    assert captured["kwargs"] == {
        "split": "train",
        "cache_dir": "/tmp/c",
        "token": None,
    }
