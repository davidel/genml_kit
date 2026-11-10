"""Tests for the text pipeline (pipelines/text.py) and text dataset proxy.

Uses an in-memory ``datasets.Dataset`` and the offline ``byte`` tokenizer
so nothing hits the network.  Covers the registry contract, plain-LM and
instruction column selection, the loader contract, and device transfer.
"""

import types

import datasets
import pytest
import torch

from genml_kit.datasets.capped import maybe_cap
from genml_kit.datasets.text_proxy import TextDatasetProxy
from genml_kit.models.tokenizer import load_tokenizer
from genml_kit.pipelines import PIPELINES
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.text import TextPipeline
from tests.cli_args import train_args


def _args(**overrides):
  """Build text-pipeline args from the real CLI parser.

    See ``tests/cli_args.py``; the previous ``types.SimpleNamespace`` mirror
    of the CLI surface could drift out of sync.
    """
  pinned = dict(
      dataset="local",
      split="train",
      val_split=None,
      text_column="text",
      dataset_arg=None,
      max_seq_len=16,
      batch_size=2,
      num_workers=0,
      seed=42,
  )
  pinned.update(overrides)
  return train_args(["--pipeline", "text"], **pinned)


def _proxy():
  hf = datasets.Dataset.from_dict({"text": ["a", "bb", "ccc", "dddd"]})
  return TextDatasetProxy(hf, text_column="text")


class TestRegistry:

  def test_text_registered(self):
    assert "text" in PIPELINES.list_names()
    assert PIPELINES.get("text") is TextPipeline


class TestTextDatasetProxy:

  def test_plain_lm_item(self):
    proxy = _proxy()
    assert proxy[0] == {"text": "a"}
    assert len(proxy) == 4

  def test_instruction_item(self):
    hf = datasets.Dataset.from_dict({"q": ["Q"], "a": ["A"]})
    proxy = TextDatasetProxy(hf, prompt_column="q", response_column="a")
    assert proxy.is_instruction
    assert proxy[0] == {"prompt": "Q", "response": "A"}

  def test_max_items_caps_length(self):
    hf = datasets.Dataset.from_dict({"text": ["a", "b", "c"]})
    proxy = TextDatasetProxy(maybe_cap(hf, 2), text_column="text")
    assert len(proxy) == 2

  def test_missing_column_is_fatal(self):
    with pytest.raises(ValueError, match="not in dataset columns"):
      TextDatasetProxy(_proxy().dataset, text_column="nope")

  def test_no_selection_is_fatal(self):
    with pytest.raises(ValueError, match="requires --text_column"):
      TextDatasetProxy(_proxy().dataset)


class TestBuildLoader:

  def _patch_load(self, monkeypatch, proxy):
    import genml_kit.pipelines.text as text_mod

    monkeypatch.setattr(text_mod,
                        "load_raw_split",
                        lambda name, split="train", **kw: proxy.dataset)

  def test_build_loader_contract(self, monkeypatch):
    self._patch_load(monkeypatch, _proxy())
    method = types.SimpleNamespace(tokenizer=load_tokenizer("byte"))
    pipeline = TextPipeline()
    loader = pipeline.build_loader(_args(), mode="train", method=method)
    assert pipeline.train_loader is loader
    batch = next(iter(loader))
    assert isinstance(batch, DataBlob)
    assert set(batch.data) == {"input_ids", "attention_mask"}
    assert batch.meta["labels"].dtype == torch.long

  def test_missing_tokenizer_is_fatal(self, monkeypatch):
    self._patch_load(monkeypatch, _proxy())
    pipeline = TextPipeline()
    method = types.SimpleNamespace(tokenizer=None)
    with pytest.raises(ValueError, match="needs a tokenizer"):
      pipeline.build_loader(_args(), mode="train", method=method)

  def test_to_device_moves_all_tensors(self, monkeypatch):
    self._patch_load(monkeypatch, _proxy())
    method = types.SimpleNamespace(tokenizer=load_tokenizer("byte"))
    pipeline = TextPipeline()
    loader = pipeline.build_loader(_args(), mode="train", method=method)
    batch = pipeline.to_device(next(iter(loader)), torch.device("cpu"))
    for tensor in batch.data.values():
      assert tensor.device.type == "cpu"
    assert batch.meta["labels"].device.type == "cpu"

  def test_val_split_caps_before_carving_holdout(self, monkeypatch):
    # --max_items must be honoured *before* the held-out fraction is carved, so
    # a budgeted run yields cap - n_val training rows plus a real val set
    # (previously --val_split silently ignored --max_items).
    raw = datasets.Dataset.from_dict({"text": [str(i) for i in range(10)]})
    self._patch_load(monkeypatch, TextDatasetProxy(raw, text_column="text"))
    method = types.SimpleNamespace(tokenizer=load_tokenizer("byte"))
    pipeline = TextPipeline()
    pipeline.build_loader(_args(val_split=0.5, max_items=4),
                          mode="train",
                          method=method)
    assert len(pipeline.train_dataset) == 2
    assert len(pipeline.val_dataset) == 2
