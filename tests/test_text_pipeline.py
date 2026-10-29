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


def _args(**overrides):
  base = dict(
      dataset="local",
      split="train",
      val_split=None,
      text_column="text",
      instruction_column=None,
      response_column=None,
      cache_dir=None,
      max_items=None,
      max_seq_len=16,
      mask_prompt=False,
      packing=False,
      pad_to_multiple_of=None,
      batch_size=2,
      num_workers=0,
      seed=42,
  )
  base.update(overrides)
  return types.SimpleNamespace(**base)


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
