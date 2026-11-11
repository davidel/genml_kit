"""Integration tests: a real streaming HF dataset through the text pipeline.

These drive the *actual* ``TextPipeline`` with a *real* ``datasets`` streaming
dataset, rather than a hand-built fake.  That gap is exactly what let an earlier
round of streaming bugs through: the proxy was tested with a fake that always
reported ``column_names``, so the real ``column_names is None`` case (a hub
stream, or anything after ``.map()``) was never exercised, and no test ever
built a loader for a stream at all.

Every test here uses ``Dataset.to_iterable_dataset()`` -- an in-memory stream
with no network -- so nothing hits the hub.  ``.map()`` without declared
features is used to reproduce the ``column_names is None`` case deterministically.
"""

import types

import datasets
import pytest

import genml_kit.pipelines.text as text_mod
from genml_kit.datasets.text_proxy import StreamingTextDataset, TextDatasetProxy
from genml_kit.models.tokenizer import load_tokenizer
from genml_kit.pipelines.text import TextPipeline
from tests.cli_args import train_args


def _stream(n=20, *, drop_features=False):
  """An in-memory streaming dataset (optionally with unknown features)."""
  hf = datasets.Dataset.from_dict({"text": [str(i) * 3 for i in range(n)]})
  stream = hf.to_iterable_dataset()
  if drop_features:
    # ``.map`` without declared features drops them -> ``column_names is None``.
    stream = stream.map(lambda row: row)
  return stream


def _args(**overrides):
  pinned = dict(
      dataset="local",
      split="train",
      val_split=None,
      text_column="text",
      dataset_arg={"streaming": True},
      max_seq_len=16,
      batch_size=4,
      num_workers=0,
      seed=42,
  )
  pinned.update(overrides)
  return train_args(["--pipeline", "text"], **pinned)


def _patch_stream(monkeypatch, stream):
  monkeypatch.setattr(text_mod,
                      "load_raw_split",
                      lambda name, split="train", **kw: stream)


def _build(monkeypatch, stream, **overrides):
  _patch_stream(monkeypatch, stream)
  method = types.SimpleNamespace(tokenizer=load_tokenizer("byte"))
  pipeline = TextPipeline()
  pipeline.build_loader(_args(**overrides), mode="train", method=method)
  return pipeline


class TestStreamingProxyIntegration:

  def test_column_names_none_constructs_and_iterates(self):
    # Regression: ``set(column_names)`` used to raise ``TypeError`` on a stream
    # whose features are unknown.
    proxy = TextDatasetProxy(_stream(drop_features=True), text_column="text")
    assert [item["text"] for item in proxy]

  def test_unknown_features_missing_column_still_fatal(self):
    proxy = TextDatasetProxy(_stream(drop_features=True), text_column="nope")
    with pytest.raises(ValueError, match="not in dataset columns"):
      list(proxy)

  def test_unknown_features_missing_layout_fatal_at_construction(self):
    with pytest.raises(ValueError, match="requires --text_column"):
      TextDatasetProxy(_stream(drop_features=True))

  def test_streaming_adapter_is_an_iterable_dataset(self):
    from torch.utils.data import IterableDataset
    proxy = TextDatasetProxy(_stream(), text_column="text")
    assert isinstance(StreamingTextDataset(proxy), IterableDataset)


class TestStreamingBuildLoader:

  def test_plain_stream_builds_and_yields_blobs(self, monkeypatch):
    pipeline = _build(monkeypatch, _stream(20))
    assert pipeline.streaming
    batches = list(pipeline.train_loader)
    assert batches
    assert sum(len(b.data["input_ids"]) for b in batches) == 20

  def test_plain_stream_with_unknown_features(self, monkeypatch):
    # The original crash: this used to raise ``TypeError`` at construction.
    pipeline = _build(monkeypatch, _stream(12, drop_features=True))
    batches = list(pipeline.train_loader)
    assert sum(len(b.data["input_ids"]) for b in batches) == 12

  def test_streaming_length_bucket_emits_every_item(self, monkeypatch):
    pipeline = _build(monkeypatch, _stream(20), length_bucket=True)
    batches = list(pipeline.train_loader)
    assert sum(len(b.data["input_ids"]) for b in batches) == 20

  def test_streaming_packing_emits_every_item(self, monkeypatch):
    # Packing concatenates documents into fixed windows (and may truncate a
    # long tail), so row count is not document count.  The invariant is that
    # every document reaches the collator exactly once: check the pipe, which
    # is the loader's dataset.
    pipeline = _build(monkeypatch, _stream(20), packing="wrapped", sequence_mode="open")
    pipe = pipeline.train_loader.dataset
    seen = [item["text"] for batch in pipe for item in batch]
    assert len(seen) == 20
    assert list(pipeline.train_loader)

  def test_streaming_val_split_is_disjoint_and_complete(self, monkeypatch):
    # A stream cannot ``.select``, so the split is positional.  Both sides must
    # be non-empty, disjoint, and cover the source exactly once.
    pipeline = _build(monkeypatch, _stream(20), val_split=0.25)
    train = [item["text"] for item in pipeline.train_dataset]
    val = [item["text"] for item in pipeline.val_dataset]
    assert len(train) + len(val) == 20
    assert set(train).isdisjoint(val)
    assert train and val

  def test_streaming_bfd_batches_by_batch_size(self, monkeypatch):
    # ``bfd`` is a collator-level strategy, not a batch sampler: a stream must
    # still use the plain loader (no ``batch_sampler``), batching by
    # ``--batch_size``.  Pinned because an earlier change routed ``bfd`` to the
    # token-budget pipe and silently altered batch composition.
    pipeline = _build(monkeypatch, _stream(20), packing="bfd", sequence_mode="open")
    # torch always wraps a ``BatchSampler`` around ``batch_size``, so the
    # distinguishing property is the *batch size*, not the sampler's absence.
    assert pipeline.train_loader.batch_size == 4
    # ``bfd`` repacks documents into windows in the collator, so row count is not
    # document count: check that every document still reaches the collator.
    assert len(list(pipeline.train_dataset)) == 20
    assert list(pipeline.train_loader)

  def test_streaming_wrapped_uses_batch_pipe(self, monkeypatch):
    # ``wrapped`` emits whole batches from a pipe, so the loader has no batch
    # size of its own (``batch_size=None``) and no sampler.
    pipeline = _build(monkeypatch, _stream(20), packing="wrapped", sequence_mode="open")
    # Whole batches come from the pipe, so the loader defers to it.
    assert pipeline.train_loader.batch_size is None
    assert list(pipeline.train_loader)

  def test_random_access_still_uses_map_style_path(self, monkeypatch):
    # A map-style dataset must NOT be wrapped as a stream: the samplers and the
    # exact ``len`` are kept.
    hf = datasets.Dataset.from_dict({"text": [str(i) for i in range(10)]})
    monkeypatch.setattr(text_mod,
                        "load_raw_split",
                        lambda name, split="train", **kw: hf)
    method = types.SimpleNamespace(tokenizer=load_tokenizer("byte"))
    pipeline = TextPipeline()
    pipeline.build_loader(_args(dataset_arg=None), mode="train", method=method)
    assert not pipeline.streaming
    assert len(pipeline.train_dataset) == 10
