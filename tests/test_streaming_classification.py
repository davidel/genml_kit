"""Streaming image classification (``--pipeline images --dataset``).

A streaming source has no index space, so the classification loader cannot
materialise it.  ``load_and_split_dataset`` accepts ``dataset_arg`` (so
``streaming=True`` reaches ``datasets.load_dataset``), routes the train/val
split through the positional :class:`IterableDatasetFilter`, and wraps the
buckets in stream-safe :class:`HFDatasetProxy`s.  The label column must be a
typed ``ClassLabel`` so the vocabulary can be read from ``features`` without
scanning, and corpus-wide features (class weights, weighted/balanced samplers)
are refused.
"""

import numpy as np
import pytest
from PIL import Image

import datasets

from genml_kit.datasets.hf_proxy import HFDatasetProxy, is_hf_stream

_IMAGES = None


def _image():
  global _IMAGES
  if _IMAGES is None:
    _IMAGES = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
  return _IMAGES


def _stream(n=60, names=("cat", "dog", "eel")):
  """A streaming HF source with a typed ``ClassLabel`` label column."""

  def gen():
    for i in range(n):
      yield {"image": _image(), "label": i % len(names)}

  return datasets.IterableDataset.from_generator(
      gen,
      features=datasets.Features({
          "image": datasets.Image(),
          "label": datasets.ClassLabel(names=list(names)),
      }))


def _untyped_stream(n=30):
  """A streaming source whose label column is a plain string (no vocabulary)."""

  def gen():
    for i in range(n):
      yield {"image": _image(), "label": ["cat", "dog"][i % 2]}

  return datasets.IterableDataset.from_generator(gen,
                                                 features=datasets.Features({
                                                     "image": datasets.Image(),
                                                     "label": datasets.Value("string"),
                                                 }))


def _import_train():
  import importlib

  return importlib.import_module("genml_kit.training.train")


class TestIsHfStream:

  def test_detects_iterable_dataset(self):
    assert is_hf_stream(_stream(3)) is True

  def test_map_style_is_not_stream(self):
    ds = datasets.Dataset.from_dict({"x": [1, 2]})
    assert is_hf_stream(ds) is False


class TestDatasetArgForwarded:

  def test_streaming_kwarg_reaches_the_loader(self):
    train_mod = _import_train()
    captured = {}

    def fake_loader(path, **kwargs):
      captured["path"] = path
      captured["kwargs"] = kwargs
      return _stream(20)

    train_p, val_p = train_mod.load_and_split_dataset("fake_stream",
                                                      loader=fake_loader,
                                                      dataset_arg={"streaming": True})

    assert captured["path"] == "fake_stream"
    assert captured["kwargs"]["streaming"] is True
    # Both halves are stream proxies (no length).
    assert train_p.label_column == "label"
    with pytest.raises(TypeError):
      len(train_p)

  def test_no_streaming_kwarg_is_not_added(self):
    train_mod = _import_train()
    captured = {}

    def fake_loader(path, **kwargs):
      captured["kwargs"] = kwargs
      return _stream(4)

    train_mod.load_and_split_dataset("fake_stream", loader=fake_loader)
    assert "streaming" not in captured["kwargs"]


class TestStreamingSplitAndWrap:

  def test_splits_positionally_and_reads_num_labels(self):
    train_mod = _import_train()
    train_p, val_p = train_mod.load_and_split_dataset(
        "fake_stream", loader=lambda *a, **k: _stream(60), test_size=0.25)

    # Vocabulary comes from the typed ClassLabel, without scanning the stream.
    assert train_p.num_labels == 3
    assert train_p.label_names == ["cat", "dog", "eel"]
    assert train_p.has_labels is True

    # The split is exhaustive and disjoint (buckets cover the corpus once).
    train = [label for _img, label in train_p]
    val = [label for _img, label in val_p]
    assert len(train) + len(val) == 60

  def test_val_split_must_be_a_fraction(self):
    train_mod = _import_train()
    with pytest.raises(ValueError, match="between 0 and 1"):
      train_mod.load_and_split_dataset("fake_stream",
                                       loader=lambda *a, **k: _stream(20),
                                       test_size=0.0)

  def test_max_items_trims_each_bucket(self):
    train_mod = _import_train()
    train_p, val_p = train_mod.load_and_split_dataset(
        "fake_stream", loader=lambda *a, **k: _stream(100), test_size=0.5, max_items=5)
    assert len([_ for _ in train_p]) == 5
    assert len([_ for _ in val_p]) == 5


class TestStreamingProxyConstraints:

  def test_untyped_label_column_is_refused(self):
    with pytest.raises(ValueError, match="typed ClassLabel"):
      HFDatasetProxy(_untyped_stream(10))

  def test_len_raises_for_stream(self):
    proxy = HFDatasetProxy(_stream(5))
    with pytest.raises(TypeError):
      len(proxy)

  def test_items_are_image_label_tuples(self):
    proxy = HFDatasetProxy(_stream(6))
    image, label = next(iter(proxy))
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
