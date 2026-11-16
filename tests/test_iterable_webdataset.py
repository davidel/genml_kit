"""Streaming over webdataset tar shards (Phase 5).

Every test builds its own shards with ``webdataset.TarWriter`` in ``tmp_path``,
so nothing here touches the network.  Each sample carries its own id in a
serialisable member, so assertions can compare the exact **multiset** of samples
that reaches the consumer -- the only property that matters for a split, and the
one a wrong split corrupts.

The contract under test: ``WebDatasetSource`` yields whole samples as
``{member_name: decoded_value}`` dicts, with webdataset's metadata entries kept
and image members converted to ``image_format``.  Field selection is
``FieldSelectorDataset``'s job, exercised at the bottom to show the two compose.
"""

import functools
import http.server
import io
import os
import threading

import numpy as np
import pytest
import torch
import webdataset as wds
from PIL import Image

from genml_kit.datasets.base import is_iterable, is_random_access, num_samples
from genml_kit.datasets.field_selector import (FieldSelectorDataset,
                                               IterableFieldSelectorDataset)
from genml_kit.datasets.iterable import StreamingDataset, is_streaming_source
from genml_kit.datasets.webdataset_source import WebDatasetSource
from genml_kit.utils import distributed


def _png(mode="RGB", size=(4, 4), color=None):
  buf = io.BytesIO()
  Image.new(mode, size, color).save(buf, format="PNG")
  return buf.getvalue()


def _write_shard(path, ids, image_key="png", label_key="cls"):
  """Write one tar shard holding *ids* as ``<image_key>``/``<label_key>``."""
  with wds.TarWriter(str(path)) as writer:
    for i in ids:
      writer.write({
          "__key__": f"s{i:04d}",
          image_key: _png(color=(i % 256, 0, 0)),
          label_key: str(i),
      })


def _make_shards(root, n, shard_count=2):
  """Split ``range(n)`` over *shard_count* shards; return the brace pattern."""
  per = n // shard_count
  for s in range(shard_count):
    _write_shard(root / f"s-{s:04d}.tar", range(s * per, (s + 1) * per))
  return str(root / f"s-{{0000..{shard_count - 1:04d}}}.tar")


def _labels(source):
  """The ``cls`` member of every sample, as a list (via FieldSelectorDataset)."""
  picked = FieldSelectorDataset(source, fields={"cls": "label"})
  return [item["label"] for item in picked]


class _LoopbackShards:
  """Serve *directory* over loopback HTTP and yield shard URLs.

  Used to exercise ``cache_dir``: webdataset caches only *remote* shards, so a
  ``file://`` path would bypass the cache under test.  Loopback keeps the test
  hermetic (no real network, no ``network`` marker).
  """

  def __init__(self, directory):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)

  def __enter__(self):
    threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
    return self

  def __exit__(self, *exc):
    self._httpd.shutdown()
    self._httpd.server_close()

  def url(self, name):
    port = self._httpd.server_address[1]
    return f"http://127.0.0.1:{port}/{name}"


class TestSampleContract:
  """A sample is a dict of decoded members; nothing is dropped."""

  def test_yields_a_dict_of_every_member(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    sample = next(iter(source))
    assert isinstance(sample, dict)
    assert {"png", "cls"} <= set(sample)

  def test_metadata_is_kept(self, tmp_path):
    """``__key__``/``__url__``/``__local_path__`` stay for debugging."""
    sample = next(iter(WebDatasetSource(_make_shards(tmp_path, 4))))
    assert "__key__" in sample
    assert "__url__" in sample
    assert sample["__key__"] == "s0000"

  def test_members_decode_by_extension(self, tmp_path):
    """Each member type arrives in the type webdataset decoded it to."""
    import json
    path = tmp_path / "types.tar"
    npbuf = io.BytesIO()
    np.save(npbuf, np.arange(3, dtype=np.float32))
    with wds.TarWriter(str(path)) as writer:
      writer.write({
          "__key__": "s0",
          "png": _png(),
          "txt": b"caption",
          "cls": b"7",
          "json": json.dumps({
              "a": [1, 2]
          }).encode(),
          "npy": npbuf.getvalue(),
          "raw": b"\x00\x01",
      })
    sample = next(iter(WebDatasetSource(str(path))))
    assert isinstance(sample["png"], Image.Image)
    assert sample["txt"] == "caption"
    assert sample["cls"] == 7
    assert sample["json"] == {"a": [1, 2]}
    # An array-valued member is a feature array, not a picture: webdataset
    # decodes npy to ndarray and the adapter must pass it through untouched.
    assert isinstance(sample["npy"], np.ndarray)
    # An unrecognised extension is left as bytes rather than guessed at.
    assert sample["raw"] == b"\x00\x01"

  def test_no_image_member_at_all(self, tmp_path):
    """A text-only corpus must work; nothing here is image-specific."""
    path = tmp_path / "text.tar"
    with wds.TarWriter(str(path)) as writer:
      for i in range(3):
        writer.write({"__key__": f"s{i}", "txt": f"line {i}".encode()})
    samples = list(WebDatasetSource(str(path)))
    assert [s["txt"] for s in samples] == ["line 0", "line 1", "line 2"]

  def test_is_a_streaming_source(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    assert isinstance(source, StreamingDataset)
    assert is_streaming_source(source)
    assert not is_random_access(source)

  def test_has_no_length(self, tmp_path):
    with pytest.raises(TypeError):
      len(WebDatasetSource(_make_shards(tmp_path, 4)))

  def test_num_samples_is_an_estimate_only(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 4), num_samples=999)
    assert source.num_samples() == 999

  def test_single_shard_path_is_accepted(self, tmp_path):
    _write_shard(tmp_path / "one.tar", range(5))
    assert _labels(WebDatasetSource(str(tmp_path / "one.tar"))) == list(range(5))

  def test_brace_pattern_expands(self, tmp_path):
    assert sorted(_labels(WebDatasetSource(_make_shards(tmp_path,
                                                        8)))) == list(range(8))

  def test_member_that_is_not_an_image_extension_stays_bytes(self, tmp_path):
    """Only real image extensions are decoded as images.

        webdataset dispatches on the member name, so a ``png``-named member is an
        image while an identically-shaped ``gray`` member is left as bytes.  Pinned
        because it silently changes what a pipeline receives.
        """
    path = tmp_path / "odd.tar"
    with wds.TarWriter(str(path)) as writer:
      writer.write({"__key__": "s0", "png": _png(), "gray": _png("L")})
    sample = next(iter(WebDatasetSource(str(path))))
    assert isinstance(sample["png"], Image.Image)
    assert isinstance(sample["gray"], bytes)


class TestImageFormat:

  @pytest.mark.parametrize("fmt", ["RGB", "L", "RGBA"])
  def test_image_members_are_converted(self, tmp_path, fmt):
    path = tmp_path / "img.tar"
    with wds.TarWriter(str(path)) as writer:
      writer.write({"__key__": "s0", "png": _png("RGB"), "jpg": _png("L")})
    sample = next(iter(WebDatasetSource(str(path), image_format=fmt)))
    assert sample["png"].mode == fmt
    assert sample["jpg"].mode == fmt

  def test_default_is_rgb(self, tmp_path):
    sample = next(iter(WebDatasetSource(_make_shards(tmp_path, 2))))
    assert sample["png"].mode == "RGB"

  def test_metadata_is_never_converted(self, tmp_path):
    """A metadata entry must not be routed through the image path."""
    sample = next(iter(WebDatasetSource(_make_shards(tmp_path, 2))))
    assert isinstance(sample["__key__"], str)

  def test_unknown_format_fails_at_construction(self, tmp_path):
    with pytest.raises(ValueError, match="Unsupported image_format"):
      WebDatasetSource(_make_shards(tmp_path, 2), image_format="BOGUS")

  def test_transform_is_applied_to_images(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 2),
                              transform=lambda img: img.resize((2, 2)))
    assert next(iter(source))["png"].size == (2, 2)


class TestIterationAndEpoch:

  def test_second_pass_replays_the_same_samples(self, tmp_path):
    """A second epoch must not silently yield nothing."""
    source = WebDatasetSource(_make_shards(tmp_path, 8))
    assert _labels(source) == _labels(source) == list(range(8))

  def test_epoch_is_exhaustion_not_a_fixed_length(self, tmp_path):
    assert len(_labels(WebDatasetSource(_make_shards(tmp_path, 7, 1)))) == 7

  def test_set_epoch_is_accepted(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    source.set_epoch(3)
    assert _labels(source) == list(range(4))


class TestCompositionWithFieldSelector:

  def test_field_selection_and_rename(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    picked = next(
        iter(
            IterableFieldSelectorDataset(source,
                                         fields={
                                             "png": "image",
                                             "cls": "label"
                                         })))
    assert sorted(picked) == ["image", "label"]
    assert isinstance(picked["image"], Image.Image)
    assert picked["label"] == 0

  def test_stream_selector_is_an_iterable_dataset(self, tmp_path):
    """The stream selector must satisfy torch's check, not just ours.

        ``DataLoader`` picks its fetch path from ``isinstance(dataset,
        IterableDataset)`` -- it never calls our ``is_iterable`` helper -- so the
        selector has to genuinely inherit a stream base class for a stream to be
        loadable at all.
        """
    from torch.utils.data import DataLoader, IterableDataset
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    picked = IterableFieldSelectorDataset(source, fields={"png": "image"})
    assert isinstance(picked, IterableDataset)
    assert is_random_access(picked) is False
    assert is_iterable(picked) is True
    assert num_samples(picked) is None
    loader = DataLoader(picked, batch_size=2, collate_fn=lambda b: len(b))
    assert sum(n for n in loader) == 4

  def test_map_selector_exposes_random_access(self):
    """The map-style selector is not an IterableDataset and keeps its length."""
    from torch.utils.data import IterableDataset

    class MapSource:

      def __len__(self):
        return 5

      def __getitem__(self, idx):
        if idx >= 5:
          raise IndexError(idx)
        return {"a": idx}

    picked = FieldSelectorDataset(MapSource(), fields={"a": "x"})
    assert not isinstance(picked, IterableDataset)
    assert is_random_access(picked) is True
    assert len(picked) == 5
    assert picked[3] == {"x": 3}
    assert list(picked) == [{"x": i} for i in range(5)]

  def test_missing_field_fails_loud(self, tmp_path):
    """A requested field absent from the tar is an error, not a silent drop."""
    source = WebDatasetSource(_make_shards(tmp_path, 4))
    with pytest.raises(KeyError):
      next(iter(FieldSelectorDataset(source, fields={"nope": "x"})))

  def test_text_field_selection(self, tmp_path):
    path = tmp_path / "text.tar"
    with wds.TarWriter(str(path)) as writer:
      for i in range(2):
        writer.write({"__key__": f"s{i}", "txt": f"l{i}".encode()})
    picked = FieldSelectorDataset(WebDatasetSource(str(path)), fields={"txt": "text"})
    assert [item["text"] for item in picked] == ["l0", "l1"]


class TestNoInternalSplitting:
  """webdataset's own splitters must stay off; ours does the work."""

  def test_source_alone_yields_every_sample_once(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path, 8))
    labels = _labels(source)
    assert len(labels) == len(set(labels)) == 8

  def test_node_splitter_disabled_under_ddp_env(self, tmp_path, monkeypatch):
    """webdataset's default node splitter raises at world_size > 1.

        This is the regression that matters: without ``nodesplitter=None`` the
        first iteration dies with "you need to add an explicit nodesplitter",
        a message telling the user to fix a pipeline they do not own.
        """
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    source = WebDatasetSource(_make_shards(tmp_path, 4, 1))
    assert _labels(source) == list(range(4))

  @pytest.mark.parametrize("world", [1, 2])
  def test_rank_sharding_composes(self, tmp_path, monkeypatch, world):
    """The Phase 4 wrapper strides this source disjointly and exhaustively."""
    from torch.utils.data import DataLoader
    n = 16
    url = _make_shards(tmp_path, n)

    def collate(batch):
      # The images pipeline supplies its own collate; this stands in for it so
      # the test exercises the split, not torch's default collate.
      images = [np.array(s["image"], copy=True) for s in batch]
      return {
          "label": torch.tensor([s["label"] for s in batch]),
          "image": torch.stack([torch.from_numpy(a) for a in images]),
      }

    monkeypatch.setattr(distributed, "world_size", lambda: world)
    per_rank = []
    for r in range(world):
      monkeypatch.setattr(distributed, "rank", lambda r=r: r)
      picked = IterableFieldSelectorDataset(WebDatasetSource(url),
                                            fields={
                                                "png": "image",
                                                "cls": "label"
                                            })
      loader = DataLoader(picked, batch_size=2, num_workers=0, collate_fn=collate)
      sharded = distributed.shard_loader(loader, seed=0, shuffle_buffer_k=2)
      per_rank.append([i for batch in sharded for i in batch["label"].tolist()])

    assert len({len(chunk) for chunk in per_rank}) == 1
    union = sorted(i for chunk in per_rank for i in chunk)
    assert union == list(range(n))
    for a in range(len(per_rank)):
      for b in range(a + 1, len(per_rank)):
        assert set(per_rank[a]).isdisjoint(per_rank[b])


class TestFailureModes:

  def test_non_image_member_raises(self, tmp_path):
    """A member named like an image but holding bytes is a config error.

        webdataset's decoder-based dispatch trusts the member name, so it fails
        while decoding rather than yielding a bad sample; assert that exact
        error rather than a bare ``Exception``.

        The failure abandons a half-consumed stream, and webdataset never closes
        the shard (upstream issue #365), so the handle is freed by refcounting
        instead.  Collect it here, inside a scope that ignores that specific
        warning, so it does not surface as an unraisable during session
        teardown, where pytest attributes it to pytest rather than webdataset.
        """
    import gc
    import warnings
    from webdataset.autodecode import DecodingError
    path = tmp_path / "broken.tar"
    with wds.TarWriter(str(path)) as writer:
      writer.write({"__key__": "s0", "png": b"not a png", "cls": "0"})
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", "unclosed file", ResourceWarning)
      with pytest.raises(DecodingError):
        list(WebDatasetSource(str(path)))
      gc.collect()


class TestCacheDir:
  """``cache_dir`` forwards to webdataset, so remote shards are cached."""

  def test_remote_shards_are_cached_when_cache_dir_is_given(self, tmp_path):
    _write_shard(tmp_path / "s-0000.tar", range(3))
    cache = tmp_path / "cache"
    with _LoopbackShards(tmp_path) as server:
      source = WebDatasetSource(server.url("s-0000.tar"), cache_dir=str(cache))
      assert len(list(source)) == 3
    cached = list(cache.rglob("*.tar"))
    assert len(cached) == 1
    assert cached[0].stat().st_size > 0

  def test_second_pass_reads_the_cached_copy(self, tmp_path):
    _write_shard(tmp_path / "s-0000.tar", range(4))
    cache = tmp_path / "cache"
    with _LoopbackShards(tmp_path) as server:
      source = WebDatasetSource(server.url("s-0000.tar"), cache_dir=str(cache))
      # First pass downloads into the cache; the server is then stopped, yet a
      # second pass still succeeds because webdataset reads the cached copy.
      assert len(list(source)) == 4
    assert len(list(source)) == 4

  def test_cache_dir_is_exposed_for_forwarding(self, tmp_path):
    _write_shard(tmp_path / "s-0000.tar", range(2))
    cache = str(tmp_path / "cache")
    source = WebDatasetSource(str(tmp_path / "s-0000.tar"), cache_dir=cache)
    assert source.cache_dir == cache
    # webdataset needs the directory to pre-exist, so it is created eagerly.
    assert os.path.isdir(cache)
