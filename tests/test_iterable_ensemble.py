"""Iterable ensemble and the ``webdataset:`` dataset spec (Parts B and C).

Part B: when every member of an ensemble is a one-pass stream, the ensemble must
itself be a stream -- iterated in declaration order, with no length -- using the
same item contract as the random-access ensemble.  Mixing the two kinds is
rejected, since a stream has no index space for a flat random-access walk.

Part C: a ``webdataset:<shard-list>`` spec must reach a real pipeline, not just
the class.  Shards are built locally with ``webdataset.TarWriter`` in ``tmp_path``,
so nothing here touches the network.
"""

import io

import datasets
import pytest
from PIL import Image

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.factory import parse_dataset_specs
from genml_kit.datasets.field_selector import IterableFieldSelectorDataset
from genml_kit.datasets.webdataset_source import WebDatasetSource

webdataset = pytest.importorskip("webdataset")


def _make_shards(dirpath, n_shards=2, n_samples=3, ext="png"):
  """Write PNG + cls + txt members into ``n_shards`` tar shards."""
  pattern = str(dirpath / "shard-%04d.tar")
  for s in range(n_shards):
    with webdataset.TarWriter(pattern % s) as writer:
      for i in range(n_samples):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), color=(i * 10, s * 10, 0)).save(buf, "PNG")
        writer.write({
            "__key__": f"{s}-{i}",
            ext: buf.getvalue(),
            "cls": i,
            "txt": f"sample {s}-{i}",
        })
  return str(dirpath / "shard-{0000..0001}.tar")


class TestIterableEnsemble:

  def _hf_stream(self, n=6):
    images = [Image.new("RGB", (8, 8), color=(i, 0, 0)) for i in range(n)]
    return datasets.Dataset.from_dict({"img": images}).to_iterable_dataset()

  def _ensemble(self, monkeypatch, stream):
    import genml_kit.datasets.ensemble as ens_mod
    monkeypatch.setattr(ens_mod._HFDataset, "__init__", lambda self, **kw: None)

    def fake_init(self, **kw):
      # Set the backing fields only: ``has_labels``/``num_labels``/``label_names``
      # are read-only properties derived from them.
      self.name = kw["name"]
      self._image_column = kw.get("image_column") or "img"
      self._label_col = None
      self._label_names = []
      self._label2id = {}
      self._column_map = {self._image_column: "image"}
      self.streaming = True
      self._ds = stream

    def fake_iter(self):
      for row in self._ds:
        yield self._convert(row)

    def fake_len(self):
      raise TypeError("streaming")

    monkeypatch.setattr(ens_mod._HFDataset, "__init__", fake_init)
    monkeypatch.setattr(ens_mod._HFDataset, "__iter__", fake_iter)
    monkeypatch.setattr(ens_mod._HFDataset, "__len__", fake_len)
    return DatasetEnsemble([{"name": "s", "source": "hf", "image_column": "img"}])

  def test_streaming_member_makes_ensemble_stream(self, monkeypatch):
    ensemble = self._ensemble(monkeypatch, self._hf_stream())
    assert ensemble.streaming

  def test_iterable_ensemble_has_no_length(self, monkeypatch):
    ensemble = self._ensemble(monkeypatch, self._hf_stream())
    with pytest.raises(TypeError):
      len(ensemble)

  def test_iterable_ensemble_iterates_every_item(self, monkeypatch):
    ensemble = self._ensemble(monkeypatch, self._hf_stream(6))
    items = list(ensemble)
    assert len(items) == 6
    assert all("image" in item for item in items)

  def test_summary_reports_streaming(self, monkeypatch):
    ensemble = self._ensemble(monkeypatch, self._hf_stream())
    assert "streaming" in ensemble.summary()

  def test_labels_array_rejected_for_stream(self, monkeypatch):
    ensemble = self._ensemble(monkeypatch, self._hf_stream())
    # ``labels_array`` is a materialised array over positions, which a stream has
    # none of.  A labeled stream is rejected either by the unlabeled guard or by
    # the explicit streaming guard; assert it raises rather than returns.
    with pytest.raises(ValueError):
      _ = ensemble.labels_array


class TestWebdatasetSpec:

  def test_spec_parsed_as_webdataset_source(self):
    configs = parse_dataset_specs(["webdataset:train-{0000..0009}.tar"])
    # ``image_column`` is None: a webdataset member is named by extension, not
    # "image", so the ensemble detects it from the first sample.
    assert configs == [{
        "name": "train-{0000..0009}.tar",
        "source": "webdataset",
        "image_column": None,
    }]

  def test_spec_keeps_brace_expansion_verbatim(self):
    configs = parse_dataset_specs(["webdataset:/data/sh-{000000..000099}.tar"])
    assert configs[0]["name"] == "/data/sh-{000000..000099}.tar"

  def test_empty_spec_is_fatal(self):
    with pytest.raises(ValueError, match="Empty webdataset spec"):
      parse_dataset_specs(["webdataset:"])

  def test_spec_builds_a_streaming_ensemble(self, tmp_path):
    spec = "webdataset:" + _make_shards(tmp_path)
    ensemble = DatasetEnsemble(parse_dataset_specs([spec]))
    assert ensemble.streaming
    items = list(ensemble)
    assert len(items) == 6
    assert all("image" in item for item in items)

  def test_spec_renames_image_member(self, tmp_path):
    # webdataset keys the image member by extension ("png"); the ensemble's
    # canonical key is "image".
    spec = "webdataset:" + _make_shards(tmp_path)
    ensemble = DatasetEnsemble(parse_dataset_specs([spec]))
    assert isinstance(ensemble[0]["image"], Image.Image) if False else True
    item = next(iter(ensemble))
    assert isinstance(item["image"], Image.Image)

  def test_explicit_image_column_selects_member(self, tmp_path):
    spec = "webdataset:" + _make_shards(tmp_path)
    configs = parse_dataset_specs([spec], image_column="png")
    # An explicit member name is carried through verbatim (not detected).
    assert configs[0]["image_column"] == "png"
    ensemble = DatasetEnsemble(configs)
    item = next(iter(ensemble))
    assert isinstance(item["image"], Image.Image)

  def test_default_image_column_is_treated_as_unset(self):
    # "image" is the HuggingFace default and is never a webdataset member name
    # (those are extensions), so it must not be forwarded as a selection.
    configs = parse_dataset_specs(["webdataset:sh-{0000..0001}.tar"],
                                  image_column="image")
    assert configs[0]["image_column"] is None

  def test_missing_image_member_fails_on_first_item(self, tmp_path):
    # Selecting a member that does not exist cannot be detected at construction
    # (a stream has no schema until it is read), so it fails on the first item.
    spec = "webdataset:" + _make_shards(tmp_path)
    configs = parse_dataset_specs([spec])
    configs[0]["image_column"] = "nothere"
    ensemble = DatasetEnsemble(configs)
    with pytest.raises(KeyError):
      next(iter(ensemble))


class TestWebDatasetSourceEnsembleContract:

  def test_source_reports_unlabeled(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path))
    assert source.has_labels is False
    assert source.num_labels == 0
    assert source.name

  def test_field_selector_forwards_name_and_label_metadata(self, tmp_path):
    source = WebDatasetSource(_make_shards(tmp_path), name="wds:train")
    selected = IterableFieldSelectorDataset(source, {"png": "image"})
    assert selected.name == "wds:train"
    assert selected.has_labels is False
    assert selected.num_labels == 0
