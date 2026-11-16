"""Exact sample-count estimation for streaming datasets.

The estimator is a *sidecar*: it reads cheap metadata (a split size on the
dataset card, or a ``parquet``/``arrow`` file footer) and never streams the
payloads.  Its contract is strict -- an ``int`` only when the count is *exact*,
``None`` otherwise -- so these tests pin every "we cannot know exactly" case to
``None`` rather than a guess.

Everything here is offline: local ``save_to_disk`` / parquet fixtures, a local
webdataset-style descriptor, and mocked Hub calls.  The one test that would hit
the network is marked ``network`` and deselected by default.
"""

import json
import tempfile

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genml_kit.datasets import counting


def _write_parquet(path, n_rows, tag=0):
  table = pa.table({"x": list(range(tag, tag + n_rows))})
  pq.write_table(table, str(path))
  return path


def _save_to_disk(n_rows):
  directory = tempfile.mkdtemp()
  datasets.Dataset.from_dict({"x": list(range(n_rows))}).save_to_disk(directory)
  return directory


class _FakeSplit:
  """Minimal stand-in for ``datasets.SplitInfo``."""

  def __init__(self, num_examples):
    self.num_examples = num_examples


class _FakeInfo:

  def __init__(self, splits):
    self.splits = splits


class _FakeConfig:

  def __init__(self, data_files):
    self.data_files = data_files


class _FakeBuilder:
  """Minimal stand-in for a ``DatasetBuilder`` (info + config.data_files)."""

  def __init__(self, splits=None, data_files=None):
    self.info = _FakeInfo(splits or {})
    self.config = _FakeConfig(data_files or {})


class TestLocalFooters:
  """Tier 2 against local files: exact, and only when the split resolves."""

  def test_local_parquet_directory(self, tmp_path):
    _write_parquet(tmp_path / "train-00000.parquet", 9)
    assert counting.hf_num_samples(str(tmp_path), split="train") == 9

  def test_local_parquet_sums_all_shards(self, tmp_path):
    _write_parquet(tmp_path / "train-00000.parquet", 4)
    _write_parquet(tmp_path / "train-00001.parquet", 6)
    assert counting.hf_num_samples(str(tmp_path), split="train") == 10

  def test_save_to_disk_arrow_is_counted(self):
    # ``save_to_disk`` writes the Arrow IPC *stream* layout, whose row count is
    # only available by summing record batches (there is no footer ``num_rows``).
    directory = _save_to_disk(7)
    assert counting.hf_num_samples(directory, split="train") == 7

  def test_unknown_split_is_none_not_a_whole_dataset_sum(self, tmp_path):
    _write_parquet(tmp_path / "train-00000.parquet", 4)
    _write_parquet(tmp_path / "validation-00000.parquet", 6)
    assert counting.hf_num_samples(str(tmp_path), split="nope") is None

  def test_split_name_must_be_a_whole_token(self):
    # ``val`` must not match ``validation`` (a substring match would).
    assert counting._matches_split("validation-0.parquet", "val") is False
    assert counting._matches_split("val-0.parquet", "val") is True

  def test_arrow_file_layout_is_also_counted(self, tmp_path):
    path = tmp_path / "train-00000.arrow"
    table = pa.table({"x": list(range(5))})
    with pa.ipc.new_file(str(path), table.schema) as writer:
      writer.write_table(table)
    assert counting.hf_num_samples(str(tmp_path), split="train") == 5


class TestDeclaredSplitSize:
  """Tier 1: the size the dataset card declares, when present."""

  def test_card_size_wins(self, monkeypatch):
    builder = _FakeBuilder(splits={"train": _FakeSplit(123)})
    monkeypatch.setattr(counting.datasets, "load_dataset_builder",
                        lambda *a, **k: builder)
    assert counting.hf_num_samples("org/name", split="train") == 123

  def test_missing_or_zero_size_falls_through(self, monkeypatch):
    builder = _FakeBuilder(splits={"train": _FakeSplit(0)})
    monkeypatch.setattr(counting.datasets, "load_dataset_builder",
                        lambda *a, **k: builder)
    # No files resolve for a repo id with no data_files, so the count is None.
    assert counting.hf_num_samples("org/name", split="train") is None

  def test_builder_failure_is_logged_not_raised(self, monkeypatch, caplog):

    def boom(*a, **k):
      raise RuntimeError("gated")

    monkeypatch.setattr(counting.datasets, "load_dataset_builder", boom)
    with caplog.at_level("DEBUG"):
      assert counting.hf_num_samples("org/private", split="train") is None
    assert any("load_dataset_builder" in record.message for record in caplog.records)


class TestHubFiles:
  """Tier 2 for a Hub repo: footer range-read, then cached download."""

  def test_resolved_data_files_drive_the_count(self, monkeypatch, tmp_path):
    _write_parquet(tmp_path / "train-00000-of-00001.parquet", 11)
    builder = _FakeBuilder(
        data_files={"train": [str(tmp_path / "train-00000-of-00001.parquet")]})
    monkeypatch.setattr(counting.datasets, "load_dataset_builder",
                        lambda *a, **k: builder)
    assert counting.hf_num_samples("org/name", split="train") == 11

  def test_footer_range_read_is_preferred(self, monkeypatch, tmp_path):
    local = _write_parquet(tmp_path / "train-00000-of-00001.parquet", 12)

    class _FakeFS:

      def open(self, url, mode="rb"):
        return open(str(local), "rb")

    called = {"download": False}

    def fake_download(*a, **k):
      called["download"] = True
      raise AssertionError("range read should have succeeded")

    monkeypatch.setattr(counting, "_downloaded_file_rows", fake_download)
    monkeypatch.setattr("huggingface_hub.HfFileSystem", _FakeFS)
    rows = counting._remote_file_rows(
        "hf://datasets/org/name/train-00000-of-00001.parquet", None)
    assert rows == 12
    assert called["download"] is False

  def test_download_fallback_populates_the_cache(self, monkeypatch, tmp_path):
    local = _write_parquet(tmp_path / "train.parquet", 13)

    monkeypatch.setattr("huggingface_hub.HfFileSystem.open", lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError()))
    captured = {}

    def fake_download(repo_id, filename, repo_type, cache_dir):
      captured["repo_id"] = repo_id
      captured["filename"] = filename
      captured["repo_type"] = repo_type
      captured["cache_dir"] = cache_dir
      return str(local)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    url = "hf://datasets/org/name@rev/sub/train-00000-of-00001.parquet"
    rows = counting._remote_file_rows(url, "/tmp/cache")
    assert rows == 13
    assert captured == {
        "repo_id": "org/name",
        "filename": "sub/train-00000-of-00001.parquet",
        "repo_type": "dataset",
        "cache_dir": "/tmp/cache",
    }

  def test_uncountable_remote_file_is_none(self, monkeypatch):
    monkeypatch.setattr("huggingface_hub.HfFileSystem.open", lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k:
                        (_ for _ in ()).throw(RuntimeError()))
    rows = counting._remote_file_rows(
        "hf://datasets/org/name/train-00000-of-00001.parquet", None)
    assert rows is None


class TestHfUrlParsing:
  """``hf://`` URLs carry a repo id and a revision-pinned path."""

  def test_revision_pinned_url(self):
    assert counting._parse_hf_url(
        "hf://datasets/org/name@abc123/sub/train.parquet") == ("org/name",
                                                               "sub/train.parquet")

  def test_plain_url(self):
    assert counting._parse_hf_url("hf://datasets/org/name/train.parquet") == (
        "org/name", "train.parquet")

  def test_non_datasets_url_is_rejected(self):
    assert counting._parse_hf_url("hf://models/org/name/file.bin") is None


class TestWebDataset:
  """webdataset: the descriptor's ``num_samples``/``nsamples``, never a tar."""

  def test_info_json_num_samples(self, tmp_path):
    _write_json(tmp_path / "_info.json", {"splits": {"train": {"num_samples": 42}}})
    assert counting.webdataset_num_samples(str(tmp_path / "shard-0.tar")) == 42

  def test_info_json_shard_lengths(self, tmp_path):
    _write_json(tmp_path / "_info.json",
                {"splits": {
                    "train": {
                        "shard_lengths": [10, 20, 30]
                    }
                }})
    assert counting.webdataset_num_samples(str(tmp_path / "shard-0.tar")) == 60

  def test_multiple_splits_need_an_explicit_split(self, tmp_path):
    _write_json(tmp_path / "_info.json",
                {"splits": {
                    "train": {
                        "num_samples": 5
                    },
                    "test": {
                        "num_samples": 3
                    },
                }})
    shard = str(tmp_path / "shard-0.tar")
    assert counting.webdataset_num_samples(shard) is None
    assert counting.webdataset_num_samples(shard, split="test") == 3

  def test_wids_shardlist_nsamples(self, tmp_path):
    _write_json(
        tmp_path / "dataset.json", {
            "shardlist": [{
                "url": "shard-0.tar",
                "nsamples": 7
            }, {
                "url": "shard-1.tar",
                "nsamples": 8
            }]
        })
    assert counting.webdataset_num_samples(str(tmp_path / "shard-0.tar")) == 15

  def test_no_descriptor_is_none(self, tmp_path):
    # An ``.idx`` (per-file offsets) or a bare ``_info.json`` filenames list
    # cannot give a sample count, so the contract is ``None``.
    (tmp_path / "shard-0.tar").write_bytes(b"")
    assert counting.webdataset_num_samples(str(tmp_path / "shard-0.tar")) is None

  def test_remote_repo_reads_info_json_through_the_cache(self, monkeypatch, tmp_path):
    local = _write_json(tmp_path / "info.json",
                        {"splits": {
                            "train": {
                                "num_samples": 99
                            }
                        }})
    captured = {}

    def fake_download(repo_id, filename, repo_type, cache_dir):
      captured.update(
          dict(repo_id=repo_id,
               filename=filename,
               repo_type=repo_type,
               cache_dir=cache_dir))
      if filename == "_info.json":
        return str(local)
      raise FileNotFoundError(filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert counting.webdataset_num_samples(repo_id="org/name",
                                           split="train",
                                           cache_dir="/tmp/c") == 99
    assert captured["cache_dir"] == "/tmp/c"
    assert captured["repo_type"] == "dataset"


def _write_json(path, payload):
  with open(path, "w") as handle:
    json.dump(payload, handle)
  return path


@pytest.mark.network
def test_real_hub_metadata_and_footer():
  """End-to-end against the Hub (deselected by default; needs network)."""
  assert counting.hf_num_samples("stanfordnlp/imdb", split="train") == 25000
  assert counting.webdataset_num_samples(repo_id="pixparse/pdfa-eng-wds",
                                         split="train") == 2159432
