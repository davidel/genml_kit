"""Sample-count estimation for streaming datasets.

The estimator is a *sidecar*: it reads cheap metadata (a split size on the
dataset card, a ``parquet``/``arrow`` footer, a datasets-server size, a
webdataset descriptor/``.idx``, or a tar header walk) and never streams the
payloads.  Its contract is strict -- an ``int`` only when the count is
*trustworthy*, ``None`` otherwise -- so these tests pin every "we cannot know"
case to ``None`` rather than a guess.  The one exception is the datasets
server's ``estimated_num_rows``, which is returned as a *labelled* fallback and
is asserted to log that it is an estimate.

Everything here is offline: local ``save_to_disk`` / parquet / tar fixtures,
local webdataset-style descriptors, and mocked Hub + datasets-server calls.  The
tests that would hit the network are marked ``network`` and deselected by
default.
"""

import json
import tempfile

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genml_kit.datasets import counting
from tests.logging_helpers import log_capture


@pytest.fixture(autouse=True)
def _no_network_by_default(request, monkeypatch):
  """Keep every test offline: the datasets-server transport returns nothing.

  Tests that exercise the server tier replace ``_http_get_json`` again, so the
  default only guarantees that a stray repo id never reaches the network.  The
  ``network``-marked test opts out and must reach the real server.
  """
  if request.node.get_closest_marker("network"):
    return
  monkeypatch.setattr(counting, "_http_get_json", lambda url, token: None)


def _write_parquet(path, n_rows, tag=0):
  table = pa.table({"x": list(range(tag, tag + n_rows))})
  pq.write_table(table, str(path))
  return path


def _save_to_disk(n_rows):
  directory = tempfile.mkdtemp()
  datasets.Dataset.from_dict({"x": list(range(n_rows))}).save_to_disk(directory)
  return directory


def _write_tar(path, names, valid=True):
  """Write a plain tar holding members called *names*.

  ``valid=False`` truncates the archive mid-data (no end-of-archive marker), so
  a header walk must reject it rather than undercount.
  """
  import io
  import tarfile

  with tarfile.open(str(path), "w") as tar:
    for name in names:
      payload = b"x" * 8
      info = tarfile.TarInfo(name)
      info.size = len(payload)
      tar.addfile(info, io.BytesIO(payload))
  if not valid:
    # Keep only the first member (one 512-byte header + its 512-byte data block)
    # and drop the end-of-archive marker, so a header walk must refuse it.
    path.write_bytes(path.read_bytes()[:1024])
  return path


def _write_dali_idx(path, count):
  """Write a DALI-style ``.idx`` whose header records *count* samples."""
  with open(path, "w") as handle:
    handle.write(f"#v1.2 {count}\n")
    handle.write("png 0 8\n")
  return path


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

  def test_builder_failure_is_logged_not_raised(self, monkeypatch):

    def boom(*a, **k):
      raise RuntimeError("gated")

    monkeypatch.setattr(counting.datasets, "load_dataset_builder", boom)
    with log_capture(level="DEBUG") as logs:
      assert counting.hf_num_samples("org/private", split="train") is None
    assert any("load_dataset_builder" in record.message for record in logs.records)


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

  def test_no_descriptor_falls_back_to_the_tar_walk(self, tmp_path):
    # With no descriptor the shard itself is walked: two samples here (one run
    # of ``s0.*`` members, one of ``s1.*``).
    _write_tar(tmp_path / "shard-0.tar", ["s0.png", "s0.cls", "s1.png"])
    assert counting.webdataset_num_samples(str(tmp_path / "shard-0.tar")) == 2

  def test_unreadable_shard_is_none(self, tmp_path):
    # A truncated shard has no end-of-archive marker, so the walk declines
    # rather than reporting the smaller count it happens to have parsed.
    _write_tar(tmp_path / "shard-0.tar", ["s0.png", "s1.png"], valid=False)
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


class _ServerResponse:
  """Canned ``/size`` payload + recorded URL/token used by ``_http_get_json``."""

  def __init__(self, payload):
    self.payload = payload
    self.urls = []
    self.tokens = []

  def __call__(self, url, token):
    self.urls.append(url)
    self.tokens.append(token)
    return self.payload


def _server_payload(config, split, num_rows, partial, estimate=None):
  entry = {
      "config": config,
      "split": split,
      "num_rows": num_rows,
      "estimated_num_rows": estimate,
  }
  return {"size": {"splits": [entry]}, "partial": partial}


class TestDatasetsServer:
  """The datasets-server ``/size`` tier: exact when full, else a labelled estimate."""

  @pytest.fixture(autouse=True)
  def _skip_the_card_tier(self, monkeypatch):
    """Make Tier 1 (the dataset card) fail so the server tier is the one under test.

    Otherwise ``load_dataset_builder`` would resolve the real card over the
    network and answer before the server is ever consulted.
    """

    def boom(*a, **k):
      raise RuntimeError("offline")

    monkeypatch.setattr(counting.datasets, "load_dataset_builder", boom)

  def test_full_scan_returns_the_exact_count(self, monkeypatch):
    response = _ServerResponse(
        _server_payload("plain_text", "train", 25000, partial=False))
    monkeypatch.setattr(counting, "_http_get_json", response)
    assert counting.hf_num_samples("org/name", name="plain_text",
                                   split="train") == 25000
    assert "dataset=org%2Fname" in response.urls[0]
    assert "config=plain_text" in response.urls[0]

  def test_partial_scan_returns_the_labelled_estimate(self, monkeypatch):
    response = _ServerResponse(
        _server_payload("en", "train", 2207027, partial=True, estimate=364695767))
    monkeypatch.setattr(counting, "_http_get_json", response)
    with log_capture(level="INFO") as logs:
      assert counting.hf_num_samples("allenai/c4", name="en",
                                     split="train") == 364695767
    # The number is an estimate and must say so.
    assert any("estimat" in record.message.lower() for record in logs.records)

  def test_partial_scan_without_estimate_is_none(self, monkeypatch):
    response = _ServerResponse(
        _server_payload("default", "train", 7100, partial=True, estimate=None))
    monkeypatch.setattr(counting, "_http_get_json", response)
    assert counting.hf_num_samples("org/name", name="default", split="train") is None

  def test_token_is_sent(self, monkeypatch):
    response = _ServerResponse(_server_payload("plain_text", "train", 10,
                                               partial=False))
    monkeypatch.setattr(counting, "_http_get_json", response)
    counting.hf_num_samples("org/gated",
                            name="plain_text",
                            split="train",
                            token="secret")
    assert response.tokens[0] == "secret"

  def test_unknown_split_is_none(self, monkeypatch):
    response = _ServerResponse(_server_payload("plain_text", "train", 10,
                                               partial=False))
    monkeypatch.setattr(counting, "_http_get_json", response)
    assert counting.hf_num_samples("org/name", name="plain_text", split="test") is None

  def test_local_path_never_asks_the_server(self, monkeypatch, tmp_path):
    _write_parquet(tmp_path / "train-00000.parquet", 4)
    response = _ServerResponse({})
    monkeypatch.setattr(counting, "_http_get_json", response)
    assert counting.hf_num_samples(str(tmp_path), split="train") == 4
    assert response.urls == []

  def test_webdataset_uses_exact_only(self, monkeypatch):
    # A webdataset repo is a partial scan, so even though the server offers an
    # estimate, the exact-only path (used for webdataset) must decline it.
    response = _ServerResponse(
        _server_payload("default", "train", 7100, partial=True, estimate=99))
    monkeypatch.setattr(counting, "_http_get_json", response)
    assert counting._server_size("org/wds", None, "train", None,
                                 exact_only=True) is None

  def test_transport_failure_is_none(self, monkeypatch):
    monkeypatch.setattr(counting, "_http_get_json", lambda url, token: None)
    assert counting.hf_num_samples("org/name", split="train") is None


class TestTarWalk:
  """The tar header walk: exact, decode-free, and matching webdataset's grouping."""

  def test_runs_of_members_make_one_sample_each(self, tmp_path):
    path = _write_tar(tmp_path / "s.tar", ["s0.png", "s0.cls", "s1.png", "s1.cls"])
    assert counting._local_tar_count(str(path)) == 2

  def test_non_adjacent_repeated_key_counts_twice(self, tmp_path):
    # webdataset groups *sequentially*: s0 reappearing after s1 is a new sample.
    path = _write_tar(tmp_path / "s.tar", ["s0.png", "s1.png", "s0.png"])
    assert counting._local_tar_count(str(path)) == 3

  def test_meta_and_extensionless_members_are_skipped(self, tmp_path):
    path = _write_tar(tmp_path / "s.tar",
                      ["__meta__", "s0.png", "noextension", "s0.cls"])
    assert counting._local_tar_count(str(path)) == 1

  def test_dots_in_key_group_by_the_first_dot(self, tmp_path):
    path = _write_tar(tmp_path / "s.tar", ["a.b.c.png", "a.b.c.json"])
    assert counting._local_tar_count(str(path)) == 1

  def test_empty_tar_is_zero(self, tmp_path):
    path = _write_tar(tmp_path / "s.tar", [])
    assert counting._local_tar_count(str(path)) == 0

  def test_truncated_tar_is_none(self, tmp_path):
    path = _write_tar(tmp_path / "s.tar", ["s0.png", "s1.png"], valid=False)
    assert counting._local_tar_count(str(path)) is None

  def test_dali_idx_short_circuits_the_walk(self, tmp_path):
    # A valid DALI ``.idx`` records the count in its header and is trusted over
    # the walk (the shard here holds two samples; the index says five).
    _write_tar(tmp_path / "s.tar", ["s0.png", "s1.png"])
    _write_dali_idx(tmp_path / "s.tar.idx", 5)
    assert counting.webdataset_num_samples(str(tmp_path / "s.tar")) == 5

  def test_non_dali_idx_is_ignored(self, tmp_path):
    # A wids pickle (or any non-DALI file) must not be parsed; the walk decides.
    _write_tar(tmp_path / "s.tar", ["s0.png", "s1.png"])
    (tmp_path / "s.tar.idx").write_bytes(b"\x80\x04pickle")
    assert counting.webdataset_num_samples(str(tmp_path / "s.tar")) == 2

  def test_brace_pattern_expands_across_shards(self, tmp_path):
    _write_tar(tmp_path / "s-0000.tar", ["s0.png", "s1.png"])
    _write_tar(tmp_path / "s-0001.tar", ["s2.png"])
    pattern = str(tmp_path / "s-{0000..0001}.tar")
    assert counting.webdataset_num_samples(pattern) == 3


class TestRemoteTarWalk:
  """A remote shard with no index is walked through ``HfFileSystem`` ranged reads."""

  def test_tar_files_are_listed_and_summed(self, monkeypatch):
    walked = {
        "a.tar": 3,
        "b.tar": 4,
    }

    monkeypatch.setattr(counting, "_repo_files", lambda repo: list(walked))
    monkeypatch.setattr(counting, "_remote_shard_count",
                        lambda repo, name, has_idx, cache: walked[name])
    assert counting._remote_tar_total("org/wds", None) == 7

  def test_one_bad_shard_poisons_the_total(self, monkeypatch):
    monkeypatch.setattr(counting, "_repo_files", lambda repo: ["a.tar", "b.tar"])
    monkeypatch.setattr(
        counting, "_remote_shard_count", lambda repo, name, has_idx, cache: 3
        if name == "a.tar" else None)
    assert counting._remote_tar_total("org/wds", None) is None

  def test_idx_is_probed_only_when_the_repo_ships_one(self, monkeypatch):
    # No ``.idx`` in the listing: no per-shard index request must be made.
    monkeypatch.setattr(counting, "_repo_files", lambda repo: ["a.tar", "b.tar"])
    monkeypatch.setattr(counting, "_fetch_repo_file",
                        lambda *a, **k: pytest.fail("should not fetch an index"))
    monkeypatch.setattr(counting, "_remote_shard_count",
                        lambda repo, name, has_idx, cache: has_idx)
    assert counting._remote_tar_total("org/wds", None) == 0

  def test_shard_uses_its_idx_when_present(self, monkeypatch, tmp_path):
    idx = _write_dali_idx(tmp_path / "a.tar.idx", 9)
    monkeypatch.setattr(counting, "_fetch_repo_file",
                        lambda repo, name, cache: str(idx))
    # No tar walk should be attempted when the index answers.
    monkeypatch.setattr("huggingface_hub.HfFileSystem",
                        lambda *a, **k: pytest.fail("should not open the tar"))
    assert counting._remote_shard_count("org/wds", "a.tar", True, None) == 9


def _write_json(path, payload):
  with open(path, "w") as handle:
    json.dump(payload, handle)
  return path


@pytest.mark.network
def test_real_hub_metadata_and_footer():
  """End-to-end against the Hub (deselected by default; needs network)."""
  assert counting.hf_num_samples("stanfordnlp/imdb", name="plain_text",
                                 split="train") == 25000
  assert counting.webdataset_num_samples(repo_id="pixparse/pdfa-eng-wds",
                                         split="train") == 2159432
