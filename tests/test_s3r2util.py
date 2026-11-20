"""Tests for scripts/s3r2util.py (CloudFlare R2 CLI utility)."""

import datetime
import importlib
import io
import os
import sys
from argparse import Namespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import s3r2util


class _RecordingS3Client:
  """Minimal stub capturing upload/download/copy/delete calls."""

  def __init__(self):
    self.uploads = []
    self.downloads = []
    self.copies = []
    self.deleted = []

  def upload_file(self, local_path, bucket, key):
    self.uploads.append((local_path, bucket, key))

  def download_file(self, bucket, key, local_path):
    self.downloads.append((bucket, key, local_path))

  def copy(self, copy_source, bucket, key):
    self.copies.append((copy_source["Bucket"], copy_source["Key"], bucket, key))

  def delete_object(self, Bucket, Key):
    self.deleted.append((Bucket, Key))


def _cp_args(sources, destination, quiet=True, progress=False):
  return Namespace(sources=sources,
                   destination=destination,
                   quiet=quiet,
                   progress=progress)


class TestJoinDirDestination:

  def test_trailing_slash(self):
    assert s3r2util.join_dir_destination("r2://destr2/", "bbbb") == ("r2://destr2/bbbb")

  def test_no_trailing_slash(self):
    assert s3r2util.join_dir_destination("r2://destr2", "bbbb") == ("r2://destr2/bbbb")

  def test_local_dir(self):
    assert s3r2util.join_dir_destination("/dest/dir", "bbbb") == ("/dest/dir/bbbb")


class TestCpDirDestination:
  """cp into a directory destination follows Unix cp: basename."""

  def test_local_to_r2_uses_basename(self, tmp_path):
    src = tmp_path / "bbbb"
    src.write_bytes(b"data")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(src), "destr2", "bbbb")]

  def test_local_nested_file_uses_basename(self, tmp_path, monkeypatch):
    src = tmp_path / "sub" / "bbbb"
    src.parent.mkdir()
    src.write_bytes(b"data")
    monkeypatch.chdir(tmp_path / "sub")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    # Unix cp places the basename in the destination directory.
    assert client.uploads == [(str(src), "destr2", "bbbb")]

  def test_r2_to_r2_uses_basename(self):
    client = _RecordingS3Client()
    args = _cp_args(["r2://srca/aaaa/bbbb"], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "bbbb")]

  def test_r2_to_local_uses_basename(self, tmp_path):
    client = _RecordingS3Client()
    dest = str(tmp_path / "destr2") + "/"
    args = _cp_args(["r2://srca/aaaa/bbbb"], dest)
    s3r2util.handle_cp(args, client)
    assert client.downloads == [("srca", "aaaa/bbbb", str(tmp_path / "destr2" / "bbbb"))
                               ]

  def test_local_file_to_r2_file_dest(self, tmp_path):
    src = tmp_path / "bbbb"
    src.write_bytes(b"data")
    client = _RecordingS3Client()
    args = _cp_args([str(src)], "r2://destr2/cccc")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(src), "destr2", "cccc")]

  def test_multiple_sources(self, tmp_path):
    s1 = tmp_path / "s1"
    s2 = tmp_path / "s2"
    s1.write_bytes(b"1")
    s2.write_bytes(b"2")
    client = _RecordingS3Client()
    args = _cp_args([str(s1), str(s2)], "r2://destr2/")
    s3r2util.handle_cp(args, client)
    assert client.uploads == [(str(s1), "destr2", "s1"), (str(s2), "destr2", "s2")]


class TestComputeDestKey:

  def test_appends_relative_to_prefix(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa/", "dest/") == ("dest/bbbb")

  def test_bucket_root_dest_no_leading_slash(self):
    # Regression: used to produce "/bbbb", an invisible-to-prefix-ops key.
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa/", "") == "bbbb"

  def test_bucket_root_dest_nested_relative(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb/cccc", "aaaa/", "") == ("bbbb/cccc")

  def test_root_dest_no_slash_in_relative(self):
    assert s3r2util.compute_dest_key("bbbb", "", "") == "bbbb"

  def test_prefix_dest_normalizes(self):
    assert s3r2util.compute_dest_key("aaaa/bbbb", "aaaa", "dest") == ("dest/bbbb")

  def test_empty_relative_returns_dst_prefix(self):
    assert s3r2util.compute_dest_key("aaaa", "aaaa", "dest/") == "dest/"


class TestMvDirDestination:
  """mv keeps its Unix semantics (regression guard against cp drift)."""

  def test_single_object_into_prefix(self):
    client = _RecordingS3Client()
    args = Namespace(sources=["r2://srca/aaaa/bbbb"],
                     destination="r2://destr2/",
                     quiet=True)
    s3r2util.handle_mv(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "bbbb")]
    assert client.deleted == [("srca", "aaaa/bbbb")]

  def test_rename_to_explicit_key(self):
    client = _RecordingS3Client()
    args = Namespace(sources=["r2://srca/aaaa/bbbb"],
                     destination="r2://destr2/cccc",
                     quiet=True)
    s3r2util.handle_mv(args, client)
    assert client.copies == [("srca", "aaaa/bbbb", "destr2", "cccc")]


_STUB_TIME = datetime.datetime(2024, 1, 1, tzinfo=datetime.timezone.utc)


@pytest.fixture(autouse=True)
def _restore_default_chunk_size():
  """Reload s3r2util after tests that touched R2_STREAM_CHUNK_SIZE."""
  yield
  if os.environ.pop("R2_STREAM_CHUNK_SIZE",
                    None) is None and (s3r2util.STREAM_CHUNK_SIZE
                                       == s3r2util.DEFAULT_STREAM_CHUNK_SIZE):
    return
  importlib.reload(s3r2util)


def _reload_with_chunk_env(value=None):
  """Set R2_STREAM_CHUNK_SIZE (or clear it) and reload the module."""
  if value is None:
    os.environ.pop("R2_STREAM_CHUNK_SIZE", None)
  else:
    os.environ["R2_STREAM_CHUNK_SIZE"] = value
  importlib.reload(s3r2util)


class _StubBody:
  """Minimal botocore StreamingBody stand-in serving in-memory bytes."""

  def __init__(self, data):
    self._stream = io.BytesIO(data)
    self.sizes = []

  def iter_chunks(self, chunk_size):
    while True:
      chunk = self._stream.read(chunk_size)
      if not chunk:
        break
      self.sizes.append(len(chunk))
      yield chunk


class _CatS3Client:
  """Stub S3 client serving objects from a dict and listing them."""

  def __init__(self, objects):
    self.objects = objects
    self.get_object_calls = []

  def get_object(self, Bucket, Key):
    self.get_object_calls.append((Bucket, Key))
    if (Bucket, Key) not in self.objects:
      raise s3r2util.ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
    body = self.objects[(Bucket, Key)]
    if not isinstance(body, _StubBody):
      body = _StubBody(body)
    return {"Body": body}

  def get_paginator(self, operation):
    assert operation == "list_objects_v2"
    return _CatPaginator(self.objects)


class _CatPaginator:

  def __init__(self, objects):
    self.objects = objects

  def paginate(self, **kwargs):
    prefix = kwargs.get("Prefix", "")
    contents = [{
        "Key": key,
        "Size": len(data),
        "LastModified": _STUB_TIME,
    }
                for (bucket, key), data in sorted(self.objects.items())
                if bucket == kwargs["Bucket"] and key.startswith(prefix)]
    yield {"Contents": contents}


class TestCat:
  """cat streams object/file bytes to stdout, silently."""

  @staticmethod
  def _args(sources):
    return Namespace(sources=sources, quiet=True)

  def test_single_object_streams_to_stdout(self, capsysbinary):
    client = _CatS3Client({("bkt", "dir/file.tar"): b"PAYLOAD-0123456789"})
    s3r2util.handle_cat(self._args(["r2://bkt/dir/file.tar"]), client)
    captured = capsysbinary.readouterr()
    assert captured.out == b"PAYLOAD-0123456789"
    assert captured.err == b""

  def test_multiple_objects_in_order(self, capsysbinary):
    client = _CatS3Client({
        ("bkt", "a.bin"): b"AAA",
        ("bkt", "b.bin"): b"BBB",
    })
    s3r2util.handle_cat(self._args(["r2://bkt/a.bin", "r2://bkt/b.bin"]), client)
    assert capsysbinary.readouterr().out == b"AAABBB"

  def test_mixed_local_and_remote(self, capsysbinary, tmp_path):
    local = tmp_path / "local.bin"
    local.write_bytes(b"LOCAL")
    client = _CatS3Client({("bkt", "remote.bin"): b"REMOTE"})
    s3r2util.handle_cat(self._args([str(local), "r2://bkt/remote.bin"]), client)
    assert capsysbinary.readouterr().out == b"LOCALREMOTE"

  def test_wildcard_expands_in_sorted_order(self, capsysbinary):
    client = _CatS3Client({
        ("bkt", "logs-001.tar.gz"): b"1",
        ("bkt", "logs-002.tar.gz"): b"2",
        ("bkt", "logs-010.tar.gz"): b"10",
        ("bkt", "other.txt"): b"X",
    })
    s3r2util.handle_cat(self._args(["r2://bkt/logs-*.tar.gz"]), client)
    captured = capsysbinary.readouterr()
    assert captured.out == b"1210"
    assert client.get_object_calls == [
        ("bkt", "logs-001.tar.gz"),
        ("bkt", "logs-002.tar.gz"),
        ("bkt", "logs-010.tar.gz"),
    ]

  def test_bucket_only_path_is_fatal(self, capsysbinary):
    client = _CatS3Client({})
    with pytest.raises(SystemExit) as excinfo:
      s3r2util.handle_cat(self._args(["r2://bkt"]), client)
    assert excinfo.value.code == 1
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"object key" in captured.err

  def test_missing_local_file_is_fatal(self, capsysbinary, tmp_path):
    client = _CatS3Client({})
    missing = tmp_path / "nope.bin"
    with pytest.raises(SystemExit) as excinfo:
      s3r2util.handle_cat(self._args([str(missing)]), client)
    assert excinfo.value.code == 1
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"No such file" in captured.err

  def test_get_object_error_is_fatal(self, capsysbinary):
    client = _CatS3Client({})
    with pytest.raises(SystemExit) as excinfo:
      s3r2util.handle_cat(self._args(["r2://bkt/gone.bin"]), client)
    assert excinfo.value.code == 1
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"cat failed" in captured.err


class TestGetenv:
  """The generic env fetch/convert helper used for configuration."""

  def test_unset_and_blank_return_default(self, monkeypatch):
    monkeypatch.delenv("R2_TEST_UNSET", raising=False)
    assert s3r2util.getenv("R2_TEST_UNSET") is None
    assert s3r2util.getenv("R2_TEST_UNSET", default="fallback") == "fallback"
    monkeypatch.setenv("R2_TEST_UNSET", "   ")
    assert s3r2util.getenv("R2_TEST_UNSET", default=7) == 7

  def test_converts_to_requested_type(self, monkeypatch):
    monkeypatch.setenv("R2_TEST_INT", " 42 ")
    assert s3r2util.getenv("R2_TEST_INT", vtype=int) == 42

  def test_conversion_failure_is_fatal(self, monkeypatch, capsysbinary):
    monkeypatch.setenv("R2_TEST_INT", "many")
    with pytest.raises(SystemExit) as excinfo:
      s3r2util.getenv("R2_TEST_INT", vtype=int)
    assert excinfo.value.code == 1
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"R2_TEST_INT" in captured.err
    assert b"int" in captured.err


class TestStreamChunkSizeEnv:
  """R2_STREAM_CHUNK_SIZE configures the `cat` streaming chunk size."""

  def test_default_is_1mib(self):
    _reload_with_chunk_env()
    assert s3r2util.STREAM_CHUNK_SIZE == 1024 * 1024

  def test_env_override_used_for_remote_reads(self):
    _reload_with_chunk_env("64")
    body = _StubBody(b"x" * 100)
    client = _CatS3Client({("bkt", "obj.bin"): body})
    args = Namespace(sources=["r2://bkt/obj.bin"], quiet=True)
    s3r2util.handle_cat(args, client)
    assert body.sizes == [64, 36]
    assert s3r2util.STREAM_CHUNK_SIZE == 64

  def test_blank_env_falls_back_to_default(self):
    _reload_with_chunk_env("   ")
    assert s3r2util.STREAM_CHUNK_SIZE == s3r2util.DEFAULT_STREAM_CHUNK_SIZE

  def test_non_integer_env_is_fatal(self, capsysbinary):
    with pytest.raises(SystemExit) as excinfo:
      _reload_with_chunk_env("64KB")
    assert excinfo.value.code == 1
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"R2_STREAM_CHUNK_SIZE" in captured.err


class TestOutputChannels:
  """stdout carries data only; status output belongs on stderr."""

  def test_log_writes_to_stderr(self, capsysbinary):
    s3r2util.log("status line", quiet=False)
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert captured.err == b"status line\n"

  def test_log_respects_quiet(self, capsysbinary):
    s3r2util.log("status line", quiet=True)
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert captured.err == b""

  def test_ls_listings_go_to_stderr(self, capsysbinary):
    client = _CatS3Client({("bkt", "prefix/one.bin"): b"A"})
    args = Namespace(path="r2://bkt/prefix", quiet=False)
    s3r2util.handle_ls(args, client)
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert b"prefix/one.bin" in captured.err

  def test_presign_url_goes_to_stdout(self, capsysbinary):
    client = _CatS3Client({})

    def fake_presign(*_args, **_kwargs):
      return "https://example.com/signed"

    client.generate_presigned_url = fake_presign
    args = Namespace(path="r2://bkt/obj.bin", expires=60)
    s3r2util.handle_presign(args, client)
    captured = capsysbinary.readouterr()
    assert captured.out == b"https://example.com/signed\n"
    assert captured.err == b""
