"""Tests for cloud storage helpers (GCS, R2, S3 URI handling and S3 upload)."""
import logging
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scdiag.storage_utils import (
    _upload_s3,
    parse_storage_uri,
    save_checkpoint,
    storage_download,
    storage_upload,
)

try:
  import boto3  # noqa: F401
  _HAS_BOTO3 = True
except ImportError:
  _HAS_BOTO3 = False


class TestParseStorageUri:

  def test_s3_uri(self):
    assert parse_storage_uri("s3://my-bucket/runs/exp1") == ("s3", "my-bucket",
                                                             "runs/exp1")

  def test_s3_uri_bucket_only(self):
    assert parse_storage_uri("s3://my-bucket") == ("s3", "my-bucket", "")

  def test_gcs_and_r2_unchanged(self):
    assert parse_storage_uri("gs://b/p") == ("gs", "b", "p")
    assert parse_storage_uri("r2://b/p") == ("r2", "b", "p")

  def test_rejects_other_schemes(self):
    for uri in ("http://b/p", "azure://b/p", "my-bucket/p"):
      with pytest.raises(ValueError, match="URI must start with"):
        parse_storage_uri(uri)


class _FakeS3Client:

  def __init__(self):
    self.calls = []

  def upload_file(self, local_path, bucket, key):
    self.calls.append((local_path, bucket, key))


@pytest.mark.skipif(not _HAS_BOTO3, reason="boto3 not installed (s3 extra)")
class TestUploadS3:

  @pytest.fixture
  def fake_client(self, monkeypatch):
    client = _FakeS3Client()
    captured = {}

    def fake_boto3_client(service, **kwargs):
      captured["service"] = service
      captured["kwargs"] = kwargs
      return client

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    client.captured = captured
    return client

  @pytest.fixture
  def local_file(self, tmp_path):
    path = tmp_path / "model_latest.pt"
    path.write_bytes(b"checkpoint-bytes")
    return str(path)

  def test_explicit_credentials_and_session_token(self, fake_client, local_file,
                                                  monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "TOKEN_TEST")

    result = _upload_s3("my-bucket", local_file, "runs/exp1")

    assert fake_client.captured["service"] == "s3"
    assert fake_client.captured["kwargs"] == {
        "aws_access_key_id": "AKIA_TEST",
        "aws_secret_access_key": "SECRET_TEST",
        "aws_session_token": "TOKEN_TEST",
    }
    assert fake_client.calls == [(local_file, "my-bucket", "runs/exp1/model_latest.pt")]
    assert result == "s3://my-bucket/runs/exp1/model_latest.pt"

  def test_session_token_defaults_to_none(self, fake_client, local_file, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_TEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SECRET_TEST")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

    _upload_s3("my-bucket", local_file, "")

    assert fake_client.captured["kwargs"]["aws_session_token"] is None
    assert fake_client.calls == [(local_file, "my-bucket", "model_latest.pt")]

  def test_falls_back_to_default_chain_without_env_creds(self, fake_client, local_file,
                                                         monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
      monkeypatch.delenv(var, raising=False)

    _upload_s3("my-bucket", local_file, "prefix")

    assert fake_client.captured["kwargs"] == {}
    assert fake_client.calls == [(local_file, "my-bucket", "prefix/model_latest.pt")]


class TestStorageUploadDispatch:

  def test_s3_dispatch(self, monkeypatch, tmp_path):
    local = tmp_path / "ckpt.pt"
    local.write_bytes(b"x")
    seen = {}

    def fake_upload_s3(bucket, path, prefix):
      seen["args"] = (bucket, path, prefix)
      return f"s3://{bucket}/{prefix}/ckpt.pt"

    monkeypatch.setattr("scdiag.storage_utils._upload_s3", fake_upload_s3)
    result = storage_upload("b", str(local), "p", scheme="s3")
    assert seen["args"] == ("b", str(local), "p")
    assert result == "s3://b/p/ckpt.pt"

  def test_unknown_scheme_fatal(self, tmp_path):
    local = tmp_path / "ckpt.pt"
    local.write_bytes(b"x")
    with pytest.raises(ValueError, match="Unsupported storage scheme"):
      storage_upload("b", str(local), "p", scheme="wasabi")


class TestSaveCheckpointS3:

  def test_save_with_s3_remote(self, tmp_path, monkeypatch):
    local = tmp_path / "nested" / "ckpt.pt"
    seen = {}

    def fake_upload_s3(bucket, path, prefix):
      seen["args"] = (bucket, path, prefix)
      return f"s3://{bucket}/{prefix}/ckpt.pt"

    monkeypatch.setattr("scdiag.storage_utils._upload_s3", fake_upload_s3)
    result = save_checkpoint({"epoch": 1}, str(local), remote_uri="s3://my-bucket/runs")
    assert os.path.isfile(result)
    assert seen["args"] == ("my-bucket", str(local), "runs")

  def test_no_remote_no_upload(self, tmp_path):
    local = tmp_path / "ckpt.pt"
    result = save_checkpoint({"epoch": 1}, str(local))
    assert os.path.isfile(result)
    assert result == str(local)


class _ClientError(Exception):
  """Stands in for the boto3 client error type."""

  def __init__(self, code):
    super().__init__(code)
    self.response = {"Error": {"Code": code}}


class _FakeDownloadClient:
  """Mimics the boto3 client surface used by the download helpers."""

  def __init__(self, objects=(), error=None):
    self.objects = set(objects)
    self.error = error
    self.calls = []

  def head_object(self, Bucket, Key):
    self.calls.append(("head", Bucket, Key))
    if self.error is not None:
      raise _ClientError(self.error)
    if Key not in self.objects:
      raise _ClientError("404")

  def download_file(self, Bucket, Key, Filename):
    self.calls.append(("get", Bucket, Key))
    with open(Filename, "wb") as handle:
      handle.write(b"ckpt")


def _use_boto3_client(fake, monkeypatch):
  """Point the ``import boto3`` inside storage_utils at *fake*."""
  fake.exceptions = SimpleNamespace(ClientError=_ClientError)
  monkeypatch.setattr("boto3.client", lambda service, **kwargs: fake)


def _write_bytes(path, data):
  """Write *data* to *path* (for stub download callbacks)."""
  with open(path, "wb") as handle:
    handle.write(data)


class TestStorageDownloadS3:

  def test_found_downloads_with_prefix_key(self, tmp_path, monkeypatch):
    fake = _FakeDownloadClient(objects=("runs/model_latest.pt",))
    _use_boto3_client(fake, monkeypatch)
    local = tmp_path / "model_latest.pt"
    assert storage_download("s3://my-bucket/runs", str(local)) is True
    assert fake.calls == [
        ("head", "my-bucket", "runs/model_latest.pt"),
        ("get", "my-bucket", "runs/model_latest.pt"),
    ]
    assert local.read_bytes() == b"ckpt"

  def test_missing_object_returns_false(self, tmp_path, monkeypatch):
    fake = _FakeDownloadClient()
    _use_boto3_client(fake, monkeypatch)
    local = tmp_path / "model_latest.pt"
    assert storage_download("s3://my-bucket/runs", str(local)) is False
    # Only the existence probe ran; nothing was written.
    assert fake.calls == [("head", "my-bucket", "runs/model_latest.pt")]
    assert not local.exists()

  def test_connection_error_warns_and_returns_false(self, tmp_path, monkeypatch,
                                                    caplog):
    fake = _FakeDownloadClient(error="403")
    _use_boto3_client(fake, monkeypatch)
    local = tmp_path / "model_latest.pt"
    with caplog.at_level(logging.WARNING):
      assert storage_download("s3://my-bucket/runs", str(local)) is False
    assert "failed" in caplog.text
    # The failed attempt must not leave a partial file behind.
    assert not local.exists()


class TestStorageDownloadDispatch:

  @staticmethod
  def _gcs_module(monkeypatch, bucket_cls):
    """Install importable google.cloud.storage stubs."""
    storage_mod = ModuleType("storage")
    storage_mod.Client = lambda: SimpleNamespace(bucket=lambda name: bucket_cls())
    cloud_mod = ModuleType("cloud")
    cloud_mod.storage = storage_mod
    google_mod = ModuleType("google")
    google_mod.cloud = cloud_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_mod)

  def test_gcs_found_downloads(self, tmp_path, monkeypatch):
    local = tmp_path / "model_latest.pt"

    class _Bucket:

      def blob(self, name):
        return SimpleNamespace(
            exists=lambda: True,
            download_to_filename=lambda path: _write_bytes(path, b"ckpt"))

    self._gcs_module(monkeypatch, _Bucket)
    assert storage_download("gs://my-bucket/runs", str(local)) is True
    assert local.read_bytes() == b"ckpt"

  def test_gcs_missing_blob_returns_false(self, tmp_path, monkeypatch):
    local = tmp_path / "model_latest.pt"

    class _Bucket:

      def blob(self, name):
        return SimpleNamespace(exists=lambda: False)

    self._gcs_module(monkeypatch, _Bucket)
    assert storage_download("gs://my-bucket/runs", str(local)) is False
    assert not local.exists()

  def test_r2_dispatch_uses_endpoint_and_prefix(self, tmp_path, monkeypatch):
    fake = _FakeDownloadClient(objects=("runs/model_latest.pt",))
    _use_boto3_client(fake, monkeypatch)
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://acc.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    local = tmp_path / "model_latest.pt"
    assert storage_download("r2://my-bucket/runs", str(local)) is True
    assert fake.calls == [
        ("head", "my-bucket", "runs/model_latest.pt"),
        ("get", "my-bucket", "runs/model_latest.pt"),
    ]
