"""Tests for the run manifest (reproducibility snapshot)."""

import argparse
import json
import os

import pytest

import genml_kit.training.run_manifest as rm


def _args(**overrides):
  """A resolved-args namespace shaped like the train driver's."""
  defaults = {
      "pipeline": "images",
      "model": "tiny_vit",
      "epochs": 3,
      "seed": 42,
      "log_dir": None,
      "checkpoint": None,
      "data_root": "/tmp/data",
      "amp_dtype": "float32",
  }
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


class TestGitSha:

  def test_none_when_not_a_repo(self, tmp_path):
    # A directory that is not a git checkout must yield None, not raise.
    assert rm.git_sha(repo_root=str(tmp_path)) is None

  def test_none_outside_checkout(self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rm.git_sha() is None

  def test_current_repo_sha(self):
    # Inside the genml_kit checkout the SHA must be a 40-hex string.
    sha = rm.git_sha()
    if sha is None:
      pytest.skip("repository exported without .git (sdist)")
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


class TestCollect:

  def test_resolved_args_roundtrip(self):
    manifest = rm.collect_run_manifest(_args())
    argv = manifest["argv_resolved"]
    assert argv["pipeline"] == "images"
    assert argv["epochs"] == 3
    assert argv["amp_dtype"] == "float32"

  def test_versions_block(self):
    manifest = rm.collect_run_manifest(_args())
    assert manifest["versions"]["python"]
    assert manifest["versions"]["torch"].startswith("2.")

  def test_missing_attrs_raise(self):
    # The manifest reads flags directly: a namespace missing a declared
    # flag must raise, so fixture drift is caught immediately instead of
    # silently producing a "seed=None" manifest.
    with pytest.raises(AttributeError):
      rm.collect_run_manifest(argparse.Namespace(model="x"))

  def test_non_jsonable_values_are_reprd(self):
    manifest = rm.collect_run_manifest(_args(custom_fn=lambda x: x))
    assert manifest["argv_resolved"]["custom_fn"].startswith("<function")


class TestWrite:

  def test_writes_json_to_log_dir(self, tmp_path):
    log_dir = str(tmp_path / "logs")
    manifest = rm.write_manifest(_args(), log_dir)
    path = os.path.join(log_dir, rm.MANIFEST_NAME)
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
      loaded = json.load(f)
    assert loaded == manifest

  def test_creates_missing_log_dir(self, tmp_path):
    log_dir = str(tmp_path / "nested" / "logs")
    rm.write_manifest(_args(), log_dir)
    assert os.path.isdir(log_dir)

  def test_read_only_log_dir_does_not_raise(self, tmp_path):
    log_dir = tmp_path / "ro"
    log_dir.mkdir()
    os.chmod(str(log_dir), 0o500)
    try:
      manifest = rm.write_manifest(_args(), str(log_dir))
    finally:
      os.chmod(str(log_dir), 0o700)
    # Manifest is still returned even when the write failed.
    assert manifest["argv_resolved"]["pipeline"] == "images"
