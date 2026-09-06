"""Tests for the shared checkpointing utilities."""

import copy
import logging
import os

import torch
from torch import nn

from scdiag.checkpointing import (
    CheckpointSaver,
    fetch_remote_checkpoint,
    filter_state_dict,
    format_count,
    load_checkpoint_weights,
    should_save_periodic,
)
from scdiag.storage_utils import save_checkpoint


class TestFormatCount:

  def test_below_1k(self):
    assert format_count(999) == "999"

  def test_kilobytes(self):
    assert format_count(1024) == "1.00K"

  def test_millions(self):
    assert format_count(25_600_000) == "24.41M"

  def test_gigabytes(self):
    assert format_count(3_221_225_472) == "3.00G"


class TestShouldSavePeriodic:

  def test_fires_at_multiple(self):
    assert should_save_periodic(500, 500) is True
    assert should_save_periodic(1000, 500) is True

  def test_no_fire_off_multiple(self):
    assert should_save_periodic(499, 500) is False
    assert should_save_periodic(501, 500) is False
    assert should_save_periodic(1, 500) is False

  def test_disabled_for_zero_or_negative_interval(self):
    assert should_save_periodic(500, 0) is False
    assert should_save_periodic(500, -1) is False

  def test_never_fires_at_step_zero(self):
    assert should_save_periodic(0, 500) is False
    assert should_save_periodic(0, 0) is False


class _TinyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(2, 3)


class _NestedWrapper(nn.Module):
  """Registers one module under two names, like the I-JEPA student."""

  def __init__(self, module):
    super().__init__()
    self.encoder = module
    self.model = module


class _WrappedAdapter(nn.Module):
  """Holds the backbone under model.*, like ConvViTAdapter for fine-tuning."""

  def __init__(self, module):
    super().__init__()
    self.model = module


class TestCheckpointSaver:

  def _make_saver(self, tmp_path):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(
        model,
        optimizer,
        None,
        root=str(tmp_path / "ckpt"),
        save_every=500,
    )
    return saver, model, optimizer

  def test_should_save_truth_table(self):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(model, optimizer, None, root="/unused", save_every=500)
    assert saver.should_save(500) is True
    assert saver.should_save(1000) is True
    assert saver.should_save(499) is False
    assert saver.should_save(501) is False
    assert saver.should_save(0) is False

  def test_should_save_disabled_by_default(self):
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saver = CheckpointSaver(model, optimizer, None, root="/unused")
    assert saver.should_save(500) is False
    assert saver.should_save(1000) is False

  def test_save_latest_and_best_write_files(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    path = saver.save_latest(2, global_step=500, best_macro_f1=90.0)
    best = saver.save_best(3, global_step=600, best_macro_f1=91.5)
    assert path.endswith("_latest.pt")
    assert best.endswith("_best.pt")
    assert os.path.exists(path)
    assert os.path.exists(best)

  def test_saved_dict_carries_extra_kv(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    saver.save_latest(1, global_step=250, best_macro_f1=42.0, custom_kv="hello")
    state = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
    assert state["epoch"] == 1
    assert state["global_step"] == 250
    assert state["best_macro_f1"] == 42.0
    assert state["custom_kv"] == "hello"
    assert "fc.weight" in state["model_state_dict"]
    assert "optimizer_state_dict" in state
    assert state["scheduler_state_dict"] is None

  def test_save_latest_is_idempotent_path(self, tmp_path):
    saver, _, _ = self._make_saver(tmp_path)
    first = saver.save_latest(1, global_step=10)
    second = saver.save_latest(1, global_step=20)
    assert first == second
    state = torch.load(str(second), weights_only=False)
    assert state["global_step"] == 20


class TestFilterStateDict:

  def test_identical_keys(self):
    state = {"a": torch.zeros(3), "b": torch.ones(5)}
    filtered, skipped = filter_state_dict(state, state)
    assert filtered == state
    assert skipped == []

  def test_shape_mismatch(self):
    ckpt = {"a": torch.zeros(3), "b": torch.ones(5)}
    model = {"a": torch.zeros(3), "b": torch.ones(10)}
    filtered, skipped = filter_state_dict(ckpt, model)
    assert "a" in filtered
    assert "b" not in filtered
    assert any("b" in s[0] for s in skipped)

  def test_missing_in_model(self):
    ckpt = {"a": torch.zeros(3), "extra": torch.ones(5)}
    model = {"a": torch.zeros(3)}
    filtered, skipped = filter_state_dict(ckpt, model)
    assert "extra" not in filtered
    assert any("extra" in s[0] for s in skipped)


class TestSaveAndLoadCheckpoint:

  def test_roundtrip(self, tmp_path):
    state = {"epoch": 5, "loss": 0.42}
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(state, path)
    assert os.path.exists(path)
    loaded = torch.load(path, weights_only=False)
    assert loaded["epoch"] == 5
    assert loaded["loss"] == 0.42

  def test_creates_parent_dirs(self, tmp_path):
    path = str(tmp_path / "subdir" / "ckpt.pt")
    save_checkpoint({"x": 1}, path)
    assert os.path.exists(path)


class TestLoadCheckpointWeights:

  def test_loads_matching_weights(self, tmp_path):
    import torch.nn as nn
    model = nn.Linear(10, 5)
    path = str(tmp_path / "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    new_model = nn.Linear(10, 5)
    load_checkpoint_weights(path, new_model)

  def test_skips_shape_mismatch(self, tmp_path):
    """Keys with wrong shapes are silently skipped by alignment."""
    import torch.nn as nn
    model = nn.Linear(10, 5)
    path = str(tmp_path / "model.pt")
    torch.save({"model_state_dict": model.state_dict()}, path)
    new_model = nn.Linear(10, 8)  # different output dim
    report = load_checkpoint_weights(path, new_model)
    assert report.unused_old or report.unmatched_new

  def _save_ijepa_like(self, tmp_path, name="ijepa.pt"):
    """Mimic an I-JEPA checkpoint: aliased student, EMA teacher, predictor."""
    backbone = nn.Linear(10, 5)
    student = _NestedWrapper(backbone)
    teacher = _NestedWrapper(copy.deepcopy(backbone))
    predictor = nn.Linear(10, 5)
    ckpt = {}
    # "student." prefix: the IJEPA wrapper holds the student as an attribute.
    ckpt.update({f"student.{k}": v for k, v in student.state_dict().items()})
    ckpt.update({f"teacher.{k}": v for k, v in teacher.state_dict().items()})
    ckpt.update({f"predictor.{k}": v for k, v in predictor.state_dict().items()})
    path = str(tmp_path / name)
    torch.save({"model_state_dict": ckpt}, path)
    return path, backbone

  def _loaded_record(self, caplog):
    """Return only the 'actually loaded' log record (not the align report)."""
    return next(r.message for r in caplog.records if "actually loaded" in r.message)

  def test_logs_actually_loaded_params(self, tmp_path, caplog):
    path, backbone = self._save_ijepa_like(tmp_path)
    # Adapter-style target (backbone under model.*), as in fine-tuning.
    new_model = _WrappedAdapter(nn.Linear(10, 5))
    with caplog.at_level(logging.INFO):
      load_checkpoint_weights(path,
                              new_model,
                              param_rename=["student\\.encoder\\.(.*);model.$1"])
    text = self._loaded_record(caplog)
    # Headline: exactly the backbone's keys, not the teacher's/predictor's.
    assert (f"param_align: {len(backbone.state_dict())} keys actually loaded" in text)
    # The renamed student keys matched exactly and are what got loaded.
    assert "model.* <- model.*  (2 keys)" in text
    # If the EMA teacher or the alias had won, their prefixes would show.
    assert "<- teacher." not in text
    assert "<- student.model." not in text
    assert "<- predictor." not in text

  def test_loaded_log_skips_head_keys(self, tmp_path, caplog):
    """A fresh head (no same-shape source key) never shows up as loaded."""
    path, _ = self._save_ijepa_like(tmp_path)

    class _WithHead(nn.Module):
      """Adapter backbone plus a freshly initialised classifier head."""

      def __init__(self):
        super().__init__()
        self.model = nn.Linear(10, 5)
        self.head = nn.Linear(5, 2)

    with caplog.at_level(logging.INFO):
      load_checkpoint_weights(path,
                              _WithHead(),
                              param_rename=["student\\.encoder\\.(.*);model.$1"])
    text = self._loaded_record(caplog)
    # Only the transferred backbone appears; the fresh head does not.
    assert "model.* <- model.*  (2 keys)" in text
    assert "head.weight" not in text
    assert "head.bias" not in text


class TestFetchRemoteCheckpoint:

  def test_no_uri_is_noop(self, tmp_path):
    latest = str(tmp_path / "run_latest.pt")
    best = str(tmp_path / "run_best.pt")
    assert fetch_remote_checkpoint(None, latest, best) == []

  def test_local_present_skips_remote(self, tmp_path, monkeypatch, caplog):
    """A locally present latest is never clobbered; only best is fetched."""
    latest = tmp_path / "run_latest.pt"
    latest.write_bytes(b"local")
    best = tmp_path / "run_best.pt"
    downloads = []
    monkeypatch.setattr("scdiag.checkpointing.storage_download",
                        lambda uri, path: downloads.append(path) or True)
    with caplog.at_level(logging.INFO):
      restored = fetch_remote_checkpoint("s3://b/runs", str(latest), str(best))
    assert restored == [str(best)]
    assert downloads == [str(best)]
    assert "skipping remote fetch" in caplog.text

  def test_downloads_missing_latest_and_best(self, tmp_path, monkeypatch):
    latest = tmp_path / "run_latest.pt"
    best = tmp_path / "run_best.pt"
    monkeypatch.setattr("scdiag.checkpointing.storage_download", lambda uri, path: True)
    restored = fetch_remote_checkpoint("s3://b/runs", str(latest), str(best))
    assert restored == [str(latest), str(best)]
    assert latest.exists() is False  # stub does not write; path bookkeeping only

  def test_latest_tried_before_best(self, tmp_path, monkeypatch, caplog):
    """Precedence mirrors resume_checkpoint: latest first, then best."""
    latest = tmp_path / "run_latest.pt"
    best = tmp_path / "run_best.pt"
    order = []
    monkeypatch.setattr("scdiag.checkpointing.storage_download",
                        lambda uri, path: order.append(path) or True)
    fetch_remote_checkpoint("s3://b/runs", str(latest), str(best))
    assert order == [str(latest), str(best)]

  def test_both_missing_silent_noop(self, tmp_path, monkeypatch, caplog):
    latest = tmp_path / "run_latest.pt"
    best = tmp_path / "run_best.pt"
    monkeypatch.setattr("scdiag.checkpointing.storage_download",
                        lambda uri, path: False)
    with caplog.at_level(logging.WARNING):
      restored = fetch_remote_checkpoint("s3://b/runs", str(latest), str(best))
    assert restored == []
    assert caplog.records == []  # no warnings: a miss is normal
    assert not latest.exists()
    assert not best.exists()
