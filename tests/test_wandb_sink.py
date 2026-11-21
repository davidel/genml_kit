"""Tests for the Weights & Biases sink.

No real ``wandb`` install or network access is needed: the sink is driven
against a monkeypatched ``wandb`` module, so the tests validate the
duck-typed protocol and the ``build_metrics_logger`` wiring, not the API.
"""

import argparse
import json
import sys
import types

import pytest

from genml_kit.training.metrics_sinks import WandBSink, build_metrics_logger


class _FakeWandb:
  """Minimal stand-in for the wandb module."""

  def __init__(self):
    self.calls = []
    self.logs = []
    self.finished = False
    self.run = types.SimpleNamespace(summary={})

  def init(self, **kwargs):
    self.calls.append(("init", kwargs))

  def log(self, payload, step=None):
    self.logs.append((payload, step))

  def Video(self, payload):
    return ("video", payload)

  def Image(self, payload):
    return ("image", payload)

  def finish(self):
    self.finished = True


def _fake_wandb_module(monkeypatch):
  fake = _FakeWandb()
  monkeypatch.setitem(sys.modules, "wandb", fake)
  return fake


def _args(**overrides):
  defaults = {
      "model": "tiny_vit",
      "epochs": 3,
      "lr": 1e-3,
      "seed": 0,
      "run_name": None,
      "checkpoint": None,
      "log_dir": None,
      "metrics_sinks": "tb,jsonl",
      "wandb_project": None,
      "wandb_mode": "online",
  }
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


class TestWandBSink:

  def test_init_carries_project_and_config(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    WandBSink(_args(run_name="my-run"), project="proj")
    (kind, kwargs), = fake.calls
    assert kind == "init"
    assert kwargs["project"] == "proj"
    assert kwargs["name"] == "my-run"
    assert kwargs["config"]["lr"] == 1e-3
    assert kwargs["config"]["model"] == "tiny_vit"

  def test_scalar_logs_with_step(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    sink = WandBSink(_args())
    sink.scalar("Train/loss", 0.5, scope="step", step=7)
    assert fake.logs == [({"Train/loss": 0.5}, 7)]

  def test_epoch_scalar_falls_back_to_epoch(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    sink = WandBSink(_args())
    sink.scalar("val/loss", 0.25, scope="epoch", epoch=3)
    assert fake.logs == [({"val/loss": 0.25}, 3)]

  def test_artifact_image_and_video(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    sink = WandBSink(_args())
    sink.artifact("img", "image", "payload", step=1)
    sink.artifact("vid", "video", "payload", step=2)
    assert fake.logs[0] == ({"img": ("image", "payload")}, 1)
    assert fake.logs[1] == ({"vid": ("video", "payload")}, 2)

  def test_run_fact_goes_to_summary(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    sink = WandBSink(_args())
    sink.run_fact("manifest/git_sha", "abc123")
    assert fake.run.summary == {"manifest/git_sha": "abc123"}

  def test_close_finishes_run(self, monkeypatch):
    fake = _fake_wandb_module(monkeypatch)
    sink = WandBSink(_args())
    sink.close()
    assert fake.finished is True


class TestBuildMetricsLogger:

  def test_wandb_sink_is_built(self, monkeypatch):
    _fake_wandb_module(monkeypatch)
    args = _args(metrics_sinks="wandb")
    logger = build_metrics_logger(args)
    wandb_sinks = [s for s in logger._sinks if isinstance(s, WandBSink)]
    assert len(wandb_sinks) == 1

  def test_wandb_mode_passed_through(self, monkeypatch):
    _fake_wandb_module(monkeypatch)
    args = _args(metrics_sinks="wandb", wandb_mode="offline")
    logger = build_metrics_logger(args)
    (sink,) = [s for s in logger._sinks if isinstance(s, WandBSink)]
    assert sink._mode == "offline"

  def test_real_wandb_offline_run(self, tmp_path, monkeypatch):
    """The sink drives the real wandb package in offline mode.

    ``wandb.init(mode="offline")`` writes a local .wandb transaction log
    and needs no network or credentials -- this exercises the genuine API
    contract (init/log/summary/finish) that the fake-module tests only
    approximate.  Requires the 'wandb' extra.
    """
    # wandb captures WANDB_* settings at import time: the env vars must be
    # in place before the package is imported.
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    wandb = pytest.importorskip("wandb")

    sink = WandBSink(_args(run_name="offline-probe"), project="probe", mode="offline")
    sink.scalar("Train/loss", 0.5, scope="step", step=3)
    sink.run_fact("manifest/hostname", "localhost")
    sink.close()
    assert wandb.run is None  # finish() closed the run
    offline_logs = [str(p) for p in tmp_path.rglob("*.wandb")]
    assert offline_logs, "offline mode must write a local .wandb transaction log"

  def test_wandb_plus_jsonl_fan_out(self, monkeypatch, tmp_path):
    fake = _fake_wandb_module(monkeypatch)
    args = _args(metrics_sinks="wandb,jsonl", log_dir=str(tmp_path))
    logger = build_metrics_logger(args)
    logger.log_step("Train/loss", 1.0, step=1, resolution="window")
    logger.close()
    # Both sinks received the same record.
    assert fake.logs == [({"Train/loss": 1.0}, 1)]
    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.is_file()
    assert json.loads(metrics_file.read_text().splitlines()[0])["tag"] == "Train/loss"
