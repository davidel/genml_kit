"""Tests for genml_kit.training.metrics (sink fan-out + records)."""

import json

import pytest

from genml_kit.training.metrics_sinks import (JsonlSink, MetricsLogger, NullSink,
                                              TensorBoardSink, build_metrics_logger,
                                              ensure_logger, parse_sink_names)


class _RecordingSink:
  """Sink that records every call for assertions."""

  def __init__(self):
    self.scalars = []
    self.artifacts = []
    self.facts = []
    self.closed = False

  def scalar(self,
             tag,
             value,
             *,
             scope,
             kind=None,
             resolution=None,
             step=None,
             epoch=None):
    self.scalars.append({
        "tag": tag,
        "value": value,
        "scope": scope,
        "kind": kind,
        "resolution": resolution,
        "step": step,
        "epoch": epoch,
    })

  def artifact(self, tag, payload_kind, payload, *, step=None):
    self.artifacts.append((tag, payload_kind, payload, step))

  def run_fact(self, key, value):
    self.facts.append((key, value))

  def close(self):
    self.closed = True


def test_parse_sink_names_handles_whitespace_and_empties():
  assert parse_sink_names("tb, jsonl ,") == ["tb", "jsonl"]
  assert parse_sink_names("") == []
  assert parse_sink_names(None) == []


def test_logger_fans_out_to_all_sinks():
  a, b = _RecordingSink(), _RecordingSink()
  logger = MetricsLogger([a, b])
  logger.log_step("Train/loss", 1.5, step=3, resolution="window")
  assert a.scalars == b.scalars
  assert a.scalars[0]["scope"] == "step"
  assert a.scalars[0]["resolution"] == "window"
  assert a.scalars[0]["step"] == 3


def test_log_epoch_carries_final_resolution_and_epoch_index():
  sink = _RecordingSink()
  logger = MetricsLogger([sink])
  logger.log_epoch("val/loss", 0.25, epoch=7)
  rec = sink.scalars[0]
  assert rec["scope"] == "epoch"
  assert rec["resolution"] == "final"
  assert rec["epoch"] == 7
  assert rec["step"] is None


def test_ensure_logger_passes_through_existing_logger():
  logger = MetricsLogger([_RecordingSink()])
  assert ensure_logger(logger) is logger


def test_ensure_logger_replaces_none_with_null_backed_logger():
  logger = ensure_logger(None)
  assert isinstance(logger, MetricsLogger)
  assert isinstance(logger._sinks[0], NullSink)
  # The dummy logger is safe to call without any guard.
  logger.log_step("x", 1.0, step=1)
  logger.log_epoch("y", 2.0, epoch=1)
  logger.close()


def test_default_logger_uses_null_sink_and_never_raises():
  logger = MetricsLogger()
  assert len(logger._sinks) == 1
  assert isinstance(logger._sinks[0], NullSink)
  logger.log_step("x", 1.0, step=1)
  logger.log_epoch("y", 2.0, epoch=1)
  logger.log_artifact("z", "image", object(), step=1)
  logger.log_run_fact("k", "v")
  logger.close()


def test_run_fact_and_artifact_fan_out():
  sink = _RecordingSink()
  logger = MetricsLogger([sink])
  logger.log_run_fact("params", 1234)
  logger.log_artifact("seg/overlay", "image", "PAYLOAD", step=9)
  assert sink.facts == [("params", 1234)]
  assert sink.artifacts == [("seg/overlay", "image", "PAYLOAD", 9)]


def test_summary_writer_passthrough_maps_to_semantic_calls():
  sink = _RecordingSink()
  logger = MetricsLogger([sink])
  logger.add_scalar("Train/loss", 0.5, 4)
  logger.add_image("seg/overlay", "IMG", 2)
  logger.add_video("eval/video", "VID", 3)
  assert sink.scalars[0]["tag"] == "Train/loss"
  assert sink.scalars[0]["scope"] == "step"
  assert sink.scalars[0]["step"] == 4
  assert sink.artifacts == [("seg/overlay", "image", "IMG", 2),
                            ("eval/video", "video", "VID", 3)]


def test_close_propagates_to_all_sinks():
  a, b = _RecordingSink(), _RecordingSink()
  MetricsLogger([a, b]).close()
  assert a.closed and b.closed


def test_jsonl_sink_writes_parseable_records(tmp_path):
  path = tmp_path / "metrics.jsonl"
  sink = JsonlSink(str(path))
  sink.scalar("Train/loss", 1.25, scope="step", kind=None, resolution="window", step=5)
  sink.run_fact("params", 99)
  sink.artifact("seg/overlay", "image", None, step=5)
  sink.close()
  rows = [json.loads(line) for line in path.read_text().splitlines()]
  assert rows[0] == {
      "type": "scalar",
      "tag": "Train/loss",
      "value": 1.25,
      "scope": "step",
      "kind": None,
      "resolution": "window",
      "step": 5,
      "epoch": None,
  }
  assert rows[1] == {"type": "run_fact", "key": "params", "value": 99}
  assert rows[2]["type"] == "artifact"


def test_jsonl_sink_renders_metric_kind_as_name(tmp_path):
  import enum

  Kind = enum.Enum("Kind", ["AVERAGE", "LAST"])
  path = tmp_path / "m.jsonl"
  sink = JsonlSink(str(path))
  sink.scalar("x", 1.0, scope="epoch", kind=Kind.AVERAGE, resolution="final", epoch=1)
  sink.close()
  row = json.loads(path.read_text().splitlines()[0])
  assert row["kind"] == "average"


def test_jsonl_sink_coerces_tensor_values(tmp_path):
  torch = pytest.importorskip("torch")
  path = tmp_path / "m.jsonl"
  sink = JsonlSink(str(path))
  sink.scalar("x", torch.tensor(2.5), scope="step", step=1)
  sink.close()
  row = json.loads(path.read_text().splitlines()[0])
  assert row["value"] == 2.5


def test_jsonl_sink_creates_parent_directory(tmp_path):
  path = tmp_path / "nested" / "deep" / "metrics.jsonl"
  JsonlSink(str(path))
  assert path.parent.is_dir()


def test_tensorboard_sink_uses_step_then_epoch_as_index():
  calls = []

  class _W:

    def add_scalar(self, tag, value, index):
      calls.append((tag, value, index))

    def add_image(self, tag, img, step):
      calls.append((tag, img, step))

    def add_video(self, tag, vid, step):
      calls.append((tag, vid, step))

    def close(self):
      calls.append(("close",))

  sink = TensorBoardSink(_W())
  sink.scalar("Train/loss", 1.0, scope="step", step=8)
  sink.scalar("val/loss", 0.5, scope="epoch", epoch=3)
  sink.artifact("seg/overlay", "image", "IMG", step=8)
  sink.artifact("eval/video", "video", "VID", step=8)
  sink.run_fact("params", 1)  # Ignored by TensorBoard.
  sink.close()
  assert calls == [("Train/loss", 1.0, 8), ("val/loss", 0.5, 3),
                   ("seg/overlay", "IMG", 8), ("eval/video", "VID", 8), ("close",)]


def test_build_metrics_logger_selects_jsonl_only(tmp_path):

  class _Args:
    metrics_sinks = "jsonl"
    log_dir = str(tmp_path)
    checkpoint = None

  logger = build_metrics_logger(_Args())
  assert len(logger._sinks) == 1
  assert isinstance(logger._sinks[0], JsonlSink)
  logger.close()


def test_build_metrics_logger_none_yields_null_sink():

  class _Args:
    metrics_sinks = "none"
    log_dir = None
    checkpoint = "/tmp/ckpt"

  logger = build_metrics_logger(_Args())
  assert isinstance(logger._sinks[0], NullSink)


def test_build_metrics_logger_unknown_name_yields_null_default():

  class _Args:
    metrics_sinks = "bogus"
    log_dir = None
    checkpoint = "/tmp/ckpt"

  logger = build_metrics_logger(_Args())
  # No recognised sinks -> MetricsLogger falls back to a NullSink.
  assert isinstance(logger._sinks[0], NullSink)
