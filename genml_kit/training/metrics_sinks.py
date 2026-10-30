"""Pluggable metric sinks: fan-out logging of training metrics.

Training code talks to a single :class:`MetricsLogger`, which forwards every
metric record to each registered :class:`MetricsSink` (TensorBoard, JSONL,
...).  This decouples *where metrics go* from *how they are computed*, so the
same run can feed a human-facing dashboard and a programmatically-queryable
structured file at once.

The logger deliberately duck-types the tiny subset of the TensorBoard
``SummaryWriter`` interface the harness uses (``add_scalar`` / ``add_image`` /
``add_video`` / ``close``); existing call sites therefore keep working while
they are migrated to the richer semantic API (``log_step`` / ``log_epoch`` /
``log_artifact``).

Beyond the value, a record carries three orthogonal tags:

- ``scope``      -- ``"step"`` or ``"epoch"``: which cadence it belongs to.
- ``kind``       -- the fold semantics (``AVERAGE`` / ``LAST`` / ``DERIVED``),
                    or ``None`` when the caller does not know.
- ``resolution`` -- ``"window"`` / ``"running"`` / ``"final"``: labels the
                    windowed-vs-cumulative mix on the per-step line without
                    changing it.
"""

import json
import os
from typing import Protocol

__all__ = [
    "MetricsSink",
    "MetricsLogger",
    "NullSink",
    "TensorBoardSink",
    "JsonlSink",
    "build_metrics_logger",
    "ensure_logger",
    "parse_sink_names",
]


def _kind_str(kind):
  """Render a ``MetricKind`` (or any value) as a lower-case string or None."""
  if kind is None:
    return None
  return str(getattr(kind, "name", kind)).lower()


def _as_float(value):
  """Coerce a scalar (float, 0-dim tensor, ...) to a Python float."""
  if hasattr(value, "detach"):
    value = value.detach()
  if hasattr(value, "item"):
    return float(value.item())
  return float(value)


class MetricsSink(Protocol):
  """A metric destination.

  Implementations receive already-tagged records; they never inspect training
  state.  ``scalar`` is the common case; ``artifact`` carries images/videos;
  ``run_fact`` carries one-shot run-level facts.
  """

  def scalar(self,
             tag,
             value,
             *,
             scope,
             kind=None,
             resolution=None,
             step=None,
             epoch=None):
    """Record a scalar time-series point."""

  def artifact(self, tag, payload_kind, payload, *, step=None):
    """Record a non-scalar payload (``payload_kind`` in {"image", "video"})."""

  def run_fact(self, key, value):
    """Record a one-shot run-level fact (config value, param count, ...)."""

  def close(self):
    """Flush and release any resources held by the sink."""


class NullSink:
  """No-op sink; makes the logger always present without special-casing."""

  def scalar(self,
             tag,
             value,
             *,
             scope,
             kind=None,
             resolution=None,
             step=None,
             epoch=None):
    return None

  def artifact(self, tag, payload_kind, payload, *, step=None):
    return None

  def run_fact(self, key, value):
    return None

  def close(self):
    return None


class TensorBoardSink:
  """Wraps a TensorBoard ``SummaryWriter``.

  TensorBoard has no notion of ``scope`` / ``kind`` / ``resolution``: a scalar
  is just ``(tag, value, index)``.  The index is the step for step-scope
  records and the epoch for epoch-scope records.  Callers must keep the two
  series under distinct tags (e.g. per-step ``Train/loss`` vs epoch-summary
  ``epoch/loss``) so the two cadences do not collide in one series.
  """

  def __init__(self, writer):
    self._writer = writer

  def scalar(self,
             tag,
             value,
             *,
             scope,
             kind=None,
             resolution=None,
             step=None,
             epoch=None):
    index = step if step is not None else epoch
    self._writer.add_scalar(tag, value, index)

  def artifact(self, tag, payload_kind, payload, *, step=None):
    if payload_kind == "video":
      self._writer.add_video(tag, payload, step)
    else:
      self._writer.add_image(tag, payload, step)

  def run_fact(self, key, value):
    # TensorBoard has no run-level scalar bucket; ignore (JSONL sink owns it).
    return None

  def close(self):
    self._writer.close()


class JsonlSink:
  """Appends one JSON object per record to a line-delimited file.

  This is the programmatic-access payoff: the file can be parsed, filtered
  and aggregated without a TensorBoard dependency.
  """

  def __init__(self, path):
    self.path = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

  def scalar(self,
             tag,
             value,
             *,
             scope,
             kind=None,
             resolution=None,
             step=None,
             epoch=None):
    self._emit({
        "type": "scalar",
        "tag": tag,
        "value": _as_float(value),
        "scope": scope,
        "kind": _kind_str(kind),
        "resolution": resolution,
        "step": step,
        "epoch": epoch,
    })

  def artifact(self, tag, payload_kind, payload, *, step=None):
    self._emit({
        "type": "artifact",
        "tag": tag,
        "payload_kind": payload_kind,
        "step": step,
    })

  def run_fact(self, key, value):
    self._emit({"type": "run_fact", "key": key, "value": value})

  def _emit(self, record):
    # Open/append/close per record: the harness already supports a flush per
    # log line, and this keeps ``records`` durable against a crash mid-run.
    with open(self.path, "a", encoding="utf-8") as f:
      f.write(json.dumps(record) + "\n")

  def close(self):
    return None


def ensure_logger(logger):
  """Return *logger*, or a no-op :class:`MetricsLogger` when it is ``None``.

  The harness guarantees a logger is always present (a dummy one when metrics
  are disabled), so every boundary that accepts an optional logger normalises
  it here.  Call sites therefore never need a ``logger is not None`` guard.
  """
  return logger if logger is not None else MetricsLogger()


class MetricsLogger:
  """Fan-out over a list of :class:`MetricsSink` objects.

  Always holds at least one sink (a :class:`NullSink` by default).  Use
  :func:`ensure_logger` to normalise an optional logger at a boundary.
  """

  def __init__(self, sinks=None):
    self._sinks = list(sinks) if sinks else [NullSink()]

  def log_step(self, tag, value, *, step, resolution=None, kind=None):
    """Record a per-step scalar (``resolution`` in {"window", "running"})."""
    for sink in self._sinks:
      sink.scalar(tag, value, scope="step", kind=kind, resolution=resolution, step=step)

  def log_epoch(self, tag, value, *, epoch, kind=None):
    """Record an epoch-scope scalar (already folded by the reporter)."""
    for sink in self._sinks:
      sink.scalar(tag, value, scope="epoch", kind=kind, resolution="final", epoch=epoch)

  def log_artifact(self, tag, payload_kind, payload, *, step=None):
    """Record a non-scalar payload (image/video)."""
    for sink in self._sinks:
      sink.artifact(tag, payload_kind, payload, step=step)

  def log_run_fact(self, key, value):
    """Record a one-shot run-level fact."""
    for sink in self._sinks:
      sink.run_fact(key, value)

  # -- SummaryWriter-compatible passthrough (keeps legacy call sites working) --
  def add_scalar(self, tag, value, step):
    self.log_step(tag, value, step=step)

  def add_image(self, tag, img, step):
    self.log_artifact(tag, "image", img, step=step)

  def add_video(self, tag, vid, step):
    self.log_artifact(tag, "video", vid, step=step)

  def close(self):
    for sink in self._sinks:
      sink.close()


def parse_sink_names(spec):
  """Parse a comma-separated sink spec into a list of lower-case names."""
  if not spec:
    return []
  return [name.strip().lower() for name in spec.split(",") if name.strip()]


def _default_jsonl_path(args):
  """Return the JSONL path, mirroring ``open_writer``'s log-dir defaulting."""
  log_dir = getattr(args, "log_dir", None)
  if log_dir is None:
    checkpoint = getattr(args, "checkpoint", None)
    log_dir = os.path.join(os.path.dirname(checkpoint) or ".", "logs")
  return os.path.join(log_dir, "metrics.jsonl")


def build_metrics_logger(args):
  """Build a :class:`MetricsLogger` from ``args.metrics_sinks``.

  Recognised sink names are ``tb`` (TensorBoard), ``jsonl`` (structured file),
  and ``none``/``null`` (explicitly nothing).  Unknown names are ignored.
  """
  from genml_kit.utils.logging import open_writer

  sinks = []
  for name in parse_sink_names(getattr(args, "metrics_sinks", "tb")):
    if name == "tb":
      sinks.append(
          TensorBoardSink(
              open_writer(log_dir=getattr(args, "log_dir", None),
                          checkpoint=getattr(args, "checkpoint", None))))
    elif name == "jsonl":
      sinks.append(JsonlSink(_default_jsonl_path(args)))
    elif name in ("none", "null"):
      sinks.append(NullSink())
  return MetricsLogger(sinks)
