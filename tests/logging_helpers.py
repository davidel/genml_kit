"""Scoped log capture for tests.

Why this exists
---------------
``caplog.at_level(INFO)`` is a *floor* on the root logger level, not a filter:
``_pytest.logging`` computes ``max(requested, logger.level)`` and installs that
on the root logger.  A record only reaches the capture handler when
``record.levelno >= root_logger.level``.  So whatever was emitted *before* the
block is captured too as soon as the ambient root level is already at or below
the requested level.

That ambient level is not stable.  CI runs pytest with
``PYTEST_ADDOPTS=--log-cli-level=INFO`` (to stream logs live), which leaves the
root logger at ``INFO`` for the whole session.  Locally the default is
``WARNING``, so the same test observes completely different sets of records: a
warm-up loop that logs at ``INFO`` is invisible locally and fully visible on
CI.  Assertions such as ``len(records) == 1`` then fail on CI only.

``log_capture`` removes the ambient level from the equation: it pins the root
logger to a fixed level for the duration of the block and collects into a
private handler.  Only records emitted *inside* the block are visible, and the
level is restored on exit.

Usage
-----
::

    with log_capture() as logs:
      monitor.step(10)

    assert len(logs.matching("Norm Trends")) == 1
    assert "top 2/4" in logs.matching("Norm Trends")[0]

Pass ``level=logging.DEBUG`` to widen capture below ``INFO``.
"""

import contextlib
import logging

DEFAULT_LEVEL = logging.INFO


class CapturedLogs:
  """Records collected by :func:`log_capture`, plus inspection helpers."""

  def __init__(self, records):
    self.records = records

  @property
  def text(self):
    """All captured messages joined by newlines."""
    return "\n".join(r.getMessage() for r in self.records)

  def matching(self, needle):
    """Messages containing *needle*."""
    return [r.getMessage() for r in self.records if needle in r.getMessage()]

  def __len__(self):
    return len(self.records)

  def __iter__(self):
    return iter(self.records)


class _CaptureHandler(logging.Handler):
  """Collects records into a list instead of formatting them."""

  def __init__(self, level):
    super().__init__(level)
    self.records = []

  def emit(self, record):
    self.records.append(record)


def _coerce_level(level):
  if level is None:
    return DEFAULT_LEVEL
  if isinstance(level, int):
    return level
  resolved = logging.getLevelName(str(level).upper())
  if not isinstance(resolved, int):
    raise ValueError(f"unknown log level: {level!r}")
  return resolved


@contextlib.contextmanager
def log_capture(level=None):
  """Capture log records emitted inside the ``with`` block only.

  The root logger level is pinned to *level* for the duration of the block and
  restored afterwards, so the captured set does not depend on how pytest was
  invoked (``--log-cli-level`` and friends).

  Parameters
  ----------
  level : int or str, optional
      Level to pin while capturing.  Defaults to ``logging.INFO``.

  Yields
  ------
  CapturedLogs
      Inspect via ``.records``, ``.text``, ``.matching(needle)`` and ``len()``.
  """
  resolved = _coerce_level(level)
  root = logging.getLogger()
  handler = _CaptureHandler(resolved)
  root.addHandler(handler)
  previous_level = root.level
  root.setLevel(resolved)
  try:
    yield CapturedLogs(handler.records)
  finally:
    root.setLevel(previous_level)
    root.removeHandler(handler)
