"""Verify tests.logging_helpers.log_capture is level-regime independent."""

import logging

from tests.logging_helpers import log_capture


def test_ignores_prior_records():
  logging.warning("noise before the block")
  logging.info("also noise before")

  with log_capture() as logs:
    logging.info("Gradient Report: hello")

  assert len(logs) == 1
  assert logs.matching("Gradient Report") == ["Gradient Report: hello"]
  assert "noise" not in logs.text


def test_root_level_restored():
  root = logging.getLogger()
  before = root.level
  with log_capture() as logs:
    logging.info("inside")
    assert root.level == logging.INFO
  assert root.level == before
  assert len(logs) == 1


def test_captures_below_info_when_asked():
  with log_capture(level=logging.DEBUG) as logs:
    logging.debug("dbg")
    logging.info("inf")
  assert len(logs) == 2


def test_accepts_string_level():
  with log_capture(level="DEBUG") as logs:
    logging.debug("dbg")
  assert len(logs) == 1


def test_rejects_bad_level():
  try:
    with log_capture(level="NOPE") as _:
      pass
  except ValueError:
    return
  raise AssertionError("expected ValueError")
