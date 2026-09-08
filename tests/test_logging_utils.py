"""Tests for genml_kit.logging_utils."""

import logging
import re

import pytest

from genml_kit.utils.logging import (
    GlogFormatter,
    parse_log_targets,
    setup_logging,
)

# GlogFormatter


class TestGlogFormatter:

  def test_basic_format_contains_level_char(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="train.py",
        lineno=42,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    out = fmt.format(record)
    # Should start with the level char 'I' followed by month/day digits.
    assert re.match(r"I\d{4} ", out)
    assert "hello world" in out
    assert "train:42]" in out

  def test_multiline_message(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.WARNING,
        pathname="foo.py",
        lineno=1,
        msg="line1\nline2\nline3",
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    lines = out.split("\n")
    assert len(lines) == 3
    # Every line should carry the glog header.
    for line in lines:
      assert re.match(r"W\d{4} ", line)

  def test_level_characters(self):
    fmt = GlogFormatter()
    expected = {
        logging.DEBUG: "D",
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.CRITICAL: "F",
    }
    for lvl, char in expected.items():
      record = logging.LogRecord(
          name="t",
          level=lvl,
          pathname="x.py",
          lineno=1,
          msg="m",
          args=(),
          exc_info=None,
      )
      assert fmt.format(record).startswith(char)

  def test_no_args_message(self):
    fmt = GlogFormatter()
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="a.py",
        lineno=10,
        msg="plain",
        args=(),
        exc_info=None,
    )
    out = fmt.format(record)
    assert "plain" in out


# setup_logging


class TestSetupLogging:

  def _clear_root(self):
    root = logging.getLogger()
    root.handlers[:] = []
    return root

  def test_adds_glog_handler(self):
    root = self._clear_root()
    setup_logging()
    assert any(isinstance(h.formatter, GlogFormatter) for h in root.handlers)

  def test_idempotent(self):
    root = self._clear_root()
    setup_logging()
    setup_logging()
    glog_handlers = [h for h in root.handlers if isinstance(h.formatter, GlogFormatter)]
    assert len(glog_handlers) == 1

  def test_replaces_existing_handlers(self):
    root = self._clear_root()
    root.addHandler(logging.StreamHandler())  # plain handler
    setup_logging()
    # The plain handler should be gone.
    assert all(isinstance(h.formatter, GlogFormatter) for h in root.handlers)

  def test_respects_level_arg(self):
    root = self._clear_root()
    setup_logging(level="DEBUG")
    assert root.level == logging.DEBUG

  def test_second_call_updates_level(self):
    root = self._clear_root()
    setup_logging(level="DEBUG")
    setup_logging(level="WARNING")
    assert root.level == logging.WARNING

  def test_actual_log_output(self, capsys):
    self._clear_root()
    setup_logging()
    logging.info("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.err


class TestParseLogTargets:

  def test_default_is_stderr(self):
    assert parse_log_targets("STDERR") == ["STDERR"]

  def test_stderr_and_file(self):
    assert parse_log_targets("STDERR,/tmp/x.log") == ["STDERR", "/tmp/x.log"]

  def test_strips_whitespace(self):
    assert parse_log_targets(" STDERR , /tmp/x.log ") == ["STDERR", "/tmp/x.log"]

  def test_drops_empty_segments(self):
    assert parse_log_targets("STDERR,,/tmp/x.log,") == ["STDERR", "/tmp/x.log"]

  def test_deduplicates_preserving_order(self):
    assert parse_log_targets("STDERR,a.log,STDERR,a.log") == ["STDERR", "a.log"]

  def test_empty_string_defaults_to_stderr(self):
    assert parse_log_targets("") == ["STDERR"]


class TestLogTargets:
  """setup_logging(destinations=...) fan-out and file handling."""

  def _clear_root(self):
    root = logging.getLogger()
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    return root

  def _glog_handlers(self, root):
    return [h for h in root.handlers if isinstance(h.formatter, GlogFormatter)]

  def test_default_single_stderr_handler(self, capsys):
    root = self._clear_root()
    setup_logging()
    handlers = self._glog_handlers(root)
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert not isinstance(handlers[0], logging.FileHandler)
    logging.info("stderr only")
    captured = capsys.readouterr()
    assert "stderr only" in captured.err

  def test_stderr_and_file_simultaneously(self, tmp_path, capsys):
    root = self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", f"STDERR,{log_file}")
    assert len(self._glog_handlers(root)) == 2
    logging.info("both destinations")
    captured = capsys.readouterr()
    assert "both destinations" in captured.err
    assert "both destinations" in log_file.read_text()

  def test_file_only_no_stderr_output(self, tmp_path, capsys):
    self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", str(log_file))
    logging.info("file only")
    captured = capsys.readouterr()
    assert "file only" not in captured.err
    assert "file only" in log_file.read_text()

  def test_file_target_creates_parent_dirs(self, tmp_path):
    self._clear_root()
    log_file = tmp_path / "deep" / "nested" / "run.log"
    setup_logging("INFO", str(log_file))
    logging.info("nested")
    assert "nested" in log_file.read_text()

  def test_file_appends_across_reconfigure(self, tmp_path):
    self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", str(log_file))
    logging.info("first")
    self._clear_root()
    setup_logging("INFO", str(log_file))
    logging.info("second")
    text = log_file.read_text()
    assert "first" in text and "second" in text

  def test_reconfigure_when_destination_set_changes(self, tmp_path, capsys):
    root = self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", "STDERR")
    setup_logging("INFO", f"STDERR,{log_file}")
    assert len(self._glog_handlers(root)) == 2
    logging.info("after change")
    captured = capsys.readouterr()
    assert "after change" in captured.err
    assert "after change" in log_file.read_text()

  def test_whitespace_and_duplicates_in_spec(self, tmp_path, capsys):
    root = self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", f" STDERR , {log_file} , STDERR ")
    handlers = self._glog_handlers(root)
    assert len(handlers) == 2
    logging.info("deduped")
    assert "deduped" in log_file.read_text()
    assert "deduped" in capsys.readouterr().err

  def test_unopenable_file_target_is_fatal(self, tmp_path):
    root = self._clear_root()
    # A file as parent directory makes FileHandler raise OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    with pytest.raises(OSError):
      setup_logging("INFO", str(blocker / "run.log"))
    assert not self._glog_handlers(root)

  def test_file_output_is_glog_formatted(self, tmp_path):
    self._clear_root()
    log_file = tmp_path / "run.log"
    setup_logging("INFO", str(log_file))
    logging.info("glog me")
    line = [ln for ln in log_file.read_text().splitlines() if "glog me" in ln][0]
    assert re.match(r"^I\d{4} \d{2}:\d{2}:\d{2}\.\d{6} \d+ \S+:\d+\] glog me$", line)
