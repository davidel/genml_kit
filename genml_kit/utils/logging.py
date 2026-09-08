"""Glog-style logging utilities (copied from conv_vit)."""

import datetime
import logging
import os
import sys

from torch.utils.tensorboard import SummaryWriter


def open_writer(log_dir=None, checkpoint=None):
  """Create the TensorBoard ``SummaryWriter`` for a training run.

  Centralises the log-directory defaulting that the two training scripts
  used to implement inline (and slightly differently): when ``log_dir``
  is not given it falls back to ``<checkpoint dir>/logs``.

  Args:
      log_dir: Directory for TensorBoard event files, or ``None``.
      checkpoint: Checkpoint path used to derive the default directory
          (only consulted when ``log_dir`` is ``None``).

  Returns:
      A ``SummaryWriter`` with the directory already created.
  """
  if log_dir is None:
    log_dir = os.path.join(os.path.dirname(checkpoint) or ".", "logs")
  os.makedirs(log_dir, exist_ok=True)
  return SummaryWriter(log_dir=log_dir)


class GlogFormatter(logging.Formatter):
  """Glog-style formatter: ``E0924 22:19:15.123456 PID MODULE:LINE] MESSAGE``"""

  LEVEL_MAP = {
      logging.DEBUG: "D",
      logging.INFO: "I",
      logging.WARNING: "W",
      logging.ERROR: "E",
      logging.CRITICAL: "F",
  }

  def format(self, record):
    now = datetime.datetime.now().astimezone()
    level_char = self.LEVEL_MAP.get(record.levelno, "I")
    month_day = now.strftime("%m%d")
    time_str = now.strftime("%H:%M:%S.%f")
    pid = os.getpid()
    # Glog header: "E0924 22:19:15.123456 12345 train:42]"
    stem = os.path.splitext(record.filename)[0]
    hdr = f"{level_char}{month_day} {time_str} {pid} {stem}:{record.lineno}]"
    msg = record.getMessage()
    lines = msg.split("\n")
    return "\n".join(f"{hdr} {line}" for line in lines)


def fatal(msg, exc=RuntimeError):
  """Log *msg* at CRITICAL level and raise *exc*.

  Use this for unrecoverable errors where the program must stop immediately.
  The exception is always raised after logging so the caller never continues.

  Args:
      msg: Message to log and include in the exception.
      exc: Exception class to raise (default: ``RuntimeError``).
  """
  logging.critical(msg)
  raise exc(msg)


STDERR_TARGET = "STDERR"


def parse_log_targets(spec):
  """Parse a ``--log_targets`` value into a list of destination tokens.

  The value is a comma-separated list.  The special token ``STDERR``
  selects standard error; every other entry is a filesystem path.
  Empty segments are dropped and duplicates removed, order preserved.

  Returns:
      A list of tokens; ``["STDERR"]`` for the default.
  """
  targets = [t.strip() for t in spec.split(",")]
  seen = []
  for t in targets:
    if t and t not in seen:
      seen.append(t)
  return seen or [STDERR_TARGET]


def _glog_handlers_match(root, targets):
  """Return True if *root* has exactly one glog handler per target."""
  glog_handlers = [h for h in root.handlers if isinstance(h.formatter, GlogFormatter)]
  if len(glog_handlers) != len(targets):
    return False
  # ``StreamHandler()`` with no stream argument writes to stderr; match it
  # to the STDERR token rather than sniffing file descriptors.
  return all(
      isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
      if t == STDERR_TARGET else isinstance(h, logging.FileHandler) and
      getattr(h, "baseFilename", None) == os.path.abspath(t)
      for h, t in zip(glog_handlers, targets))


def setup_logging(level="INFO", destinations="STDERR"):
  """Configure the root logger with glog-style formatting.

  Args:
      level: Minimum logging level name (e.g. ``"INFO"``).
      destinations: Comma-separated list of log destinations.  The
          special token ``"STDERR"`` logs to standard error (the
          default); every other entry is a file path appended to.
          Multiple destinations receive the same records simultaneously,
          e.g. ``"STDERR,/tmp/train.log"``.  Parent directories of file
          targets are created as needed.

  Raises:
      SystemExit: via :func:`fatal` when a file target cannot be opened.

  Safe to call more than once: if the attached glog handlers already
  match *destinations*, only the level is updated.  Otherwise every
  existing handler is removed and replaced with ours.
  """
  targets = parse_log_targets(destinations) if isinstance(destinations,
                                                          str) else list(destinations)
  root = logging.getLogger()
  root.setLevel(getattr(logging, level))
  if _glog_handlers_match(root, targets):
    return
  # Remove all existing handlers so we don't double-log.  Close the
  # handlers we own (GlogFormatter-formatted) so file handles are not
  # leaked when the destination set changes.
  for h in root.handlers[:]:
    root.removeHandler(h)
    if isinstance(h.formatter, GlogFormatter):
      h.close()
  formatter = GlogFormatter()
  for target in targets:
    if target == STDERR_TARGET:
      handler = logging.StreamHandler(sys.stderr)
    else:
      try:
        parent = os.path.dirname(os.path.abspath(target))
        os.makedirs(parent, exist_ok=True)
        handler = logging.FileHandler(target, mode="a", encoding="utf-8")
      except OSError as e:
        fatal(f"Cannot open log file {target!r}: {e}", OSError)
    handler.setFormatter(formatter)
    root.addHandler(handler)
