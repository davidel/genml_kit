"""Trial bookkeeping: one resolved config and its observed outcome.

A :class:`Trial` is the unit of work in a sweep.  It carries the resolved
hyperparameter config, the per-trial seed/argv/paths, and the objective
*series* extracted from the trial's ``metrics.jsonl``.  It round-trips through
the study ledger as a plain dict so a run can be resumed verbatim.

The objective is read from the trainer's existing JSONL metric sink: every
epoch-scope scalar is a ``{"type": "scalar", "tag": ..., "scope": "epoch",
"value": ..., "epoch": ...}`` record, and the study's ``objective`` names the
tag to track (e.g. ``val/macro_f1``).
"""

import json
from dataclasses import dataclass, field

#: Terminal trial states.
STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_PRUNED = "pruned"
STATUS_FAILED = "failed"
STATUS_RUNNING = "running"

_TERMINAL = (STATUS_COMPLETED, STATUS_PRUNED, STATUS_FAILED)


@dataclass
class Trial:
  """One evaluated (or to-be-evaluated) configuration."""

  trial_id: int
  config: dict
  seed: int = 42
  status: str = STATUS_PENDING
  series: list = field(default_factory=list)  # [(epoch, value)] chronological
  best: float = None
  exit_code: int = None
  duration: float = None
  trial_dir: str = None
  error: str = None

  @property
  def is_terminal(self):
    return self.status in _TERMINAL

  def observe(self, epoch, value):
    """Append an epoch-scope objective observation."""
    self.series.append((int(epoch), float(value)))

  def best_objective(self, direction="max"):
    """Best objective seen so far, honoring the study *direction*."""
    if not self.series:
      return None
    values = [value for _, value in self.series]
    return max(values) if direction == "max" else min(values)

  def to_record(self):
    """Serialise to a JSON-ready dict for the study ledger."""
    return {
        "trial_id": self.trial_id,
        "config": self.config,
        "seed": self.seed,
        "status": self.status,
        "series": self.series,
        "best": self.best,
        "exit_code": self.exit_code,
        "duration": self.duration,
        "trial_dir": self.trial_dir,
        "error": self.error,
    }

  @classmethod
  def from_record(cls, record):
    """Reconstruct a trial from a ledger record."""
    return cls(
        trial_id=record["trial_id"],
        config=record.get("config", {}),
        seed=record.get("seed", 42),
        status=record.get("status", STATUS_PENDING),
        series=[tuple(pair) for pair in record.get("series", [])],
        best=record.get("best"),
        exit_code=record.get("exit_code"),
        duration=record.get("duration"),
        trial_dir=record.get("trial_dir"),
        error=record.get("error"),
    )


def read_objective_series(metrics_path, objective):
  """Parse a ``metrics.jsonl`` file into ``[(epoch, value)]`` for *objective*.

  Only ``type == "scalar"`` records whose ``tag`` equals *objective* and whose
  ``scope`` is ``epoch`` are considered; each contributes one point keyed by
  its ``epoch`` field.  A missing file yields an empty series (the trial is
  still starting up).
  """
  series = []
  try:
    with open(metrics_path, encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        try:
          record = json.loads(line)
        except json.JSONDecodeError:
          continue
        if record.get("type") != "scalar":
          continue
        if record.get("tag") != objective or record.get("scope") != "epoch":
          continue
        value = record.get("value")
        epoch = record.get("epoch")
        if value is None or epoch is None:
          continue
        series.append((int(epoch), float(value)))
  except FileNotFoundError:
    return []
  series.sort(key=lambda pair: pair[0])
  return series
