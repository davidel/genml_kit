"""Append-only study ledger: persistence, resume and leaderboard.

The store keeps one JSON line per trial under ``<study_dir>/study.jsonl``,
preceded by a single header line with the effective study spec.  Appending a
record is atomic at the line level, which is enough for a crash-resumable
sweep: on ``--resume`` the ledger is replayed and every terminal trial is
skipped.

Layout::

    <study_dir>/
      study.jsonl            # header + one line per trial
      trials/<id>/           # per-trial log_dir (metrics.jsonl, checkpoints)
"""

import json
import os

from genml_kit.search.errors import StudyNotFound
from genml_kit.search.trial import Trial

LEDGER_NAME = "study.jsonl"
_HEADER_TYPE = "study"


class StudyStore:
  """File-backed ledger of trials for one study directory."""

  def __init__(self, study_dir):
    self.study_dir = study_dir
    self.path = os.path.join(study_dir, LEDGER_NAME)
    self.trials = {}  # trial_id -> Trial
    self.header = None

  # -- lifecycle -----------------------------------------------------------

  @classmethod
  def create(cls, study_dir, spec_dict=None):
    """Create (or open) a study directory, writing the header once."""
    os.makedirs(study_dir, exist_ok=True)
    os.makedirs(os.path.join(study_dir, "trials"), exist_ok=True)
    store = cls(study_dir)
    if os.path.isfile(store.path):
      store.load()
    if store.header is None and spec_dict is not None:
      store.header = spec_dict
      store._append_line({"type": _HEADER_TYPE, "spec": spec_dict})
    return store

  @classmethod
  def open(cls, study_dir):
    """Open an existing study directory for resume/report."""
    if not os.path.isfile(os.path.join(study_dir, LEDGER_NAME)):
      raise StudyNotFound(f"No study ledger under {study_dir!r}.")
    store = cls(study_dir)
    store.load()
    return store

  def load(self):
    """Replay the ledger; later records for a trial supersede earlier ones."""
    self.trials = {}
    self.header = None
    with open(self.path, encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        record = json.loads(line)
        if record.get("type") == _HEADER_TYPE:
          self.header = record.get("spec")
          continue
        self.trials[record["trial_id"]] = Trial.from_record(record)

  # -- access --------------------------------------------------------------

  @property
  def next_trial_id(self):
    return (max(self.trials) + 1) if self.trials else 0

  def completed_ids(self):
    """Ids of trials that reached a terminal state (skipped on resume)."""
    return {tid for tid, trial in self.trials.items() if trial.is_terminal}

  def observed(self):
    """``[(config, best_score)]`` for every trial with a usable score."""
    pairs = []
    for trial in self.trials.values():
      if trial.best is not None:
        pairs.append((trial.config, trial.best))
    return pairs

  # -- mutation ------------------------------------------------------------

  def record(self, trial):
    """Persist (or update) *trial* in the ledger."""
    self.trials[trial.trial_id] = trial
    self._append_line({"type": "trial", **trial.to_record()})

  def _append_line(self, record):
    with open(self.path, "a", encoding="utf-8") as handle:
      handle.write(json.dumps(record) + "\n")

  # -- reporting -----------------------------------------------------------

  def leaderboard(self, top=None, direction="max"):
    """Return scored trials sorted best-first for *direction*."""
    scored = [t for t in self.trials.values() if t.best is not None]
    scored.sort(key=lambda t: t.best, reverse=(direction == "max"))
    return scored[:top] if top else scored
