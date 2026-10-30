"""Tests for the trial records and the study ledger (store)."""

import json
import os

import pytest

from genml_kit.search.errors import StudyNotFound
from genml_kit.search.store import StudyStore
from genml_kit.search.trial import (STATUS_COMPLETED, Trial, read_objective_series)


class TestTrial:

  def test_best_objective_max(self):
    trial = Trial(trial_id=0, config={"x": 1}, series=[(0, 0.1), (1, 0.3)])
    assert trial.best_objective("max") == 0.3

  def test_best_objective_min(self):
    trial = Trial(trial_id=0, config={"x": 1}, series=[(0, 0.1), (1, 0.3)])
    assert trial.best_objective("min") == 0.1

  def test_empty_series_has_no_best(self):
    assert Trial(trial_id=0, config={}).best_objective() is None

  def test_record_roundtrip(self):
    trial = Trial(trial_id=3, config={"a": 1}, series=[(0, 0.5)], best=0.5)
    restored = Trial.from_record(trial.to_record())
    assert restored.trial_id == 3
    assert restored.config == {"a": 1}
    assert restored.series == [(0, 0.5)]


class TestReadObjectiveSeries:

  def test_filters_by_tag_and_scope(self, tmp_path):
    path = tmp_path / "metrics.jsonl"
    records = [
        {
            "type": "scalar",
            "tag": "val/macro_f1",
            "scope": "epoch",
            "value": 0.5,
            "epoch": 0
        },
        {
            "type": "scalar",
            "tag": "val/loss",
            "scope": "epoch",
            "value": 9.0,
            "epoch": 0
        },
        {
            "type": "scalar",
            "tag": "val/macro_f1",
            "scope": "step",
            "value": 0.9,
            "step": 5
        },
        {
            "type": "config",
            "tag": "val/macro_f1",
            "value": 1.0
        },
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    series = read_objective_series(str(path), "val/macro_f1")
    assert series == [(0, 0.5)]

  def test_missing_file_is_empty(self, tmp_path):
    assert read_objective_series(str(tmp_path / "nope.jsonl"), "x") == []

  def test_malformed_lines_are_skipped(self, tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("not json\n" + json.dumps({
        "type": "scalar",
        "tag": "x",
        "scope": "epoch",
        "value": 1.0,
        "epoch": 0
    }),
                    encoding="utf-8")
    assert read_objective_series(str(path), "x") == [(0, 1.0)]


class TestStudyStore:

  def test_create_writes_header(self, tmp_path):
    store_dir = str(tmp_path / "study")
    store = StudyStore.create(store_dir, {"objective": "x"})
    assert store.header == {"objective": "x"}
    assert os.path.isfile(os.path.join(store_dir, "study.jsonl"))

  def test_record_and_reload(self, tmp_path):
    store_dir = str(tmp_path / "study")
    store = StudyStore.create(store_dir, {"objective": "x"})
    trial = Trial(trial_id=0,
                  config={"a": 1},
                  status=STATUS_COMPLETED,
                  series=[(0, 0.4)],
                  best=0.4)
    store.record(trial)
    reloaded = StudyStore.open(store_dir)
    assert reloaded.header == {"objective": "x"}
    assert reloaded.trials[0].best == 0.4
    assert reloaded.completed_ids() == {0}

  def test_next_trial_id(self, tmp_path):
    store = StudyStore.create(str(tmp_path / "s"), {"objective": "x"})
    assert store.next_trial_id == 0
    store.record(Trial(trial_id=0, config={}))
    assert store.next_trial_id == 1

  def test_leaderboard_orders_by_direction(self, tmp_path):
    store = StudyStore.create(str(tmp_path / "s"), {"objective": "x"})
    store.record(Trial(trial_id=0, config={"a": 1}, best=0.2))
    store.record(Trial(trial_id=1, config={"a": 2}, best=0.8))
    board = store.leaderboard(direction="max")
    assert [t.trial_id for t in board] == [1, 0]
    board_min = store.leaderboard(direction="min")
    assert [t.trial_id for t in board_min] == [0, 1]

  def test_open_missing_raises(self, tmp_path):
    with pytest.raises(StudyNotFound):
      StudyStore.open(str(tmp_path / "nothing"))
