"""Tests for study specs and the pruners."""

import json
import textwrap

import pytest

from genml_kit.search.pruners import (AshaPruner, MedianPruner, NoPruner, PRUNERS,
                                      build_pruner)
from genml_kit.search.study import StudySpec


class TestStudySpec:

  def _data(self):
    return {
        "objective": "val/macro_f1",
        "direction": "max",
        "sampler": "tpe",
        "pruner": "asha",
        "max_trials": 8,
        "space": {
            "opt.lr": "log(1e-4:1e-1)",
            "batch_size": "32,64,128",
        },
        "base_args": ["--pipeline", "images"],
    }

  def test_from_dict(self):
    spec = StudySpec(self._data())
    assert spec.objective == "val/macro_f1"
    assert spec.direction == "max"
    assert spec.space.names == ["opt.lr", "batch_size"]
    assert spec.base_args == ["--pipeline", "images"]

  def test_from_space_pairs(self):
    spec = StudySpec.from_space_pairs(["opt.lr=1e-4,1e-3"], objective="val/macro_f1")
    assert spec.space.names == ["opt.lr"]
    assert spec.sampler == "tpe"  # default

  def test_missing_objective_raises(self):
    data = self._data()
    del data["objective"]
    with pytest.raises(ValueError):
      StudySpec(data)

  def test_missing_space_raises(self):
    with pytest.raises(ValueError):
      StudySpec({"objective": "x"})

  def test_bad_direction_raises(self):
    data = self._data()
    data["direction"] = "sideways"
    with pytest.raises(ValueError):
      StudySpec(data)

  def test_from_file_yaml(self, tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(textwrap.dedent("""
        objective: val/macro_f1
        space:
          a: "1,2,3"
        """),
                    encoding="utf-8")
    spec = StudySpec.from_file(str(path))
    assert spec.space.names == ["a"]

  def test_from_file_json(self, tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps(self._data()), encoding="utf-8")
    spec = StudySpec.from_file(str(path))
    assert spec.sampler == "tpe"

  def test_roundtrip_to_namespace(self):
    spec = StudySpec(self._data())
    ns = spec.as_namespace()
    assert ns.objective == "val/macro_f1"
    assert ns.max_trials == 8


class TestPruners:

  def test_registry_names(self):
    names = PRUNERS.list_names()
    for expected in ("none", "median", "asha"):
      assert expected in names

  def test_no_pruner_never_stops(self):
    pruner = NoPruner()
    assert pruner.should_stop(0, [(0, 0.0), (1, 0.0), (2, 0.0)]) is False

  def test_median_pruner_stops_below_median(self):
    pruner = MedianPruner(min_resource=2, reduction_factor=2, n_warmup=1)
    # Two well-performing warmup trials establish a median.
    assert pruner.should_stop(0, [(0, 10.0), (1, 10.0)]) is False
    assert pruner.should_stop(1, [(0, 12.0), (1, 12.0)]) is False
    # A poor trial at the same rung should be pruned.
    assert pruner.should_stop(2, [(0, 1.0), (1, 1.0)]) is True

  def test_asha_pruner_promotes_top_fraction(self):
    pruner = AshaPruner(min_resource=1, reduction_factor=3, min_trials_per_rung=3)
    pruner.should_stop(0, [(0, 10.0)])
    pruner.should_stop(1, [(0, 9.0)])
    pruner.should_stop(2, [(0, 8.0)])
    # Fourth trial is worst -> pruned at the first rung.
    assert pruner.should_stop(3, [(0, 1.0)]) is True

  def test_asha_min_direction(self):
    pruner = AshaPruner(objective_is_min=True,
                        min_resource=1,
                        reduction_factor=3,
                        min_trials_per_rung=3)
    pruner.should_stop(0, [(0, 1.0)])
    pruner.should_stop(1, [(0, 2.0)])
    pruner.should_stop(2, [(0, 3.0)])
    assert pruner.should_stop(3, [(0, 9.0)]) is True

  def test_build_pruner(self):
    assert isinstance(build_pruner("none"), NoPruner)
