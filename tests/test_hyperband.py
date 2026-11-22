"""Tests for the Hyperband pruner."""

from genml_kit.search.pruners import (
    HyperbandPruner,
    PRUNERS,
    build_pruner,
)


def _bracket_smax():
  pruner = HyperbandPruner(min_resource=1, reduction_factor=3)
  pruner._ensure_bracket(9)  # rungs 1,3,9 -> s_max = 2
  return pruner


class TestRegistration:

  def test_registered_as_hyperband(self):
    assert PRUNERS.get("hyperband") is HyperbandPruner

  def test_build_pruner_roundtrip(self):
    assert isinstance(build_pruner("hyperband"), HyperbandPruner)


class TestBrackets:

  def test_bracket_count_is_smax_plus_one(self):
    pruner = _bracket_smax()
    assert pruner._n_brackets() == 3

  def test_smax_derived_from_epoch_budget(self):
    pruner = _bracket_smax()
    assert pruner._s_max == 2

  def test_explicit_smax_respected(self):
    pruner = HyperbandPruner(min_resource=1, reduction_factor=3, s_max=4)
    pruner._ensure_bracket(9)
    assert pruner._n_brackets() == 5

  def test_bracket_first_rungs_are_geometric(self):
    pruner = _bracket_smax()
    # Bracket 0 starts at min_resource (deepest, most halvings);
    # bracket s starts at min_resource * rf**s.
    assert pruner._asha[0].min_resource == 1
    assert pruner._asha[1].min_resource == 3
    assert pruner._asha[2].min_resource == 9

  def test_bracket_assignment_is_deterministic(self):
    pruner = _bracket_smax()
    assert pruner._bracket_of(4) == pruner._bracket_of(4)
    assert pruner._bracket_of(7) == 7 % 3


class TestDecisions:

  def test_deep_bracket_prunes_early(self):
    # Bracket 0 (trials 0, 3, 6) records at rung 1 once past epoch 1.
    pruner = _bracket_smax()
    pruner.should_stop(0, [(0, 10.0), (1, 10.0)])
    pruner.should_stop(3, [(0, 9.0), (1, 9.0)])
    pruner.should_stop(6, [(0, 8.0), (1, 8.0)])
    # A below-cutoff fourth trial (bracket 0) is pruned at rung 1.
    assert pruner.should_stop(9, [(0, 1.0), (1, 1.0)]) is True

  def test_shallow_bracket_needs_its_first_rung(self):
    # Bracket 2 (trial 2) starts at rung 9: nothing before epoch 10.
    pruner = _bracket_smax()
    pruner.should_stop(2, [(9, 5.0)])
    assert pruner.should_stop(2, [(9, 1.0)]) is False

  def test_below_first_rung_never_prunes(self):
    pruner = _bracket_smax()
    assert pruner.should_stop(0, [(0, 99.0)]) is False  # single trial

  def test_max_direction(self):
    pruner = HyperbandPruner(objective_is_min=False, min_resource=1, reduction_factor=2)
    pruner._ensure_bracket(4)
    pruner.should_stop(0, [(0, 0.1), (1, 0.1)])
    pruner.should_stop(2, [(0, 0.2), (1, 0.2)])
    pruner.should_stop(4, [(0, 0.3), (1, 0.3)])
    # A *low* value is bad when maximizing.
    assert pruner.should_stop(6, [(0, 0.01), (1, 0.01)]) is True

  def test_no_interference_between_brackets(self):
    # Bracket 0's rung-1 records must not leak into bracket 1's grid
    # (whose first rung is 3), and vice versa.
    pruner = _bracket_smax()
    pruner.should_stop(1, [(2, 5.0), (3, 5.0)])  # bracket 1 only
    pruner.should_stop(0, [(0, 10.0), (1, 10.0)])
    pruner.should_stop(3, [(0, 9.0), (1, 9.0)])
    pruner.should_stop(6, [(0, 8.0), (1, 8.0)])
    # Bracket 1 saw exactly one trial (trial 1) at its rung 3, and none at
    # its own rung-1 key (bracket 0's rung 1 is not part of bracket 1's grid).
    assert pruner._asha[1]._rung_values.get(1) in (None, [])
    assert pruner._asha[0]._rung_values.get(3) in (None, [])
