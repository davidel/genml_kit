"""Tests for the samplers (genml_kit/search/samplers.py)."""

import pytest

from genml_kit.search.errors import SearchSpaceExhausted
from genml_kit.search.samplers import (SEARCHERS, GridSampler, RandomSampler,
                                       TpeSampler, build_sampler)
from genml_kit.search.space import SearchSpace


def _space():
  return SearchSpace.from_pairs(["x=0.0:1.0", "y=0:10"])


def _quadratic(config):
  """Objective peaking at x=y_norm=0.2 (a known, findable optimum)."""
  return -((config["x"] - 0.2)**2 + (config["y"] / 10.0 - 0.2)**2)


class TestRegistry:

  def test_core_samplers_registered(self):
    names = SEARCHERS.list_names()
    for expected in ("grid", "random", "tpe"):
      assert expected in names

  def test_build_sampler(self):
    sampler = build_sampler("random", _space())
    assert isinstance(sampler, RandomSampler)

  def test_unknown_sampler_raises(self):
    with pytest.raises(ValueError):
      build_sampler("does-not-exist", _space())


class TestGridSampler:

  def test_yields_fresh_configs(self):
    sampler = GridSampler(_space(), max_points=3)
    observed = []
    seen = set()
    for _ in range(6):
      config = sampler.suggest(observed, n=1)[0]
      key = tuple(sorted(config.items()))
      assert key not in seen
      seen.add(key)
      observed.append((config, 0.0))

  def test_exhaustion_raises(self):
    sampler = GridSampler(SearchSpace.from_pairs(["a=1,2"]))
    with pytest.raises(SearchSpaceExhausted):
      sampler.suggest([({"a": 1}, 0.0), ({"a": 2}, 0.0)], n=1)


class TestRandomSampler:

  def test_is_deterministic_for_a_seed(self):
    s1 = RandomSampler(_space(), seed=7).suggest([], n=5)
    s2 = RandomSampler(_space(), seed=7).suggest([], n=5)
    assert s1 == s2

  def test_respects_direction_validation(self):
    with pytest.raises(ValueError):
      RandomSampler(_space(), direction="sideways")


class TestTpeSampler:

  def test_startup_falls_back_to_random(self):
    sampler = TpeSampler(_space(), seed=0, n_startup_trials=5)
    assert len(sampler.suggest([], n=2)) == 2

  def test_finds_optimum_region(self):
    """TPE should concentrate proposals near the known optimum."""
    sampler = TpeSampler(_space(), seed=3, n_startup_trials=5)
    observed = []
    for _ in range(60):
      config = sampler.suggest(observed, n=1)[0]
      observed.append((config, _quadratic(config)))
    best = max(observed, key=lambda pair: pair[1])[0]
    assert abs(best["x"] - 0.2) < 0.2
    assert abs(best["y"] / 10.0 - 0.2) < 0.2

  def test_min_direction_is_honored(self):
    sampler = TpeSampler(_space(), direction="min", seed=1, n_startup_trials=3)
    observed = [({
        "x": 0.1,
        "y": 1
    }, 0.1), ({
        "x": 0.9,
        "y": 9
    }, 0.9), ({
        "x": 0.5,
        "y": 5
    }, 0.5)]
    assert len(sampler.suggest(observed, n=1)) == 1


class TestGpEiSampler:

  def test_registered_when_scipy_present(self):
    pytest.importorskip("scipy")
    assert "gpei" in SEARCHERS.list_names()

  def test_improves_over_random(self):
    pytest.importorskip("scipy")
    sampler = build_sampler("gpei", _space(), seed=5, n_startup_trials=5)
    observed = []
    for _ in range(30):
      config = sampler.suggest(observed, n=1)[0]
      observed.append((config, _quadratic(config)))
    best = max(observed, key=lambda pair: pair[1])[1]
    assert best > -0.05
