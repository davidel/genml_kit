"""Tests for SpaceSpec / space_spec (RL model space carrier)."""

import pytest

from genml_kit.models.rl.spaces import SpaceSpec, space_spec


class _Space:

  def __init__(self, **attrs):
    self.__dict__.update(attrs)


class TestSpaceSpec:

  def test_discrete(self):
    obs = _Space(shape=(4,))
    act = _Space(n=2)
    spec = space_spec(obs, act)
    assert isinstance(spec, SpaceSpec)
    assert spec.obs_shape == (4,)
    assert spec.obs_dim == 4
    assert spec.n_actions == 2
    assert spec.action_dim is None
    assert spec.is_discrete is True
    assert spec.action_low is None
    assert spec.action_high is None

  def test_continuous(self):
    obs = _Space(shape=(4,))
    act = _Space(shape=(2,), low=-1.0, high=1.0)
    spec = space_spec(obs, act)
    assert spec.n_actions is None
    assert spec.action_dim == 2
    assert spec.is_discrete is False
    assert spec.action_low == -1.0
    assert spec.action_high == 1.0

  def test_image_obs_shape(self):
    obs = _Space(shape=(3, 84, 84))
    act = _Space(n=10)
    spec = space_spec(obs, act)
    assert spec.obs_shape == (3, 84, 84)
    assert spec.obs_dim == 3 * 84 * 84
    assert spec.n_actions == 10

  def test_obs_dim_override(self):
    obs = _Space(shape=(4,))
    act = _Space(n=2)
    spec = space_spec(obs, act, obs_dim=17)
    assert spec.obs_shape == (4,)
    assert spec.obs_dim == 17

  def test_missing_obs_shape_raises_without_override(self):
    obs = _Space()
    act = _Space(n=2)
    with pytest.raises(ValueError, match="--obs_dim"):
      space_spec(obs, act)

  def test_missing_obs_shape_with_override_ok(self):
    obs = _Space()
    act = _Space(n=2)
    spec = space_spec(obs, act, obs_dim=4)
    assert spec.obs_dim == 4

  def test_fallback_no_n_no_shape(self):
    obs = _Space(shape=(4,))
    act = _Space()
    spec = space_spec(obs, act)
    assert spec.n_actions == 2
    assert spec.action_dim is None
    assert spec.is_discrete is True

  def test_action_low_high_none_for_discrete(self):
    obs = _Space(shape=(4,))
    act = _Space(n=2, low=-1.0, high=1.0)
    spec = space_spec(obs, act)
    assert spec.action_low is None
    assert spec.action_high is None
