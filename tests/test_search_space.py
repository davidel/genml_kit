"""Tests for the search-space model (genml_kit/search/space.py)."""

import numpy as np
import pytest

from genml_kit.search.space import Dimension, SearchSpace, format_value


class TestDimensionParsing:

  def test_choice_of_floats(self):
    dim = Dimension.from_spec("lr", "1e-4,1e-3,1e-2")
    assert dim.kind == "choice"
    assert dim.values == [1e-4, 1e-3, 1e-2]

  def test_choice_of_ints(self):
    dim = Dimension.from_spec("batch_size", "32,64,128")
    assert dim.kind == "choice"
    assert dim.values == [32, 64, 128]

  def test_choice_of_strings(self):
    dim = Dimension.from_spec("optimizer", "adam,sgd")
    assert dim.values == ["adam", "sgd"]

  def test_int_range(self):
    dim = Dimension.from_spec("depth", "2:12")
    assert dim.kind == "int"
    assert (dim.low, dim.high) == (2, 12)

  def test_float_range(self):
    dim = Dimension.from_spec("dropout", "0.0:0.5")
    assert dim.kind == "float"
    assert (dim.low, dim.high) == (0.0, 0.5)

  def test_log_range(self):
    dim = Dimension.from_spec("lr", "log(1e-4:1e-1)")
    assert dim.kind == "float"
    assert dim.log is True

  def test_float_hint_on_int_literals(self):
    dim = Dimension.from_spec("temp", "float(1:3)")
    assert dim.kind == "float"
    assert (dim.low, dim.high) == (1.0, 3.0)

  def test_step_range(self):
    dim = Dimension.from_spec("depth", "2:10:step=2")
    assert dim.grid() == [2, 4, 6, 8, 10]


class TestSampling:

  def test_int_sample_within_bounds(self):
    dim = Dimension.from_spec("depth", "2:12")
    rng = np.random.default_rng(0)
    for _ in range(200):
      value = dim.sample(rng)
      assert isinstance(value, int)
      assert 2 <= value <= 12

  def test_log_sample_stays_in_range(self):
    dim = Dimension.from_spec("lr", "log(1e-4:1e-1)")
    rng = np.random.default_rng(1)
    for _ in range(200):
      value = dim.sample(rng)
      assert 1e-4 <= value <= 1e-1

  def test_choice_sample_is_a_member(self):
    dim = Dimension.from_spec("optimizer", "adam,sgd")
    rng = np.random.default_rng(2)
    for _ in range(50):
      assert dim.sample(rng) in ("adam", "sgd")

  def test_unit_roundtrip(self):
    for spec in ("32,64,128", "2:12", "0.0:0.5", "log(1e-4:1e-1)", "adam,sgd"):
      dim = Dimension.from_spec("x", spec)
      for value in dim.grid():
        assert dim.from_unit(dim.to_unit(value)) == value


class TestSearchSpace:

  def test_from_pairs_and_argv_encoding(self):
    space = SearchSpace.from_pairs(
        ["opt.lr=1e-4,1e-3", "batch_size=32,64", "model.embed_dim=64:128"])
    argv = space.decode_to_argv({
        "opt.lr": 1e-4,
        "batch_size": 64,
        "model.embed_dim": 96,
    })
    assert "--batch_size" in argv
    assert "64" in argv
    assert "--opt_arg" in argv
    assert "lr=0.0001" in argv
    assert "--model_arg" in argv
    assert "embed_dim=96" in argv

  def test_grid_product(self):
    space = SearchSpace.from_pairs(["a=1,2", "b=x,y"])
    grid = space.grid()
    assert len(grid) == 4
    assert {"a": 1, "b": "x"} in grid

  def test_sample_covers_all_dimensions(self):
    space = SearchSpace.from_pairs(["a=1,2", "b=0.0:1.0"])
    config = space.sample(np.random.default_rng(0))
    assert set(config) == {"a", "b"}

  def test_bad_token_raises(self):
    with pytest.raises(ValueError):
      SearchSpace.from_pairs(["no_equals_sign"])


class TestFormatValue:

  def test_bool(self):
    assert format_value(True) == "true"

  def test_list(self):
    assert format_value([1, 2]) == "[1,2]"

  def test_float_roundtrips_through_parse(self):
    from genml_kit.utils.cli import parse_value
    assert parse_value(format_value(1e-4)) == 1e-4
