"""Tests for genml_kit.cli_utils — CLI parsing utilities."""

import argparse

import pytest

from genml_kit.utils.cli import KVPairAction, parse_value


class TestParseValue:
  """Tests for the auto-converting parse_value helper."""

  def test_true(self):
    assert parse_value("true") is True

  def test_false(self):
    assert parse_value("false") is False

  case_insensitive_bools = [
      ("TRUE", True),
      ("True", True),
      ("False", False),
      ("FALSE", False),
  ]

  @pytest.mark.parametrize("input_val,expected", case_insensitive_bools)
  def test_bool_case_insensitive(self, input_val, expected):
    assert parse_value(input_val) is expected

  none_cases = ["none", "None", "NONE"]

  @pytest.mark.parametrize("input_val", none_cases)
  def test_none_case_insensitive(self, input_val):
    assert parse_value(input_val) is None

  def test_positive_int(self):
    assert parse_value("42") == 42
    assert isinstance(parse_value("42"), int)

  def test_negative_int(self):
    assert parse_value("-7") == -7

  def test_positive_float(self):
    assert parse_value("0.1") == pytest.approx(0.1)
    assert isinstance(parse_value("0.1"), float)

  def test_scientific_notation(self):
    assert parse_value("1e-3") == pytest.approx(0.001)

  def test_negative_float(self):
    assert parse_value("-0.5") == pytest.approx(-0.5)

  def test_plain_string(self):
    assert parse_value("hello") == "hello"

  def test_empty_string_is_string(self):
    # Empty string doesn't match int/float, stays str
    assert parse_value("") == ""

  def test_list_with_ints(self):
    result = parse_value("[3,6,12,6]")
    assert result == [3, 6, 12, 6]
    assert all(isinstance(v, int) for v in result)

  def test_list_with_floats(self):
    result = parse_value("[0.1, 0.2, 0.3]")
    assert result == pytest.approx([0.1, 0.2, 0.3])

  def test_list_with_mixed_types(self):
    result = parse_value("[1, hello, 0.5]")
    assert result == [1, "hello", 0.5]

  def test_empty_list(self):
    assert parse_value("[]") == []

  def test_tuple_syntax_returns_list(self):
    result = parse_value("(1, 2, 3)")
    assert result == [1, 2, 3]

  def test_nested_list(self):
    result = parse_value("[[1, 2], [3, 4]]")
    assert result == [[1, 2], [3, 4]]

  def test_bare_bracket_string_single_element(self):
    # Single-element bracket expression — parsed as a 1-element list
    assert parse_value("[hello]") == ["hello"]


class TestKVPairAction:
  """Tests for the argparse KVPairAction."""

  def _make_parser(self):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_arg",
        nargs="+",
        action=KVPairAction,
        default={},
    )
    return parser

  def test_single_pair_int(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "depth=6"])
    assert args.model_arg == {"depth": 6}

  def test_multiple_pairs(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "depth=6", "num_heads=8", "dropout=0.2"])
    assert args.model_arg == {"depth": 6, "num_heads": 8, "dropout": 0.2}

  def test_float_value(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "lr=3e-4"])
    assert args.model_arg == {"lr": pytest.approx(3e-4)}

  def test_bool_value(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "use_cls=true"])
    assert args.model_arg == {"use_cls": True}

  def test_list_value(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "depths=[2,6,2]"])
    assert args.model_arg == {"depths": [2, 6, 2]}

  def test_string_value(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "name=convvit"])
    assert args.model_arg == {"name": "convvit"}

  def test_duplicate_key_last_wins(self):
    parser = self._make_parser()
    args = parser.parse_args(["--model_arg", "depth=12", "--model_arg", "depth=6"])
    assert args.model_arg == {"depth": 6}

  def test_missing_equals_raises_error(self):
    parser = self._make_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(["--model_arg", "noequals"])

  def test_multiple_flags_accumulate(self):
    parser = self._make_parser()
    args = parser.parse_args(
        ["--model_arg", "a=1", "--model_arg", "b=2", "--model_arg", "c=3"])
    assert args.model_arg == {"a": 1, "b": 2, "c": 3}
