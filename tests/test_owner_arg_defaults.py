"""Tests for owner-declared CLI defaults (``arg_defaults``).

A pipeline or method may declare default values for existing CLI flags;
the explicit command line always wins.  Dict-valued flags (``KVPairAction``,
e.g. ``--dist_arg``) merge per key instead of being replaced wholesale.
"""

import argparse

import pytest

from genml_kit.cli.train import parse_args
from genml_kit.methods import METHODS
from genml_kit.methods.base import Method
from genml_kit.pipelines import PIPELINES
from genml_kit.pipelines.base import DataPipeline
from genml_kit.utils.cli import KVPairAction, apply_arg_defaults


class _Owner:
  """Minimal stand-in for a pipeline/method declaration."""

  def __init__(self, defaults):
    self._defaults = defaults

  def arg_defaults(self):
    return self._defaults


def _parser(with_kv=True):
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--batch_size", type=int, default=32)
  if with_kv:
    parser.add_argument("--dist_arg", nargs="+", action=KVPairAction)
  return parser


class TestApplyArgDefaults:

  def test_empty_hook_is_identity(self):
    parser = _parser()
    assert not apply_arg_defaults(parser, [_Owner({})])
    assert parser.get_default("epochs") == 5

  def test_none_hook_is_tolerated(self):
    parser = _parser()
    assert not apply_arg_defaults(parser, [_Owner(None)])

  def test_scalar_default_applied(self):
    parser = _parser()
    declared = apply_arg_defaults(parser, [_Owner({"epochs": 7})])
    assert declared == {"epochs"}
    assert parser.get_default("epochs") == 7
    assert parser.parse_args([]).epochs == 7

  def test_explicit_cli_beats_owner_default(self):
    parser = _parser()
    apply_arg_defaults(parser, [_Owner({"epochs": 7})])
    assert parser.parse_args(["--epochs", "3"]).epochs == 3

  def test_later_owner_wins_on_scalar(self):
    parser = _parser()
    apply_arg_defaults(parser, [_Owner({"epochs": 7}), _Owner({"epochs": 9})])
    assert parser.get_default("epochs") == 9

  def test_dict_default_merges_per_key(self):
    parser = _parser()
    apply_arg_defaults(
        parser, [_Owner({"dist_arg": {"find_unused_parameters": True}})])
    args = parser.parse_args(["--dist_arg", "static_graph"])
    assert args.dist_arg == {
        "find_unused_parameters": True,
        "static_graph": True,
    }

  def test_dict_default_alone(self):
    parser = _parser()
    apply_arg_defaults(parser, [_Owner({"dist_arg": {"static_graph": True}})])
    assert parser.parse_args([]).dist_arg == {"static_graph": True}

  def test_dict_same_key_override(self):
    parser = _parser()
    apply_arg_defaults(
        parser, [_Owner({"dist_arg": {"find_unused_parameters": True}})])
    args = parser.parse_args(["--dist_arg", "find_unused_parameters=false"])
    assert args.dist_arg == {"find_unused_parameters": False}

  def test_mismatched_types_replace_outright(self):
    parser = _parser()
    apply_arg_defaults(parser, [_Owner({"epochs": None})])
    assert parser.get_default("epochs") is None

  def test_declared_names_accumulate(self):
    parser = _parser()
    declared = apply_arg_defaults(
        parser,
        [_Owner({"epochs": 7}),
         _Owner({"batch_size": 8, "dist_arg": {
             "static_graph": True
         }})])
    assert declared == {"epochs", "batch_size", "dist_arg"}

  def test_dict_defaults_merge_across_owners(self):
    parser = _parser()
    apply_arg_defaults(
        parser, [
            _Owner({"dist_arg": {"find_unused_parameters": True}}),
            _Owner({"dist_arg": {"static_graph": True}}),
        ])
    assert parser.get_default("dist_arg") == {
        "find_unused_parameters": True,
        "static_graph": True,
    }

  def test_unknown_dest_is_fatal(self):
    parser = _parser()
    with pytest.raises(RuntimeError, match="arg_defaults.*nope"):
      apply_arg_defaults(parser, [_Owner({"nope": 1})])

  def test_unknown_dest_writes_nothing(self):
    """Validation runs before any mutation, so a typo changes no default."""
    parser = _parser()
    with pytest.raises(RuntimeError):
      apply_arg_defaults(parser, [_Owner({"epochs": 7, "nope": 1})])
    assert parser.get_default("epochs") == 5


class _DefaultMethod(Method):
  NAME = "test_default_method"
  _DEFAULTS = {}

  @classmethod
  def arg_defaults(cls):
    return dict(cls._DEFAULTS)

  def build_model(self, args, device):  # pragma: no cover - never built
    raise AssertionError("not built in these tests")

  def train_step(self, model, blob, global_step, *, labels=None):
    raise AssertionError("not built in these tests")


class _DefaultPipeline(DataPipeline):
  NAME = "test_default_pipeline"
  _DEFAULTS = {}

  @classmethod
  def arg_defaults(cls):
    return dict(cls._DEFAULTS)

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    raise AssertionError("not built in these tests")


@pytest.fixture
def owners():
  """Register transient owners and clean them out of the registries."""
  METHODS.register(_DefaultMethod)
  PIPELINES.register(_DefaultPipeline)
  yield _DefaultMethod, _DefaultPipeline
  METHODS.unregister(_DefaultMethod.NAME)
  PIPELINES.unregister(_DefaultPipeline.NAME)


class TestParseArgsIntegration:

  def test_base_hooks_are_empty(self):
    assert Method.arg_defaults() == {}
    assert DataPipeline.arg_defaults() == {}

  def test_method_default_applied(self, owners):
    method, _ = owners
    method._DEFAULTS = {"epochs": 7}
    args = parse_args(["--method", _DefaultMethod.NAME])
    assert args.epochs == 7

  def test_explicit_cli_still_wins(self, owners):
    method, _ = owners
    method._DEFAULTS = {"epochs": 7}
    args = parse_args(["--method", _DefaultMethod.NAME, "--epochs", "3"])
    assert args.epochs == 3

  def test_equals_form_selects_owner(self, owners):
    method, _ = owners
    method._DEFAULTS = {"epochs": 7}
    args = parse_args([f"--method={_DefaultMethod.NAME}"])
    assert args.epochs == 7

  def test_pipeline_default_applied(self, owners):
    _, pipeline = owners
    pipeline._DEFAULTS = {"batch_size": 8}
    args = parse_args(["--pipeline", _DefaultPipeline.NAME])
    assert args.batch_size == 8

  def test_unselected_owner_does_not_leak(self, owners):
    method, pipeline = owners
    method._DEFAULTS = {"epochs": 7}
    pipeline._DEFAULTS = {"batch_size": 8}
    # Selected: the fake pipeline, but the default method (classification).
    args = parse_args(["--pipeline", _DefaultPipeline.NAME])
    assert args.batch_size == 8
    assert args.epochs == 5

  def test_dist_arg_per_key_merge_end_to_end(self, owners):
    method, _ = owners
    method._DEFAULTS = {"dist_arg": {"find_unused_parameters": True}}
    args = parse_args(["--method", _DefaultMethod.NAME])
    assert args.dist_arg == {"find_unused_parameters": True}
    args = parse_args(
        ["--method", _DefaultMethod.NAME, "--dist_arg", "static_graph"])
    assert args.dist_arg == {
        "find_unused_parameters": True,
        "static_graph": True,
    }

  def test_unknown_dest_is_fatal_end_to_end(self, owners):
    method, _ = owners
    method._DEFAULTS = {"not_a_flag": 1}
    with pytest.raises(RuntimeError, match="not_a_flag"):
      parse_args(["--method", _DefaultMethod.NAME])
