"""Tests for the shared-args registration contract.

``add_shared_arg`` is idempotent per parser: a shared argument that is
already registered anywhere on the parser (including via one of its
groups) is skipped instead of raising argparse's duplicate-option
conflict.  That is what lets composable callers -- e.g. several methods
chaining a common mixin's ``add_args`` -- re-request the same shared
flags.
"""

import argparse

import pytest

from genml_kit.utils.shared_args import ARG_GRAD_CLIP, add_shared_arg


def test_first_registration_returns_action():
  parser = argparse.ArgumentParser()
  action = add_shared_arg(parser, ARG_GRAD_CLIP)
  assert action is not None
  assert "--grad_clip" in parser.format_help()


def test_re_registration_is_skipped_not_conflicting():
  parser = argparse.ArgumentParser()
  first = add_shared_arg(parser, ARG_GRAD_CLIP)
  # A second request for the same shared arg is a no-op, not an
  # ArgumentError from argparse's duplicate-option check.
  second = add_shared_arg(parser, ARG_GRAD_CLIP)
  assert first is not None
  assert second is None
  # Exactly one --grad_clip remains registered.
  hits = [a for a in parser._actions if "--grad_clip" in a.option_strings]
  assert len(hits) == 1


def test_group_registration_visible_to_root_parser():
  """Registered via a group, re-requested via the parser: skipped."""
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group("my group")
  assert add_shared_arg(group, ARG_GRAD_CLIP) is not None
  assert add_shared_arg(parser, ARG_GRAD_CLIP) is None


def test_parser_registration_visible_to_group():
  """Registered via the parser, re-requested via a group: skipped."""
  parser = argparse.ArgumentParser()
  group = parser.add_argument_group("my group")
  assert add_shared_arg(parser, ARG_GRAD_CLIP) is not None
  assert add_shared_arg(group, ARG_GRAD_CLIP) is None


def test_override_on_first_registration_only():
  """The per-call override is honoured on the first registration only."""
  parser = argparse.ArgumentParser()
  first = add_shared_arg(parser, ARG_GRAD_CLIP, default=5.0)
  assert first.default == 5.0
  assert parser.get_default("grad_clip") == 5.0


def test_differing_spec_raises_value_error():
  """A re-request with a *different* spec is a registration-order bug."""
  parser = argparse.ArgumentParser()
  add_shared_arg(parser, ARG_GRAD_CLIP, default=5.0)
  with pytest.raises(ValueError, match="different spec"):
    add_shared_arg(parser, ARG_GRAD_CLIP, default=9.0)


def test_differing_help_raises_value_error():
  """The description is part of the spec: changing it is an error too."""
  parser = argparse.ArgumentParser()
  add_shared_arg(parser, ARG_GRAD_CLIP)
  with pytest.raises(ValueError, match="different spec"):
    add_shared_arg(parser, ARG_GRAD_CLIP, help="a different description")


def test_store_true_re_request_without_default_is_consistent():
  """``store_true`` implies ``default=False``; a bare re-request matches."""
  parser = argparse.ArgumentParser()
  spec = ("my_flag", {"action": "store_true", "help": "h"})
  assert add_shared_arg(parser, spec) is not None
  assert add_shared_arg(parser, spec) is None  # consistent re-request skips
  # A re-request that spells out the same implied default is also fine.
  assert add_shared_arg(parser, ("my_flag", {
      "action": "store_true",
      "help": "h",
      "default": False,
  })) is None
  # But a contradictory one is an error.
  with pytest.raises(ValueError, match="different spec"):
    add_shared_arg(parser, ("my_flag", {
        "action": "store_true",
        "help": "h",
        "default": True,
    }))


def test_independent_parsers_do_not_share_state():
  """Idempotency is per parser, not process-global."""
  parser_a = argparse.ArgumentParser()
  parser_b = argparse.ArgumentParser()
  assert add_shared_arg(parser_a, ARG_GRAD_CLIP) is not None
  assert add_shared_arg(parser_b, ARG_GRAD_CLIP) is not None
