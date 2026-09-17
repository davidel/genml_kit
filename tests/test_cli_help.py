"""Tests for the full-CLI --help render (TODO #1)."""

import io
from contextlib import redirect_stdout

import pytest

from genml_kit.training.train import build_parser, parse_args, register_all_owners


class TestRegisterAllOwners:

  def test_no_option_string_collisions(self):
    """Every pipeline and method can coexist on one parser (single-pass).

    If a future owner registers an option string already owned by another,
    ``add_args`` raises ``ArgumentError`` here -- the explicit signal to
    rename one of the flags.
    """
    parser = build_parser()
    register_all_owners(parser)
    assert len(parser._actions) > 80

  def test_owner_groups_are_registered(self):
    parser = build_parser()
    register_all_owners(parser)
    titles = [g.title for g in parser._action_groups]
    for expected in ("images pipeline", "vo_pair pipeline", "DINO", "BYOL", "SimMIM",
                     "I-JEPA", "classification method"):
      assert expected in titles


class TestHelpRender:

  def test_help_prints_full_cli(self):
    """--help must list every owner's flags plus shared groups."""
    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
      parse_args(["--help"])
    out = buf.getvalue()
    for token in ("--dataset", "--vo_length", "--dino_local_num", "--lr",
                  "--checkpoint", "images pipeline", "DINO"):
      assert token in out, f"missing {token!r} in --help output"

  def test_help_short_flag(self):
    buf = io.StringIO()
    with pytest.raises(SystemExit), redirect_stdout(buf):
      parse_args(["-h"])
    assert "--dataset" in buf.getvalue()
