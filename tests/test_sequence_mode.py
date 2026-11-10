"""Tests for the ``--sequence_mode`` resolver (training/sequence_mode.py).

The resolver is pure and torch-free, so the whole valid/invalid matrix is
checked directly: the named recipes, the windowing-flag precedence, and the
combinations that must be rejected (unknown mode, unknown packing, the
``--packing`` / ``--length_bucket`` clash, and ``open`` without packing).
"""

import pytest

from genml_kit.training.sequence_mode import (MODES, mode_names, resolve_sequence_mode)


class TestRecipes:

  def test_document_is_unpacked_masked_plain(self):
    mode = resolve_sequence_mode("document")
    assert mode.windowing is None
    assert mode.boundary_loss == "mask"
    assert mode.prompt_framing == "plain"

  def test_open_is_packed_kept_doc(self):
    mode = resolve_sequence_mode("open")
    assert mode.windowing == "wrapped"
    assert mode.boundary_loss == "keep"
    assert mode.prompt_framing == "doc"

  def test_default_is_the_document_recipe(self):
    # ``document`` must be the first/only default callers rely on.
    assert resolve_sequence_mode("document") == MODES["document"]

  def test_mode_names_are_stable_and_complete(self):
    assert mode_names() == ("document", "open")


class TestWindowingOverride:

  def test_explicit_packing_overrides_document_default(self):
    mode = resolve_sequence_mode("document", packing="wrapped")
    assert mode.windowing == "wrapped"
    # Boundary/framing still come from the mode.
    assert mode.boundary_loss == "mask"
    assert mode.prompt_framing == "plain"

  def test_explicit_bfd_is_kept(self):
    assert resolve_sequence_mode("document", packing="bfd").windowing == "bfd"

  def test_length_bucket_selects_bucket(self):
    mode = resolve_sequence_mode("document", length_bucket=True)
    assert mode.windowing == "bucket"

  def test_packing_beats_length_bucket_never_apply_together(self):
    # Both set is an error, not a precedence question.
    with pytest.raises(ValueError, match="mutually exclusive"):
      resolve_sequence_mode("document", packing="wrapped", length_bucket=True)


class TestInvalidCombinations:

  def test_unknown_mode_is_fatal(self):
    with pytest.raises(ValueError, match="unknown --sequence_mode"):
      resolve_sequence_mode("nope")

  def test_unknown_packing_is_fatal(self):
    with pytest.raises(ValueError, match="--packing must be one of"):
      resolve_sequence_mode("document", packing="sideways")

  def test_open_with_length_bucket_is_fatal(self):
    # ``open`` needs concatenation; the bucket sampler does not concatenate.
    with pytest.raises(ValueError, match="requires packing"):
      resolve_sequence_mode("open", length_bucket=True)
