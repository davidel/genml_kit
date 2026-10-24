"""Tests for layer-matching helpers used by feature distillation (§26).

Pure CPU, no dataset, no GPU.  A fake backbone with a ``blocks`` ModuleList
stands in for a real teacher/student transformer.
"""

import logging

import pytest
import torch.nn as nn

from genml_kit.models.layer_match import (
    LayerPair,
    _even_indices,
    iter_named_layers,
    match_layers,
)


def _make_backbone(depth, prefix=""):
  """A tiny model whose ``blocks.<i>`` are the distillation candidates."""

  class _Backbone(nn.Module):

    def __init__(self):
      super().__init__()
      self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(depth)])

  model = _Backbone()
  if prefix:
    holder = nn.Module()
    setattr(holder, prefix, model)
    return holder
  return model


class TestIterNamedLayers:

  def test_returns_registration_order(self):
    layers = iter_named_layers(_make_backbone(3), pattern=r"blocks\.\d+$")
    assert [name for name, _ in layers] == ["blocks.0", "blocks.1", "blocks.2"]

  def test_skips_root_container(self):
    layers = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    assert "" not in [name for name, _ in layers]

  def test_loose_pattern_also_matches_the_container(self):
    # Footgun worth pinning: `re.search` matches the ModuleList `blocks`
    # itself, so the caller's pattern must be specific enough to select the
    # leaves it actually wants to distil.
    names = [name for name, _ in _make_backbone(2).named_modules()]
    assert names == ["", "blocks", "blocks.0", "blocks.1"]
    loose = iter_named_layers(_make_backbone(2), pattern=r"blocks")
    assert "blocks" in [name for name, _ in loose]

  def test_pattern_filters_by_name(self):
    layers = iter_named_layers(_make_backbone(5), pattern=r"blocks\.[12]$")
    assert [name for name, _ in layers] == ["blocks.1", "blocks.2"]

  def test_no_pattern_match_raises(self):
    with pytest.raises(ValueError, match="No module name matched"):
      iter_named_layers(_make_backbone(3), pattern="does_not_exist")

  def test_tied_module_listed_once(self):
    model = nn.Module()
    shared = nn.Linear(4, 4)
    model.a = shared
    model.b = shared
    layers = iter_named_layers(model)
    assert [name for name, _ in layers] == ["a"]


class TestEvenIndices:

  def test_spans_first_and_last(self):
    assert _even_indices(12, 6) == [0, 2, 4, 7, 9, 11]
    assert _even_indices(12, 6)[0] == 0
    assert _even_indices(12, 6)[-1] == 11

  def test_single_index_is_middle(self):
    assert _even_indices(5, 1) == [2]

  def test_strictly_increasing_for_many_depths(self):
    for total in range(2, 40):
      for count in range(1, total + 1):
        idx = _even_indices(total, count)
        assert len(idx) == count
        assert idx == sorted(set(idx)), (total, count, idx)


class TestMatchLayersByIndex:

  def test_equal_depth_positional(self):
    teacher = iter_named_layers(_make_backbone(3), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(3), pattern=r"blocks\.\d+$")
    pairs = match_layers(teacher, student, strategy="by_index")
    assert len(pairs) == 3
    assert all(isinstance(p, LayerPair) for p in pairs)
    assert pairs[0].teacher is teacher[0][1]
    assert pairs[1].student is student[1][1]

  def test_unequal_depth_raises(self):
    teacher = iter_named_layers(_make_backbone(3), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    with pytest.raises(ValueError, match="needs equal depth"):
      match_layers(teacher, student, strategy="by_index")


class TestMatchLayersUniform:

  def test_equal_depth_is_identity(self):
    teacher = iter_named_layers(_make_backbone(4), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(4), pattern=r"blocks\.\d+$")
    pairs = match_layers(teacher, student, strategy="uniform")
    assert [p.teacher for p in pairs] == [m for _, m in teacher]
    assert [p.student for p in pairs] == [m for _, m in student]

  def test_unequal_depth_spans_both(self):
    teacher = iter_named_layers(_make_backbone(12), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(6), pattern=r"blocks\.\d+$")
    pairs = match_layers(teacher, student, strategy="uniform")
    assert len(pairs) == 6
    # First and last teacher layer are always distilled.
    assert pairs[0].teacher is teacher[0][1]
    assert pairs[-1].teacher is teacher[11][1]

  def test_default_strategy_is_uniform(self):
    teacher = iter_named_layers(_make_backbone(4), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    assert len(match_layers(teacher, student)) == 2


class TestMatchLayersByName:

  def _both(self, tdepth, sdepth, tprefix="", sprefix=""):
    teacher = iter_named_layers(_make_backbone(tdepth, tprefix),
                                pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(sdepth, sprefix),
                                pattern=r"blocks\.\d+$")
    return teacher, student

  def test_pairs_on_matched_key(self):
    teacher, student = self._both(3, 3)
    pairs = match_layers(teacher, student, strategy="by_name", pattern=r"blocks\.\d+")
    assert len(pairs) == 3
    assert pairs[1].teacher is teacher[1][1]
    assert pairs[1].student is student[1][1]

  def test_pattern_tolerates_differing_prefixes(self):
    teacher, student = self._both(2, 2, tprefix="encoder", sprefix="model")
    pairs = match_layers(teacher, student, strategy="by_name", pattern=r"blocks\.\d+")
    assert len(pairs) == 2

  def test_skips_extras_and_logs_warning(self, caplog):
    teacher, student = self._both(5, 3)
    with caplog.at_level(logging.WARNING):
      pairs = match_layers(teacher, student, strategy="by_name", pattern=r"blocks\.\d+")
    assert len(pairs) == 3
    assert any("Skipped 2 unmatched layer(s)" in r.message for r in caplog.records)

  def test_logs_matches_at_info(self, caplog):
    teacher, student = self._both(2, 2)
    with caplog.at_level(logging.INFO):
      match_layers(teacher, student, strategy="by_name", pattern=r"blocks\.\d+")
    assert any("Matched 2 layer(s)" in r.message for r in caplog.records)

  def test_zero_pairs_raises(self):
    teacher, student = self._both(2, 2)
    with pytest.raises(ValueError, match="matched no teacher/student layers"):
      match_layers(teacher, student, strategy="by_name", pattern=r"nope")


class TestMatchLayersErrors:

  def test_unknown_strategy_raises(self):
    teacher = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    student = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    with pytest.raises(ValueError, match="Unknown strategy"):
      match_layers(teacher, student, strategy="bogus")

  def test_empty_input_raises(self):
    student = iter_named_layers(_make_backbone(2), pattern=r"blocks\.\d+$")
    with pytest.raises(ValueError, match="must be non-empty"):
      match_layers([], student, strategy="uniform")
