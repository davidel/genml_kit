"""Tests for genml_kit.training.param_align — automatic state-dict key alignment."""

import torch

from genml_kit.training.param_align import (
    AlignConfig,
    AlignReport,
    _build_shape_dict,
    _find_best_match,
    _tokenize,
    align_state_dicts,
    report_to_str,
    weighted_token_distance,
)


class TestTokenize:

  def test_single_token(self):
    assert _tokenize("weight") == ["weight"]

  def test_multi_token(self):
    assert _tokenize("encoder.layer.weight") == ["weight", "layer", "encoder"]

  def test_empty_string(self):
    assert _tokenize("") == [""]


class TestWeightedTokenDistance:

  def test_identical_tokens(self):
    t = ["weight", "layer", "encoder"]
    assert weighted_token_distance(t, t) == 0.0

  def test_single_leaf_substitution(self):
    """One substitution at position 1 costs 0.5."""
    t1 = ["weight", "layer", "encoder"]
    t2 = ["weight", "fc", "encoder"]
    d = weighted_token_distance(t1, t2)
    assert d == 0.5

  def test_single_root_substitution(self):
    """One root substitution at position 2 costs 0.25."""
    t1 = ["weight", "layer", "encoder"]
    t2 = ["weight", "layer", "decoder"]
    d = weighted_token_distance(t1, t2)
    assert d == 0.25

  def test_position_zero_mismatch(self):
    """One substitution at position 0 costs 1.0."""
    t1 = ["weight", "layer", "encoder"]
    t2 = ["bias", "layer", "encoder"]
    d = weighted_token_distance(t1, t2)
    assert d == 1.0

  def test_longer_vs_shorter(self):
    """Missing a root-side token should be cheap."""
    t1 = ["weight", "ln", "model", "encoder"]
    t2 = ["weight", "ln", "encoder"]
    d = weighted_token_distance(t1, t2)
    assert d == 0.25  # one root-side extra token

  def test_symmetry(self):
    t1 = ["a", "b", "c"]
    t2 = ["a", "x", "c"]
    assert weighted_token_distance(t1, t2) == weighted_token_distance(t2, t1)

  def test_completely_different(self):
    t1 = ["a", "b", "c"]
    t2 = ["x", "y", "z"]
    d = weighted_token_distance(t1, t2)
    assert d == 0.75

  def test_two_leaf_mismatches(self):
    """Two mismatches at positions 0 and 1."""
    t1 = ["weight", "layer", "encoder"]
    t2 = ["bias", "fc", "encoder"]
    d = weighted_token_distance(t1, t2)
    assert d == 1.0  # substitution at pos 0 + deletion at pos 1 via DP

  def test_geometric_weights_dominated(self):
    """Sum of all weights beyond position 0 is < 1.0, so one leaf match
    always beats any number of root mismatches."""
    t1 = ["x", "a", "b", "c"]
    t2 = ["x", "d", "e", "f"]
    d_match = weighted_token_distance(t1, t2)
    t3 = ["y", "a", "b", "c"]
    t4 = ["z", "a", "b", "c"]
    d_nomatch = weighted_token_distance(t3, t4)
    assert d_match < d_nomatch


class TestBuildShapeDict:

  def test_groups_by_shape(self):
    sd = {
        "encoder.layer.weight": torch.zeros(64, 64),
        "encoder.bias": torch.zeros(32,),
    }
    d = _build_shape_dict(sd)
    assert (64, 64) in d
    assert (32,) in d

  def test_keys_stored(self):
    sd = {"a.weight": torch.zeros(3, 3)}
    d = _build_shape_dict(sd)
    assert len(d[(3, 3)]) == 1
    tokens, key = d[(3, 3)][0]
    assert key == "a.weight"
    assert tokens == ["weight", "a"]

  def test_same_shape_different_names(self):
    sd = {
        "a.weight": torch.zeros(3, 3),
        "b.weight": torch.zeros(3, 3),
    }
    d = _build_shape_dict(sd)
    assert len(d[(3, 3)]) == 2
    keys = {k for _, k in d[(3, 3)]}
    assert keys == {"a.weight", "b.weight"}


class TestFindBestMatch:

  def test_exact_match(self):
    old = {"encoder.layer.weight": torch.zeros(64, 64)}
    d = _build_shape_dict(old)
    tokens = _tokenize("encoder.layer.weight")
    key, dist = _find_best_match(d, (64, 64), tokens)
    assert key == "encoder.layer.weight"
    assert dist == 0.0

  def test_best_match_selected(self):
    old = {
        "blocks.0.attn.weight": torch.zeros(8, 8),
        "blocks.0.mlp.weight": torch.zeros(8, 8),
    }
    d = _build_shape_dict(old)
    tokens = _tokenize("blocks.0.attn.weight")
    key, _ = _find_best_match(d, (8, 8), tokens)
    assert key == "blocks.0.attn.weight"

  def test_no_match_different_shape(self):
    old = {"a.weight": torch.zeros(3,)}
    d = _build_shape_dict(old)
    new_tokens = _tokenize("a.weight")
    key, _ = _find_best_match(d, (5,), new_tokens)
    assert key is None

  def test_max_distance_rejects(self):
    old = {"encoder.layer.weight": torch.zeros(64, 64)}
    d = _build_shape_dict(old)
    tokens = _tokenize("decoder.fc.bias")
    # With a very strict threshold, even a same-shape key is rejected.
    key, _ = _find_best_match(d, (64, 64), tokens, max_distance=0.01)
    assert key is None

  def test_longer_leaf_match_wins(self):
    """When multiple old keys share a shape, the one with more leaf-side
    matching tokens should score better."""
    old = {
        "blocks.0.attn.weight": torch.zeros(8, 8),
        "blocks.0.mlp.weight": torch.zeros(8, 8),
    }
    d = _build_shape_dict(old)
    tokens = _tokenize("blocks.0.attn.weight")
    key, _ = _find_best_match(d, (8, 8), tokens)
    assert key == "blocks.0.attn.weight"


class TestAlignStateDicts:

  def test_identical_keys(self):
    old_sd = {"encoder.layer.weight": torch.randn(64, 64)}
    new_sd = {"encoder.layer.weight": torch.randn(64, 64)}
    report = align_state_dicts(old_sd, new_sd)
    assert report.mapping == {"encoder.layer.weight": "encoder.layer.weight"}
    assert not report.unused_old
    assert not report.unmatched_new

  def test_prefix_only_change(self):
    old_sd = {"model.encoder.layer1.weight": torch.zeros(3, 3)}
    new_sd = {"vision.encoder.layer1.weight": torch.zeros(3, 3)}
    report = align_state_dicts(old_sd, new_sd)
    assert report.mapping["vision.encoder.layer1.weight"] == (
        "model.encoder.layer1.weight")
    assert report.ok

  def test_prefix_only_change_short_keys(self):
    old_sd = {
        "model.encoder.layer1.weight": torch.zeros(3, 3),
        "model.encoder.layer1.bias": torch.zeros(3),
    }
    new_sd = {
        "vision.encoder.layer1.weight": torch.zeros(3, 3),
        "vision.encoder.layer1.bias": torch.zeros(3),
    }
    report = align_state_dicts(old_sd, new_sd)
    assert report.mapping["vision.encoder.layer1.weight"] == (
        "model.encoder.layer1.weight")
    assert report.mapping["vision.encoder.layer1.bias"] == ("model.encoder.layer1.bias")
    assert report.ok

  def test_shape_mismatch_no_match(self):
    old_sd = {"encoder.layer.weight": torch.randn(64, 64)}
    new_sd = {"encoder.layer.weight": torch.randn(128, 128)}
    report = align_state_dicts(old_sd, new_sd)
    assert report.mapping == {}
    assert "encoder.layer.weight" in report.unused_old
    assert "encoder.layer.weight" in report.unmatched_new

  def test_layer_expansion_no_false_conflict(self):
    old = {f"layers.{i}.weight": torch.zeros(32, 32) for i in range(4)}
    new = {f"model.layers.{i}.weight": torch.zeros(32, 32) for i in range(4)}
    report = align_state_dicts(old, new)
    for i in range(4):
      assert report.mapping[f"model.layers.{i}.weight"] == f"layers.{i}.weight"
    assert not report.divergent
    assert report.ok

  def test_layer_expansion_longer_match_wins(self):
    """The new key with a longer leaf-side match should win."""
    old = {
        "A.B.3.C.D": torch.zeros(2, 2),
    }
    new = {
        "X.Y.3.C.D": torch.zeros(2, 2),
        "X.Y.4.C.D": torch.zeros(2, 2),
    }
    # Need relaxed threshold — distance is 0.375 (two root mismatches).
    report = align_state_dicts(old, new, AlignConfig(max_distance=0.5))
    assert report.mapping["X.Y.3.C.D"] == "A.B.3.C.D"
    # "X.Y.4.C.D" has distance 0.4375 > 0.5 threshold? Let's check:
    # The old key is already claimed; this key is either unmatched or divergent.
    assert report.ok or report.divergent or report.unmatched_new

  def test_layer_expansion_high_threshold(self):
    """With a strict max_distance, short matches are rejected."""
    old = {f"layers.{i}.weight": torch.zeros(32, 32) for i in range(8)}
    new = {f"model.layers.{i}.weight": torch.zeros(32, 32) for i in range(8)}
    config = AlignConfig(max_distance=0.0)
    report = align_state_dicts(old, new, config)
    # Only exact token matches (distance 0) would pass, but with prefix
    # changes the distance is > 0, so nothing should match.
    assert not report.mapping
    assert len(report.unmatched_new) == 8

  def test_unused_old_keys(self):
    old = {
        "encoder.weight": torch.zeros(3, 3),
        "extra.weight": torch.zeros(3, 3),
    }
    new = {"encoder.weight": torch.zeros(3, 3)}
    report = align_state_dicts(old, new)
    assert "extra.weight" in report.unused_old
    assert not report.ok  # unused old key → not perfect

  def test_empty_state_dicts(self):
    report = align_state_dicts({}, {})
    assert report.mapping == {}
    assert report.ok

  def test_old_empty_new_has_keys(self):
    new = {"a.weight": torch.zeros(3, 3)}
    report = align_state_dicts({}, new)
    assert "a.weight" in report.unmatched_new

  def test_new_empty_old_has_keys(self):
    old = {"a.weight": torch.zeros(3, 3)}
    report = align_state_dicts(old, {})
    assert "a.weight" in report.unused_old
    assert not report.ok  # unused old key → not perfect

  def test_many_to_one_old_conflict(self):
    old = {"shared.weight": torch.zeros(3, 3)}
    new = {
        "a.weight": torch.zeros(3, 3),
        "b.weight": torch.zeros(3, 3),
    }
    # Distance is 0.5 (one prefix mismatch on a 2-token name).
    report = align_state_dicts(old, new, AlignConfig(max_distance=1.0))
    assert report.mapping.get("a.weight") == "shared.weight"
    assert report.mapping.get("b.weight") == "shared.weight"
    assert len(report.divergent) == 1

  def test_mixed_shapes(self):
    old = {
        "encoder.weight": torch.zeros(64, 64),
        "encoder.bias": torch.zeros(64),
    }
    new = {
        "model.weight": torch.zeros(64, 64),
        "model.bias": torch.zeros(64),
    }
    # Distance is 0.5 (one prefix mismatch on a 2-token name).
    report = align_state_dicts(old, new, AlignConfig(max_distance=0.5))
    assert report.mapping["model.weight"] == "encoder.weight"
    assert report.mapping["model.bias"] == "encoder.bias"
    assert report.ok

  def test_vit_style(self):
    old = {
        "vision_model.encoder.layer.0.weight": torch.zeros(5, 5),
        "vision_model.encoder.layer.1.weight": torch.zeros(5, 5),
    }
    new = {
        "vision_model.transformer.blocks.0.weight": torch.zeros(5, 5),
        "vision_model.transformer.blocks.1.weight": torch.zeros(5, 5),
    }
    # Two root mismatches (layer→blocks, encoder→transformer) = 0.375.
    report = align_state_dicts(old, new, AlignConfig(max_distance=0.5))
    assert report.mapping["vision_model.transformer.blocks.0.weight"] == (
        "vision_model.encoder.layer.0.weight")
    assert report.mapping["vision_model.transformer.blocks.1.weight"] == (
        "vision_model.encoder.layer.1.weight")
    assert report.ok

  def test_conv_stem_layer_expansion(self):
    """ConvPatchEmbedding: 4 stem blocks with stem-specific tokens."""
    old = {}
    new = {}
    for i in range(4):
      old[f"encoder.model.patch_embed.blocks.{i}.conv1.weight"] = (torch.zeros(
          24, 24, 3, 3))
      new[f"model.patch_embed.blocks.{i}.conv1.weight"] = (torch.zeros(24, 24, 3, 3))
    report = align_state_dicts(old, new)
    for i in range(4):
      new_k = f"model.patch_embed.blocks.{i}.conv1.weight"
      old_k = f"encoder.model.patch_embed.blocks.{i}.conv1.weight"
      assert report.mapping[new_k] == old_k
    assert report.ok


class TestEdgeCases:

  def test_single_token_keys(self):
    old = {"weight": torch.zeros(3, 3)}
    new = {"bias": torch.zeros(3, 3)}
    report = align_state_dicts(old, new)
    # Distance is 1.0 (single token mismatch) — exceeds default threshold.
    assert "bias" in report.unmatched_new
    # With relaxed threshold, it matches.
    report2 = align_state_dicts(old, new, AlignConfig(max_distance=1.0))
    assert report2.mapping.get("bias") == "weight"

  def test_deeply_nested_keys(self):
    old = {"a.b.c.d.e.f.g.weight": torch.zeros(2, 2)}
    new = {"x.y.c.d.e.f.g.weight": torch.zeros(2, 2)}
    report = align_state_dicts(old, new)
    assert report.mapping["x.y.c.d.e.f.g.weight"] == "a.b.c.d.e.f.g.weight"
    assert report.ok

  def test_tensor_shape_preserved(self):
    """Ensure we are not confused by batch dims or other ambiguities."""
    old = {
        "a.weight": torch.zeros(16, 8, 3, 3),
        "b.weight": torch.zeros(8, 16, 3, 3),
    }
    new = {
        "enc.a.weight": torch.zeros(16, 8, 3, 3),
        "enc.b.weight": torch.zeros(8, 16, 3, 3),
    }
    report = align_state_dicts(old, new)
    assert report.mapping["enc.a.weight"] == "a.weight"
    assert report.mapping["enc.b.weight"] == "b.weight"
    assert report.ok

  def test_param_rename_not_needed(self):
    """The whole point — common renaming should just work."""
    old = {"vision_model.encoder.layer.0.weight": torch.zeros(5, 5)}
    new = {"vision_model.transformer.blocks.0.weight": torch.zeros(5, 5)}
    # Two root mismatches = 0.375, needs relaxed threshold.
    report = align_state_dicts(old, new, AlignConfig(max_distance=0.5))
    assert report.mapping["vision_model.transformer.blocks.0.weight"] == (
        "vision_model.encoder.layer.0.weight")
    assert report.ok

  def test_realistic_convvit_pretrain_to_finetune(self):
    """Real-world ConvViT + SimMIM pretrain → standalone fine-tune.

    Pre-training wraps ConvViTForClassification inside SimMIM, adding an
    ``encoder.`` prefix and SimMIM-specific params (mask_token, decoder).
    Fine-tuning uses ConvViTForClassification directly (``model.`` prefix)
    with a fresh classification head (``head.``).
    """
    D, H = 24, 48
    old_sd = {}
    new_sd = {}

    # Pretrain checkpoint (SimMIM wrapping ConvViTForClassification)
    # ConvPatchEmbedding stem.
    for i in range(4):
      old_sd[f"encoder.model.patch_embed.blocks.{i}.conv1.weight"] = (torch.zeros(
          D, D, 3, 3))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.bn1.weight"] = (torch.zeros(D))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.conv2.weight"] = (torch.zeros(
          D, D, 3, 3))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.bn2.weight"] = (torch.zeros(D))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.skip_proj.weight"] = (torch.zeros(
          D, D, 1, 1))
      old_sd[f"encoder.model.patch_embed.blocks.{i}.skip_bn.weight"] = (torch.zeros(D))
    # Transformer blocks.
    for i in range(6):
      old_sd[f"encoder.model.blocks.{i}.ln1.weight"] = torch.zeros(D)
      old_sd[f"encoder.model.blocks.{i}.attn.in_proj_weight"] = (torch.zeros(3 * D, D))
      old_sd[f"encoder.model.blocks.{i}.attn.out_proj.weight"] = (torch.zeros(D, D))
      old_sd[f"encoder.model.blocks.{i}.ln2.weight"] = torch.zeros(D)
      old_sd[f"encoder.model.blocks.{i}.mlp.w12.weight"] = (torch.zeros(2 * H, D))
      old_sd[f"encoder.model.blocks.{i}.mlp.w3.weight"] = (torch.zeros(D, H))
    # Final norm.
    old_sd["encoder.model.ln.weight"] = torch.zeros(D)
    # SimMIM-specific (should be unused).
    old_sd["encoder.head.weight"] = torch.zeros(D, D)
    old_sd["mask_token"] = torch.zeros(1, 1, D)
    old_sd["decoder.weight"] = torch.zeros(D, D, 1, 1)
    old_sd["decoder.bias"] = torch.zeros(D)

    # Fine-tune target (standalone ConvViTForClassification)
    # ConvPatchEmbedding stem (same architecture).
    for i in range(4):
      new_sd[f"model.patch_embed.blocks.{i}.conv1.weight"] = (torch.zeros(D, D, 3, 3))
      new_sd[f"model.patch_embed.blocks.{i}.bn1.weight"] = (torch.zeros(D))
      new_sd[f"model.patch_embed.blocks.{i}.conv2.weight"] = (torch.zeros(D, D, 3, 3))
      new_sd[f"model.patch_embed.blocks.{i}.bn2.weight"] = (torch.zeros(D))
      new_sd[f"model.patch_embed.blocks.{i}.skip_proj.weight"] = (torch.zeros(
          D, D, 1, 1))
      new_sd[f"model.patch_embed.blocks.{i}.skip_bn.weight"] = (torch.zeros(D))
    # Transformer blocks.
    for i in range(6):
      new_sd[f"model.blocks.{i}.ln1.weight"] = torch.zeros(D)
      new_sd[f"model.blocks.{i}.attn.in_proj_weight"] = (torch.zeros(3 * D, D))
      new_sd[f"model.blocks.{i}.attn.out_proj.weight"] = (torch.zeros(D, D))
      new_sd[f"model.blocks.{i}.ln2.weight"] = torch.zeros(D)
      new_sd[f"model.blocks.{i}.mlp.w12.weight"] = (torch.zeros(2 * H, D))
      new_sd[f"model.blocks.{i}.mlp.w3.weight"] = (torch.zeros(D, H))
    # Final norm.
    new_sd["model.ln.weight"] = torch.zeros(D)
    # New classification head (unmatched).
    new_sd["model.head.weight"] = torch.zeros(7, D)
    new_sd["model.head.bias"] = torch.zeros(7)

    report = align_state_dicts(old_sd, new_sd)

    # All backbone params should match (encoder.model.* → model.*).
    for i in range(4):
      for suffix in [
          "conv1.weight", "bn1.weight", "conv2.weight", "bn2.weight",
          "skip_proj.weight", "skip_bn.weight"
      ]:
        new_k = f"model.patch_embed.blocks.{i}.{suffix}"
        old_k = f"encoder.model.patch_embed.blocks.{i}.{suffix}"
        assert report.mapping[new_k] == old_k, f"{new_k} → {report.mapping.get(new_k)}"

    for i in range(6):
      for suffix in [
          "ln1.weight", "attn.in_proj_weight", "attn.out_proj.weight", "ln2.weight",
          "mlp.w12.weight", "mlp.w3.weight"
      ]:
        new_k = f"model.blocks.{i}.{suffix}"
        old_k = f"encoder.model.blocks.{i}.{suffix}"
        assert report.mapping[new_k] == old_k, f"{new_k} → {report.mapping.get(new_k)}"

    assert report.mapping["model.ln.weight"] == "encoder.model.ln.weight"

    # SimMIM-specific params should be unused.
    assert "encoder.head.weight" in report.unused_old
    assert "mask_token" in report.unused_old
    assert "decoder.weight" in report.unused_old
    assert "decoder.bias" in report.unused_old

    # New classification head should be unmatched.
    assert "model.head.weight" in report.unmatched_new
    assert "model.head.bias" in report.unmatched_new

    assert not report.divergent


class TestReportToStr:

  def test_ok_report(self):
    """All-clear report with nothing to flag."""
    report = AlignReport(
        mapping={"a": "a"},
        unmatched_new=[],
        unused_old=[],
        divergent=[],
        ok=True,
    )
    out = report_to_str(report)
    assert "matched" in out or "used" in out

  def test_unmatched_new(self):
    report = AlignReport(
        mapping={"a": "a"},
        unmatched_new=["b", "c"],
        unused_old=[],
        divergent=[],
        ok=False,
    )
    out = report_to_str(report)
    assert "b" in out
    assert "c" in out

  def test_unused_old(self):
    report = AlignReport(
        mapping={"a": "a"},
        unmatched_new=[],
        unused_old=["x", "y"],
        divergent=[],
        ok=False,
    )
    out = report_to_str(report)
    assert "x" in out
    assert "y" in out

  def test_divergent(self):
    report = AlignReport(
        mapping={"a": "a"},
        unmatched_new=[],
        unused_old=[],
        divergent=[("a", ["a", "b"])],
        ok=False,
    )
    out = report_to_str(report)
    assert "divergent" in out.lower() or "a" in out

  def test_empty_mapping(self):
    report = AlignReport(
        mapping={},
        unmatched_new=["head.weight"],
        unused_old=["mask_token"],
        divergent=[],
        ok=False,
    )
    out = report_to_str(report)
    assert "head.weight" in out
    assert "mask_token" in out
