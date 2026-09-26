"""Tests for pre-training method dispatch and SimMIM round-trip."""
import argparse

import pytest
import torch

from genml_kit.methods import METHODS
from genml_kit.methods.simmim import SimMIMMethod, make_mask


class TestRegistry:

  def test_list_methods(self):
    methods = METHODS.list_names()
    assert "simmim" in methods
    assert "ijepa" in methods
    assert "byol" in methods
    assert "dino" in methods
    assert "supcon" in methods

  def test_get_method(self):
    cls = METHODS.get("simmim")
    assert cls.NAME == "simmim"

  def test_unknown_method(self):
    with pytest.raises(ValueError, match="Unknown method"):
      METHODS.get("nonexistent")


class TestMakeMask:

  def test_mask_shape(self):
    images = torch.randn(2, 3, 448, 448)
    mask = make_mask(images, patch_size=16, mask_ratio=0.6)
    expected_patches = (448 // 16) * (448 // 16)
    assert mask.shape == (2, expected_patches)
    assert mask.dtype == torch.bool

  def test_mask_ratio(self):
    images = torch.randn(4, 3, 448, 448)
    mask = make_mask(images, patch_size=16, mask_ratio=0.6)
    ratios = mask.float().mean(dim=1)
    # Allow some variance due to block quantisation.
    for r in ratios:
      assert 0.3 < r < 0.9

  def test_mask_ratio_zero(self):
    images = torch.randn(2, 3, 224, 224)
    mask = make_mask(images, patch_size=16, mask_ratio=0.0)
    assert mask.sum() == 0 or mask.float().mean() < 0.05


class TestSimMIMMethod:

  def test_add_args(self):
    parser = argparse.ArgumentParser()
    method = METHODS.get("simmim")()
    method.add_args(parser)
    args = parser.parse_args([])
    assert args.mask_ratio == 0.6
    assert args.decoder_dim == 768

  def test_checkpoint_round_trip(self):
    """get_checkpoint_state / load_checkpoint_state preserves mask_ratio."""
    method = SimMIMMethod()

    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args(["--mask_ratio", "0.8"])

    # Create a minimal model with a fake mask_ratio.
    class FakeModel:
      mask_ratio = 0.8

    model = FakeModel()
    state = method.get_checkpoint_state(model, args)
    assert state["mask_ratio"] == 0.8

    # Simulate loading: reset mask_ratio, then load.  The checkpoint
    # restore touches the model only -- it must NOT write back into
    # args (plans/FIX_FRAP.md).
    model.mask_ratio = 0.6
    method.load_checkpoint_state(model, state, args)
    assert model.mask_ratio == 0.8
    # The args namespace is never mutated by the checkpoint restore: the
    # CLI flag keeps its parsed value (0.8 from --mask_ratio), while the
    # model carries the restored value (plans/FIX_FRAP.md).
    assert args.mask_ratio == 0.8

  def test_backward_compat_old_checkpoint(self):
    """Old checkpoints with _mask_ratio are handled gracefully."""
    method = SimMIMMethod()

    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([])

    class FakeModel:
      mask_ratio = 0.6
      # Old-style attribute.
      _mask_ratio = 0.75

    model = FakeModel()
    # Empty state (no method_state in checkpoint).
    method.load_checkpoint_state(model, {}, args)
    assert model.mask_ratio == 0.75
