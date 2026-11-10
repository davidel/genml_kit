"""Tests for the I-JEPA predictor and end-to-end forward path.

Covers the two defects fixed while bringing I-JEPA under DDP coverage:

* ``_make_block_mask`` wrote through an ``expand``ed broadcast view (illegal
  aliasing) and indexed a flat 1-D tensor as if it were 2-D rows.  Geometry
  tests for the mask itself live in ``tests/test_pretrain_methods.py``; this
  module exercises the predictor and the full ``IJEPA.forward``.
* ``_Predictor`` gathered *target*-view indices from the *source* sequence, so
  ``torch.gather`` ran out of bounds at every image size.  It now takes
  explicit absolute ``(row, col)`` positions for context and target tokens and
  cross-attends mask queries to the context.
"""

import pytest
import torch

import genml_kit.methods.ijepa as ijepa_mod
from genml_kit.methods import METHODS
from genml_kit.methods.ijepa import _sincos_2d
from genml_kit.models.convvit.loader import load_convvit
from tests.cli_args import train_args_full

_ENCODER = dict(embed_dim=64, num_heads=4, depth=2, num_conv_layers=2)
_IMAGE_SIZE = 32


def _load_convvit(*_args, **_kwargs):
  """Stand-in for ``load_model`` returning a tiny ConvViT backbone."""
  return load_convvit(
      num_labels=0,
      id2label={},
      label2id={},
      image_size=_IMAGE_SIZE,
      device="cpu",
      **_ENCODER,
  )


def _make_args(image_size=_IMAGE_SIZE):
  args = train_args_full(["--model", "convvit", "--method", "ijepa"],
                         image_size=image_size,
                         model_arg=dict(_ENCODER),
                         cache_dir=None)
  # 512-dim predictor with 12 heads does not divide evenly at this width.
  args.predictor_dim = 64
  args.predictor_heads = 4
  return args


class TestSincos2d:

  def test_shape_and_dtype(self):
    pos = torch.tensor([[0, 0], [3, 7], [12, 1]])
    emb = _sincos_2d(pos, 16)
    assert emb.shape == (3, 16)
    assert emb.dtype == torch.float32

  def test_is_position_dependent(self):
    pos = torch.tensor([[0, 0], [0, 1]])
    emb = _sincos_2d(pos, 16)
    assert not torch.allclose(emb[0], emb[1])

  def test_size_agnostic(self):
    # Absolute coordinates well beyond any fixed grid must still encode.
    pos = torch.tensor([[0, 0], [99, 250]])
    emb = _sincos_2d(pos, 16)
    assert torch.isfinite(emb).all()
    assert not torch.allclose(emb[0], emb[1])


class TestPredictor:

  def _predictor(self):
    return ijepa_mod._Predictor(
        embed_dim=32,
        depth=2,
        num_heads=4,
        predictor_dim=32,
    )

  def test_forward_shape(self):
    predictor = self._predictor()
    context = torch.randn(2, 5, 32)
    context_pos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    target_pos = torch.tensor([[3, 3], [4, 4], [5, 5]])
    out = predictor(context, context_pos, target_pos)
    assert out.shape == (2, 3, 32)

  def test_backward_runs(self):
    predictor = self._predictor()
    context = torch.randn(2, 4, 32, requires_grad=True)
    context_pos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_pos = torch.tensor([[2, 2], [2, 3]])
    out = predictor(context, context_pos, target_pos)
    out.sum().backward()
    assert predictor.input_proj.weight.grad is not None
    assert predictor.mask_token.grad is not None

  def test_accepts_arbitrary_target_count(self):
    # A regression guard: the old gather-based predictor required the target
    # count to match the source sequence length.
    predictor = self._predictor()
    context = torch.randn(2, 4, 32)
    context_pos = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_pos = torch.tensor([[5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
    out = predictor(context, context_pos, target_pos)
    assert out.shape == (2, 5, 32)


class TestIJEPAForward:

  def _build(self, image_size=_IMAGE_SIZE):
    args = _make_args(image_size)
    method = METHODS.get("ijepa")()
    model = method.build_model(args, torch.device("cpu"))
    return method, model

  def test_forward_runs_and_is_finite(self):
    method, model = self._build()
    images = torch.randn(2, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    loss, info = model(images)
    assert torch.isfinite(loss)
    assert info["n_masked"] > 0

  def test_forward_backward_two_steps(self):
    # The second iteration is where DDP's unused-parameter check would fire.
    method, model = self._build()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(2, 3, _IMAGE_SIZE, _IMAGE_SIZE)
    for _ in range(2):
      optimizer.zero_grad()
      loss, _info = model(images)
      loss.backward()
      optimizer.step()

  @pytest.mark.parametrize("size", [16, 32])
  def test_forward_multiple_sizes(self, size):
    # Crops are derived from the image size, so several sizes must run.  The
    # backbone is built per size: ConvViT's absolute pos_embedding table is
    # fixed to the construction grid (pre-existing limitation, out of scope).
    method, model = self._build(size)
    images = torch.randn(2, 3, size, size)
    loss, _info = model(images)
    assert torch.isfinite(loss)
