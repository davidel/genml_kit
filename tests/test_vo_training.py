"""Unit tests for the VO training/evaluation pieces."""

import collections

import pytest
import torch

from genml_kit.geometry.similarity import SimilarityParams
from genml_kit.models.vo.vo_similar import VOModelOutput
from genml_kit.training.vo.train_vo import (
    STAGES,
    photometric_residual,
    vo_losses,
)


class _Cfg:
  w_log_s = 1.0
  w_theta = 1.0
  w_conf = 0.5
  w_photo = 0.1


class _Meta(
    collections.namedtuple("_Meta", [
        "gt",
        "gt_residual",
        "corners_src",
        "corners_dst",
    ])):
  """Stand-in for VOPairMeta, including the ground-truth corner fields."""

  def __new__(cls, log_s, theta, t, residual, corners_src, corners_dst):
    return super().__new__(
        cls, {
            "log_s": torch.tensor([log_s]),
            "theta": torch.tensor([theta]),
            "t": torch.tensor([t]),
        }, torch.tensor([residual]), corners_src, corners_dst)


def _batch(log_s=0.1, theta=0.2, t=(1.0, -1.0), residual=0.4):
  image_a = torch.rand(1, 1, 32, 32) + 0.1
  image_b = torch.rand(1, 1, 32, 32) + 0.1
  corners = torch.tensor([[[0.0, 0.0], [31.0, 0.0], [31.0, 31.0], [0.0, 31.0]]])
  # GT destination corners are produced the same way the dataset produces
  # them: fit a similarity to (src, a known-dst) pair.  Deriving them via
  # umeyama keeps them exactly consistent with ``gt`` -- which is what
  # makes "a perfect prediction scores mce=0" meaningful -- and matches
  # the dataset's own relationship rather than re-deriving the matrix
  # convention by hand (a hand-rolled matmul here silently transposed the
  # batch axis and produced nonsense).
  from genml_kit.geometry.similarity import params_to_matrix
  mat = params_to_matrix(torch.tensor([log_s], dtype=torch.float64),
                         torch.tensor([theta], dtype=torch.float64),
                         torch.tensor([t], dtype=torch.float64))
  ones = torch.ones(1, 4, 1, dtype=torch.float64)
  hom = torch.cat([corners.to(torch.float64), ones], dim=-1)  # (1, 4, 3)
  # mat is (1, 3, 3): contract the corner index, not the batch axis.
  corners_dst = torch.einsum("bij,bnj->bni", mat, hom)[..., :2]
  return {
      "image_a": image_a,
      "image_b": image_b,
      "corners": corners,
      "dc": torch.zeros_like(corners),
      "meta": _Meta(log_s, theta, t, residual, corners, corners_dst.to(corners.dtype)),
  }


def _pred(batch, log_s, theta, t, conf=0.3):
  return VOModelOutput(
      params=SimilarityParams(log_s=torch.tensor([log_s], requires_grad=True),
                              theta=torch.tensor([theta], requires_grad=True),
                              t=torch.tensor([t], requires_grad=True)),
      corners=batch["corners"],
      dc=torch.zeros_like(batch["corners"]),
      conf=torch.tensor([[conf]], requires_grad=True),
  )


def test_stages_constant_matches_plan():
  assert STAGES["supervised"] == 0
  assert STAGES["photometric"] == 1


def test_supervised_stage_has_no_photometric_part():
  batch = _batch()
  pred = _pred(batch, 0.1, 0.2, (1.0, -1.0))
  total, parts = vo_losses(pred, batch, _Cfg(), STAGES["supervised"])
  assert "photo" not in parts
  assert torch.isfinite(total)


def test_photometric_stage_adds_photo_part():
  batch = _batch()
  pred = _pred(batch, 0.1, 0.2, (1.0, -1.0))
  total, parts = vo_losses(pred, batch, _Cfg(), STAGES["photometric"])
  assert "photo" in parts
  assert torch.isfinite(total)
  total.backward()
  assert pred.params.log_s.grad is not None


def test_loss_zero_for_exact_prediction():
  """MCE = 0 when the GT corners are the src transformed by the params.

  This is the round-trip guarantee of vo/README.md sB.6: a *perfect*
  prediction must score exactly zero MCE.  It used to pass a
  ``corners_dst`` batch key, which ``_dst`` ignored in favour of the
  meta; the ground-truth corners now live on the meta where the
  dataset actually produces them.
  """
  batch = _batch(log_s=0.3, theta=0.5, t=(2.0, 3.0), residual=0.0)
  pred = _pred(batch, 0.3, 0.5, (2.0, 3.0))
  total, parts = vo_losses(pred, batch, _Cfg(), STAGES["supervised"])
  # float32 corners vs. float64 fit: agreement is to float32 precision.
  assert parts["mce"].item() == pytest.approx(0.0, abs=1e-5)
  assert parts["dlog_s"].item() == pytest.approx(0.0, abs=1e-6)
  assert parts["dtheta"].item() == pytest.approx(0.0, abs=1e-6)


def test_mce_is_large_for_a_wrong_prediction():
  """MCE must grow when the prediction is wrong.

  Regression test for the original defect: ``vo_metrics_from_output``
  defaulted ``src``/``dst`` to the network's *own* corner offsets, so
  ``mce`` measured how well the Umeyama solve reproduced its own input.
  It stayed near 0.03 no matter how wrong the prediction was, and the
  epoch summary then reported that constant as if it were accuracy.
  """
  from genml_kit.training.vo.train_vo import vo_metrics_from_output
  batch = _batch(log_s=0.3, theta=0.5, t=(2.0, 3.0), residual=0.0)
  gt = batch["meta"].gt
  good = _pred(batch, 0.3, 0.5, (2.0, 3.0))
  bad = _pred(batch, -0.3, -0.5, (-2.0, -3.0))
  src, dst = batch["meta"].corners_src, batch["meta"].corners_dst
  good_mce = vo_metrics_from_output(good, gt, batch["meta"].gt_residual, src, dst).mce
  bad_mce = vo_metrics_from_output(bad, gt, batch["meta"].gt_residual, src, dst).mce
  assert bad_mce > 5.0, "a grossly wrong prediction must not score ~0.03"
  assert bad_mce > good_mce


def test_mce_rejects_missing_ground_truth_corners():
  """Calling without GT corners must fail loudly, not silently self-score.

  ``src``/``dst`` are required positional parameters, so omitting them is
  a TypeError; passing an explicit ``None`` hits the guard.  Either way
  the call cannot silently fall back to the prediction's own offsets.
  """
  from genml_kit.training.vo.train_vo import vo_metrics_from_output
  batch = _batch()
  pred = _pred(batch, 0.1, 0.2, (1.0, -1.0))
  with pytest.raises(TypeError):
    vo_metrics_from_output(pred, batch["meta"].gt, batch["meta"].gt_residual)
  with pytest.raises(ValueError, match="ground-truth corners"):
    vo_metrics_from_output(pred, batch["meta"].gt, batch["meta"].gt_residual, None,
                           None)


def test_dataset_gt_corners_span_image_and_match_gt():
  """The dataset must expose true pixel corners, and they must be real.

  ``corners_dst`` is the homography destination, so re-deriving it from
  the fitted similarity must land within ``gt_residual`` -- the
  irreducible foreshortening the similarity family cannot absorb.
  """
  from genml_kit.datasets.vo_pairs import VOPairDataset
  ds = VOPairDataset(length=4, size=(64, 64), seed=0)
  item = ds[0]
  meta = item["meta"]
  assert meta.corners_src.shape == (4, 2)
  assert meta.corners_dst.shape == (4, 2)
  assert torch.allclose(
      meta.corners_src,
      torch.tensor([[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]]))
  from genml_kit.geometry.similarity import params_to_matrix
  gt = meta.gt
  mat = params_to_matrix(gt["log_s"].reshape(1).to(torch.float64),
                         gt["theta"].reshape(1).to(torch.float64),
                         gt["t"].reshape(1, 2).to(torch.float64))
  ones = torch.ones(1, 4, 1, dtype=torch.float64)
  hom = torch.cat([meta.corners_src.to(torch.float64).unsqueeze(0), ones],
                  dim=-1)  # (1, 4, 3)
  fit = torch.einsum("bij,bnj->bni", mat, hom)[..., :2]
  err = (fit[0] - meta.corners_dst.to(torch.float64)).norm(dim=-1).mean()
  assert err <= meta.gt_residual.to(torch.float64).item() + 1e-3


def test_photometric_residual_bounds():
  torch.manual_seed(0)
  image_a = torch.rand(2, 1, 24, 24)
  image_b = torch.rand(2, 1, 24, 24)
  params = SimilarityParams(log_s=torch.zeros(2),
                            theta=torch.zeros(2),
                            t=torch.zeros(2, 2))
  residual = photometric_residual(image_a, image_b, params)
  assert residual.shape == (2,)
  assert torch.all((residual >= 0) & (residual <= 2))


def test_photometric_residual_zero_for_identical_frames():
  image = torch.rand(1, 1, 24, 24) + 0.1
  params = SimilarityParams(log_s=torch.zeros(1),
                            theta=torch.zeros(1),
                            t=torch.zeros(1, 2))
  residual = photometric_residual(image, image, params)
  assert residual.item() == pytest.approx(0.0, abs=1e-5)
