"""Unit tests for the VO training/evaluation pieces."""

import pytest
import torch

from genml_kit.geometry.similarity import SimilarityParams
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


class _Meta:

  def __init__(self, log_s, theta, t, residual):
    self.gt = {
        "log_s": torch.tensor([log_s]),
        "theta": torch.tensor([theta]),
        "t": torch.tensor([t]),
    }
    self.gt_residual = torch.tensor([residual])


def _batch(log_s=0.1, theta=0.2, t=(1.0, -1.0), residual=0.4):
  image_a = torch.rand(1, 1, 32, 32) + 0.1
  image_b = torch.rand(1, 1, 32, 32) + 0.1
  corners = torch.tensor([[[0.0, 0.0], [31.0, 0.0], [31.0, 31.0], [0.0, 31.0]]])
  return {
      "image_a": image_a,
      "image_b": image_b,
      "corners": corners,
      "dc": torch.zeros_like(corners),
      "meta": _Meta(log_s, theta, t, residual),
  }


def _pred(batch, log_s, theta, t, conf=0.3):
  return {
      "params":
          SimilarityParams(log_s=torch.tensor([log_s], requires_grad=True),
                           theta=torch.tensor([theta], requires_grad=True),
                           t=torch.tensor([t], requires_grad=True)),
      "conf":
          torch.tensor([[conf]], requires_grad=True),
  }


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
  assert pred["params"].log_s.grad is not None


def test_loss_zero_for_exact_prediction():
  """MCE = 0 when dst is exactly the transformed src by the params."""
  batch = _batch(log_s=0.3, theta=0.5, t=(2.0, 3.0), residual=0.0)
  pred = _pred(batch, 0.3, 0.5, (2.0, 3.0))
  from genml_kit.geometry.similarity import params_to_matrix
  mat = params_to_matrix(pred["params"].log_s, pred["params"].theta, pred["params"].t)
  ones = torch.ones_like(batch["corners"][..., :1])
  batch["corners_dst"] = (
      mat @ torch.cat([batch["corners"], ones], dim=-1).transpose(1, 2)).transpose(
          1, 2)[..., :2]
  total, parts = vo_losses(pred, batch, _Cfg(), STAGES["supervised"])
  assert parts["mce"].item() == pytest.approx(0.0, abs=1e-6)
  assert parts["dlog_s"].item() == pytest.approx(0.0, abs=1e-6)
  assert parts["dtheta"].item() == pytest.approx(0.0, abs=1e-6)


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
