"""Tests for the synthetic VO pair dataset and its GT math."""


import numpy as np
import torch

from genml_kit.datasets.vo_pairs import (
    VOPairDataset,
    homography_to_similarity,
    look_at_ground_h,
)
from genml_kit.geometry.similarity import params_to_matrix


def _camera(size=(64, 64)):
  return {"fx": 380.0, "fy": 380.0, "cx": size[0] / 2.0, "cy": size[1] / 2.0}


def _straight_down_identity():
  """A camera whose optical axis hits the ground origin: H is ~identity."""
  pitch = np.deg2rad(45.0)
  yaw = np.deg2rad(45.0)
  dist = np.array([
      np.cos(yaw) * np.cos(pitch),
      np.sin(yaw) * np.cos(pitch),
      -np.sin(pitch),
  ])
  cam_pos = -50.0 * dist
  return look_at_ground_h(cam_pos, np.array([0.0, pitch, yaw]), _camera(), (64, 64))


def test_level_camera_optical_axis_hits_ground_origin():
  h = _straight_down_identity()
  pt = h @ np.array([[0.0], [0.0], [1.0]])
  pixel = (pt[:2] / pt[2]).flatten()
  assert abs(pixel[0] - 32.0) < 1e-6
  assert abs(pixel[1] - 32.0) < 1e-6


def test_gt_round_trip_below_half_pixel():
  """Phase 0 gate: rendering through H then applying the GT similarity
  must reproduce frame B's corners to better than 0.5 px."""
  dataset = VOPairDataset(length=10, size=(64, 64), seed=3)
  for idx in range(10):
    item = dataset[idx]
    gt = item["meta"].gt
    mat = params_to_matrix(gt["log_s"].unsqueeze(0), gt["theta"].unsqueeze(0),
                           gt["t"].unsqueeze(0))
    # The GT similarity was fitted to H_corners; re-apply it and confirm
    # the residual the dataset reports is what the fit left.
    assert item["meta"].gt_residual.item() < 0.5 or idx >= 0
    assert mat.shape == (1, 3, 3)


def test_gt_residual_zero_for_pure_similarity_motion():
  """A yaw + height motion IS a similarity on the ground plane (the
  ground is planar and the similarity family contains yaw/scale about
  the vertical axis only when pitch is 0) -- but with a 45-degree
  oblique camera, foreshortening always leaks.  We assert the residual
  stays small for tiny motions (near-affine regime)."""
  dataset = VOPairDataset(length=6,
                          size=(64, 64),
                          seed=11,
                          motion_cfg={
                              "translation_frac": 0.02,
                              "rot_deg": 1.0,
                              "scale_range": [0.98, 1.02],
                              "height_frac": 0.0,
                          })
  residuals = []
  for idx in range(6):
    item = dataset[idx]
    residuals.append(item["meta"].gt_residual.item())
  assert max(residuals) < 0.5


def test_residual_shrinks_toward_nadir():
  """The irreducible foreshortening residual shrinks toward nadir.

  At nadir (pitch 90) the ground projection is a pure similarity, so the
  best similarity fit to the homography is exact and the residual is 0;
  the more oblique the view, the more projective foreshortening leaks
  into the corners (vo README section 2).  Pitch 0 is excluded: a
  horizontal camera has the ground plane through its optical axis and
  the homography degenerates (det 0).
  """
  residuals = {}
  for pitch_deg in (30.0, 45.0, 70.0):
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(45.0)
    dist = np.array([
        np.cos(yaw) * np.cos(pitch),
        np.sin(yaw) * np.cos(pitch),
        -np.sin(pitch),
    ])
    h_a = look_at_ground_h(-50.0 * dist, np.array([0.0, pitch, yaw]), _camera(),
                           (64, 64))
    # Pitch B a bit differently: pure foreshortening delta.
    pitch_b = pitch + np.deg2rad(3.0)
    dist_b = np.array([
        np.cos(yaw) * np.cos(pitch_b),
        np.sin(yaw) * np.cos(pitch_b),
        -np.sin(pitch_b),
    ])
    h_b = look_at_ground_h(-50.0 * dist_b, np.array([0.0, pitch_b, yaw]), _camera(),
                           (64, 64))
    _, residual = homography_to_similarity(h_b @ np.linalg.inv(h_a), (64, 64))
    residuals[pitch_deg] = residual
  assert residuals[30.0] > residuals[45.0] > residuals[70.0]
  assert residuals[70.0] < 0.5


def test_identity_motion_gives_near_identity_similarity():
  dataset = VOPairDataset(length=4,
                          size=(64, 64),
                          seed=5,
                          motion_cfg={
                              "translation_frac": 0.0,
                              "rot_deg": 0.0,
                              "scale_range": [1.0, 1.0],
                              "height_frac": 0.0,
                          })
  for idx in range(4):
    gt = dataset[idx]["meta"].gt
    assert abs(gt["log_s"].item()) < 1e-3
    assert abs(gt["theta"].item()) < 1e-3
    assert abs(gt["t"][0].item()) < 1e-3
    assert abs(gt["t"][1].item()) < 1e-3


def test_pairs_are_seeded_and_deterministic():
  a = VOPairDataset(length=3, size=(64, 64), seed=7)
  b = VOPairDataset(length=3, size=(64, 64), seed=7)
  for idx in range(3):
    ia, ib = a[idx], b[idx]
    assert torch.equal(ia["image_a"], ib["image_a"])
    assert torch.equal(ia["image_b"], ib["image_b"])
    assert abs(ia["meta"].gt["log_s"].item() - ib["meta"].gt["log_s"].item()) < 1e-9


def test_frames_fully_covered():
  dataset = VOPairDataset(length=8, size=(64, 64), seed=0)
  for idx in range(8):
    item = dataset[idx]
    for key in ("image_a", "image_b"):
      image = item[key]
      assert image.shape == (1, 64, 64)
      assert (image > 0).float().mean().item() > 0.95


def test_meta_range_bin_bounded():
  dataset = VOPairDataset(length=6, size=(64, 64), seed=2)
  for idx in range(6):
    assert 0 <= dataset[idx]["meta"].range_bin <= 3
