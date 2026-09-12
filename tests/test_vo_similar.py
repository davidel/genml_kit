"""Unit tests for the VO similarity model pieces."""

import torch

from genml_kit.geometry.similarity import params_to_matrix
from genml_kit.models.vo.vo_similar import (
    VOSimilarityConfig,
    VOSimilarityNet,
    correlate,
)


def test_correlate_matches_direct_semantics():
  torch.manual_seed(0)
  fa = torch.rand(2, 3, 6, 6)
  fb = torch.rand(2, 3, 6, 6)
  padded = torch.nn.functional.pad(fb, (2, 2, 2, 2))
  disps = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)]
  vol = correlate(fa, fb, 2)
  assert vol.shape == (2, 25, 6, 6)
  for k, (dy, dx) in enumerate(disps):
    ref = (fa * padded[:, :, 2 + dy:2 + dy + 6, 2 + dx:2 + dx + 6]).sum(dim=1)
    assert torch.isclose(vol[:, k], ref, atol=1e-5).all()


def test_correlate_zero_outside_frame():
  """Displacements pointing out of frame produce exact zero scores."""
  fa = torch.ones(1, 1, 4, 4)
  fb = torch.ones(1, 1, 4, 4)
  vol = correlate(fa, fb, 1)
  # channel k=0 is (dy, dx) = (-1, -1): at pixel (0, 0) the partner pixel
  # is (-1, -1), outside the frame -> score 0.
  assert vol[0, 0, 0, 0].item() == 0.0
  # (dy, dx) = (0, 0) everywhere inside frame -> score 1.
  assert vol[0, 4, :, :].min().item() == 1.0


def test_network_forward_backward_shapes():
  net = VOSimilarityNet(VOSimilarityConfig())
  a = torch.rand(2, 1, 128, 128)
  b = torch.rand(2, 1, 128, 128)
  out = net(a, b)
  assert out["params"].log_s.shape == (2,)
  assert out["params"].theta.shape == (2,)
  assert out["params"].t.shape == (2, 2)
  assert out["corners"].shape == (2, 4, 2)
  assert out["dc"].shape == (2, 4, 2)
  assert out["conf"].shape == (2, 2)
  loss = out["params"].log_s.sum() + out["conf"].sum()
  loss.backward()


def test_network_output_is_valid_similarity():
  """The closed-form solve guarantees a valid (theta, s) for ANY offsets.

  Random adversarial corner offsets (mirrored geometry, huge jumps) must
  still yield a rotation matrix with positive determinant and finite,
  positive scale -- the network cannot emit an invalid similarity.
  """
  net = VOSimilarityNet(VOSimilarityConfig())
  net.eval()
  with torch.no_grad():
    net.corner_mlp[0].reset_parameters()
    net.corner_mlp[2].reset_parameters()
  a = torch.rand(3, 1, 128, 128)
  b = torch.rand(3, 1, 128, 128)
  out = net(a, b)
  src = out["corners"]
  dst = src + out["dc"] * 100.0  # exaggerate the offsets
  from genml_kit.geometry.similarity import umeyama_similarity
  params = umeyama_similarity(src, dst)
  scale = torch.exp(params.log_s)
  assert torch.all(torch.isfinite(scale))
  assert torch.all(scale > 0)
  # The plan invariant: the linear part is s * R with R a *proper*
  # rotation, i.e. R^T R = s^2 I (never -s^2, which would be a mirror).
  # det(sR) = s^2 >= 0 for every output by construction.
  mat = params_to_matrix(params.log_s, params.theta, params.t)
  rot = mat[:, :2, :2]
  rtr = rot.transpose(1, 2) @ rot
  s_sq = torch.exp(params.log_s)**2
  expected = s_sq.view(-1, 1, 1) * torch.eye(2).expand_as(rtr)
  assert torch.allclose(rtr, expected, rtol=1e-3, atol=1e-3)


def test_profile_param_counts_ordered():
  """The three profiles grow monotonically in capacity."""
  counts = []
  for profile in ("npu-small", "edge-mid", "station"):
    in_ch = 1 if profile != "station" else 3
    net = VOSimilarityNet(VOSimilarityConfig(profile=profile, in_ch=in_ch))
    net(torch.rand(1, in_ch, 128, 128), torch.rand(1, in_ch, 128, 128))
    counts.append(sum(p.numel() for p in net.parameters()))
  assert counts[0] < counts[1] < counts[2]
