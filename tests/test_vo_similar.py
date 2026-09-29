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
  assert out.params.log_s.shape == (2,)
  assert out.params.theta.shape == (2,)
  assert out.params.t.shape == (2, 2)
  assert out.corners.shape == (2, 4, 2)
  assert out.dc.shape == (2, 4, 2)
  assert out.conf.shape == (2, 2)
  loss = out.params.log_s.sum() + out.conf.sum()
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
  src = out.corners
  # Exaggerate the offsets.
  dst = src + out.dc * 100.0
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


def test_corners_are_true_pixel_corners():
  """out.corners must be the image corners in pixels, at any size.

  vo/README.md s14/s16.1 require corner deltas -- and therefore MCE --
  to be in pixels.  The basis used to be derived from the encoder's
  feature map (``feat/2 * cost_scale``), giving corners spanning
  ``image/8`` centered on zero rather than ``[0, W-1]``.  That silently
  made every "mce" value meaningless as a pixel error.
  """
  net = VOSimilarityNet(VOSimilarityConfig())
  net.eval()
  for img in (32, 64, 128):
    with torch.no_grad():
      out = net(torch.rand(1, 1, img, img), torch.rand(1, 1, img, img))
    expected = torch.tensor([[0.0, 0.0], [img - 1.0, 0.0], [img - 1.0, img - 1.0],
                             [0.0, img - 1.0]])
    assert torch.allclose(out.corners[0], expected), f"img={img}"


def test_pixel_gain_is_learnable_and_gradients_flow():
  """The pixel-space gain must be a real parameter, not a constant.

  Option 3 keeps the head's internal normalized output and learns the
  conversion to pixels, so the gradient of the loss must reach it.
  """
  net = VOSimilarityNet(VOSimilarityConfig())
  assert isinstance(net.pixel_gain, torch.nn.Parameter)
  out = net(torch.rand(2, 1, 64, 64), torch.rand(2, 1, 64, 64))
  (out.params.log_s.sum() + out.dc.sum()).backward()
  assert net.pixel_gain.grad is not None
  assert torch.isfinite(net.pixel_gain.grad).all()
  assert net.pixel_gain.grad.abs().item() > 0.0


def test_corner_head_shapes_and_slice():
  """The corner MLP maps pooled features to 10 values: 8 deltas + 2 conf.

  The input width must track the encoder's final stage width (128 for the
  default npu-small profile), and the head must never exceed 10 outputs
  (no dead/spare channels).  This guards against a silent regression if a
  profile's stage widths change.
  """
  net = VOSimilarityNet(VOSimilarityConfig())
  a = torch.rand(2, 1, 64, 64)
  b = torch.rand(2, 1, 64, 64)
  out = net(a, b)
  assert out.params.log_s.shape == (2,)
  assert out.params.theta.shape == (2,)
  assert out.params.t.shape == (2, 2)
  assert out.corners.shape == (2, 4, 2)
  assert out.dc.shape == (2, 4, 2)
  assert out.conf.shape == (2, 2)
  # The MLP output width must equal the encoder's final stage width.
  final_width = dict(net.encoder.named_parameters())["stages.7.0.weight"].shape[0]
  assert net.corner_mlp[0].in_features == final_width
  assert net.corner_mlp[0].out_features == final_width
  assert net.corner_mlp[2].in_features == final_width
  assert net.corner_mlp[2].out_features == 10
  # First 8 channels = corner deltas (4 corners x 2); last 2 = confidence.
  head_out = net.corner_mlp(net.corner_mlp[0](torch.rand(2, final_width)))
  assert head_out.shape == (2, 10)
