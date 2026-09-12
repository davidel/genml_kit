"""Unit tests for the 2-D similarity-transform algebra.

Every numeric case here doubles as a documented example in
``vo/README.md`` sections 1 and 3, so doc and tests cannot drift.
"""

import math

import torch

from genml_kit.geometry.similarity import (
    SimilarityParams,
    corner_residual,
    params_from_matrix,
    params_to_matrix,
    umeyama_similarity,
    warp_similarity,
    wrap_angle,
)

BATCH = 7
POINTS = 11


def _params(log_s, theta, tx, ty):
  return SimilarityParams(log_s=torch.full((BATCH,), log_s, dtype=torch.float64),
                          theta=torch.full((BATCH,), theta, dtype=torch.float64),
                          t=torch.tensor([[tx, ty]],
                                         dtype=torch.float64).expand(BATCH, 2))


def _random_params(generator):
  log_s = torch.empty((BATCH,), dtype=torch.float64).uniform_(math.log(0.5),
                                                              math.log(2.0),
                                                              generator=generator)
  theta = torch.empty((BATCH,), dtype=torch.float64).uniform_(-math.pi,
                                                              math.pi,
                                                              generator=generator)
  t = torch.empty((BATCH, 2), dtype=torch.float64).uniform_(-40.0,
                                                            40.0,
                                                            generator=generator)
  return SimilarityParams(log_s=log_s, theta=theta, t=t)


def _apply(params, pts):
  mat = params_to_matrix(params.log_s, params.theta, params.t)
  ones = torch.ones_like(pts[..., :1])
  return torch.einsum("bij,bnj->bni", mat, torch.cat([pts, ones], dim=-1))[..., :2]


def test_params_matrix_round_trip():
  generator = torch.Generator().manual_seed(0)
  params = _random_params(generator)
  recovered = params_from_matrix(params_to_matrix(params.log_s, params.theta, params.t))
  assert torch.allclose(recovered.log_s, params.log_s, atol=1e-12)
  assert torch.allclose(recovered.t, params.t, atol=1e-12)
  assert torch.allclose(wrap_angle(recovered.theta - params.theta),
                        torch.zeros(BATCH, dtype=torch.float64),
                        atol=1e-12)


def test_quarter_turn_sign_convention():
  """The worked 90-degree case from plan section 9.1 / vo README section 1.

  Counter-clockwise quarter turn about the origin must send +x onto +y
  with unit scale; scale/angle readout must not fold the sign.
  """
  params = _params(0.0, math.pi / 2, 0.0, 0.0)
  pts = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=torch.float64)
  out = _apply(params, pts)
  expected = torch.tensor([[[0.0, 1.0], [-1.0, 0.0], [-1.0, 1.0]]], dtype=torch.float64)
  assert torch.allclose(out, expected, atol=1e-12)
  read = params_from_matrix(params_to_matrix(params.log_s, params.theta, params.t))
  assert torch.allclose(read.theta, params.theta)
  assert torch.allclose(read.log_s, params.log_s)


def test_wrap_angle_branch_cut():
  """Values inside (-pi, pi] pass through; beyond, reduce modulo 2*pi."""
  wrapped = wrap_angle(torch.tensor([math.pi - 0.1, -math.pi + 0.1, 3 * math.pi]))
  assert torch.allclose(wrapped[:2],
                        torch.tensor([math.pi - 0.1, -math.pi + 0.1]),
                        atol=1e-12)
  assert torch.allclose(wrapped[2], torch.tensor(-math.pi), atol=1e-12)


def test_compose_matches_matrix_product():
  generator = torch.Generator().manual_seed(1)
  first = _random_params(generator)
  second = _random_params(generator)
  pts = torch.empty((BATCH, POINTS, 2),
                    dtype=torch.float64).uniform_(-10.0, 10.0, generator=generator)
  via_params = _apply(second, _apply(first, pts))
  mat = torch.bmm(params_to_matrix(second.log_s, second.theta, second.t),
                  params_to_matrix(first.log_s, first.theta, first.t))
  via_matrix = torch.einsum("bij,bnj->bni", mat,
                            torch.cat([pts, torch.ones_like(pts[..., :1])],
                                      dim=-1))[..., :2]
  assert torch.allclose(via_params, via_matrix, atol=1e-9)


def test_umeyama_recovers_exact_transform():
  generator = torch.Generator().manual_seed(2)
  params = _random_params(generator)
  src = torch.empty((BATCH, POINTS, 2),
                    dtype=torch.float64).uniform_(-50.0, 50.0, generator=generator)
  dst = _apply(params, src)
  fit = umeyama_similarity(src, dst)
  assert torch.allclose(fit.log_s, params.log_s, atol=1e-9)
  assert torch.allclose(fit.t, params.t, atol=1e-9)
  assert torch.allclose(wrap_angle(fit.theta - params.theta),
                        torch.zeros(BATCH, dtype=torch.float64),
                        atol=1e-9)


def test_umeyama_matches_brute_force_least_squares():
  generator = torch.Generator().manual_seed(3)
  src = torch.empty((1, 40, 2), dtype=torch.float64).uniform_(-1.0,
                                                              1.0,
                                                              generator=generator)
  noise = torch.empty((1, 40, 2), dtype=torch.float64).normal_(std=0.01,
                                                               generator=generator)
  dst = _apply(_params(0.3, 0.7, 5.0, -2.0), src) + noise

  # Independent dense least-squares reference: stack both coordinates into
  # one 80-row system over (a, b, tx, ty) with
  # x' = a x - b y + tx,  y' = b x + a y + ty.
  x = src[0, :, 0]
  y = src[0, :, 1]
  xstar = dst[0, :, 0]
  ystar = dst[0, :, 1]
  zeros = torch.zeros_like(x)
  ones = torch.ones_like(x)
  amat = torch.cat([
      torch.stack([x, -y, ones, zeros], dim=-1),
      torch.stack([y, x, zeros, ones], dim=-1),
  ],
                   dim=0)
  bvec = torch.cat([xstar, ystar], dim=0)
  sol = torch.linalg.lstsq(amat, bvec).solution
  a_ref, b_ref = sol[0], sol[1]

  fit = umeyama_similarity(src, dst)
  s = torch.exp(fit.log_s[0])
  assert torch.allclose(s, torch.hypot(a_ref, b_ref), atol=1e-9)
  assert torch.allclose(fit.theta[0], torch.atan2(b_ref, a_ref), atol=1e-9)


def test_umeyama_weighted_ignores_zero_weight_points():
  generator = torch.Generator().manual_seed(4)
  params = _params(0.2, -0.5, 3.0, 4.0)
  src = torch.empty((BATCH, POINTS, 2),
                    dtype=torch.float64).uniform_(-50.0, 50.0, generator=generator)
  dst = _apply(params, src)
  weights = torch.ones((BATCH, POINTS), dtype=torch.float64)
  outlier = _apply(params, src[:, :1]) + 1e3
  polluted = torch.cat([dst[:, :1], dst[:, 1:]], dim=1)
  weights[:, 0] = 0.0
  polluted[:, 0] = outlier[:, 0]
  fit = umeyama_similarity(src, polluted, weights=weights)
  clean = umeyama_similarity(src[:, 1:], dst[:, 1:])
  assert torch.allclose(fit.log_s, clean.log_s, atol=1e-9)
  assert torch.allclose(fit.t, clean.t, atol=1e-9)


def test_umeyama_never_emits_reflection():
  generator = torch.Generator().manual_seed(5)
  src = torch.empty((1, 12, 2), dtype=torch.float64).uniform_(-1.0,
                                                              1.0,
                                                              generator=generator)
  mirrored = src * torch.tensor([1.0, -1.0])
  fit = umeyama_similarity(src, mirrored)
  mat = params_to_matrix(fit.log_s, fit.theta, fit.t)[:, :2, :2]
  dets = torch.det(mat)
  assert torch.all(dets >= 0)


def test_umeyama_gradients_flow():
  src = torch.randn(BATCH, POINTS, 2, dtype=torch.float64)
  dst = torch.randn(BATCH, POINTS, 2, dtype=torch.float64)
  src.requires_grad_(True)
  dst.requires_grad_(True)
  fit = umeyama_similarity(src, dst)
  (fit.log_s.sum() + fit.theta.sum() + fit.t.sum()).backward()
  assert src.grad is not None and torch.all(torch.isfinite(src.grad))
  assert dst.grad is not None and torch.all(torch.isfinite(dst.grad))


def test_warp_identity_round_trip():
  generator = torch.Generator().manual_seed(6)
  img = torch.rand((2, 3, 24, 32), generator=generator)
  identity = SimilarityParams(log_s=torch.zeros((2,)),
                              theta=torch.zeros((2,)),
                              t=torch.zeros((2, 2)))
  out = warp_similarity(img, identity, (24, 32))
  assert out.shape == img.shape
  interior = img[..., 2:-2, 2:-2]
  assert torch.allclose(out[..., 2:-2, 2:-2], interior, atol=1e-5)


def test_warp_similarity_matches_point_map():
  """Warp by M then sample at mapped points == sample source at points."""
  generator = torch.Generator().manual_seed(7)
  params = SimilarityParams(log_s=torch.empty(
      (BATCH,), dtype=torch.float64).uniform_(math.log(0.7),
                                              math.log(1.4),
                                              generator=generator),
                            theta=torch.empty(
                                (BATCH,),
                                dtype=torch.float64).uniform_(-1.0,
                                                              1.0,
                                                              generator=generator),
                            t=torch.empty(
                                (BATCH, 2),
                                dtype=torch.float64).uniform_(-3.0,
                                                              3.0,
                                                              generator=generator))
  # Smooth (band-limited) image: resampling then resampling again must
  # track the direct pullback closely; with per-pixel noise the two
  # bilinear compositions are legitimately different signals.
  coords = torch.arange(32, dtype=torch.float64)
  img = (0.5 + 0.5 * torch.sin(0.21 * coords)[None, None, None, :] *
         torch.cos(0.17 * coords)[None, None, :, None]).expand(BATCH, 1, 32,
                                                               32).contiguous()
  # Keep points well inside the frame: the warped image is zero-padded
  # outside, and the two sampling paths only agree where no padding and
  # no bilinear edge clamp is involved.
  src = torch.empty((BATCH, 32, 2), dtype=torch.float64).uniform_(8.0,
                                                                  23.0,
                                                                  generator=generator)
  warped = warp_similarity(img, params, (32, 32))
  ones = torch.ones_like(src[..., :1])
  mapped = torch.einsum("bij,bnj->bni",
                        params_to_matrix(params.log_s, params.theta, params.t),
                        torch.cat([src, ones], dim=-1))[..., :2]
  # grid_sample with align_corners=True samples at normalized coords of
  # the pixel-center grid, i.e. index (x, y) maps to (2x/(W-1)-1, ...).
  # The warp resamples via the backward map, so the warped image at
  # output pixel p holds the source content at inv(M) @ p.  Sampling
  # the warped image at inv(M) @ src must therefore reproduce the
  # source content at src.  (Earlier forward-map interpretation was
  # the sign bug this test originally had.)
  # sample(W_M img, r) == sample(img, inv(M) @ r) by construction, so
  # reading the warp at the forward-mapped points reproduces the source
  # content at src.  Compare only where both read points sit inside the
  # frame: the warp zero-pads content pushed outside the viewport.
  keep = ((mapped > 2.0) & (mapped < 29.0)).all(dim=-1)
  assert keep.any()
  gx_map = 2.0 * mapped[..., 0] / 31.0 - 1.0
  gy_map = 2.0 * mapped[..., 1] / 31.0 - 1.0
  picked_from_warp = torch.nn.functional.grid_sample(
      warped,
      torch.stack([gx_map, gy_map], dim=-1).unsqueeze(-2),
      mode="bilinear",
      padding_mode="zeros",
      align_corners=True).squeeze(-2).squeeze(1)
  gx_src = 2.0 * src[..., 0] / 31.0 - 1.0
  gy_src = 2.0 * src[..., 1] / 31.0 - 1.0
  sampled_at_src = torch.nn.functional.grid_sample(
      img,
      torch.stack([gx_src, gy_src], dim=-1).unsqueeze(-2),
      mode="bilinear",
      padding_mode="zeros",
      align_corners=True).squeeze(-2).squeeze(1)
  # Float32 resampling of a band-limited signal agrees with the direct
  # pullback only to a few 1e-3 -- the composition of two bilinear
  # interpolants is not the identity interpolation of the same signal.
  assert torch.allclose(sampled_at_src[keep], picked_from_warp[keep], atol=2e-2)


def test_warp_gradients_flow():
  img = torch.rand((1, 1, 16, 16), requires_grad=True)
  params = SimilarityParams(log_s=torch.tensor([0.1], requires_grad=True),
                            theta=torch.tensor([0.05], requires_grad=True),
                            t=torch.tensor([[1.0, -1.0]], requires_grad=True))
  out = warp_similarity(img, params, (16, 16))
  out.sum().backward()
  assert img.grad is not None and torch.all(torch.isfinite(img.grad))
  for field in (params.log_s, params.theta, params.t):
    assert field.grad is not None and torch.all(torch.isfinite(field.grad))


def test_corner_residual_zero_for_exact_fit():
  generator = torch.Generator().manual_seed(8)
  params = _random_params(generator)
  src = torch.empty((BATCH, 4, 2), dtype=torch.float64).uniform_(0.0,
                                                                 100.0,
                                                                 generator=generator)
  assert torch.allclose(corner_residual(params, src, _apply(params, src)),
                        torch.zeros(BATCH, dtype=torch.float64),
                        atol=1e-9)


def test_corner_residual_matches_manual():
  generator = torch.Generator().manual_seed(9)
  params = _random_params(generator)
  src = torch.empty((BATCH, 4, 2), dtype=torch.float64).uniform_(0.0,
                                                                 100.0,
                                                                 generator=generator)
  manual = (_apply(params, src) - src).norm(dim=-1).mean(dim=-1)
  assert torch.allclose(corner_residual(params, src, src), manual)
