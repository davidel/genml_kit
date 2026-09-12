"""2-D similarity-transform algebra for the visual-odometry front-end.

A similarity is parameterized by ``(log_s, theta, t)`` and acts on image
points as ``p' = s * R(theta) @ p + t``.  Scale is carried in log space and
angles in radians; both conventions (and the sign conventions pinned by the
worked 90-degree rotation case in ``vo/README.md`` section 1) are the single
source of truth for every consumer of this module.

All functions are batched over a leading dimension and differentiable with
respect to their tensor inputs unless stated otherwise.  See ``vo/README.md``
sections 1 (parameterization) and 3 (Umeyama derivation) for the math.
"""

import collections

import torch

SimilarityParams = collections.namedtuple("SimilarityParams", ["log_s", "theta", "t"])


def params_to_matrix(log_s, theta, t):
  """Build the homogeneous 3x3 similarity matrices from parameters.

  Args:
      log_s: (B,) natural log of the uniform scale.
      theta: (B,) rotation angle in radians (counter-clockwise).
      t: (B, 2) translation applied after scaling and rotation.

  Returns:
      (B, 3, 3) homogeneous matrices with the (0, 0, 1) bottom row.
  """
  batch = log_s.shape[0]
  scale = torch.exp(log_s)
  cos = torch.cos(theta)
  sin = torch.sin(theta)
  zero = torch.zeros_like(cos)
  one = torch.ones_like(cos)
  rot_scale = torch.stack([scale * cos, -scale * sin, scale * sin, scale * cos],
                          dim=-1).view(batch, 2, 2)
  top = torch.cat([rot_scale, t.unsqueeze(-1)], dim=-1)
  bottom = torch.stack([zero, zero, one], dim=-1).unsqueeze(1)
  return torch.cat([top, bottom], dim=1)


def params_from_matrix(mat):
  """Read ``(log_s, theta, t)`` back out of homogeneous similarity matrices.

  The angle is recovered as ``atan2(M[1, 0], M[0, 0])`` so the full
  (-pi, pi] range maps back one-to-one; an ``arccos``-based readout would
  silently fold negative angles onto positive ones.

  Args:
      mat: (B, 2, 3) or (B, 3, 3) similarity matrices.

  Returns:
      SimilarityParams with fields shaped (B,), (B,), (B, 2).
  """
  a = mat[:, 0, 0]
  b = mat[:, 1, 0]
  theta = torch.atan2(b, a)
  log_s = 0.5 * torch.log(a * a + b * b)
  t = torch.stack([mat[:, 0, 2], mat[:, 1, 2]], dim=-1)
  return SimilarityParams(log_s=log_s, theta=theta, t=t)


def wrap_angle(delta):
  """Wrap angle differences into (-pi, pi].

  Args:
      delta: (...,) angle values in radians.

  Returns:
      Tensor of the same shape with values in (-pi, pi].
  """
  return (delta + torch.pi) % (2.0 * torch.pi) - torch.pi


def compose_similarity(first, second):
  """Compose two similarities: ``second`` is applied after ``first``.

  With ``M1`` and ``M2`` the homogeneous matrices, the composition is the
  matrix product ``M2 @ M1`` because homogeneous points multiply on the
  right; log-scale and angle add exactly, translation composes affinely.

  Args:
      first: SimilarityParams applied first.
      second: SimilarityParams applied second.

  Returns:
      SimilarityParams of the composed transform.
  """
  m = torch.bmm(params_to_matrix(second.log_s, second.theta, second.t),
                params_to_matrix(first.log_s, first.theta, first.t))
  return params_from_matrix(m)


def umeyama_similarity(src, dst, weights=None, eps=1e-8):
  """Batched closed-form similarity fit (Umeyama) that backpropagates.

  Minimizes ``sum_i w_i * || s * R @ src_i + t - dst_i ||^2`` in closed
  form.  Gradients flow through the SVD, so the fit can sit inside a
  training graph (this is the plan section 4 closed-form head).

  The reflection guard flips the singular value associated with the
  smallest singular direction whenever ``U V^T`` has negative
  determinant, so the fit can never emit a mirrored similarity.

  Args:
      src: (B, N, 2) source points.
      dst: (B, N, 2) target points.
      weights: optional (B, N) non-negative weights; uniform when None.
      eps: numerical floor for variance and scale to keep logs finite.

  Returns:
      SimilarityParams with fields shaped (B,), (B,), (B, 2).
  """
  if src.dim() != 3 or dst.dim() != 3:
    raise ValueError("src and dst must be (B, N, 2), got "
                     f"{tuple(src.shape)} and {tuple(dst.shape)}")
  w = torch.ones_like(src[..., 0]) if weights is None else weights
  w_sum = w.sum(dim=-1, keepdim=True).clamp_min(eps)
  mu_src = (w.unsqueeze(-1) * src).sum(dim=1) / w_sum
  mu_dst = (w.unsqueeze(-1) * dst).sum(dim=1) / w_sum
  sc = src - mu_src.unsqueeze(1)
  dc = dst - mu_dst.unsqueeze(1)
  cov = torch.einsum("bni,bnj->bij", sc, w.unsqueeze(-1) * dc)
  u, sv, vt = torch.linalg.svd(cov)
  det = torch.sign(torch.det(torch.matmul(u, vt)))
  guard = torch.diag_embed(torch.stack([torch.ones_like(det), det], dim=-1))
  rot = torch.matmul(vt.transpose(-1, -2), torch.matmul(guard, u))
  var = (w * (sc * sc).sum(dim=-1)).sum(dim=-1).clamp_min(eps)
  scale = (sv * torch.stack([torch.ones_like(det), det], dim=-1)).sum(dim=-1) / var
  log_s = torch.log(scale.clamp_min(eps))
  theta = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])
  t = mu_dst - torch.einsum("bij,bj->bi", rot, mu_src) * scale.unsqueeze(-1)
  return SimilarityParams(log_s=log_s, theta=theta, t=t)


def warp_similarity(img, params, size):
  """Backward-map resample of ``img`` by the similarity.

  For every output pixel ``p`` the input is sampled at ``inv(M) @ p``;
  this is the backward map of vo/README.md section 3 and it is
  differentiable with respect to both ``img`` and ``params``.  Pixel
  coordinates are integer grid coordinates (pixel (0, 0) center at 0).

  Args:
      img: (B, C, H, W) input tensor.
      params: SimilarityParams describing img -> output.
      size: (H_out, W_out) of the output grid.

  Returns:
      (B, C, H_out, W_out) resampled tensor.
  """
  batch = img.shape[0]
  ho, wo = size
  inv = torch.linalg.inv(params_to_matrix(params.log_s, params.theta, params.t))
  ys = torch.arange(ho, device=img.device, dtype=img.dtype)
  xs = torch.arange(wo, device=img.device, dtype=img.dtype)
  gy, gx = torch.meshgrid(ys, xs, indexing="ij")
  ones = torch.ones_like(gx)
  pts = torch.stack([gx, gy, ones], dim=-1).view(1, -1, 3)
  pts = pts.expand(batch, -1, -1)
  mapped = torch.einsum("bij,bnj->bni", inv, pts)[..., :2]
  sx = 2.0 * mapped[..., 0] / max(wo - 1, 1) - 1.0
  sy = 2.0 * mapped[..., 1] / max(ho - 1, 1) - 1.0
  grid = torch.stack([sx, sy], dim=-1).view(batch, ho, wo, 2)
  return torch.nn.functional.grid_sample(img,
                                         grid,
                                         mode="bilinear",
                                         padding_mode="zeros",
                                         align_corners=True)


def corner_residual(params, src, dst):
  """Corner reprojection error -- the MCE metric itself (plan section 7).

  Args:
      params: SimilarityParams.
      src: (B, N, 2) reference points.
      dst: (B, N, 2) target points.

  Returns:
      (B,) mean pixel distance of transformed src to dst.
  """
  mat = params_to_matrix(params.log_s, params.theta, params.t)
  ones = torch.ones_like(src[..., :1])
  pts = torch.cat([src, ones], dim=-1)
  proj = torch.einsum("bij,bnj->bni", mat, pts)[..., :2]
  return (proj - dst).norm(dim=-1).mean(dim=-1)
