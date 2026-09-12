"""VO similarity network -- Siamese encoder + cost volume + corner head.

The head predicts corner offsets and the closed-form Umeyama solve maps
them to a *valid* similarity: the network cannot emit an invalid
(theta, s) pair.  See ``vo/README.md`` section 3 for the estimator math
and ``genml_kit.geometry.similarity`` for the algebra itself.
"""

import collections

import torch
from torch import nn

from genml_kit.geometry.similarity import umeyama_similarity

VOSimilarityConfig = collections.namedtuple(
    "VOSimilarityConfig", ["profile", "in_ch", "cost_range", "cost_scale"],
    defaults=["npu-small", 1, 6, 8])

_PROFILES = {
    # profile:    (stage widths,           blocks/stage)
    "npu-small": ((16, 32, 64, 128), (2, 2, 2, 2)),
    "edge-mid": ((32, 64, 128, 256), (2, 2, 3, 3)),
    "station": ((48, 96, 192, 384), (3, 3, 4, 4)),
}


def _conv_block(in_ch, out_ch, stride):
  """Plain conv + ReLU: NPU-friendly by construction (no norm layers)."""
  return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
                       nn.ReLU(inplace=True))


class _Encoder(nn.Module):
  """Strided conv/ReLU stack producing features at 1/cost_scale."""

  def __init__(self, in_ch, widths, blocks):
    super().__init__()
    stages = []
    current = in_ch
    for width, count in zip(widths, blocks):
      for index in range(count):
        stride = 2 if index == 0 else 1
        stages.append(_conv_block(current, width, stride))
        current = width
    self.stages = nn.Sequential(*stages)

  def forward(self, image):
    return self.stages(image)


def correlate(fa, fb, radius):
  """Plain correlation cost volume (plan section 4 / ILOC Ch. 3.3).

  For every displacement (dy, dx) with |dy|, |dx| <= radius computes the
  channelwise dot product

      score(y, x; dy, dx) = sum_c fa[c, y, x] * fb[c, y+dy, x+dx]

  with zero padding wherever ``y+dy`` / ``x+dx`` falls outside the frame
  (torch.roll would wrap content around and fabricate border matches).
  Only pad + mul + sum are used -- no custom ops, so an int8 exporter
  sees plain conv-shaped work.

  Args:
      fa: (B, C, h, w) reference features.
      fb: (B, C, h, w) moving features.
      radius: max displacement in feature-map pixels.

  Returns:
      (B, (2r+1)^2, h, w) correlation score per displacement, ordered
      dy-major: displacement (dy, dx) sits at channel
      (dy + r) * (2r + 1) + (dx + r).
  """
  padded = nn.functional.pad(fb, (radius,) * 4)
  disps = [(dy, dx)
           for dy in range(-radius, radius + 1)
           for dx in range(-radius, radius + 1)]
  scores = []
  for dy, dx in disps:
    shifted = padded[:, :, radius + dy:radius + dy + fa.shape[2],
                     radius + dx:radius + dx + fa.shape[3]]
    scores.append((fa * shifted).sum(dim=1))
  return torch.stack(scores, dim=1)


class VOSimilarityNet(nn.Module):
  """Estimates the inter-frame similarity + confidence of an image pair.

  forward() returns a plain dict (not a namedtuple): tensors of varying
  shape, consumed by name in the loss functions.  The similarity itself is
  produced by the closed-form Umeyama solve over predicted corner
  correspondences, so the output is always a *valid* similarity (any
  rotation, any positive scale) -- the network cannot emit an invalid
  (theta, s) pair.
  """

  def __init__(self, config):
    super().__init__()
    self.cfg = config
    widths, blocks = _PROFILES[config.profile]
    self.scale = config.cost_scale
    self.encoder = _Encoder(config.in_ch, widths, blocks)
    self.ref_corners = nn.Buffer(torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
                                               [-1.0, 1.0]]),
                                 persistent=False)
    self.corner_mlp = nn.Sequential(nn.LazyLinear(64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 12))
    self.head = nn.LazyConv2d(widths[-1], 1)

  def forward(self, a, b):
    fa = self.encoder(a)
    fb = self.encoder(b)
    vol = correlate(fa, fb, self.cfg.cost_range)
    feats = self.head(torch.cat([vol, fa], dim=1))
    pooled = feats.mean(dim=(2, 3))
    corners = self.ref_corners.unsqueeze(0).expand(a.shape[0], -1, -1)
    deltas = self.corner_mlp(pooled)[:, :8].view(-1, 4, 2)
    size = torch.tensor(
        [fa.shape[-1], fa.shape[-2]], device=a.device, dtype=deltas.dtype) / 2.0
    src = corners * size.unsqueeze(0) * self.scale + size.unsqueeze(0)
    params = umeyama_similarity(src, src + deltas)
    conf = self.corner_mlp(pooled)[:, 8:10]
    return {"params": params, "corners": src, "dc": deltas, "conf": conf}
