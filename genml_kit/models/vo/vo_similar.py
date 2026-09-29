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

# What forward() returns.  A namedtuple (not a dict) so consumers write
# ``out.params`` -- and so a mistyped key raises AttributeError at the
# call site instead of returning None deep inside a loss.
VOModelOutput = collections.namedtuple("VOModelOutput",
                                       ["params", "corners", "dc", "conf"])

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
  """Strided conv/ReLU stack producing features at 1/cost_scale.

  Each stage starts with a stride-2 conv (halving the spatial size) and
  continues with stride-1 convs at the same resolution; the final stage
  width must match the MLP input width expected by ``VOSimilarityNet``.
  """

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
    # Strided conv stack: (B, in_ch, H, W) -> (B, widths[-1], H/2^n, W/2^n).
    return self.stages(image)


def correlate(fa, fb, radius):
  """Plain correlation cost volume (vo/README.md section 12).

  For every displacement (dy, dx) with |dy|, |dx| <= radius computes the
  channelwise dot product

      score(y, x; dy, dx) = sum_c fa[c, y, x] * fb[c, y+dy, x+dx]

  with zero padding wherever ``y+dy`` / ``x+dx`` falls outside the frame
  (torch.roll would wrap content around and fabricate border matches).
  Only pad + mul + sum are used -- no custom ops, so an int8 exporter
  sees plain conv-shaped work.

  Args:
      fa: (B, C, H, W) reference features.
      fb: (B, C, H, W) moving features.
      radius: max displacement (R) in feature-map pixels.

  Returns:
      (B, (2R+1)^2, H, W) correlation score per displacement, ordered
      dy-major: displacement (dy, dx) sits at channel
      (dy + R) * (2R + 1) + (dx + R).
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

  A Siamese encoder lifts both frames to a shared feature space, a
  correlation cost volume (see :func:`correlate`) scores every
  displacement, and a 1x1-conv head + small MLP predict per-image corner
  offsets and a confidence pair.  The similarity itself is produced by the
  closed-form Umeyama solve over the predicted corner correspondences, so
  the output is always a *valid* similarity (any rotation, any positive
  scale) -- the network cannot emit an invalid (theta, s) pair.

  forward() returns a :class:`VOModelOutput` namedtuple:

  * ``params``: ``SimilarityParams`` -- the closed-form similarity fit.
  * ``corners``: ``(B, 4, 2)`` reference corner coordinates in pixels.
  * ``dc``: ``(B, 4, 2)`` predicted corner deltas (in pixels).
  * ``conf``: ``(B, 2)`` predicted confidence pair.
  """

  def __init__(self, config):
    super().__init__()
    self.cfg = config
    widths, blocks = _PROFILES[config.profile]
    self.scale = config.cost_scale
    self.encoder = _Encoder(config.in_ch, widths, blocks)
    # Learnable gain converting the head's normalized output into pixel
    # corner deltas.  vo/README.md s14 and s16.1 require those deltas --
    # and therefore L_mce -- to be in pixels.
    #
    # The old basis was derived from the encoder's feature map instead
    # (``size = feat/2``, then ``* cost_scale``), which made it
    # image_size/8, centered on zero, and dependent on encoder stride
    # rather than on the image.  ``cost_scale`` is a correlation search
    # *radius*, not a length, and the encoder's four stride-2 stages put
    # features at H/16 regardless of it, so that basis was never a pixel
    # count.  Initialising the gain to the old effective scale (8.0,
    # i.e. image/8) keeps the head's starting output magnitude so it
    # does not have to relearn a much larger range from scratch.
    self.pixel_gain = nn.Parameter(torch.tensor(float(self.scale * 8.0)))
    self.ref_corners = nn.Buffer(torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0],
                                               [-1.0, 1.0]]),
                                 persistent=False)
    # The MLP reads the pooled feature vector and outputs a flat head
    # vector of 10 values: 8 corner deltas (4 corners x 2) + 2 confidence
    # channels.  The input width MUST equal the encoder's final stage
    # width (`widths[-1]`, 128 for the default npu-small profile) -- it is
    # the feature map pooled across space, not the per-pixel feature depth
    # of an intermediate stage.
    self.corner_mlp = nn.Sequential(nn.LazyLinear(widths[-1]), nn.ReLU(inplace=True),
                                    nn.Linear(widths[-1], 10))
    self.head = nn.LazyConv2d(widths[-1], 1)

  def forward(self, a, b):
    # B = batch, H = W for square frames; C = final encoder width
    # (e.g. 128 for the default npu-small profile).
    # Siamese encode: (B, in_ch, H, W) -> (B, C, H/16, W/16) per frame.
    fa = self.encoder(a)
    fb = self.encoder(b)
    # Correlation volume: (B, C, H/16, W/16) x (B, C, H/16, W/16) ->
    # (B, (2R+1)^2, H/16, W/16), score per displacement (dy, dx).
    vol = correlate(fa, fb, self.cfg.cost_range)
    # Concatenate volume with reference features and refine with a 1x1
    # conv: (B, C + (2R+1)^2, H/16, W/16) -> (B, C, H/16, W/16).
    feats = self.head(torch.cat([vol, fa], dim=1))
    # Global average pooling: (B, C, H/16, W/16) -> (B, C).
    pooled = feats.mean(dim=(2, 3))
    # Corner MLP reads the pooled vector: (B, C) -> (B, 10); the first 8
    # values are corner deltas (4 corners x 2), the last 2 the confidence
    # pair.
    head_out = self.corner_mlp(pooled)
    # Split off per-corner deltas: (B, 10) -> (B, 4, 2), scaled from the
    # head's normalized output into pixels.
    deltas = head_out[:, :8].view(-1, 4, 2) * self.pixel_gain
    # Reference corners in *pixel* coordinates, built from the input
    # shape: the integer-corner convention of vo/README.md s11.4, which
    # is also what ``homography_to_similarity`` uses to build the ground
    # truth (genml_kit.datasets.vo_pairs).  Deliberately independent of
    # the feature map: the corner basis is a property of the image, not
    # of the encoder stride.
    height, width = a.shape[-2], a.shape[-1]
    src = self.ref_corners.new_tensor([[0.0, 0.0], [width - 1, 0.0],
                                       [width - 1, height - 1], [0.0, height - 1]
                                      ]).unsqueeze(0).expand(a.shape[0], -1, -1)
    # Closed-form Umeyama least-squares similarity from src to src + deltas.
    params = umeyama_similarity(src, src + deltas)
    # Confidence pair: (B, 10) -> (B, 2).
    conf = head_out[:, 8:10]
    return VOModelOutput(params=params, corners=src, dc=deltas, conf=conf)
