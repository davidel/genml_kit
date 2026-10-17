"""FPN decoder: feature-pyramid fusion into a single high-resolution map.

Reference: Lin et al., "Feature Pyramid Networks for Object Detection",
CVPR 2017.  Each encoder level is projected to a common width, fused
top-down with element-wise addition, upsampled to a shared resolution, and
concatenated before a segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from genml_kit.models.decoders import DECODERS
from genml_kit.models.decoders.base import BaseDecoder


@DECODERS.register("fpn")
class FPNDecoder(BaseDecoder):
  """Feature-pyramid decoder with additive top-down fusion.

  Parameters
  ----------
  in_channels : list[int]
      Encoder channel counts, finest level first.
  num_classes : int
      Output classes.
  fpn_channels : int
      Common width for all pyramid levels.
  depth : int | None
      Number of encoder levels to use (default: all).
  """

  def __init__(self, in_channels, num_classes, fpn_channels=256, depth=None):
    super().__init__(in_channels, num_classes)
    if depth is not None:
      keep = min(depth, len(in_channels))
      in_channels = in_channels[len(self.in_channels) - keep:]
      self.in_channels = list(in_channels)
    if len(self.in_channels) < 2:
      raise ValueError("FPNDecoder needs at least 2 encoder levels "
                       f"(got {len(self.in_channels)})")

    self.depth = len(self.in_channels)
    self.lateral = nn.ModuleList(
        [nn.Conv2d(c, fpn_channels, kernel_size=1) for c in self.in_channels])
    self.smooth = nn.ModuleList([
        nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        for _ in range(self.depth)
    ])
    self.head = nn.Conv2d(fpn_channels * self.depth, num_classes, kernel_size=1)

  def forward(self, pyramid):
    """Return ``(B, num_classes, H, W)`` logits at the finest level."""
    levels = pyramid.levels
    if len(levels) < self.depth:
      raise ValueError(f"FPNDecoder expects >= {self.depth} encoder levels, "
                       f"got {len(levels)}")
    levels = levels[len(levels) - self.depth:]

    # Lateral 1x1 projections: (B, C_i, H_i, W_i) -> (B, F, H_i, W_i).
    laterals = [lat(level.feature) for lat, level in zip(self.lateral, levels)]

    # Top-down pass: fuse each level with the upsampled coarser level.
    # Each coarser level is upsampled to the finer level's resolution.
    top_down = [laterals[-1]]
    for i in range(self.depth - 2, -1, -1):
      # (B, F, H_i/2, W_i/2) -> (B, F, H_i, W_i).
      up = F.interpolate(top_down[-1],
                         size=laterals[i].shape[-2:],
                         mode="bilinear",
                         align_corners=False)
      # Additive fusion of two (B, F, H_i, W_i) maps -> (B, F, H_i, W_i).
      top_down.append(laterals[i] + up)
    # Re-order from coarsest-first to finest-first.
    top_down.reverse()

    smoothed = [sm(lvl) for sm, lvl in zip(self.smooth, top_down)]

    # Upsample every level to the finest resolution and concatenate:
    # list of (B, F, H, W) -> (B, depth * F, H, W).
    target = smoothed[0].shape[-2:]
    fused = torch.cat([
        F.interpolate(s, size=target, mode="bilinear", align_corners=False)
        for s in smoothed
    ],
                      dim=1)
    # 1x1 head: (B, depth * F, H, W) -> (B, num_classes, H, W).
    return self.head(fused)
