"""U-Net decoder: top-down pathway with horizontal skip connections.

Consumes the full encoder pyramid (highest resolution first), fuses each
upsampled feature with the matching encoder level via concatenation, and
emits per-pixel logits at the finest pyramid resolution.

This is the classic skip-connection decoder; it requires a *hierarchical*
encoder that exposes multiple spatial resolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from genml_kit.models.decoders import DECODERS
from genml_kit.models.decoders.base import BaseDecoder


def _double_conv(in_ch, out_ch):
  """Two 3x3 conv-BN-ReLU blocks (the U-Net building block)."""
  return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
  )


@DECODERS.register("unet")
class UNetDecoder(BaseDecoder):
  """Skip-connection U-Net decoder.

  Parameters
  ----------
  in_channels : list[int]
      Encoder channel counts, finest level first.
  num_classes : int
      Output classes.
  decoder_channels : int
      Width of the decoder at every level.
  depth : int | None
      Number of encoder levels to use (default: all).  The selected
      levels are the *deepest* ones (lowest resolution), so the shallowest
      stride-2 map (often redundant with the input) can be dropped.
  """

  def __init__(self, in_channels, num_classes, decoder_channels=256, depth=None):
    super().__init__(in_channels, num_classes)
    if depth is not None:
      keep = min(depth, len(in_channels))
      in_channels = in_channels[len(self.in_channels) - keep:]
      self.in_channels = list(in_channels)
    if len(self.in_channels) < 2:
      raise ValueError("UNetDecoder needs at least 2 encoder levels "
                       f"(got {len(self.in_channels)})")

    self.depth = len(self.in_channels)

    # Lateral 1x1 projections adapt arbitrary encoder widths to the
    # decoder width, so the same decoder works for ResNet-18/50/Swin.
    self.lateral = nn.ModuleList(
        [nn.Conv2d(c, decoder_channels, kernel_size=1) for c in self.in_channels])

    # Top-down fusion: at each step concat(upsampled_prev, lateral_level).
    self.fuse = nn.ModuleList()
    for _ in range(self.depth - 1):
      self.fuse.append(_double_conv(decoder_channels * 2, decoder_channels))

    self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

  def forward(self, pyramid):
    """Return ``(B, num_classes, H, W)`` logits at the finest level.

    Each level's feature ``(B, C_i, H_i, W_i)`` is first projected to the
    decoder width by a 1x1 lateral conv.
    """
    levels = pyramid.levels
    if len(levels) < self.depth:
      raise ValueError(f"UNetDecoder expects >= {self.depth} encoder levels, got "
                       f"{len(levels)}")
    # Keep the deepest ``depth`` levels (finest first ordering preserved).
    levels = levels[len(levels) - self.depth:]

    # Lateral projections: (B, C_i, H_i, W_i) -> (B, D, H_i, W_i) per level.
    feats = [lat(level.feature) for lat, level in zip(self.lateral, levels)]

    # Coarsest level starts the top-down pathway: (B, D, H/32, W/32).
    x = feats[-1]
    for i in range(self.depth - 2, -1, -1):
      # Upsample to the skip size: (B, D, H_i/2, W_i/2) -> (B, D, H_i, W_i),
      # concat with the lateral skip: (B, 2D, H_i, W_i), then fuse back to
      # (B, D, H_i, W_i).
      x = F.interpolate(x,
                        size=feats[i].shape[-2:],
                        mode="bilinear",
                        align_corners=False)
      x = self.fuse[self.depth - 2 - i](torch.cat([x, feats[i]], dim=1))

    # 1x1 head: (B, D, H, W) -> (B, num_classes, H, W).
    return self.head(x)
