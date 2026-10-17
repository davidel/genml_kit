"""Linear (a.k.a. UPerNet-lite / SETR-pixel) decoder.

A lightweight decoder that projects the *deepest* encoder feature to
``num_classes`` with a single 1x1 convolution and bilinearly upsamples it
to full resolution.  It makes no use of skip connections, so it is the
natural choice for *single-scale* encoders (plain ViTs) as well as a fast
baseline for hierarchical ones.
"""

import torch.nn as nn
import torch.nn.functional as F

from genml_kit.models.decoders import DECODERS
from genml_kit.models.decoders.base import BaseDecoder


@DECODERS.register("linear")
class LinearDecoder(BaseDecoder):
  """1x1-conv classification head plus bilinear upsampling.

  Parameters
  ----------
  in_channels : list[int]
      Encoder channel counts, finest level first.
  num_classes : int
      Output classes.
  hidden_channels : int | None
      Optional intermediate width; when given, a 3x3 conv-BN-ReLU is
      inserted before the classifier to add capacity.
  """

  def __init__(self, in_channels, num_classes, hidden_channels=None):
    super().__init__(in_channels, num_classes)
    deepest = self.in_channels[-1]
    if hidden_channels:
      self.proj = nn.Sequential(
          nn.Conv2d(deepest, hidden_channels, kernel_size=3, padding=1, bias=False),
          nn.BatchNorm2d(hidden_channels),
          nn.ReLU(inplace=True),
      )
      self.head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
    else:
      self.proj = None
      self.head = nn.Conv2d(deepest, num_classes, kernel_size=1)

  def forward(self, pyramid):
    """Return ``(B, num_classes, H, W)`` upsampled to the finest level."""
    # Deepest (lowest-resolution, most semantic) level: (B, C, h, w) with
    # h, w = H/32, W/32.
    x = pyramid.levels[-1].feature
    if self.proj is not None:
      # Optional 3x3 stem: (B, C, h, w) -> (B, hidden, h, w).
      x = self.proj(x)
    # Classifier: (B, hidden or C, h, w) -> (B, num_classes, h, w).
    logits = self.head(x)
    # Upsample to the finest pyramid resolution so the output matches the
    # input image for single-scale (stride == patch) encoders too:
    # (B, num_classes, h, w) -> (B, num_classes, H, W).
    target = pyramid.reference_size()
    return F.interpolate(logits, size=target, mode="bilinear", align_corners=False)
