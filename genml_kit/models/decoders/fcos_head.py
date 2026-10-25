"""FCOS detection head: anchor-free dense per-location box prediction.

Reference: Tian et al., "FCOS: Fully Convolutional One-Stage Object
Detection", ICCV 2019.

For every spatial location of every pyramid level the head predicts three
things from a small shared convolutional tower:

* a ``num_classes``-way classification logit vector (focal-loss targets),
* four positive "distance" values ``(l, t, r, b)`` -- how far the box edges
  are from the location, in *pixels* of the input image,
* a single centerness logit that down-weights low-quality locations.

Unlike the segmentation decoders this head does not collapse the pyramid into
one map: detection needs per-level predictions so each location keeps its
own stride (used to convert distances back into absolute ``xyxy`` boxes and
to build the location grid).  It therefore returns a :class:`DetHeadOutput`
container rather than a single tensor, and is *not* a
:class:`~genml_kit.models.decoders.base.BaseDecoder` subclass.
"""

import math
from collections import namedtuple

import torch
import torch.nn as nn

from genml_kit.models.decoders import DECODERS

# Per-level dense predictions.  Each list is ordered finest-level-first,
# matching ``EncoderPyramid.levels``:
#   cls_logits[i] : (B, num_classes, H_i, W_i) raw class scores
#   box_pred[i]   : (B, 4, H_i, W_i) positive (l, t, r, b) distances in pixels
#   centerness[i] : (B, 1, H_i, W_i) raw centerness logits
#   strides[i]    : int, input-to-feature downsampling factor for level i
DetHeadOutput = namedtuple("DetHeadOutput",
                           ["cls_logits", "box_pred", "centerness", "strides"])


def _groups(channels, max_groups=32):
  """Return a GroupNorm group count that divides *channels* evenly.

  ``GroupNorm`` needs ``channels % groups == 0``.  Using ``gcd`` keeps the
  usual 32 groups for wide towers (256 -> 32) while still working for the
  narrow towers used in unit tests (e.g. 8 -> 8), so the head never fails
  just because a test chose a small width.
  """
  return math.gcd(int(channels), int(max_groups)) or 1


def _make_tower(in_channels, hidden, num_convs):
  """Build one convolutional tower per pyramid level.

  Each level has a different input width, so the input projection cannot be
  shared; the tower is a :class:`~torch.nn.ModuleList` indexed by level
  order.  Deeper towers give a larger receptive field at the cost of speed.
  """
  towers = nn.ModuleList()
  for channels in in_channels:
    layers = []
    prev = int(channels)
    for _ in range(int(num_convs)):
      # (B, prev, H, W) -> (B, hidden, H, W): 3x3 conv keeps H, W.
      layers.append(nn.Conv2d(prev, hidden, kernel_size=3, padding=1))
      layers.append(nn.GroupNorm(_groups(hidden), hidden))
      layers.append(nn.ReLU(inplace=True))
      prev = int(hidden)
    towers.append(nn.Sequential(*layers))
  return towers


@DECODERS.register("fcos_head")
class FCOSHead(nn.Module):
  """Anchor-free FCOS detection head over an encoder feature pyramid.

  Parameters
  ----------
  in_channels : list[int]
      Encoder channel counts, finest level first (``pyramid.channels``).
  num_classes : int
      Number of foreground classes (background is implicit).
  hidden : int
      Tower width; also the width of the three output branches.
  num_convs : int
      Convolutions per tower (2 is the FCOS default; more is slower).
  """

  def __init__(self, in_channels, num_classes, hidden=256, num_convs=2):
    super().__init__()
    self.in_channels = list(in_channels)
    self.num_classes = int(num_classes)
    self.hidden = int(hidden)
    self.num_convs = int(num_convs)

    # Separate towers for classification and localisation: the FCOS paper
    # found sharing them hurts, because the two tasks want different
    # features.  Centerness shares the regression tower.
    self.cls_tower = _make_tower(self.in_channels, self.hidden, self.num_convs)
    self.reg_tower = _make_tower(self.in_channels, self.hidden, self.num_convs)

    # Branch heads: each maps (B, hidden, H_i, W_i) -> (B, out, H_i, W_i).
    self.cls_logits = nn.Conv2d(self.hidden, self.num_classes, kernel_size=3, padding=1)
    self.bbox_pred = nn.Conv2d(self.hidden, 4, kernel_size=3, padding=1)
    self.centerness = nn.Conv2d(self.hidden, 1, kernel_size=3, padding=1)

    self._init_weights()

  def _init_weights(self):
    """Initialise branches so training starts stable.

    The classification branch is biased towards "background" (``-log((1 -
    pi) / pi)`` with ``pi = 0.01``) so the initial loss is not dominated by
    the huge number of negative locations -- the standard FCOS/RetinaNet
    prior.  Box distances start near 1 pixel (``bias = -log(1) = 0`` after the
    ``exp``) and centerness starts at 0.5.
    """
    for module in (self.cls_logits, self.bbox_pred, self.centerness):
      for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
          nn.init.normal_(layer.weight, std=0.01)
          nn.init.constant_(layer.bias, 0.0)
    prior = 0.01
    nn.init.constant_(self.cls_logits.bias, -math.log((1.0 - prior) / prior))

  def forward(self, pyramid):
    """Return per-level class / box / centerness predictions.

    Parameters
    ----------
    pyramid : EncoderPyramid
        Ordered encoder levels, finest first.

    Returns
    -------
    DetHeadOutput
        Per-level lists of class logits, box distances and centerness, plus
        the per-level strides needed to decode the distances to pixels.
    """
    levels = pyramid.levels
    strides = list(pyramid.strides)
    if len(self.cls_tower) != len(levels):
      raise ValueError(f"FCOSHead built for {len(self.cls_tower)} levels but "
                       f"got a pyramid with {len(levels)} levels")

    cls_out, box_out, ctr_out = [], [], []
    for i, level in enumerate(levels):
      feature = level.feature
      stride = int(strides[i])
      # Tower: (B, C_i, H_i, W_i) -> (B, hidden, H_i, W_i).
      cls_feat = self.cls_tower[i](feature)
      # Tower: (B, C_i, H_i, W_i) -> (B, hidden, H_i, W_i).
      reg_feat = self.reg_tower[i](feature)
      # Branch: (B, hidden, H_i, W_i) -> (B, num_classes, H_i, W_i).
      cls_out.append(self.cls_logits(cls_feat))
      # Branch: (B, hidden, H_i, W_i) -> (B, 4, H_i, W_i), raw distances.
      raw = self.bbox_pred(reg_feat)
      # exp() makes the distances strictly positive; multiplying by the
      # stride rescales them from "level units" into input-image pixels, so
      # the loss can compare them against boxes in the original frame.
      # (B, 4, H_i, W_i) -> (B, 4, H_i, W_i).
      box_out.append(torch.exp(raw) * stride)
      # Branch: (B, hidden, H_i, W_i) -> (B, 1, H_i, W_i).
      ctr_out.append(self.centerness(reg_feat))

    return DetHeadOutput(cls_logits=cls_out,
                         box_pred=box_out,
                         centerness=ctr_out,
                         strides=strides)
