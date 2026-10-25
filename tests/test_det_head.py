"""Tests for the FCOS detection head and the detection model wrapper."""

import torch

from genml_kit.models.decoders import DECODERS, build_decoder
from genml_kit.models.decoders.base import EncoderLevel, EncoderPyramid
from genml_kit.models.decoders.fcos_head import DetHeadOutput, FCOSHead, _groups


def _pyramid(channels=(16, 32, 64), base=32):
  """Build a synthetic pyramid with strides 8/16/32 from *channels*."""
  levels = []
  for i, c in enumerate(channels):
    size = max(1, base // (8 * (2**i)))
    feature = torch.randn(2, c, size, size)
    levels.append(EncoderLevel(feature=feature, stride=8 * (2**i)))
  return EncoderPyramid(levels)


class TestGroups:

  def test_divides_wide_channel_count(self):
    assert _groups(256) == 32

  def test_handles_narrow_channel_count(self):
    # 8 is not divisible by 32; gcd(8, 32) = 8 keeps GroupNorm valid.
    assert _groups(8) == 8

  def test_never_zero(self):
    assert _groups(1) == 1


class TestFCOSHead:

  def test_output_shapes(self):
    head = FCOSHead(in_channels=[16, 32, 64], num_classes=3, hidden=16, num_convs=1)
    out = head(_pyramid())
    assert isinstance(out, DetHeadOutput)
    assert len(out.cls_logits) == 3
    # Per-level class logits: (B, num_classes, H_i, W_i).
    assert out.cls_logits[0].shape[1] == 3
    assert out.cls_logits[0].shape[0] == 2
    # Box distances and centerness keep the level's spatial size.
    assert out.box_pred[0].shape[1] == 4
    assert out.centerness[0].shape[1] == 1
    assert out.strides == [8, 16, 32]

  def test_box_distances_positive(self):
    head = FCOSHead(in_channels=[16], num_classes=2, hidden=16, num_convs=1)
    out = head(_pyramid(channels=(16,), base=32))
    # exp() * stride guarantees strictly positive distances.
    assert (out.box_pred[0] > 0).all()

  def test_registered_in_registry(self):
    assert DECODERS.contains("fcos_head")

  def test_build_via_registry(self):
    head = build_decoder("fcos_head", in_channels=[16, 32], num_classes=4)
    assert isinstance(head, FCOSHead)
    assert head.num_classes == 4

  def test_level_mismatch_raises(self):
    head = FCOSHead(in_channels=[16, 32], num_classes=2, hidden=16, num_convs=1)
    try:
      head(_pyramid(channels=(16, 32, 64)))
    except ValueError as exc:
      assert "levels" in str(exc)
    else:
      raise AssertionError("expected ValueError on level mismatch")

  def test_inline_spec_kwargs(self):
    head = build_decoder("fcos_head:num_convs=3", in_channels=[16], num_classes=2)
    assert head.num_convs == 3


class _FakePyramidEncoder(torch.nn.Module):
  """Encoder stub that returns a fixed-shape EncoderPyramid.

  ``DetModelWrapper`` only requires ``encoder(pixel_values)`` to return an
  :class:`EncoderPyramid`; using this instead of a real timm backbone keeps
  the test fast and dependency-free.
  """

  def __init__(self, channels=(16, 32, 64), base=32):
    super().__init__()
    self._channels = list(channels)
    self._base = base

  def forward(self, x):
    levels = []
    for i, c in enumerate(self._channels):
      size = max(1, self._base // (8 * (2**i)))
      levels.append(
          EncoderLevel(feature=torch.randn(x.shape[0], c, size, size),
                       stride=8 * (2**i)))
    return EncoderPyramid(levels)


class TestDetModelWrapper:

  def _encoder(self, channels=(16, 32, 64), base=32):
    return _FakePyramidEncoder(channels=channels, base=base)

  def test_forward_returns_det_head_output(self):
    from genml_kit.models.det_model_wrapper.model import DetModelWrapper

    enc = self._encoder()
    head = FCOSHead(in_channels=[16, 32, 64], num_classes=2, hidden=16, num_convs=1)
    model = DetModelWrapper(enc, head, num_labels=2)
    out = model(pixel_values=torch.randn(2, 3, 32, 32))
    assert isinstance(out.logits, DetHeadOutput)

  def test_config_exposes_num_labels(self):
    from genml_kit.models.det_model_wrapper.model import DetModelWrapper

    enc = self._encoder()
    head = FCOSHead(in_channels=[16, 32, 64], num_classes=5, hidden=16, num_convs=1)
    model = DetModelWrapper(enc, head, num_labels=5, id2label={0: "a"})
    assert model.config.num_labels == 5
    assert model.config.id2label == {0: "a"}

  def test_extract_backbone_features(self):
    from genml_kit.models.det_model_wrapper.model import DetModelWrapper

    enc = self._encoder()
    head = FCOSHead(in_channels=[16, 32, 64], num_classes=2, hidden=16, num_convs=1)
    model = DetModelWrapper(enc, head, num_labels=2)
    feats = model.extract_backbone_features(torch.randn(1, 3, 32, 32))
    assert feats.shape[1] == 64
