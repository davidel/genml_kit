"""Tests for the decoder registry, EncoderPyramid, and SegModelWrapper."""

import pytest
import torch

from genml_kit.models.decoders import DECODERS, build_decoder, parse_decoder_spec
from genml_kit.models.decoders.base import EncoderLevel, EncoderPyramid, upsample_to
from genml_kit.models.seg_model_wrapper.encoder import TimmEncoder, probe_encoder
from genml_kit.models.seg_model_wrapper.model import SegModelWrapper


def _pyramid(channels=(64, 128, 256, 512), base=32, batch=2):
  levels = []
  h = w = base
  for i, c in enumerate(channels):
    levels.append(EncoderLevel(torch.randn(batch, c, h, w), 2**(i + 2)))
    h //= 2
    w //= 2
  return EncoderPyramid(levels)


class TestEncoderPyramid:

  def test_properties(self):
    p = _pyramid()
    assert len(p) == 4
    assert p.channels == [64, 128, 256, 512]
    assert p.strides == [4, 8, 16, 32]
    assert p.reference_size() == (32, 32)

  def test_empty_raises(self):
    with pytest.raises(ValueError):
      EncoderPyramid([])


class TestDecoders:

  @pytest.mark.parametrize("spec", ["unet:depth=4", "fpn:depth=4", "linear"])
  def test_output_shape(self, spec):
    p = _pyramid()
    dec = build_decoder(spec, in_channels=p.channels, num_classes=5)
    out = dec(p)
    assert out.shape == (2, 5, 32, 32)

  def test_registry_contents(self):
    for name in ("unet", "fpn", "linear"):
      assert name in DECODERS.list_names()

  def test_parse_decoder_spec(self):
    name, kwargs = parse_decoder_spec("unet:depth=4,decoder_channels=128")
    assert name == "unet"
    assert kwargs == {"depth": 4, "decoder_channels": 128}

  def test_unknown_decoder_raises(self):
    with pytest.raises(ValueError, match="Unknown decoder"):
      build_decoder("nope", in_channels=[16, 32], num_classes=2)

  def test_unet_needs_two_levels(self):
    with pytest.raises(ValueError, match="at least 2"):
      build_decoder("unet", in_channels=[16], num_classes=2)

  def test_custom_decoder_from_script(self, tmp_path):
    script = tmp_path / "my_dec.py"
    script.write_text(
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "from genml_kit.models.decoders.base import BaseDecoder\n"
        "class Decoder(BaseDecoder):\n"
        "    def forward(self, pyramid):\n"
        "        x = self.head(pyramid.levels[-1].feature)\n"
        "        return F.interpolate(x, size=pyramid.reference_size(),\n"
        "                             mode='bilinear', align_corners=False)\n"
        "def create_decoder(in_channels, num_classes, **kwargs):\n"
        "    d = Decoder(in_channels, num_classes)\n"
        "    d.head = nn.Conv2d(in_channels[-1], num_classes, 1)\n"
        "    return d\n")
    dec = build_decoder(str(script), in_channels=[8, 16], num_classes=3)
    p = _pyramid(channels=(8, 16), base=16)
    assert dec(p).shape == (2, 3, 16, 16)


class TestUpsampleTo:

  def test_noop_when_matching(self):
    x = torch.randn(1, 2, 8, 8)
    assert upsample_to(x, (8, 8)) is x

  def test_resizes(self):
    x = torch.randn(1, 2, 4, 4)
    assert upsample_to(x, (16, 16)).shape == (1, 2, 16, 16)


class TestTimmEncoder:

  def test_probe_and_forward(self):
    import timm
    model = timm.create_model("resnet18", features_only=True, pretrained=False)
    enc = TimmEncoder(model)
    p = probe_encoder(enc, 64)
    assert len(p) == 5
    assert p.channels[0] == 64
    out = enc(torch.randn(1, 3, 64, 64))
    assert out.channels == p.channels

  def test_out_indices_selection(self):
    import timm
    model = timm.create_model("resnet18", features_only=True, pretrained=False)
    enc = TimmEncoder(model, out_indices=[1, 2, 3])
    p = probe_encoder(enc, 64)
    assert len(p) == 3


class TestSegModelWrapper:

  def test_forward_protocol(self):
    import timm
    enc = TimmEncoder(
        timm.create_model("resnet18", features_only=True, pretrained=False))
    dec = build_decoder("unet:depth=4", in_channels=[64, 128, 256, 512], num_classes=4)
    model = SegModelWrapper(enc, dec, num_labels=4)
    out = model(pixel_values=torch.randn(1, 3, 64, 64))
    assert out.logits.shape == (1, 4, 16, 16)
    assert model.config.num_labels == 4

  def test_extract_backbone_features(self):
    import timm
    enc = TimmEncoder(
        timm.create_model("resnet18", features_only=True, pretrained=False))
    dec = build_decoder("linear", in_channels=[64, 64, 128, 256, 512], num_classes=2)
    model = SegModelWrapper(enc, dec, num_labels=2)
    feats = model.extract_backbone_features(torch.randn(1, 3, 64, 64))
    assert feats.shape[:2] == (1, 512)
