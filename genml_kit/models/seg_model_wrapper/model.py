"""SegModelWrapper -- encoder backbone + pluggable segmentation decoder.

The wrapper keeps the two halves orthogonal: the *encoder* produces an
:class:`EncoderPyramid` and the *decoder* turns it into per-pixel logits.
Swapping either is independent, exactly as :class:`ClsModelWrapper`
separates a backbone from its classifier head.

Protocol parity with the other wrappers:

* ``forward(pixel_values)`` returns a :class:`ModelOutput` whose
  ``.logits`` is ``(B, num_classes, H, W)``;
* the wrapper exposes ``.config`` (``num_labels``, ``id2label``,
  ``label2id``) and ``extract_backbone_features``;
* ``.num_labels`` is available for checkpoint metadata.
"""

from types import SimpleNamespace

import torch.nn as nn

from genml_kit.models.registry import ModelOutput
from genml_kit.models.seg_model_wrapper.encoder import probe_encoder


class SegModelWrapper(nn.Module):
  """Encoder + decoder segmentation model.

  Parameters
  ----------
  encoder : nn.Module
      A callable returning an :class:`EncoderPyramid`, e.g.
      :class:`~genml_kit.models.seg_model_wrapper.encoder.TimmEncoder`.
  decoder : BaseDecoder
      Decoder built for the encoder's channel counts.
  num_labels : int
      Number of segmentation classes.
  id2label : dict | None
      Optional class-id -> name map (HF compatibility).
  """

  def __init__(self, encoder, decoder, num_labels, id2label=None):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.num_labels = num_labels
    self.id2label = id2label or {i: f"class_{i}" for i in range(num_labels)}
    self.label2id = {v: k for k, v in self.id2label.items()}
    self.config = SimpleNamespace(num_labels=num_labels,
                                  id2label=self.id2label,
                                  label2id=self.label2id)

  def forward(self, pixel_values=None, **kwargs):
    """Return per-pixel logits.

    Accepts ``pixel_values`` positionally or by keyword for HF-style call
    sites (``model(pixel_values=x)``).
    """
    if pixel_values is None:
      pixel_values = kwargs.get("images")
    if pixel_values is None:
      raise ValueError("SegModelWrapper.forward requires pixel_values")
    # Encoder: (B, 3, H, W) -> feature pyramid of (B, C_i, H_i, W_i).
    pyramid = self.encoder(pixel_values)
    # Decoder: pyramid -> (B, num_classes, H, W) logits.
    logits = self.decoder(pyramid)
    return ModelOutput(logits)

  def extract_backbone_features(self, pixel_values):
    """Return the deepest encoder feature map (for probes / feature reuse).

    Shape ``(B, C, H/32, W/32)`` -- the coarsest, most semantic level.
    """
    pyramid = self.encoder(pixel_values)
    return pyramid.levels[-1].feature


def build_seg_model(encoder,
                    decoder_spec,
                    num_classes,
                    image_size,
                    device="cpu",
                    id2label=None):
  """Probe *encoder*, build the decoder for its channels, and wrap both.

  This is the single place where decoding depends on the runtime encoder:
  the pyramid shapes are discovered by :func:`probe_encoder`, then the
  concrete ``in_channels`` list is injected into the decoder.

  Parameters
  ----------
  encoder : nn.Module
      Encoder returning an :class:`EncoderPyramid`.
  decoder_spec : str
      Decoder name/spec accepted by
      :func:`genml_kit.models.decoders.build_decoder`.
  num_classes : int
      Output classes.
  image_size : int
      Input resolution used for the probe.
  device : str | torch.device
      Device for the probe and the resulting model.
  id2label : dict | None
      Optional label map.

  Returns
  -------
  SegModelWrapper
  """
  from genml_kit.models.decoders import build_decoder

  pyramid = probe_encoder(encoder, image_size, device=device)
  decoder = build_decoder(decoder_spec,
                          in_channels=pyramid.channels,
                          num_classes=num_classes)
  model = SegModelWrapper(encoder, decoder, num_labels=num_classes, id2label=id2label)
  model.to(device)
  return model
