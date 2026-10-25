"""DetModelWrapper -- encoder backbone + pluggable detection head.

The wrapper keeps the two halves orthogonal exactly as
:class:`~genml_kit.models.seg_model_wrapper.model.SegModelWrapper` does for
segmentation: the *encoder* produces an :class:`EncoderPyramid` and the
*head* turns it into per-level detection predictions.  The difference is that
a detection head cannot collapse the pyramid into one map -- it must keep the
per-level structure so locations retain their strides -- so the wrapper
forwards the head's structured output rather than a single ``logits`` tensor.

The encoder half (``TimmEncoder`` / ``probe_encoder``) is imported from the
segmentation wrapper rather than re-implemented: a feature pyramid is a
feature pyramid, and duplicating it would let the two drift apart.
"""

import torch.nn as nn

from genml_kit.models.registry import ModelOutput
from genml_kit.models.seg_model_wrapper.encoder import TimmEncoder, probe_encoder  # noqa: F401


class DetModelWrapper(nn.Module):
  """Encoder backbone + detection head.

  ``forward(pixel_values)`` returns a :class:`~genml_kit.models.registry.ModelOutput`
  whose ``.logits`` attribute holds the :class:`DetHeadOutput` container (the
  generic training loop reads ``.logits`` uniformly across tasks; the
  detection method unpacks the container).

  Parameters
  ----------
  encoder : nn.Module
      Backbone returning an :class:`EncoderPyramid`.
  head : nn.Module
      Detection head (e.g. :class:`~genml_kit.models.decoders.fcos_head.FCOSHead`).
  num_labels : int
      Number of foreground classes.
  id2label : dict, optional
      Compact-label to name map, stored for reporting.
  """

  def __init__(self, encoder, head, num_labels, id2label=None):
    super().__init__()
    self.encoder = encoder
    self.head = head
    self.num_labels = int(num_labels)
    self.id2label = dict(id2label) if id2label else {}
    self.config = _DetConfig(self.num_labels, self.id2label)

  def forward(self, pixel_values=None, **kwargs):
    """Return ``ModelOutput`` carrying the structured detection predictions."""
    pixel_values = kwargs.get("images") if pixel_values is None else pixel_values
    if pixel_values is None:
      raise ValueError("DetModelWrapper.forward requires pixel_values")
    # Encoder: (B, 3, H, W) -> feature pyramid of (B, C_i, H_i, W_i).
    pyramid = self.encoder(pixel_values)
    # Head: pyramid -> per-level class / box / centerness predictions.
    det_out = self.head(pyramid)
    return ModelOutput(det_out)

  def extract_backbone_features(self, pixel_values):
    """Return the deepest encoder feature map (for probes / feature reuse).

    Shape ``(B, C, H/32, W/32)`` -- the coarsest, most semantic level.
    """
    pyramid = self.encoder(pixel_values)
    return pyramid.levels[-1].feature


class _DetConfig:
  """Minimal HF-style ``config`` shim exposing ``num_labels`` / ``id2label``.

  The checkpoint and a few shared helpers read ``model.config.num_labels``
  (as the segmentation wrapper's HF models do); this keeps that contract
  without depending on ``transformers``.
  """

  def __init__(self, num_labels, id2label):
    self.num_labels = num_labels
    self.id2label = id2label


def build_det_model(encoder,
                    head_spec,
                    num_classes,
                    image_size,
                    device="cpu",
                    id2label=None):
  """Probe *encoder*, build the head for its channels, and wrap both.

  This is the single place where the encoder's channel counts are discovered
  (by a dummy forward pass) and threaded into the head, so no caller has to
  know the backbone's shapes ahead of time.

  Parameters
  ----------
  encoder : nn.Module
      Backbone returning an :class:`EncoderPyramid`.
  head_spec : str
      Head spec (registered name or inline ``name:kwargs``, or a ``.py``
      path defining ``create_decoder()``).
  num_classes : int
      Number of foreground classes.
  image_size : int
      Input resolution; the encoder is probed at this size.
  device : str or torch.device
      Device to build and probe on.
  id2label : dict | None
      Optional label map.

  Returns
  -------
  DetModelWrapper
  """
  from genml_kit.models.decoders import build_decoder

  pyramid = probe_encoder(encoder, image_size, device=device)
  head = build_decoder(head_spec, in_channels=pyramid.channels, num_classes=num_classes)
  model = DetModelWrapper(encoder, head, num_labels=num_classes, id2label=id2label)
  model.to(device)
  return model
