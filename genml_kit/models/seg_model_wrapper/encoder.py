"""Encoder feature extraction for segmentation.

Wraps a *hierarchical* image backbone (timm ``features_only`` models are
the canonical case) so it returns an :class:`EncoderPyramid`.  The wrapper
also records channels/strides by probing the encoder once with a dummy
input -- the shapes depend on the runtime model string, so they cannot be
known statically.
"""

import logging

import torch
import torch.nn as nn

from genml_kit.models.decoders.base import EncoderLevel, EncoderPyramid


class TimmEncoder(nn.Module):
  """Adapter returning a feature pyramid from a timm ``features_only`` model.

  Parameters
  ----------
  model : nn.Module
      A timm model created with ``features_only=True``.  Its forward
      returns a list of ``(B, C, H, W)`` feature maps.
  out_indices : tuple[int] | None
      Which pyramid levels to expose (default: all the model provides).
  """

  def __init__(self, model, out_indices=None):
    super().__init__()
    self.model = model
    info = getattr(model, "feature_info", None)
    if info is None:
      raise ValueError("TimmEncoder requires a model with feature_info "
                       "(create it with features_only=True)")
    self._all_indices = list(info.out_indices)
    if out_indices is None:
      self._selected = list(range(len(self._all_indices)))
    else:
      self._selected = list(out_indices)
    all_channels = list(info.channels())
    reductions = list(info.reduction())
    self._reductions = [reductions[i] for i in self._selected]
    self._expected_channels = [all_channels[i] for i in self._selected]
    self._channels = None

  @property
  def out_channels(self):
    """Per-level channel counts (available after the first forward)."""
    if self._channels is None:
      raise RuntimeError("TimmEncoder.out_channels is only known after a "
                         "forward pass; call probe() first.")
    return list(self._channels)

  def _to_nchw(self, feature, expected_channels):
    """Normalise a timm feature map to ``(B, C, H, W)``.

    Some timm families (e.g. Swin) emit channels-last ``(B, H, W, C)``
    tensors.  We disambiguate using the model's declared channel count:
    whichever trailing/interior dimension matches is the channel axis.
    """
    if feature.shape[1] == expected_channels:
      return feature
    if feature.shape[-1] == expected_channels:
      return feature.permute(0, 3, 1, 2).contiguous()
    # Shapes should always match; fail loudly if timm changes its contract.
    raise ValueError(f"TimmEncoder could not locate the channel axis (expected "
                     f"{expected_channels}) in feature with shape "
                     f"{tuple(feature.shape)}")

  def forward(self, x):
    """Return an :class:`EncoderPyramid` for ``(B, 3, H, W)`` inputs.

    Each timm feature map is normalised to ``(B, C_i, H_i, W_i)``.
    """
    feats = self.model(x)
    if not isinstance(feats, (list, tuple)):
      feats = [feats]
    selected = [feats[i] for i in self._selected]
    selected = [self._to_nchw(f, c) for f, c in zip(selected, self._expected_channels)]
    self._channels = [f.shape[1] for f in selected]
    levels = [EncoderLevel(f, stride) for f, stride in zip(selected, self._reductions)]
    return EncoderPyramid(levels)


def probe_encoder(encoder, image_size, device="cpu"):
  """Run a dummy forward through *encoder* and return an ``EncoderPyramid``.

  Puts the encoder in eval mode for the probe and restores the previous
  training flag, so probing is side-effect free.

  Parameters
  ----------
  encoder : callable
      Returns an :class:`EncoderPyramid` for a ``(1, 3, H, W)`` input.
  image_size : int
      Spatial size of the dummy input.
  device : str | torch.device
      Device to run the probe on.

  Returns
  -------
  EncoderPyramid
  """
  was_training = getattr(encoder, "training", False)
  if hasattr(encoder, "eval"):
    encoder.eval()
  with torch.no_grad():
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    pyramid = encoder(dummy)
  if hasattr(encoder, "train"):
    encoder.train(was_training)
  logging.info("Probed encoder pyramid: %s", pyramid.summary())
  return pyramid
