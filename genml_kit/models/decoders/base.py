"""Base interfaces for segmentation decoders and encoder feature access.

A segmentation model is ``encoder + decoder``.  The encoder produces a
feature pyramid (or, for single-scale backbones, a set of transformer
hidden states); the decoder consumes those features and returns per-pixel
class logits.  This module defines the two contracts that make the two
halves independent:

* :class:`EncoderPyramid` -- the ordered list of ``(feature, stride)``
  levels handed to the decoder.  Decoder authors read channel counts and
  strides from here instead of hard-coding backbone shapes.
* :class:`BaseDecoder` -- what a decoder must implement: take a pyramid
  and produce ``(B, num_classes, H, W)`` logits.

The channel/stride information is discovered at *build* time by probing
the encoder with a dummy input (see
:func:`genml_kit.models.seg_model_wrapper.probe_encoder`), because the
encoder is chosen at runtime from a string and its shapes cannot be known
statically.
"""

import abc
from collections import namedtuple

import torch.nn as nn

from genml_kit.utils.logging import fatal
from genml_kit.utils.seg import upsample_to

# One feature-pyramid level: ``(feature, stride)`` where ``feature`` is
# ``(B, C, H, W)`` and ``stride`` is the input-to-feature downsampling
# factor (e.g. 4 for a stride-4 map).
EncoderLevel = namedtuple("EncoderLevel", ["feature", "stride"])


class EncoderPyramid:
  """Ordered collection of encoder levels fed to a decoder.

  Parameters
  ----------
  levels : list[EncoderLevel]
      Ordered from highest resolution (small stride) to lowest.
  """

  def __init__(self, levels):
    if not levels:
      fatal("EncoderPyramid requires at least one level", ValueError)
    self._levels = list(levels)

  def __len__(self):
    return len(self._levels)

  def __iter__(self):
    return iter(self._levels)

  def __getitem__(self, idx):
    return self._levels[idx]

  @property
  def levels(self):
    """Return the level list (read-only copy of the container)."""
    return list(self._levels)

  @property
  def channels(self):
    """Return the per-level channel counts, ordered as the levels are."""
    return [level.feature.shape[1] for level in self._levels]

  @property
  def strides(self):
    """Return the per-level strides."""
    return [level.stride for level in self._levels]

  @property
  def spatial_sizes(self):
    """Return ``(H, W)`` per level."""
    return [tuple(level.feature.shape[-2:]) for level in self._levels]

  def reference_size(self):
    """Return the ``(H, W)`` of the finest level (largest resolution)."""
    return tuple(self._levels[0].feature.shape[-2:])

  def summary(self):
    """One-line description for logging."""
    parts = [f"s{level.stride}:C{level.feature.shape[1]}" for level in self._levels]
    return "[{}]".format(", ".join(parts))


class BaseDecoder(nn.Module, abc.ABC):
  """Abstract base for all genml_kit segmentation decoders.

  A decoder receives an :class:`EncoderPyramid` and returns class logits
  shaped ``(B, num_classes, H, W)`` at the resolution of the input image
  (or at the finest pyramid level; the method upsamples if needed).

  The decoder owns no reference to the encoder -- feature extraction is
  the wrapper's responsibility, mirroring :class:`BaseClassifier`.
  """

  def __init__(self, in_channels, num_classes):
    """
    Parameters
    ----------
    in_channels : list[int]
        Per-level encoder channel counts (same order as the pyramid).
        Injected by the wrapper after probing the encoder.
    num_classes : int
        Number of output classes.
    """
    super().__init__()
    self.in_channels = list(in_channels)
    self.num_classes = num_classes

  @abc.abstractmethod
  def forward(self, pyramid):
    """Return ``(B, num_classes, H, W)`` logits from *pyramid*.

    Parameters
    ----------
    pyramid : EncoderPyramid
        Ordered encoder levels.

    Returns
    -------
    torch.Tensor
      ``(B, num_classes, H, W)`` raw scores.
    """


# ``upsample_to`` is re-exported for backwards compatibility; the
# implementation lives in a model-package-free module so the metrics and
# method can share it without importing ``genml_kit.models``.
__all__ = ["EncoderLevel", "EncoderPyramid", "BaseDecoder", "upsample_to"]
