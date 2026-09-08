"""TimmProcessor — bridges timm's data config to the genml_kit processor protocol.

timm embeds preprocessing metadata (mean, std, interpolation, crop
percentage) inside each model's ``pretrained_cfg`` dict.  There is no
standalone "processor" object like HuggingFace's ``AutoImageProcessor``.

This class wraps the resolved data config so that:

* ``.image_mean`` / ``.image_std`` satisfy the normalization protocol
  used by ``build_transforms`` and ``build_val_transform``.
* ``__call__(image)`` applies the full model-specific transform pipeline
  (resize, crop, normalize) via ``timm.data.create_transform``.
"""

import timm.data

from genml_kit.models.processors.base import BaseImageProcessor


class TimmProcessor(BaseImageProcessor):
  """Thin adapter that exposes timm preprocessing as a processor object."""

  def __init__(self, data_config, image_size):
    """
    Parameters
    ----------
    data_config : dict
      Output of ``timm.data.resolve_data_config()`` — contains
      ``mean``, ``std``, ``input_size``, ``interpolation``,
      ``crop_pct``, ``crop_mode``.
    image_size : int
      Target resolution (pixels) after resize / crop.
    """
    self._data_config = data_config
    super().__init__(
        image_size,
        data_config["mean"],
        data_config["std"],
    )

  def _build_transform(self):
    """Build a timm-native transform respecting the model's own
    interpolation and crop_pct settings."""
    cfg = dict(self._data_config)
    cfg["input_size"] = (3, self.image_size, self.image_size)
    return timm.data.create_transform(**cfg)

  def __call__(self, image):
    """Apply the full model-specific transform pipeline.

    Parameters
    ----------
    image : PIL.Image.Image
      Raw input image.

    Returns
    -------
    torch.Tensor
      Shape ``(C, H, W)`` — normalized and resized to ``image_size``.
    """
    return self._transform(image)
