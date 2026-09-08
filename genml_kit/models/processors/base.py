"""Base image processor shared by custom model backends.

All custom processors (ConvViT, UVito, timm) follow the same protocol:
``__call__`` accepts a PIL image and returns a normalised tensor, and
exposes ``image_mean`` / ``image_std`` / ``size`` properties for
HuggingFace-compatibility.

This base class encapsulates the shared property logic and the standard
torchvision ``Resize → CenterCrop → ToImage → ToDtype → Normalize``
pipeline.  Subclasses can override ``_build_transform`` for custom
pipelines (e.g. timm's native transform).
"""

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode


class BaseImageProcessor:
  """Shared image-processor logic for custom model backends.

  Subclasses set ``self._mean`` / ``self._std`` / ``self.image_size``
  (typically by calling ``super().__init__``) and may override
  ``_build_transform`` to customise the transform pipeline.
  """

  def __init__(
      self,
      image_size,
      mean,
      std,
      interpolation=InterpolationMode.BICUBIC,
      antialias=None,
  ):
    self.image_size = image_size
    self._mean = mean
    self._std = std
    self._interpolation = interpolation
    self._antialias = antialias
    self._transform = self._build_transform()

  def _build_transform(self):
    """Return the standard torchvision v2 transform pipeline."""
    resize_kwargs = {"interpolation": self._interpolation}
    if self._antialias is not None:
      resize_kwargs["antialias"] = self._antialias
    return v2.Compose([
        v2.Resize(self.image_size, **resize_kwargs),
        v2.CenterCrop(self.image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=self._mean, std=self._std),
    ])

  def __call__(self, image):
    """Apply the transform pipeline to a single image."""
    return self._transform(image)

  @property
  def image_mean(self):
    """HF-compatible normalization mean."""
    return list(self._mean)

  @property
  def image_std(self):
    """HF-compatible normalization std."""
    return list(self._std)

  @property
  def size(self):
    """Return processor resolution as a dict (HF compatibility)."""
    return {"height": self.image_size, "width": self.image_size}

  def __repr__(self):
    return (f"{type(self).__name__}(image_size={self.image_size}, "
            f"mean={self._mean}, std={self._std})")
