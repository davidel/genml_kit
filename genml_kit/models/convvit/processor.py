"""ConvViT image processor — mirrors the HF AutoImageProcessor interface.

Applies the same Resize → CenterCrop → ToImage → ToDtype → Normalize
pipeline that HuggingFace uses for ConvNeXtV2 models.
"""

import torch

from genml_kit.models.processors.base import BaseImageProcessor

# Default ImageNet normalization used by ConvNeXtV2 pre-training.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ConvViTProcessor(BaseImageProcessor):
  """Image processor for ConvViT.

  Accepts PIL images (single or list) and returns normalised tensors
  ready for the model's ``forward()`` method.
  """

  def __init__(
      self,
      image_size=224,
      mean=IMAGENET_MEAN,
      std=IMAGENET_STD,
  ):
    super().__init__(image_size, mean, std)

  def __call__(self, images):
    """Process one or more PIL images into a ``pixel_values`` tensor.

    Parameters
    ----------
    images : PIL.Image or list[PIL.Image]
        Raw input image(s).

    Returns
    -------
    torch.Tensor
        Shape ``[B, 3, H, W]`` (always 4-D, even for a single image).
    """
    if isinstance(images, list):
      return torch.stack([self._transform(img) for img in images])
    # Single image → unsqueeze to add batch dim.
    return self._transform(images).unsqueeze(0)
