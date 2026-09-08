"""UVito image processor — mirrors the HF AutoImageProcessor interface.

Applies the same Resize → CenterCrop → ToImage → ToDtype → Normalize
pipeline that HuggingFace uses for ConvNeXtV2 models.
"""

from torchvision.transforms.functional import InterpolationMode

from genml_kit.models.processors.base import BaseImageProcessor

# Default ImageNet normalization used by SMP encoder weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.228, 0.227)


class UVitoProcessor(BaseImageProcessor):
  """Stateless image pre-processor for UVito.

  Mirrors the interface of a HuggingFace ``AutoImageProcessor``: the
  caller passes ``processor(image)`` and receives a ``[C, H, W]``
  ``float32`` tensor ready for the model.
  """

  def __init__(
      self,
      image_size=384,
      mean=IMAGENET_MEAN,
      std=IMAGENET_STD,
  ):
    super().__init__(image_size, mean, std, InterpolationMode.BICUBIC, antialias=True)

  def __call__(self, image, **kwargs):
    return self._transform(image)
