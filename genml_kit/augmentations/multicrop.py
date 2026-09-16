"""Multi-crop augmentation for DINO.

Produces 2 global crops (large) and N local crops (small) from a single
image, with separate augmentation pipelines for each.

The split follows the DINO recipe (Caron et al., ICCV 2021 —
https://arxiv.org/abs/2104.14294): global crops cover a large fraction
of the original image area (default 0.4–1.0), local crops a small one
(default 0.05–0.4).  The contrast between large and small resolutions is
what teaches the student that local views must match the global
representation.
"""

import torch
from torchvision.transforms import v2


class MultiCropTransform:
  """Generate global + local crops with distinct augmentation strengths.

  Calling the transform on a PIL image returns a list of
  ``2 + local_num`` tensors: index 0 and 1 are the global crops
  (``global_size`` px, `global_transform` pipeline), indices 2.. are the
  local crops (``local_size`` px, `local_transform` pipeline).  Use
  :meth:`split_crops` to convert the list into the
  ``(global_crops, local_crops)`` stacked tensors expected by
  :meth:`genml_kit.models.dino.DINO.forward`.

  Args:
      global_size: Spatial size of global crops.
      local_size: Spatial size of local crops.
      global_scale: (min, max) fraction of image area for global crops.
      local_scale: (min, max) fraction of image area for local crops.
      local_num: Number of local crops.
      global_transform: Transform applied to global crops.
      local_transform: Transform applied to local crops.
  """

  def __init__(self,
               global_size,
               local_size=96,
               global_scale=(0.4, 1.0),
               local_scale=(0.05, 0.4),
               local_num=8,
               global_transform=None,
               local_transform=None):
    self.global_size = global_size
    self.local_size = local_size
    self.local_num = local_num
    self.global_crop = v2.RandomResizedCrop(global_size,
                                            scale=global_scale,
                                            interpolation=v2.InterpolationMode.BICUBIC)
    self.local_crop = v2.RandomResizedCrop(local_size,
                                           scale=local_scale,
                                           interpolation=v2.InterpolationMode.BICUBIC)
    self.global_transform = global_transform or _default_global_transform(global_size)
    self.local_transform = local_transform or _default_local_transform()

  def __call__(self, image):
    """Return list of tensors: [global1, global2, local1, ..., localN]."""
    crops = []
    for _ in range(2):
      crop = self.global_crop(image)
      crops.append(self.global_transform(crop))
    for _ in range(self.local_num):
      crop = self.local_crop(image)
      crops.append(self.local_transform(crop))
    return crops

  def split_crops(self, crops):
    """Split a list of crops into (global_crops, local_crops) tensors."""
    global_crops = torch.stack(crops[:2])
    local_crops = torch.stack(crops[2:])
    return global_crops, local_crops


def _default_global_transform(image_size):
  """Standard DINO global-view pipeline: flip, jitter, blur, normalise."""
  return v2.Compose([
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomApply([
          v2.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.8, hue=0.1),
      ],
                     p=0.8),
      v2.RandomGrayscale(p=0.2),
      v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
      v2.RandomSolarize(threshold=0.5, p=0.2),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])


def _default_local_transform():
  """Standard DINO local-view pipeline: same recipe as the global one."""
  return v2.Compose([
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomApply([
          v2.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.8, hue=0.1),
      ],
                     p=0.8),
      v2.RandomGrayscale(p=0.2),
      v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])
