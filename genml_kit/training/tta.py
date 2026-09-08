"""Test-Time Augmentation (TTA) for inference / evaluation.

TTA runs each image through multiple augmented views, averages the
softmax probabilities across views, and returns the most-likely class.

Usage::

    from genml_kit.training.tta import create_default_tta_transform

    tta = create_default_tta_transform(image_size=448)
    # tta: (B, C, H, W) -> (B, N, C, H, W)

External scripts
----------------
A user-supplied script must define ``create_tta_transform(image_size, **kwargs)``
that returns a per-sample callable ``(C, H, W) -> (N, C, H, W)``.
"""

import torch
import torchvision.transforms.v2 as v2

from genml_kit.utils.script import load_extern


class TTATransform:
  """Wrap a per-sample transform into a batched TTA transform.

  Args:
      fn: A callable ``(C, H, W) -> (N, C, H, W)`` that applies N
          augmentations to a single image and returns a stack.
  """

  def __init__(self, fn):
    self._fn = fn

  def __call__(self, images):
    """Apply *fn* to every sample and stack.

    Args:
        images: ``(B, C, H, W)`` tensor.

    Returns:
        ``(B, N, C, H, W)`` tensor where *N* is the number of
        views produced by the underlying per-sample function.
    """
    views = [self._fn(img) for img in images]
    return torch.stack(views)  # (B, N, C, H, W)


def create_default_tta_transform(image_size=None, **kwargs):
  """Create the default TTA transform (identity + flips).

  Returns a batched callable ``(B, C, H, W) -> (B, N=4, C, H, W)``.

  The 4 views are:

  1. Identity (original image)
  2. Horizontal flip
  3. Vertical flip
  4. Horizontal + vertical flip (180° rotation)
  """
  h_flip = v2.RandomHorizontalFlip(p=1.0)
  v_flip = v2.RandomVerticalFlip(p=1.0)

  def per_sample(image):
    return torch.stack([
        image,
        h_flip(image),
        v_flip(image),
        v_flip(h_flip(image)),
    ])

  return TTATransform(per_sample)


def load_tta_transform(path_or_url, image_size=None):
  """Load an external TTA transform script and return a batched transform.

  The script must define ``create_tta_transform(image_size, **kwargs)``
  returning a per-sample callable ``(C, H, W) -> (N, C, H, W)``.

  Args:
      path_or_url: Local path or HTTP/HTTPS URL.
      image_size: Forwarded to the script's ``create_tta_transform``.

  Returns:
      A ``TTATransform`` that operates on ``(B, C, H, W)`` batches.
  """
  fn = load_extern(path_or_url, "create_tta_transform")
  per_sample = fn(image_size=image_size)
  return TTATransform(per_sample)
