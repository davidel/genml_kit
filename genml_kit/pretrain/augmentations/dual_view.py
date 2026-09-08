"""Dual-view augmentation: apply the same transform twice for contrastive learning."""


class DualViewTransform:
  """Apply the same transform twice to produce two augmented views."""

  def __init__(self, transform):
    self._transform = transform

  def __call__(self, image):
    return self._transform(image), self._transform(image)
