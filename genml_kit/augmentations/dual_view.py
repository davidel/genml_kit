"""Dual-view augmentation: apply the same transform twice for contrastive learning."""

from torch import nn


class DualViewTransform(nn.Module):
  """Apply the same transform twice to produce two augmented views."""

  def __init__(self, transform):
    super().__init__()
    self._transform = transform

  def forward(self, image):
    return self._transform(image), self._transform(image)
