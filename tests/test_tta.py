"""Tests for Test-Time Augmentation (TTA)."""

import torch

from genml_kit.training.tta import TTATransform, create_default_tta_transform


class TestDefaultTTATransform:
  """Tests for the built-in default TTA transform."""

  def test_output_shape(self):
    """Default transform should produce (B, 4, C, H, W)."""
    tta = create_default_tta_transform()
    images = torch.randn(2, 3, 64, 64)
    views = tta(images)
    assert views.shape == (2, 4, 3, 64, 64)

  def test_single_image(self):
    """Should work with batch size 1."""
    tta = create_default_tta_transform()
    images = torch.randn(1, 3, 32, 32)
    views = tta(images)
    assert views.shape == (1, 4, 3, 32, 32)

  def test_identity_is_first_view(self):
    """The first view must be the original image unchanged."""
    tta = create_default_tta_transform()
    images = torch.randn(3, 3, 48, 48)
    views = tta(images)
    assert torch.equal(views[:, 0], images)

  def test_views_are_distinct(self):
    """Flip views should differ from identity on non-symmetric input."""
    tta = create_default_tta_transform()
    images = torch.randn(1, 3, 64, 64)
    views = tta(images)
    # At least views 1, 2 should differ from view 0 (identity).
    assert not torch.equal(views[:, 1], views[:, 0])
    assert not torch.equal(views[:, 2], views[:, 0])

  def test_preserves_batch_independence(self):
    """Different batch elements should remain independent."""
    tta = create_default_tta_transform()
    a = torch.randn(1, 3, 32, 32)
    b = torch.randn(1, 3, 32, 32)
    views_a = tta(a)
    views_b = tta(b)
    assert not torch.equal(views_a, views_b)

  def test_dtype_preserved(self):
    """Output dtype should match input dtype."""
    tta = create_default_tta_transform()
    images = torch.randn(2, 3, 32, 32, dtype=torch.float16)
    views = tta(images)
    assert views.dtype == torch.float16


class TestTTATransform:
  """Tests for the TTATransform wrapper class."""

  def test_custom_transform(self):
    """A user-defined 3-view transform should produce (B, 3, C, H, W)."""

    def my_fn(image):
      return torch.stack([image, image.flip(-1), image.flip(-2)])

    tta = TTATransform(my_fn)
    images = torch.randn(4, 3, 16, 16)
    views = tta(images)
    assert views.shape == (4, 3, 3, 16, 16)

  def test_single_view(self):
    """N=1 should just add a view dimension."""

    def identity(image):
      return image.unsqueeze(0)

    tta = TTATransform(identity)
    images = torch.randn(5, 3, 8, 8)
    views = tta(images)
    assert views.shape == (5, 1, 3, 8, 8)
    assert torch.equal(views[:, 0], images)
