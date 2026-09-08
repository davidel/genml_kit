"""Tests for the SimMIM pre-training components."""

import pytest
import torch

from genml_kit.models.convvit.loader import load_convvit
from genml_kit.models.convvit.masked_encoder import ConvViTMaskedImageEncoder
from genml_kit.models.simmim import (
    SimMIM,
    patchify,
    random_mask,
    simmim_loss,
    unpatchify,
)


class ConvViTConfig:
  """Minimal ConvViT configuration for testing purposes."""

  def __init__(
      self,
      image_size=224,
      patch_size=16,
      in_chans=3,
      depths=None,
      embed_dims=None,
      vit_hidden_dim=768,
      vit_num_heads=12,
      vit_num_layers=12,
      vit_mlp_ratio=4.0,
      drop_rate=0.0,
      drop_path_rate=0.1,
      head_hidden_dim=512,
      num_labels=7,
      id2label=None,
      label2id=None,
  ):
    self.image_size = image_size
    self.patch_size = patch_size
    self.in_chans = in_chans
    self.depths = depths if depths is not None else [2, 2, 6, 2]
    self.embed_dims = embed_dims if embed_dims is not None else [128, 256, 512, 768]
    self.vit_hidden_dim = vit_hidden_dim
    self.vit_num_heads = vit_num_heads
    self.vit_num_layers = vit_num_layers
    self.vit_mlp_ratio = vit_mlp_ratio
    self.drop_rate = drop_rate
    self.drop_path_rate = drop_path_rate
    self.head_hidden_dim = head_hidden_dim
    self.num_labels = num_labels
    self.id2label = id2label if id2label is not None else {}
    self.label2id = label2id if label2id is not None else {}


class TestPatchify:

  def test_roundtrip_224(self):
    img = torch.randn(2, 3, 224, 224)
    patches = patchify(img, patch_size=16)
    restored = unpatchify(patches, patch_size=16, img_size=224, channels=3)
    assert restored.shape == img.shape
    assert torch.allclose(restored, img, atol=1e-5)

  def test_roundtrip_448(self):
    img = torch.randn(2, 3, 448, 448)
    patches = patchify(img, patch_size=16)
    restored = unpatchify(patches, patch_size=16, img_size=448, channels=3)
    assert restored.shape == img.shape
    assert torch.allclose(restored, img, atol=1e-5)

  def test_output_shape(self):
    img = torch.randn(4, 3, 448, 448)
    patches = patchify(img, patch_size=16)
    # 28 * 28 = 784 patches, each 16*16*3 = 768
    assert patches.shape == (4, 784, 768)


class TestRandomMask:

  def test_shape(self):
    mask = random_mask(batch_size=8, num_patches=784, mask_ratio=0.60)
    assert mask.shape == (8, 784)
    assert mask.dtype == torch.bool

  def test_ratio(self):
    mask = random_mask(batch_size=100, num_patches=784, mask_ratio=0.60)
    # Each row should have exactly int(784 * 0.60) = 470 masked patches
    counts = mask.sum(dim=1)
    assert (counts == 470).all()

  def test_different_ratios(self):
    for ratio in [0.30, 0.50, 0.70]:
      mask = random_mask(batch_size=10, num_patches=100, mask_ratio=ratio)
      expected = int(100 * ratio)
      assert (mask.sum(dim=1) == expected).all()


class TestSimMIMLoss:

  def test_perfect_reconstruction(self):
    """Loss should be zero when prediction equals target at masked positions."""
    B, N, D = 4, 784, 768
    target = torch.randn(B, N, D)
    mask = random_mask(B, N, mask_ratio=0.60)
    loss = simmim_loss(target.clone(), target, mask)
    assert loss.item() < 1e-6

  def test_non_negative(self):
    B, N, D = 4, 784, 768
    pred = torch.randn(B, N, D)
    target = torch.randn(B, N, D)
    mask = random_mask(B, N, mask_ratio=0.60)
    loss = simmim_loss(pred, target, mask)
    assert loss.item() >= 0.0

  def test_no_mask_no_loss(self):
    """With no masked patches, loss should be zero (0/0 → 0)."""
    B, N, D = 2, 100, 768
    pred = torch.randn(B, N, D)
    target = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    loss = simmim_loss(pred, target, mask)
    assert loss.item() == 0.0


class TestSimMIM:

  @pytest.fixture
  def simmim_model(self):
    """Build a small headless ConvViT encoder and wrap in SimMIM."""
    encoder = load_convvit(
        image_size=224,
        num_labels=0,
        id2label={},
        label2id={},
        device="cpu",
    )
    model = SimMIM(
        ConvViTMaskedImageEncoder(encoder),
        decoder_dim=768,
        decoder_depth=2,
    )
    return model

  def test_forward_output_shapes(self, simmim_model):
    images = torch.randn(2, 3, 224, 224)
    num_patches = simmim_model.num_patches
    mask = random_mask(2, num_patches, mask_ratio=0.60)
    pred, target = simmim_model(images, mask)
    assert pred.shape == target.shape
    assert pred.shape == (2, num_patches, 768)

  def test_loss_finite(self, simmim_model):
    images = torch.randn(2, 3, 224, 224)
    num_patches = simmim_model.num_patches
    mask = random_mask(2, num_patches, mask_ratio=0.60)
    pred, target = simmim_model(images, mask)
    loss = simmim_loss(pred, target, mask)
    assert torch.isfinite(loss)

  def test_backward(self, simmim_model):
    images = torch.randn(2, 3, 224, 224)
    num_patches = simmim_model.num_patches
    mask = random_mask(2, num_patches, mask_ratio=0.60)
    pred, target = simmim_model(images, mask)
    loss = simmim_loss(pred, target, mask)
    loss.backward()
    # Check that gradients exist for encoder parameters
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in simmim_model.encoder.parameters())
    assert has_grad
