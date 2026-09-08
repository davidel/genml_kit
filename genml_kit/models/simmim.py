"""Generic SimMIM masked-image-modeling components."""

import torch
import torch.nn as nn

from genml_kit.models.masked_encoder import MaskedImageEncoder
from genml_kit.utils.logging import fatal


def patchify(images, patch_size, channels=None):
  """Convert ``(B, C, H, W)`` images to flattened patch pixels."""
  if images.ndim != 4:
    fatal(f"Expected images with shape (B, C, H, W), got {images.shape}", ValueError)
  if isinstance(patch_size, tuple):
    if patch_size[0] != patch_size[1]:
      fatal("SimMIM currently requires square patches", ValueError)
    patch_size = patch_size[0]
  B, C, H, W = images.shape
  if channels is not None and channels != C:
    fatal(f"Expected {channels} channels, got {C}", ValueError)
  if H % patch_size or W % patch_size:
    fatal("Image dimensions must be divisible by patch_size", ValueError)
  h, w = H // patch_size, W // patch_size
  x = images.reshape(B, C, h, patch_size, w, patch_size)
  x = x.permute(0, 2, 4, 3, 5, 1)
  return x.reshape(B, h * w, patch_size * patch_size * C)


def unpatchify(patches, patch_size, img_size, channels=3):
  """Convert flattened patch pixels back to ``(B, C, H, W)`` images."""
  if isinstance(patch_size, tuple):
    if patch_size[0] != patch_size[1]:
      fatal("SimMIM currently requires square patches", ValueError)
    patch_size = patch_size[0]
  if isinstance(img_size, tuple):
    if img_size[0] != img_size[1]:
      fatal("SimMIM currently requires square images", ValueError)
    img_size = img_size[0]
  B, N, D = patches.shape
  expected = patch_size * patch_size * channels
  if expected != D:
    fatal(f"Expected patch dimension {expected}, got {D}", ValueError)
  grid = img_size // patch_size
  if grid * patch_size != img_size or grid * grid != N:
    fatal("Patch count and image size do not match", ValueError)
  x = patches.reshape(B, grid, grid, patch_size, patch_size, channels)
  x = x.permute(0, 5, 1, 3, 2, 4)
  return x.reshape(B, channels, img_size, img_size)


def random_mask(batch_size, num_patches, mask_ratio=0.60, device=None):
  """Create a boolean ``(B, N)`` mask; ``True`` denotes a masked patch."""
  num_masked = int(num_patches * mask_ratio)
  mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
  if num_masked:
    noise = torch.rand(batch_size, num_patches, device=device)
    indices = noise.argsort(dim=1)[:, :num_masked]
    mask.scatter_(1, indices, True)
  return mask


def simmim_loss(pred, target, mask):
  """Compute mean squared reconstruction error over masked patches."""
  if pred.shape != target.shape or mask.shape != pred.shape[:2]:
    fatal("Prediction, target, and mask shapes are incompatible", ValueError)
  loss = (pred - target)**2
  loss = loss.mean(dim=-1)
  mask_float = mask.float()
  return (loss * mask_float).sum() / mask_float.sum().clamp(min=1)


class SimMIM(nn.Module):
  """Generic SimMIM wrapper around a masked-image encoder adapter."""

  def __init__(self, encoder: MaskedImageEncoder, decoder_dim=768, decoder_depth=2):
    super().__init__()
    self.encoder = encoder.model
    self._encoder_api = encoder
    self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
    patch_size = encoder.patch_size
    if isinstance(patch_size, tuple):
      if patch_size[0] != patch_size[1]:
        fatal("SimMIM currently requires square patches", ValueError)
      patch_size = patch_size[0]
    self._patch_pixel_dim = patch_size * patch_size * encoder.in_channels

    layers = []
    in_dim = encoder.embed_dim
    for _ in range(decoder_depth):
      layers.extend([nn.Linear(in_dim, decoder_dim), nn.GELU()])
      in_dim = decoder_dim
    layers.append(nn.Linear(decoder_dim, self._patch_pixel_dim))
    self.decoder = nn.Sequential(*layers)

  @property
  def patch_size(self):
    """Return the encoder's input patch size."""
    return self._encoder_api.patch_size

  @property
  def num_patches(self):
    """Return the encoder's patch count."""
    return self._encoder_api.num_patches

  @property
  def in_channels(self):
    """Return the encoder's input channel count."""
    return self._encoder_api.in_channels

  def forward(self, images, mask):
    target = patchify(
        images,
        patch_size=self._encoder_api.patch_size,
        channels=self._encoder_api.in_channels,
    )
    embeddings = self._encoder_api.patch_embed(images)
    if embeddings.shape[:2] != mask.shape:
      fatal(f"Expected mask shape {embeddings.shape[:2]}, got {mask.shape}", ValueError)
    mask_tokens = self.mask_token.expand(embeddings.shape[0], embeddings.shape[1], -1)
    embeddings = torch.where(mask.unsqueeze(-1), mask_tokens, embeddings)
    features = self._encoder_api.encode_embeddings(embeddings)
    return self.decoder(features), target
