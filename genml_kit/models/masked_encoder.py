"""Interfaces for encoders used by masked image modeling."""

from abc import ABC, abstractmethod


class MaskedImageEncoder(ABC):
  """Adapter interface for one-token-per-patch image encoders.

  Implementations keep architecture-specific details out of the generic
  SimMIM wrapper. ``patch_embed`` must return one token per reconstruction
  patch, and ``encode_embeddings`` must return the corresponding spatial
  features after the encoder stack.

  Implementations must expose the underlying model as ``model``, either as an
  explicit field or as a property; the generic SimMIM wrapper accesses
  ``encoder.model`` directly.
  """

  @property
  @abstractmethod
  def embed_dim(self):
    """Return the encoder token dimension."""

  @property
  @abstractmethod
  def patch_size(self):
    """Return the square patch size in input pixels."""

  @property
  @abstractmethod
  def num_patches(self):
    """Return the number of tokens for the configured image size."""

  @property
  @abstractmethod
  def in_channels(self):
    """Return the number of input image channels."""

  @abstractmethod
  def patch_embed(self, images):
    """Convert images to ``(B, N, D)`` patch embeddings."""

  @abstractmethod
  def encode_embeddings(self, embeddings):
    """Encode modified embeddings and return ``(B, N, D)`` features."""
