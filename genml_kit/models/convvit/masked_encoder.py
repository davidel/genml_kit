"""ConvViT adapter for the generic masked-image encoder interface."""

from genml_kit.models.masked_encoder import MaskedImageEncoder
from genml_kit.utils.logging import fatal


class ConvViTMaskedImageEncoder(MaskedImageEncoder):
  """Expose ConvViT's patch stem and transformer representation API."""

  def __init__(self, encoder):
    self.model = getattr(encoder, "model", encoder)

  @property
  def embed_dim(self):
    """Embedding width of the wrapped model.

    ConvViT stores it in ``pos_embedding``; a HuggingFace ViT has no such
    attribute and exposes its width via ``config.hidden_size`` instead.
    """
    pos_embedding = getattr(self.model, "pos_embedding", None)
    if pos_embedding is not None:
      return pos_embedding.shape[-1]
    config = getattr(self.model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
      return hidden_size
    fatal(
        "Cannot infer encoder embed_dim: model exposes neither "
        "pos_embedding nor config.hidden_size.", ValueError)

  @property
  def patch_size(self):
    return self.model.patch_embed.patch_size

  @property
  def num_patches(self):
    return self.model.patch_embed.num_patches

  @property
  def in_channels(self):
    return 3

  def patch_embed(self, images):
    return self.model.patch_embed(images)

  def encode_embeddings(self, embeddings):
    return self.model.encoder_forward(embeddings)
