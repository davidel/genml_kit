"""ConvViT adapter for the generic masked-image encoder interface."""

from genml_kit.models.masked_encoder import MaskedImageEncoder


class ConvViTMaskedImageEncoder(MaskedImageEncoder):
  """Expose ConvViT's patch stem and transformer representation API."""

  def __init__(self, encoder):
    self.model = getattr(encoder, "model", encoder)

  @property
  def embed_dim(self):
    return self.model.pos_embedding.shape[-1]

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
