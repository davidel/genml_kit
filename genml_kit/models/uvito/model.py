"""UVito — SMP encoder + Transformer for image classification.

Architecture::

    backbone (frozen SMP encoder) → patch projection → [CLS tokens + pos]
    → TransformerEncoder → per-token head_norm → CLS flatten → MLP head → logits
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from genml_kit.models.transformer_utils import TransformerBlock


class UVito(nn.Module):

  def __init__(
      self,
      num_classes,
      encoder_name="resnet50",
      encoder_weights="imagenet",
      img_size=384,
      num_cls_tokens=1,
      transformer_dim=512,
      num_transformer_layers=6,
      nhead=8,
      dim_feedforward=2048,
      dropout=0.1,
      drop_path_rate=0.1,
      use_grad_checkpoint=False,
  ):
    super().__init__()
    self.use_grad_checkpoint = use_grad_checkpoint
    self.num_cls_tokens = num_cls_tokens

    # Step 1: Load the pretrained SMP segmentation encoder
    base_model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )

    # Step 2: Freeze the entire CNN backbone and keep it always in eval mode
    self.frozen_encoder = base_model.encoder
    for param in self.frozen_encoder.parameters():
      param.requires_grad = False
    self.frozen_encoder.eval()

    # Step 3: Determine bottleneck channels from the encoder.
    # SMP encoders expose the feature pyramid as a 5-element sequence:
    #   features[0]  = the input image itself            (B, 3,   H,     W)
    #   features[1..3] = intermediate stage outputs      (B, C_i, H/s_i, W/s_i)
    #   features[-1] = the final (deepest) stage output, i.e. the bottleneck
    # UVito only consumes the bottleneck, hence the [-1] index. For
    # ResNet-50 @ 384px that is (1, 2048, 12, 12).
    # The dummy forward pass is needed because encoder_name is a runtime
    # string, so (c, h, w) cannot be known statically; (h, w) also become
    # the number of spatial transformer tokens below.
    dummy_input = torch.randn(1, 3, img_size, img_size)  # (1, 3, H, W)
    with torch.no_grad():
      bottleneck = self.frozen_encoder(dummy_input)[-1]  # (1, C, H/s, W/s)
    c, h, w = bottleneck.shape[1], bottleneck.shape[2], bottleneck.shape[3]

    # Step 4: Patch Projection — map CNN channels → transformer dimensions
    self.patch_projection = nn.Linear(c, transformer_dim)

    # Step 5: Learnable CLS tokens + positional embeddings
    # Initialized to zero (standard in DINOv2, MAE, DeiT) so the
    # transformer starts from actual image content, not random noise.
    # cls_tokens:   (1, num_cls_tokens, D) — broadcast over the batch
    # pos_embedding: (1, num_cls_tokens + T, D) with T = h * w spatial
    # tokens; leading slots position-encode the CLS tokens, the rest
    # position-encode the h * w spatial tokens.
    self.cls_tokens = nn.Parameter(torch.zeros(1, num_cls_tokens, transformer_dim))
    self.pos_embedding = nn.Parameter(
        torch.zeros(1, num_cls_tokens + h * w, transformer_dim))

    # Step 6: Pre-norm Transformer blocks with DropPath (linear ramp)
    dpr = torch.linspace(0, drop_path_rate, num_transformer_layers).tolist()
    self.transformer_layers = nn.ModuleList([
        TransformerBlock(
            embed_dim=transformer_dim,
            num_heads=nhead,
            dropout=dropout,
            drop_path=dpr[i],
            dim_feedforward=dim_feedforward,
        ) for i in range(num_transformer_layers)
    ])
    self.transformer_norm = nn.LayerNorm(transformer_dim)

    # Step 7: Dropout and final head
    self.pos_drop = nn.Dropout(p=dropout)
    self.head_norm = nn.LayerNorm(transformer_dim)
    if num_classes == 0:
      # timm convention: no classification head (self-supervised mode).
      self.mlp_head = None
    else:
      # Xavier-init classification head (DINOv2 convention: trunc_normal 0.02)
      # Maps the flattened CLS states (B, D * num_cls_tokens) -> (B, num_classes).
      self.mlp_head = nn.Linear(transformer_dim * num_cls_tokens, num_classes)
      nn.init.trunc_normal_(self.mlp_head.weight, std=0.02)
      nn.init.zeros_(self.mlp_head.bias)

  def backbone_features(self, x):
    """Run everything up to the CLS-flattened representation.

    Returns
    -------
    torch.Tensor
        Shape ``(B, num_cls_tokens * transformer_dim)`` — the
        penultimate representation before ``head_norm`` / ``mlp_head``.
    """
    batch_size = x.shape[0]

    # Encode via frozen CNN backbone: (B, 3, H, W) -> (B, C, H/s, W/s).
    # features[-1] = deepest pyramid stage (the bottleneck), same indexing
    # rationale as in __init__ Step 3.
    features = self.frozen_encoder(x)  # list of (B, C_i, H/s_i, W/s_i)
    bottleneck = features[-1]  # (B, C, h, w)
    b, c, h, w = bottleneck.shape

    # Reshape spatial dims → tokens: flatten h * w positions into a
    # sequence so each spatial cell becomes one transformer token.
    spatial_tokens = bottleneck.view(b, c, h * w).permute(0, 2, 1)  # (B, T, C)
    # Per-token channel projection C -> D (the analog of ViT patch embedding).
    spatial_tokens = self.patch_projection(spatial_tokens)  # (B, T, D)

    # Prepend CLS tokens: (B, num_cls, D) + (B, T, D) -> (B, num_cls + T, D)
    cls_tokens_expanded = self.cls_tokens.expand(batch_size, -1, -1)  # (B, num_cls, D)
    tokens = torch.cat((cls_tokens_expanded, spatial_tokens),
                       dim=1)  # (B, num_cls + T, D)

    # Add positional embeddings & dropout (broadcast over the batch dim).
    tokens = tokens + self.pos_embedding  # (B, num_cls + T, D)
    tokens = self.pos_drop(tokens)  # (B, num_cls + T, D)

    # Transformer blocks: sequence length is preserved, so the shape stays
    # (B, num_cls + T, D) throughout.
    for layer in self.transformer_layers:
      if self.use_grad_checkpoint and self.training:
        tokens = torch.utils.checkpoint.checkpoint(
            layer,
            tokens,
            use_reentrant=False,
        )
      else:
        tokens = layer(tokens)
    transformer_output = self.transformer_norm(tokens)  # (B, num_cls + T, D)

    # Extract CLS tokens and flatten: keep the leading num_cls slots,
    # then collapse them into one vector per sample. The downstream _head
    # un-flattens them again so head_norm normalizes per token.
    final_cls_states = transformer_output[:, :self.num_cls_tokens, :]  # (B, num_cls, D)
    return final_cls_states.reshape(batch_size, -1)  # (B, num_cls * D)

  def _head(self, cls_features):
    """Classification head: per-token LayerNorm → Linear (passthrough if headless).

    *cls_features* is the flattened CLS representation ``(B,
    num_cls_tokens * transformer_dim)``. The LayerNorm normalizes each
    CLS token's ``transformer_dim`` features, so the tensor is un-flattened
    to ``(B, num_cls_tokens, transformer_dim)`` first and re-flattened
    after; with ``num_cls_tokens > 1`` normalizing the flat vector would
    fail (wrong ``normalized_shape``) and, even if shapes had matched,
    would mix statistics across independent tokens.
    """
    if self.mlp_head is None:
      return cls_features
    batch_size = cls_features.shape[0]
    # Un-flatten so head_norm normalizes each CLS token independently:
    # (B, num_cls * D) -> (B, num_cls, D)
    tokens = cls_features.reshape(batch_size, self.num_cls_tokens, -1)
    # head_norm: (B, num_cls, D) -> (B, num_cls, D); re-flatten for the head.
    # mlp_head: (B, num_cls * D) -> (B, num_classes) logits.
    return self.mlp_head(self.head_norm(tokens).reshape(batch_size, -1))

  def forward(self, x):
    return self._head(self.backbone_features(x))

  def train(self, mode=True):
    """Override train mode to ensure the frozen CNN backbone strictly
    remains in eval mode.
    """
    super().train(mode)
    self.frozen_encoder.eval()
    return self
