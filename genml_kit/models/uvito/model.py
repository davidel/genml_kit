"""UVito — SMP encoder + Transformer for image classification.

Architecture::

    backbone (frozen SMP encoder) → patch projection → [CLS tokens + pos]
    → TransformerEncoder → CLS flatten → head_norm → MLP head → logits
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

    # Step 3: Determine bottleneck channels from the encoder
    dummy_input = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
      bottleneck = self.frozen_encoder(dummy_input)[-1]
    c, h, w = bottleneck.shape[1], bottleneck.shape[2], bottleneck.shape[3]

    # Step 4: Patch Projection — map CNN channels → transformer dimensions
    self.patch_projection = nn.Linear(c, transformer_dim)

    # Step 5: Learnable CLS tokens + positional embeddings
    # Initialized to zero (standard in DINOv2, MAE, DeiT) so the
    # transformer starts from actual image content, not random noise.
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

    # Encode via frozen CNN backbone
    features = self.frozen_encoder(x)
    bottleneck = features[-1]
    b, c, h, w = bottleneck.shape

    # Reshape spatial dims → tokens
    spatial_tokens = bottleneck.view(b, c, h * w).permute(0, 2, 1)
    spatial_tokens = self.patch_projection(spatial_tokens)

    # Prepend CLS tokens
    cls_tokens_expanded = self.cls_tokens.expand(batch_size, -1, -1)
    tokens = torch.cat((cls_tokens_expanded, spatial_tokens), dim=1)

    # Add positional embeddings & dropout
    tokens = tokens + self.pos_embedding
    tokens = self.pos_drop(tokens)

    # Transformer blocks
    for layer in self.transformer_layers:
      if self.use_grad_checkpoint and self.training:
        tokens = torch.utils.checkpoint.checkpoint(
            layer,
            tokens,
            use_reentrant=False,
        )
      else:
        tokens = layer(tokens)
    transformer_output = self.transformer_norm(tokens)

    # Extract CLS tokens and flatten
    final_cls_states = transformer_output[:, :self.num_cls_tokens, :]
    return final_cls_states.reshape(batch_size, -1)

  def _head(self, cls_features):
    """Classification head: LayerNorm → Linear (passthrough if headless)."""
    return (cls_features if self.mlp_head is None else self.mlp_head(
        self.head_norm(cls_features)))

  def forward(self, x):
    return self._head(self.backbone_features(x))

  def train(self, mode=True):
    """Override train mode to ensure the frozen CNN backbone strictly
    remains in eval mode.
    """
    super().train(mode)
    self.frozen_encoder.eval()
    return self
