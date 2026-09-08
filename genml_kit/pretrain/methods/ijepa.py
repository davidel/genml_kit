"""I-JEPA: Image-based Joint-Embedding Predictive Architecture.

Reference: Assran et al., "I-JEPA: Image-based Joint-Embedding
Predictive Architecture", CVPR 2023.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method
from genml_kit.training.model_utils import set_train_mode
from genml_kit.utils.transformer import build_transformer_encoder


class _PatchEmbedder(nn.Module):
  """Thin wrapper that exposes a (patch_embed + encode) interface."""

  def __init__(self, encoder):
    super().__init__()
    self.encoder = encoder
    self.model = getattr(encoder, "model", encoder)

  @property
  def patch_size(self):
    return self.model.patch_embed.patch_size

  @property
  def embed_dim(self):
    return self.model.pos_embedding.shape[-1]

  @property
  def num_patches(self):
    return self.model.patch_embed.num_patches

  def forward(self, images):
    """Return ``(B, N, D)`` patch features."""
    embeds = self.model.patch_embed(images)
    return self.model.encoder_forward(embeds)


class _Predictor(nn.Module):
  """Small transformer that predicts target embeddings from context."""

  def __init__(self, embed_dim, num_patches, depth=6, num_heads=12, predictor_dim=512):
    super().__init__()
    self.embed_dim = embed_dim
    self.predictor_dim = predictor_dim

    # Project down to predictor dimension.
    self.input_proj = nn.Linear(embed_dim, predictor_dim)
    # Positional bias (learnable).
    self.pos_bias = nn.Parameter(torch.zeros(1, num_patches, predictor_dim))
    nn.init.trunc_normal_(self.pos_bias, std=0.02)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=predictor_dim,
        nhead=num_heads,
        dim_feedforward=predictor_dim * 4,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    # The nested-tensor fast path is post-norm only and unused here (no
    # padding mask), so disable it to avoid the construction-time warning.
    self.transformer = build_transformer_encoder(
        encoder_layer,
        num_layers=depth,
        enable_nested_tensor=False,
    )
    self.output_proj = nn.Linear(predictor_dim, embed_dim)

  def forward(self, context, mask_indices):
    """Predict embeddings at *mask_indices* from *context*.

        Args:
          context: ``(B, N, D)`` student encoder output.
          mask_indices: ``(B, M)`` long tensor of masked patch indices.

        Returns:
          ``(B, M, D)`` predicted embeddings.
        """
    _B, N, _ = context.shape
    x = self.input_proj(context) + self.pos_bias[:, :N, :]
    x = self.transformer(x)

    # Gather predictions at mask positions.
    idx = mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, M, D)
    return self.output_proj(torch.gather(x, 1, idx))


def _random_crop(image_size, crop_size):
  """Return random crop coordinates (top, left, h, w)."""
  h = w = image_size
  th = torch.randint(0, h - crop_size + 1, (1,)).item()
  tw = torch.randint(0, w - crop_size + 1, (1,)).item()
  return th, tw, crop_size, crop_size


def _make_block_mask(num_h, num_w, block_size_h, block_size_w, n_blocks, device):
  """Create block masks for I-JEPA target view.

    Returns a boolean mask of shape ``(num_h * num_w,)`` where True = masked.
    """
  mask = torch.zeros(num_h * num_w, dtype=torch.bool, device=device)
  # Generate candidate block top-left corners.
  rows = torch.arange(0, num_h - block_size_h + 1, block_size_h, device=device)
  cols = torch.arange(0, num_w - block_size_w + 1, block_size_w, device=device)
  grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
  grid = grid.reshape(-1, 2)  # (num_candidates, 2)
  n_blocks = min(n_blocks, grid.shape[0])
  idx = torch.randperm(grid.shape[0], device=device)[:n_blocks]
  for r, c in grid[idx]:
    mask[r:r + block_size_h].unsqueeze(1).expand(-1, num_w)[:,
                                                            c:c + block_size_w] = True
  return mask


@register_method
class IJEPAMethod(PretrainMethod):
  """I-JEPA: Image-based Joint-Embedding Predictive Architecture."""

  NAME = "ijepa"

  def add_args(self, parser):
    p = parser.add_argument_group("I-JEPA")
    p.add_argument(
        "--teacher_momentum",
        type=float,
        default=0.996,
        help="Initial EMA momentum for teacher (default: 0.996).",
    )
    p.add_argument(
        "--teacher_final_momentum",
        type=float,
        default=1.0,
        help="Final EMA momentum after cosine ramp (default: 1.0).",
    )
    p.add_argument(
        "--predictor_depth",
        type=int,
        default=6,
        help="Transformer depth of the predictor MLP (default: 6).",
    )
    p.add_argument(
        "--predictor_dim",
        type=int,
        default=512,
        help="Hidden dimension of the predictor (default: 512).",
    )
    p.add_argument(
        "--predictor_heads",
        type=int,
        default=12,
        help="Number of attention heads in the predictor (default: 12).",
    )
    p.add_argument(
        "--ijepa_weight",
        type=float,
        default=1.0,
        help="Scalar weight for the I-JEPA loss (default: 1.0).",
    )

  def build(self, args, encoder, device):
    student = _PatchEmbedder(encoder).to(device)
    teacher = copy.deepcopy(student)
    # Teacher starts identical to student, no grad.
    for p in teacher.parameters():
      p.requires_grad = False

    num_patches = student.num_patches
    predictor = _Predictor(
        embed_dim=student.embed_dim,
        num_patches=num_patches,
        depth=args.predictor_depth,
        num_heads=args.predictor_heads,
        predictor_dim=args.predictor_dim,
    ).to(device)

    return IJEPA(
        student=student,
        teacher=teacher,
        predictor=predictor,
        momentum=args.teacher_momentum,
        final_momentum=args.teacher_final_momentum,
        weight=args.ijepa_weight,
    )

  def train_step(self, model, images, global_step, *, labels=None):
    set_train_mode(model, 'train')
    loss, info = model(images)
    return loss, info

  def get_checkpoint_state(self, model, args):
    return {
        "method": "ijepa",
        "teacher_momentum": args.teacher_momentum,
        "teacher_final_momentum": args.teacher_final_momentum,
        "predictor_depth": args.predictor_depth,
        "predictor_dim": args.predictor_dim,
        "predictor_heads": args.predictor_heads,
    }

  def load_checkpoint_state(self, model, state, args):
    # Momentum / hyperparams restored automatically on next build.
    pass

  def on_epoch_end(self, model, epoch, writer):
    """Ramp teacher EMA momentum."""
    if isinstance(model, IJEPA):
      model.update_momentum(epoch)


class IJEPA(nn.Module):
  """Full I-JEPA model with student, teacher, and predictor."""

  def __init__(
      self,
      student,
      teacher,
      predictor,
      momentum=0.996,
      final_momentum=1.0,
      weight=1.0,
  ):
    super().__init__()
    self.student = student
    self.teacher = teacher
    self.predictor = predictor
    self.momentum = momentum
    self._final_momentum = final_momentum
    self.weight = weight
    # Block masking defaults (matching common I-JEPA configuration).
    self._n_mask_blocks = 4
    self._block_size = 6

  def update_momentum(self, epoch, total_epochs=200):
    """Ramp momentum from ``self.momentum`` to ``self._final_momentum``."""
    t = min(epoch / total_epochs, 1.0)
    m = self.momentum + (self._final_momentum - self.momentum) * t
    for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
      pt.data.mul_(m).add_(ps.data, alpha=1 - m)

  def forward(self, images):
    _B, _C, H, W = images.shape
    patch_size = self.student.patch_size

    # -- Random crops (source and target views).
    source_size = int(min(H, W) * 0.5)
    target_size = int(min(H, W) * 0.85)
    source_size = max(source_size, patch_size * 2)
    target_size = max(target_size, patch_size * 2)

    source_crop = _random_crop(min(H, W), source_size)
    target_crop = _random_crop(min(H, W), target_size)

    source_view = images[
        :,
        :,
        source_crop[0]:source_crop[0] + source_crop[2],
        source_crop[1]:source_crop[1] + source_crop[3],
    ]
    target_view = images[
        :,
        :,
        target_crop[0]:target_crop[0] + target_crop[2],
        target_crop[1]:target_crop[1] + target_crop[3],
    ]

    # -- Source: full context through student.
    z_s = self.student(source_view)  # (B, N_s, D)

    # -- Target: full view through teacher (no grad).
    with torch.no_grad():
      z_t = self.teacher(target_view)  # (B, N_t, D)

    # -- Block mask on target space.
    t_h = target_view.shape[2] // patch_size
    t_w = target_view.shape[3] // patch_size
    n_mask = self._n_mask_blocks
    block_size = self._block_size
    mask = _make_block_mask(
        t_h,
        t_w,
        min(block_size, t_h),
        min(block_size, t_w),
        n_mask,
        images.device,
    )  # (N_t,)

    mask_indices = mask.nonzero(as_tuple=False).squeeze(1)  # (M,)

    # -- Predict masked target embeddings from source context.
    context = z_s  # (B, N_s, D)
    idx_expanded = mask_indices.unsqueeze(0).expand(z_s.shape[0], -1)
    z_pred = self.predictor(context, idx_expanded)  # (B, M, D)

    # -- Loss: L2 between prediction and teacher target.
    z_t_masked = z_t[:, mask_indices, :]  # (B, M, D)
    loss = self.weight * F.mse_loss(z_pred, z_t_masked)

    info = {
        "loss": loss.item(),
        "n_masked": float(mask_indices.numel()),
    }
    return loss, info
