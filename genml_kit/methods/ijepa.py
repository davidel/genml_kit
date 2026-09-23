"""I-JEPA: Image-based Joint-Embedding Predictive Architecture.

Reference: Assran et al., "I-JEPA: Image-based Joint-Embedding
Predictive Architecture", CVPR 2023.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models import load_model
from genml_kit.pipelines.contracts import LossOutput
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
    # Patch embed + encoder: (B, C, H, W) -> (B, N, D) embeddings -> (B, N, D).
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
    # Project to the predictor dim and add the per-position bias:
    # (B, N, D) -> (B, N, D_P).
    x = self.input_proj(context) + self.pos_bias[:, :N, :]
    # Transformer over the context sequence: (B, N, D_P) unchanged.
    x = self.transformer(x)

    # Gather the (B, M) masked indices, then project to the target dim:
    # (B, M, D_P) -> (B, M, D).
    idx = mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
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
class IJEPAMethod(Method):
  """I-JEPA: Image-based Joint-Embedding Predictive Architecture."""

  NAME = "ijepa"
  NEEDS_LABELS = False
  METRIC_KEY = "loss"
  METRIC_MINIMIZE = True  # loss is minimized

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("I-JEPA")
    group.add_argument(
        "--teacher_momentum",
        type=float,
        default=0.996,
        help="Initial EMA momentum for teacher.",
    )
    group.add_argument(
        "--teacher_final_momentum",
        type=float,
        default=1.0,
        help="Final EMA momentum after cosine ramp.",
    )
    group.add_argument(
        "--predictor_depth",
        type=int,
        default=6,
        help="Transformer depth of the predictor MLP.",
    )
    group.add_argument(
        "--predictor_dim",
        type=int,
        default=512,
        help="Hidden dimension of the predictor.",
    )
    group.add_argument(
        "--predictor_heads",
        type=int,
        default=12,
        help="Number of attention heads in the predictor.",
    )
    group.add_argument(
        "--ijepa_weight",
        type=float,
        default=1.0,
        help="Scalar weight for the I-JEPA loss.",
    )

  def build_model(self, args, device):
    encoder = load_model(
        args.model,
        num_labels=0,
        id2label={},
        label2id={},
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        **getattr(args, "model_arg", {}),
    )
    # Apply LoRA / freeze / checkpointing to the student's encoder BEFORE
    # building the student/teacher pair: IJEPA deep-copies the student into
    # the EMA teacher, so adapting afterwards would double the adapters and
    # break teacher/student state-dict symmetry.  The teacher copy inherits
    # the (frozen, never-gradient-updated) adapter weights -- correct EMA
    # semantics.
    encoder = self._apply_model_extras(args, encoder, device)
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

  def train_step(self, model, blob, global_step, *, labels=None):
    set_train_mode(model, 'train')
    loss, info = model(blob.data)
    return LossOutput(
        loss=loss,
        metrics={
            k: torch.tensor(v, dtype=torch.float32) for k, v in info.items()
        })

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
    # Input frames: (B, C, H, W).
    _B, _C, H, W = images.shape
    patch_size = self.student.patch_size

    # -- Random crops (source and target views): each view stays (B, C, H, W).
    source_size = int(min(H, W) * 0.5)
    target_size = int(min(H, W) * 0.85)
    source_size = max(source_size, patch_size * 2)
    target_size = max(target_size, patch_size * 2)

    source_crop = _random_crop(min(H, W), source_size)
    target_crop = _random_crop(min(H, W), target_size)

    # Crop the two views: (B, C, H, W) -> (B, C, Hs, Ws) and (B, C, Ht, Wt).
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
    # Student encoder: (B, C, Hs, Ws) -> (B, N_s, D) patch features.
    z_s = self.student(source_view)

    # -- Target: full view through teacher (no grad).
    # Teacher encoder: (B, C, Ht, Wt) -> (B, N_t, D) patch features.
    with torch.no_grad():
      z_t = self.teacher(target_view)

    # -- Block mask on target space.
    t_h = target_view.shape[2] // patch_size  # target grid height
    t_w = target_view.shape[3] // patch_size  # target grid width
    n_mask = self._n_mask_blocks
    block_size = self._block_size
    mask = _make_block_mask(
        t_h,
        t_w,
        min(block_size, t_h),
        min(block_size, t_w),
        n_mask,
        images.device,
    )  # (N_t,) boolean mask over target patches

    # Flatten the masked positions: (N_t,) -> (M,).
    mask_indices = mask.nonzero(as_tuple=False).squeeze(1)

    # -- Predict masked target embeddings from source context.
    context = z_s  # (B, N_s, D)
    # Broadcast mask indices over the batch: (M,) -> (B, M).
    idx_expanded = mask_indices.unsqueeze(0).expand(z_s.shape[0], -1)
    # Predictor: (B, N_s, D) context + (B, M) indices -> (B, M, D).
    z_pred = self.predictor(context, idx_expanded)

    # -- Loss: L2 between prediction and teacher target.
    # Gather teacher targets at the masked positions: (B, N_t, D) -> (B, M, D).
    z_t_masked = z_t[:, mask_indices, :]
    # Scalar MSE loss over (B, M, D), scaled by self.weight.
    loss = self.weight * F.mse_loss(z_pred, z_t_masked)

    info = {
        "loss": loss.item(),
        "n_masked": float(mask_indices.numel()),
    }
    return loss, info
