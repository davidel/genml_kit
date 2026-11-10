"""I-JEPA: Image-based Joint-Embedding Predictive Architecture.

Reference: Assran et al., "I-JEPA: Image-based Joint-Embedding
Predictive Architecture", CVPR 2023.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.models import load_model
from genml_kit.models.ema import EMAModel, linear_momentum
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


def _sincos_2d(positions, dim):
  """Fixed 2-D sinusoidal positional embeddings for patch coordinates.

    Each coordinate ``v`` is encoded with the classic sin/cos geometric
    frequency ladder; the row and column axes each contribute half of *dim*.
    Fixed (parameter-free) so any crop size or offset is representable without
    learning a fixed-grid table.

    Args:
      positions: ``(N, 2)`` long tensor of absolute ``(row, col)`` patch
        coordinates.
      dim: Output embedding width.

    Returns:
      ``(N, dim)`` float32 tensor.
    """
  device = positions.device
  d_row = dim // 2
  d_col = dim - d_row

  def _encode(coord, width):
    if width == 0:
      return torch.zeros(coord.shape[0], 0, device=device)
    half = width // 2
    freq = torch.arange(half, dtype=torch.float32, device=device)
    freq = 1.0 / (10000.0**(freq / max(half, 1)))
    ang = coord[:, None].float() * freq[None, :]
    emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
    # The ladder produces 2 * half <= width channels; pad the (rare) odd tail.
    if emb.shape[-1] < width:
      pad = torch.zeros(emb.shape[0], width - emb.shape[-1], device=device)
      emb = torch.cat([emb, pad], dim=-1)
    return emb

  return torch.cat([_encode(positions[:, 0], d_row),
                    _encode(positions[:, 1], d_col)],
                   dim=-1)


class _Predictor(nn.Module):
  """Transformer predictor: masked target embeddings from context.

    Follows the I-JEPA predictor.  The visible context tokens and a set of
    learnable mask-token queries are concatenated and processed by a single
    self-attention transformer; the outputs at the mask-token positions are the
    predictions.  Every token carries a fixed 2-D sinusoidal positional
    embedding of its *absolute* patch coordinate, so the context and the
    target queries share one coordinate frame even when they come from
    different crops.
    """

  def __init__(self, embed_dim, depth=6, num_heads=12, predictor_dim=512):
    super().__init__()
    self.embed_dim = embed_dim
    self.predictor_dim = predictor_dim

    # Project context tokens down to the predictor dimension.
    self.input_proj = nn.Linear(embed_dim, predictor_dim)
    # Shared learnable query placed at each masked target position.
    self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
    nn.init.trunc_normal_(self.mask_token, std=0.02)

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

  def forward(self, context, context_pos, target_pos):
    """Predict embeddings at *target_pos* from *context*.

        Args:
          context: ``(B, N_s, D)`` student encoder output (source tokens).
          context_pos: ``(N_s, 2)`` absolute ``(row, col)`` of each source
            token.
          target_pos: ``(N_t, 2)`` absolute ``(row, col)`` of the masked
            target tokens.

        Returns:
          ``(B, N_t, D)`` predicted embeddings at the target positions.
        """
    b, n_s, _ = context.shape
    n_t = target_pos.shape[0]

    # Context tokens + their absolute positions: (B, N_s, D_P).
    x_ctx = self.input_proj(context)
    x_ctx = x_ctx + _sincos_2d(context_pos, self.predictor_dim).to(x_ctx.dtype)

    # Shared mask query broadcast to the target positions: (B, N_t, D_P).
    x_mask = self.mask_token.expand(b, n_t, -1)
    x_mask = x_mask + _sincos_2d(target_pos, self.predictor_dim).to(x_mask.dtype)

    # Self-attention over [context ; mask queries]; read out the query slots.
    x = torch.cat([x_ctx, x_mask], dim=1)
    x = self.transformer(x)
    return self.output_proj(x[:, n_s:])


def _random_crop(image_size, crop_size):
  """Return random crop coordinates (top, left, h, w)."""
  h = w = image_size
  th = torch.randint(0, h - crop_size + 1, (1,)).item()
  tw = torch.randint(0, w - crop_size + 1, (1,)).item()
  return th, tw, crop_size, crop_size


def _patch_positions(num_h, num_w, row0, col0, device):
  """Absolute ``(row, col)`` grid coordinates of a crop's patches.

    Args:
      num_h: Number of patch rows in the crop.
      num_w: Number of patch columns in the crop.
      row0: Absolute patch-row of the crop's top edge.
      col0: Absolute patch-column of the crop's left edge.
      device: Device for the returned tensor.

    Returns:
      ``(num_h * num_w, 2)`` long tensor in row-major order.
    """
  rows = torch.arange(num_h, device=device) + row0
  cols = torch.arange(num_w, device=device) + col0
  grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
  return grid.reshape(-1, 2)


def _make_block_mask(num_h, num_w, block_size_h, block_size_w, n_blocks, device):
  """Create block masks for I-JEPA target view.

    Returns a boolean mask of shape ``(num_h * num_w,)`` where True = masked.
    """
  # Write whole rectangular blocks into a 2-D grid (writing through an
  # ``expand``ed view aliases memory and raises), then flatten to the
  # ``(num_h * num_w,)`` row-major contract the caller relies on.
  mask = torch.zeros(num_h, num_w, dtype=torch.bool, device=device)
  # Generate candidate block top-left corners.
  rows = torch.arange(0, num_h - block_size_h + 1, block_size_h, device=device)
  cols = torch.arange(0, num_w - block_size_w + 1, block_size_w, device=device)
  grid = torch.stack(torch.meshgrid(rows, cols, indexing="ij"), dim=-1)
  # (num_candidates, 2)
  grid = grid.reshape(-1, 2)
  n_blocks = min(n_blocks, grid.shape[0])
  idx = torch.randperm(grid.shape[0], device=device)[:n_blocks]
  for r, c in grid[idx]:
    mask[r:r + block_size_h, c:c + block_size_w] = True
  return mask.flatten()


@METHODS.register
class IJEPAMethod(Method):
  """I-JEPA: Image-based Joint-Embedding Predictive Architecture."""

  NAME = "ijepa"
  NEEDS_LABELS = False
  METRIC_KEY = "loss"
  # Loss is minimized.
  METRIC_MINIMIZE = True

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
        cache_dir=args.cache_dir,
        device=device,
        **args.model_arg,
    )
    # Apply LoRA / freeze / checkpointing to the student's encoder BEFORE
    # building the student/teacher pair: IJEPA deep-copies the student into
    # the EMA teacher, so adapting afterwards would double the adapters and
    # break teacher/student state-dict symmetry.  The teacher copy inherits
    # the (frozen, never-gradient-updated) adapter weights -- correct EMA
    # semantics.
    encoder = self._apply_model_extras(args, encoder, device)
    student = _PatchEmbedder(encoder).to(device)
    # Teacher starts identical to student; IJEPA freezes it via EMAModel.
    teacher = copy.deepcopy(student)

    predictor = _Predictor(
        embed_dim=student.embed_dim,
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

  def on_epoch_end(self, model, epoch, logger):
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
    self._ema = EMAModel(self.teacher, self.student)
    self.predictor = predictor
    self.momentum = momentum
    self._final_momentum = final_momentum
    self.weight = weight
    # Block masking defaults (matching common I-JEPA configuration).
    self._n_mask_blocks = 4
    self._block_size = 6

  def update_momentum(self, epoch, total_epochs=200):
    """Ramp momentum from ``self.momentum`` to ``self._final_momentum``."""
    m = linear_momentum(epoch, total_epochs, self.momentum, self._final_momentum)
    self._ema.update(m)

  def forward(self, images):
    # Input frames: (B, C, H, W).
    _B, _C, H, W = images.shape
    patch_size = self.student.patch_size

    # Random crops (source and target views): each view stays (B, C, H, W).
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

    # Source: full context through student.
    # Student encoder: (B, C, Hs, Ws) -> (B, N_s, D) patch features.
    z_s = self.student(source_view)

    # Target: full view through teacher (no grad).
    # Teacher encoder: (B, C, Ht, Wt) -> (B, N_t, D) patch features.
    with torch.no_grad():
      z_t = self.teacher(target_view)

    # Block mask on target space.
    # Target grid height.
    t_h = target_view.shape[2] // patch_size
    # Target grid width.
    t_w = target_view.shape[3] // patch_size
    n_mask = self._n_mask_blocks
    block_size = self._block_size
    # (N_t,) boolean mask over target patches
    mask = _make_block_mask(
        t_h,
        t_w,
        min(block_size, t_h),
        min(block_size, t_w),
        n_mask,
        images.device,
    )

    # Flatten the masked positions: (N_t,) -> (M,).
    mask_indices = mask.nonzero(as_tuple=False).squeeze(1)

    # Absolute (row, col) patch coordinates in the *full image* frame, so the
    # context and the target queries are expressed in one shared coordinate
    # system even though the two views are independent crops.
    source_pos = _patch_positions(
        source_view.shape[2] // patch_size,
        source_view.shape[3] // patch_size,
        source_crop[0] // patch_size,
        source_crop[1] // patch_size,
        images.device,
    )
    target_pos = _patch_positions(
        t_h,
        t_w,
        target_crop[0] // patch_size,
        target_crop[1] // patch_size,
        images.device,
    )[mask_indices]

    # Predictor: context (B, N_s, D) + target query positions -> (B, M, D).
    z_pred = self.predictor(z_s, source_pos, target_pos)

    # Loss: L2 between prediction and teacher target.
    # Gather teacher targets at the masked positions: (B, N_t, D) -> (B, M, D).
    z_t_masked = z_t[:, mask_indices, :]
    # Scalar MSE loss over (B, M, D), scaled by self.weight.
    loss = self.weight * F.mse_loss(z_pred, z_t_masked)

    info = {
        "loss": loss.item(),
        "n_masked": float(mask_indices.numel()),
    }
    return loss, info
