"""SimMIM masked-image modeling pre-training method."""
import torch

from genml_kit.models.convvit.masked_encoder import ConvViTMaskedImageEncoder
from genml_kit.models.simmim import SimMIM, simmim_loss, unpatchify
from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method
from genml_kit.training.model_utils import model_mode


def make_mask(images, patch_size, mask_ratio, patch_size_multiplier=1):
  """Create a random block mask for SimMIM.

  Each sample gets ``mask_ratio`` of its patches masked.  Masks are
  drawn per-block of ``patch_size_multiplier`` patches for contiguous
  corruption.

  Args:
    images: ``(B, C, H, W)`` input images.
    patch_size: Patch size used by the encoder.
    mask_ratio: Fraction of patches to mask (0-1).
    patch_size_multiplier: Group patches in blocks of this size.

  Returns:
    Bool mask of shape ``(B, N)`` where ``N`` = number of encoder
    patches.  ``True`` = masked.
  """
  _, _, H, W = images.shape
  num_h = H // patch_size
  num_w = W // patch_size
  num_patches = num_h * num_w

  # Block-granularity mask.
  block = patch_size * patch_size_multiplier
  num_h_blocks = H // block
  num_w_blocks = W // block
  num_blocks = num_h_blocks * num_w_blocks
  patches_per_block = (block // patch_size)**2

  # Number of block-patches to mask per sample.
  n_mask = max(1, int(num_blocks * mask_ratio))

  # (B, num_blocks)
  block_mask = torch.zeros(images.shape[0],
                           num_blocks,
                           dtype=torch.bool,
                           device=images.device)
  for i in range(images.shape[0]):
    idx = torch.randperm(num_blocks, device=images.device)[:n_mask]
    block_mask[i, idx] = True

  # Expand to per-patch mask: (B, num_blocks, patches_per_block)
  block_mask = block_mask.unsqueeze(-1).expand(-1, -1, patches_per_block)
  # (B, num_patches)
  mask = block_mask.reshape(images.shape[0], num_patches)
  return mask


@register_method
class SimMIMMethod(PretrainMethod):
  """SimMIM: simple masked image modeling."""

  NAME = "simmim"

  def add_args(self, parser):
    p = parser.add_argument_group("SimMIM")
    p.add_argument(
        "--mask_ratio",
        type=float,
        default=0.6,
        help="Fraction of patches to mask (default: 0.6).",
    )
    p.add_argument(
        "--decoder_dim",
        type=int,
        default=768,
        help="Hidden dimension of the decoder MLP (default: 768).",
    )
    p.add_argument(
        "--decoder_depth",
        type=int,
        default=2,
        help="Number of Linear->GELU layers in the decoder (default: 2).",
    )

  def build(self, args, encoder, device):
    enc = ConvViTMaskedImageEncoder(encoder)
    model = SimMIM(
        enc,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
    ).to(device)
    model.mask_ratio = args.mask_ratio
    return model

  def train_step(self, model, images, global_step, *, labels=None):
    mask = make_mask(images, model.patch_size, model.mask_ratio)
    output, target = model(images, mask)
    loss = simmim_loss(output, target, mask)
    return loss, {
        "loss": loss.item(),
        "mask_ratio": model.mask_ratio,
    }

  def get_checkpoint_state(self, model, args):
    return {
        "method": "simmim",
        "mask_ratio": model.mask_ratio,
        "decoder_dim": args.decoder_dim,
        "decoder_depth": args.decoder_depth,
    }

  def load_checkpoint_state(self, model, state, args):
    # New-style: restore from method_state in checkpoint.
    if (mask_ratio := state.get("mask_ratio")) is not None:
      model.mask_ratio = mask_ratio
      args.mask_ratio = mask_ratio
      return
    # Backward compat: old checkpoints saved _mask_ratio on the
    # SimMIM model directly (not in method_state).
    old_val = getattr(model, "_mask_ratio", None)
    if old_val is not None:
      model.mask_ratio = old_val
      args.mask_ratio = old_val

  def validate(self, model, images, num_samples):
    """Return reconstructed images for validation logging."""
    with model_mode(model, "eval"):
      samples = images[:num_samples].to(next(model.parameters()).device)
      mask = make_mask(samples, model.patch_size, model.mask_ratio)
      output, _target = model(samples, mask)
      recon = unpatchify(
          output,
          patch_size=model.patch_size,
          img_size=samples.shape[2],
          channels=model.in_channels,
      )
      return recon
