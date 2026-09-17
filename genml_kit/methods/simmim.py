"""SimMIM masked-image modeling pre-training method."""

import torch

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models import load_model
from genml_kit.models.convvit.masked_encoder import ConvViTMaskedImageEncoder
from genml_kit.models.simmim import SimMIM, simmim_loss, unpatchify
from genml_kit.pipelines.contracts import LossOutput
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
class SimMIMMethod(Method):
  """SimMIM: simple masked image modeling."""

  NAME = "simmim"
  NEEDS_LABELS = False
  METRIC_KEY = "loss"
  METRIC_MINIMIZE = True  # loss is minimized

  @classmethod
  def add_args(cls, parser):
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

  def build_model(self, args, device):
    base_model = load_model(
        args.model,
        num_labels=0,
        id2label={},
        label2id={},
        image_size=args.image_size,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        **getattr(args, "model_arg", {}),
    )
    enc = ConvViTMaskedImageEncoder(base_model)
    model = SimMIM(
        enc,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
    ).to(device)
    model.mask_ratio = args.mask_ratio
    # LoRA / freeze / checkpointing target the whole wrapper: PEFT injects
    # into the encoder's Linear layers through it, and the SimMIM decoder
    # stays trainable either way.
    return self._apply_model_extras(args, model, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    images = blob.data
    mask = make_mask(images, model.patch_size, model.mask_ratio)
    output, target = model(images, mask)
    loss = simmim_loss(output, target, mask)
    return LossOutput(loss=loss,
                      metrics={
                          "loss": loss.detach(),
                          "mask_ratio": model.mask_ratio,
                      })

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

  def log_validation(self,
                     model,
                     loader,
                     to_device,
                     writer,
                     global_step,
                     device,
                     num_samples=8):
    """Log one reconstructed vs. original image pair to TensorBoard.

    Pulls a single batch from *loader* (which yields ``DataBlob``
    namedtuples via the production collate), reconstructs the first
    ``num_samples`` images under eval mode, and writes
    ``recon/original`` / ``recon/reconstructed``.
    """
    try:
      blob = next(iter(loader))
    except StopIteration:
      return
    blob = to_device(blob, device)
    images = blob.data
    if isinstance(images, (tuple, list)):
      images = images[0]  # dual-view/multi-crop: log the first view
    images = images[:num_samples]
    with model_mode(model, "eval"):
      mask = make_mask(images, model.patch_size, model.mask_ratio)
      output, _target = model(images, mask)
      recon = unpatchify(
          output,
          patch_size=model.patch_size,
          img_size=images.shape[2],
          channels=model.in_channels,
      )
    writer.add_image("recon/original", images[0], global_step)
    writer.add_image("recon/reconstructed", recon[0].clamp(0, 1), global_step)
