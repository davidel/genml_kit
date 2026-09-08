"""Supervised Contrastive pre-training method.

Reference: Khosla et al., "Supervised Contrastive Learning",
NeurIPS 2020.
"""

from genml_kit.models.contrastive import ContrastiveEncoder
from genml_kit.pretrain.losses.contrastive import supcon_loss
from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method


@register_method
class SupConMethod(PretrainMethod):
  """Supervised contrastive pre-training via NT-Xent loss."""

  NAME = "supcon"
  needs_labels = True

  def add_args(self, p):
    p.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="Projection head output dimension (default: 256).",
    )
    p.add_argument(
        "--proj_hidden",
        type=int,
        default=2048,
        help="Projection head hidden dimension (default: 2048).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="NT-Xent temperature (default: 0.07).",
    )

  def build(self, args, encoder, device):
    model = ContrastiveEncoder(
        encoder,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
    ).to(device)
    model.temperature = args.temperature
    return model

  def train_step(self, model, images, global_step, *, labels=None):
    features = model(images)
    loss = supcon_loss(features, labels, temperature=model.temperature)
    return loss, {
        "loss": loss.item(),
        "temperature": model.temperature,
    }

  def get_checkpoint_state(self, model, args):
    return {
        "method": "supcon",
        "proj_dim": args.proj_dim,
        "proj_hidden": args.proj_hidden,
        "temperature": args.temperature,
    }

  def load_checkpoint_state(self, model, state, args):
    pass  # Projection head is part of the model state dict.

  def validate(self, model, images, num_samples):
    return None  # No pixel-space visualization for contrastive.

  def on_epoch_end(self, model, epoch, writer):
    pass
