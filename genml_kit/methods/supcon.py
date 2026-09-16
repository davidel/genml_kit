"""Supervised Contrastive pre-training method.

Reference: Khosla et al., "Supervised Contrastive Learning",
NeurIPS 2020.
"""

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models import load_model
from genml_kit.models.contrastive import ContrastiveEncoder
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.losses.contrastive import supcon_loss


@register_method
class SupConMethod(Method):
  """Supervised contrastive pre-training via NT-Xent loss."""

  NAME = "supcon"
  METRIC_KEY = "loss"
  NEEDS_LABELS = True

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
    model = ContrastiveEncoder(
        encoder,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
    ).to(device)
    model.temperature = args.temperature
    # LoRA / freeze / checkpointing target the whole wrapper: the projection
    # head must stay trainable, PEFT reaches the backbone through it.
    return self._apply_model_extras(args, model, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    images = blob.data
    labels_ = blob.meta.get("labels", labels)
    if labels_ is None:
      raise ValueError(
          "SupConMethod requires labels (blob.meta['labels'] or labels kwarg).")
    features = model(images)
    loss = supcon_loss(features, labels_, temperature=model.temperature)
    return LossOutput(loss=loss,
                      metrics={
                          "loss": loss.detach(),
                          "temperature": model.temperature,
                      })

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
