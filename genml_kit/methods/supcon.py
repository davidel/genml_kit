"""Supervised Contrastive pre-training method.

Reference: Khosla et al., "Supervised Contrastive Learning",
NeurIPS 2020.
"""

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.models import load_model
from genml_kit.models.contrastive import ContrastiveEncoder
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.losses.contrastive import supcon_loss
from genml_kit.utils.logging import fatal


@METHODS.register
class SupConMethod(Method):
  """Supervised contrastive pre-training via NT-Xent loss."""

  NAME = "supcon"
  METRIC_KEY = "loss"
  # Loss is minimized.
  METRIC_MINIMIZE = True
  NEEDS_LABELS = True

  @classmethod
  def arg_defaults(cls):
    """Declare ``find_unused_parameters`` for DDP.

        Declared *defensively*, not because the default path needs it: the
        contrastive encoder routes every trainable parameter through
        ``encode_with_backbone`` (which now reaches the full backbone),
        so a stock run leaves nothing unused and completes with the DDP
        default ``find_unused_parameters=False``.

        The flag is kept so that a non-default configuration which leaves
        some trainable parameter out of the loss -- without aborting on
        DDP's unused-parameter check on the second iteration -- keeps
        working.  Users who want the lower-overhead strict check on a
        config they know is fully used can pass ``--dist_arg
        find_unused_parameters=false``.
        """
    return {"dist_arg": {"find_unused_parameters": True}}

  @classmethod
  def add_args(cls, parser):
    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="Projection head output dimension.",
    )
    parser.add_argument(
        "--proj_hidden",
        type=int,
        default=2048,
        help="Projection head hidden dimension.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="NT-Xent temperature.",
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
      fatal("SupConMethod requires labels (blob.meta['labels'] or labels kwarg).",
            ValueError)
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
