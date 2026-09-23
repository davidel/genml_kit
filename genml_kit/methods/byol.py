"""BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020)."""

import contextlib

import torch
from torchvision.transforms import v2

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models import load_model
from genml_kit.models.byol import BYOL
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.augmentations.dual_view import DualViewTransform
from genml_kit.training.model_utils import set_train_mode


@register_method
class BYOLMethod(Method):
  """Self-supervised contrastive pre-training via BYOL."""

  NAME = "byol"
  NEEDS_LABELS = False
  METRIC_KEY = "loss"
  # Loss is minimized.
  METRIC_MINIMIZE = True

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("BYOL")
    group.add_argument("--byol_proj_dim",
                       type=int,
                       default=256,
                       help="Projection head output dimension.")
    group.add_argument("--byol_proj_hidden",
                       type=int,
                       default=2048,
                       help="Projection head hidden dimension.")
    group.add_argument("--byol_predictor_hidden",
                       type=int,
                       default=2048,
                       help="Predictor MLP hidden dimension.")
    group.add_argument("--byol_momentum",
                       type=float,
                       default=0.996,
                       help="Initial EMA momentum for target encoder.")
    group.add_argument("--byol_final_momentum",
                       type=float,
                       default=1.0,
                       help="Final EMA momentum (ramped up over training).")

  def build_model(self, args, device):
    self._byol_momentum = args.byol_momentum
    self._byol_final_momentum = args.byol_final_momentum
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
    # Apply LoRA / freeze / checkpointing to the online encoder BEFORE
    # constructing the composite: BYOL deep-copies the online encoder into
    # the EMA target, so adapting afterwards would double the adapters and
    # break online/target state-dict symmetry.  The target copy inherits the
    # (frozen, never-gradient-updated) adapter weights -- correct EMA
    # semantics; update_momentum blends them with the online branch's.
    encoder = self._apply_model_extras(args, encoder, device)
    return BYOL(
        encoder,
        proj_dim=args.byol_proj_dim,
        proj_hidden=args.byol_proj_hidden,
        predictor_hidden=args.byol_predictor_hidden,
    ).to(device)

  def build_transform(self, args, image_size):
    base = v2.Compose([
        v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        ],
                       p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return DualViewTransform(base)

  def wire_data(self, args, pipeline):
    """Compute the optimizer-step budget for the EMA momentum ramp.

    Lifecycle position 2: both loaders exist, so ``len(train_loader)`` is
    final.  ``global_step`` is the optimizer-step counter (the trainer
    increments it once per ``grad_accum_steps`` flushes), so the budget
    is the micro-batch total divided by ``grad_accum_steps`` -- the ramp
    completes exactly at the last optimizer step of the last epoch.
    """
    total = None
    with contextlib.suppress(TypeError):
      # Iterable-only datasets: no len().
      total = len(pipeline.train_loader)
    if total:
      total = total * args.epochs // max(getattr(args, "grad_accum_steps", 1), 1)
    self._total_steps = total

  def train_step(self, model, blob, global_step, *, labels=None):
    """Compute the BYOL loss for one dual-view blob (train path).

    Delegates to the shared :meth:`_run_loss` and additionally advances
    the target EMA with the momentum for this optimizer step.  The EMA
    update is the train-only side effect that :meth:`eval_step` omits.
    """
    out = self._run_loss(model, blob.data)
    model.update_momentum(self._current_momentum(global_step, model))
    return out

  def eval_step(self, model, blob, global_step=0, *, labels=None):
    """Compute the BYOL loss with NO training side-effects.

    Unlike :meth:`train_step`, this does not advance the target EMA.
    ``evaluate()`` calls this instead of ``train_step``.
    """
    return self._run_loss(model, blob.data)

  def _run_loss(self, model, images):
    """Forward + LossOutput wrapping (shared by train/eval)."""
    set_train_mode(model, "train")
    loss, info = model(images)
    return LossOutput(
        loss=loss,
        metrics={
            k: (v.detach() if hasattr(v, "detach") else v) for k, v in info.items()
        })

  def _current_momentum(self, global_step, model):
    total = getattr(self, "_total_steps", None)
    if not total:
      return self._momentum_start()
    ratio = min(global_step / total, 1.0)
    start = self._momentum_start()
    end = self._momentum_end()
    return end + (start - end) * (1.0 - ratio)

  def _momentum_start(self):
    return getattr(self, "_byol_momentum", 0.996)

  def _momentum_end(self):
    return getattr(self, "_byol_final_momentum", 1.0)

  def get_checkpoint_state(self, model, args):
    return {
        "method": "byol",
        "momentum": self._momentum_start(),
        "final_momentum": self._momentum_end(),
    }

  def load_checkpoint_state(self, model, state, args):
    self._byol_momentum = state.get("momentum", args.byol_momentum)
    self._byol_final_momentum = state.get("final_momentum", args.byol_final_momentum)

  def validate(self, model, images, num_samples):
    return None
