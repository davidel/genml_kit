"""DINO: Self-Distillation with No Labels (Caron et al., ICCV 2021).

Pre-training method wiring the :class:`~genml_kit.models.dino.DINO`
student/teacher module to the multi-crop data pipeline.

Reference: Caron et al., *"Emerging Properties in Self-Supervised Vision
Transformers"*, ICCV 2021 -- https://arxiv.org/abs/2104.14294
"""

import contextlib

import torch

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models import load_model
from genml_kit.models.dino import DINO
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.augmentations.multicrop import MultiCropTransform
from genml_kit.training.model_utils import set_train_mode
from genml_kit.utils.attr import get_attribute, MISSING


@register_method
class DINOMethod(Method):
  """Self-distillation pre-training via DINO with multi-crop.

  Responsibilities per training step:

  1. **Data**: :meth:`build_transform` returns a
     :class:`~genml_kit.augmentations.multicrop.MultiCropTransform`
     producing 2 global crops and ``--dino_local_num`` local crops per
     image; ``train_step`` splits ``blob.data`` (a stacked crop tensor)
     into the ``(global_crops, local_crops)`` pair expected by
     ``DINO.forward``.
  2. **Model step**: ``model(global_crops, local_crops)`` computes the
     DINO loss (student vs. teacher).
  3. **Teacher update**: the EMA momentum is linearly scheduled from
     ``--dino_momentum`` to ``--dino_final_momentum`` over the total
     number of optimiser steps (:meth:`_current_momentum`), then applied
     with ``model.update_momentum(momentum)``.

  Checkpointing: :meth:`get_checkpoint_state` persists the loss center
  and the momentum schedule endpoints so resumed runs continue with an
  identical teacher state.
  """

  NAME = "dino"
  NEEDS_LABELS = False
  METRIC_KEY = "loss"
  # Loss is minimized.
  METRIC_MINIMIZE = True

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("DINO")
    group.add_argument("--dino_proj_dim",
                       type=int,
                       default=256,
                       help="Projection head output dimension.")
    group.add_argument("--dino_proj_hidden",
                       type=int,
                       default=2048,
                       help="Projection head hidden dimension.")
    group.add_argument("--dino_student_temp",
                       type=float,
                       default=0.1,
                       help="Student temperature.")
    group.add_argument("--dino_teacher_temp",
                       type=float,
                       default=0.04,
                       help="Teacher temperature.")
    group.add_argument("--dino_center_momentum",
                       type=float,
                       default=0.9,
                       help="EMA momentum for center update.")
    group.add_argument("--dino_momentum",
                       type=float,
                       default=0.996,
                       help="Initial EMA momentum for teacher encoder.")
    group.add_argument("--dino_final_momentum",
                       type=float,
                       default=1.0,
                       help="Final EMA momentum.")
    group.add_argument("--dino_global_size",
                       type=int,
                       default=224,
                       help="Spatial size of global crops.")
    group.add_argument("--dino_local_size",
                       type=int,
                       default=96,
                       help="Spatial size of local crops.")
    group.add_argument("--dino_local_num",
                       type=int,
                       default=8,
                       help="Number of local crops.")

  def build_model(self, args, device):
    self._dino_global_size = getattr(args, "dino_global_size", 224)
    self._dino_local_size = getattr(args, "dino_local_size", 96)
    self._dino_local_num = getattr(args, "dino_local_num", 8)
    self._dino_momentum = args.dino_momentum
    self._dino_final_momentum = args.dino_final_momentum
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
    # constructing the composite: DINO deep-copies the student into the EMA
    # teacher, so adapting afterwards would double the adapters and break
    # teacher/student state-dict symmetry.  The teacher copy inherits the
    # (frozen, never-gradient-updated) adapter weights, which is correct
    # EMA semantics; update_momentum blends them with the student's.
    encoder = self._apply_model_extras(args, encoder, device)
    return DINO(
        encoder,
        proj_dim=args.dino_proj_dim,
        proj_hidden=args.dino_proj_hidden,
        teacher_temp=args.dino_teacher_temp,
        student_temp=args.dino_student_temp,
        center_momentum=args.dino_center_momentum,
    ).to(device)

  def build_transform(self, args, image_size):
    return MultiCropTransform(
        global_size=min(image_size, self._global_size),
        local_size=self._local_size,
        local_num=self._local_num,
    )

  @property
  def _global_size(self):
    return getattr(self, "_dino_global_size", 224)

  @property
  def _local_size(self):
    return getattr(self, "_dino_local_size", 96)

  @property
  def _local_num(self):
    return getattr(self, "_dino_local_num", 8)

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
    """Compute the DINO loss for one multi-crop blob (train path).

    Delegates to the shared :meth:`_run_loss` and additionally advances
    the teacher EMA with the momentum for this optimizer step.  The EMA
    update is the train-only side effect that :meth:`eval_step` omits.
    """
    out = self._run_loss(model, blob.data)
    model.update_momentum(self._current_momentum(global_step, model))
    return out

  def eval_step(self, model, blob, global_step=0, *, labels=None):
    """Compute the DINO loss with NO training side-effects.

    Unlike :meth:`train_step`, this does not advance the teacher EMA and
    runs the model with ``update_center=False`` so the loss center is left
    untouched.  ``evaluate()`` calls this instead of ``train_step``.
    """
    return self._run_loss(model, blob.data, update_center=False)

  def _run_loss(self, model, images, *, update_center=True):
    """Crop split + forward + LossOutput wrapping (shared by train/eval).

    MultiCropTransform yields a per-item tuple ``(global_1, global_2,
    local_1..N)``; collation stacks each position, so *images* is a tuple
    of ``2 + local_num`` tensors each of shape ``(B, C, S, S)``.  The
    first two positions are the global pair, the rest the locals.  A
    single tensor is treated as ``global_crops == local_crops``.
    """
    set_train_mode(model, "train")
    if isinstance(images, (tuple, list)):
      # (2B, C, H, W)
      global_crops = torch.stack(images[:2]).flatten(0, 1)
      if len(images) > 2:
        # (N*B, C, h, w)
        local_crops = torch.stack(images[2:]).flatten(0, 1)
      else:
        local_crops = global_crops
    else:
      global_crops = local_crops = images
    loss, info = model(global_crops, local_crops, update_center=update_center)
    return LossOutput(
        loss=loss,
        metrics={
            k: (v.detach() if hasattr(v, "detach") else v) for k, v in info.items()
        })

  def _current_momentum(self, global_step, model):
    """Compute the teacher EMA momentum for this optimiser step.

    Linearly interpolates from ``--dino_momentum`` (at step 0) to
    ``--dino_final_momentum`` (at the last step).  The total step budget
    is computed in :meth:`wire_data` from the real loader length and the
    optimizer-step semantics of ``global_step`` (see
    :meth:`BaseTrainer.train_epoch`).  A missing budget never freezes the
    teacher: it falls back to the *start* momentum so the teacher still
    tracks the student.
    """
    total = getattr(self, "_total_steps", None)
    if not total:
      return self._momentum_start()
    ratio = min(global_step / total, 1.0)
    start = self._momentum_start()
    end = self._momentum_end()
    return end + (start - end) * (1.0 - ratio)

  def _momentum_start(self):
    return getattr(self, "_dino_momentum", 0.996)

  def _momentum_end(self):
    return getattr(self, "_dino_final_momentum", 1.0)

  def get_checkpoint_state(self, model, args):
    state = {
        "method": "dino",
        "momentum": self._momentum_start(),
        "final_momentum": self._momentum_end(),
    }
    center = get_attribute(model, "loss.center")
    if center is not MISSING:
      state["center"] = center.clone()
    return state

  def load_checkpoint_state(self, model, state, args):
    self._dino_momentum = state.get("momentum", args.dino_momentum)
    self._dino_final_momentum = state.get("final_momentum", args.dino_final_momentum)
    if "center" in state:
      model.loss.center.copy_(state["center"])

  def validate(self, model, images, num_samples):
    return None
