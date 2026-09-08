"""DINO: Self-Distillation with No Labels (Caron et al., ICCV 2021).

Pre-training method wiring the :class:`~genml_kit.models.dino.DINO`
student/teacher module to the multi-crop data pipeline.

Reference: Caron et al., *"Emerging Properties in Self-Supervised Vision
Transformers"*, ICCV 2021 — https://arxiv.org/abs/2104.14294
"""

import torch

from genml_kit.models.dino import DINO
from genml_kit.pretrain.augmentations.multicrop import MultiCropTransform
from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method
from genml_kit.training.model_utils import set_train_mode


@register_method
class DINOMethod(PretrainMethod):
  """Self-distillation pre-training via DINO with multi-crop.

  Responsibilities per training step (see :meth:`PretrainMethod` hooks):

  1. **Data**: :meth:`build_transform` returns a
     :class:`~genml_kit.pretrain.augmentations.multicrop.MultiCropTransform`
     producing 2 global crops and ``--dino_local_num`` local crops per
     image; the training loop stacks them and
     :meth:`~genml_kit.pretrain.augmentations.multicrop.MultiCropTransform.split_crops`
     splits the list into the ``(global_crops, local_crops)`` tensors
     expected by ``DINO.forward``.
  2. **Model step**: ``model(global_crops, local_crops)`` computes the
     DINO loss (student vs. teacher, see
     :class:`~genml_kit.losses.dino.DINOLoss`).
  3. **Teacher update**: the EMA momentum is linearly scheduled from
     ``--dino_momentum`` to ``--dino_final_momentum`` over the total
     number of optimiser steps (:meth:`_current_momentum`), then applied
     with ``model.update_momentum(momentum)``.  Scheduling the momentum
     toward 1.0 progressively freezes the teacher.

  Checkpointing: :meth:`get_checkpoint_state` persists the loss center
  and the momentum schedule endpoints so resumed runs continue with an
  identical teacher state.
  """

  NAME = "dino"
  needs_labels = False

  @classmethod
  def add_args(cls, parser):
    g = parser.add_argument_group("DINO")
    g.add_argument("--dino_proj_dim",
                   type=int,
                   default=256,
                   help="Projection head output dimension.")
    g.add_argument("--dino_proj_hidden",
                   type=int,
                   default=2048,
                   help="Projection head hidden dimension.")
    g.add_argument("--dino_student_temp",
                   type=float,
                   default=0.1,
                   help="Student temperature.")
    g.add_argument("--dino_teacher_temp",
                   type=float,
                   default=0.04,
                   help="Teacher temperature.")
    g.add_argument("--dino_center_momentum",
                   type=float,
                   default=0.9,
                   help="EMA momentum for center update.")
    g.add_argument("--dino_momentum",
                   type=float,
                   default=0.996,
                   help="Initial EMA momentum for teacher encoder.")
    g.add_argument("--dino_final_momentum",
                   type=float,
                   default=1.0,
                   help="Final EMA momentum.")
    g.add_argument("--dino_global_size",
                   type=int,
                   default=224,
                   help="Spatial size of global crops.")
    g.add_argument("--dino_local_size",
                   type=int,
                   default=96,
                   help="Spatial size of local crops.")
    g.add_argument("--dino_local_num",
                   type=int,
                   default=8,
                   help="Number of local crops.")

  def build_transform(self, image_size):
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

  def build(self, args, encoder, device):
    self._dino_global_size = getattr(args, "dino_global_size", 224)
    self._dino_local_size = getattr(args, "dino_local_size", 96)
    self._dino_local_num = getattr(args, "dino_local_num", 8)
    return DINO(
        encoder,
        proj_dim=args.dino_proj_dim,
        proj_hidden=args.dino_proj_hidden,
        teacher_temp=args.dino_teacher_temp,
        student_temp=args.dino_student_temp,
        center_momentum=args.dino_center_momentum,
    ).to(device)

  def train_step(self, model, images, global_step, *, labels=None):
    set_train_mode(model, "train")
    if isinstance(images, (list, tuple)):
      global_crops = torch.cat(images[:2], dim=0)
      local_crops = torch.cat(images[2:], dim=0)
    else:
      global_crops = images
      local_crops = images
    loss, info = model(global_crops, local_crops)
    momentum = self._current_momentum(global_step, model)
    model.update_momentum(momentum)
    return loss, info

  def _current_momentum(self, global_step, model):
    """Compute the teacher EMA momentum for this optimiser step.

    Linearly interpolates from ``--dino_momentum`` (at step 0) to
    ``--dino_final_momentum`` (at the last step).  If the model does not
    expose a total step count, the final momentum is used.
    """
    total = getattr(model, "_total_steps", 0)
    if total <= 0:
      return self._momentum_end()
    ratio = min(global_step / total, 1.0)
    start = self._momentum_start()
    end = self._momentum_end()
    return end + (start - end) * (1.0 - ratio)

  def _momentum_start(self):
    return getattr(self, "_dino_momentum", 0.996)

  def _momentum_end(self):
    return getattr(self, "_dino_final_momentum", 1.0)

  def on_epoch_end(self, model, epoch, writer):
    pass

  def get_checkpoint_state(self, model, args):
    state = {
        "method": "dino",
        "momentum": self._momentum_start(),
        "final_momentum": self._momentum_end(),
    }
    if hasattr(model, "loss"):
      state["center"] = model.loss.center.clone()
    return state

  def load_checkpoint_state(self, model, state, args):
    self._dino_momentum = state.get("momentum", args.dino_momentum)
    self._dino_final_momentum = state.get("final_momentum", args.dino_final_momentum)
    if "center" in state and hasattr(model, "loss"):
      model.loss.center.copy_(state["center"])

  def validate(self, model, images, num_samples):
    return None
