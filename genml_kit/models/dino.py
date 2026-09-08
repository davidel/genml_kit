"""DINO module — student + teacher networks for self-distillation.

Implements the DINO pre-training scheme from Caron et al., *"Emerging
Properties in Self-Supervised Vision Transformers"*, ICCV 2021
(https://arxiv.org/abs/2104.14294).

DINO trains a **student** network to match the output distribution of a
**teacher** network on the same image.  The teacher is not trained by
gradient descent: it is an exponential moving average (EMA) of the
student, and it only sees the global views of each image while the
student additionally sees small local crops.  Because the teacher is an
EMA of the student, its predictions are a stable, slowly-moving target;
the resulting per-crop output distributions concentrate on different
"salient" image regions, which is the property DINO is named after.

See :class:`~genml_kit.models.dino.DINO` for the training-step mechanics
and :class:`~genml_kit.losses.dino.DINOLoss` for the collapse-prevention
mechanics (centering and sharpening).
"""

import copy

import torch
import torch.nn as nn

from genml_kit.models.contrastive import ProjectionHead
from genml_kit.models.encoder_utils import detect_backbone_dim, encode_with_backbone
from genml_kit.pretrain.losses.dino import DINOLoss


class DINO(nn.Module):
  """DINO: student sees all crops, teacher sees only global crops.

  The model wraps a single backbone encoder in two identical
  :class:`_EncoderWithHead` towers (backbone + MLP projection head):

  * the **student** receives every crop of the batch (global *and*
    local) and is the only tower that receives gradients;
  * the **teacher** is initialised as a deep copy of the student, its
    parameters are frozen (``requires_grad=False``) and it is updated
    only through :meth:`update_momentum`, i.e. an EMA blend of the
    student weights.  It runs under ``torch.no_grad()``.

  ``forward`` performs one training step:

  1. encode the 2 global crops with both towers;
  2. encode the ``N`` local crops with the student only;
  3. update the loss center from the teacher's global outputs (EMA);
  4. repeat the teacher's global-crop projections to obtain one target
     row per student crop (``t_all`` below) — the teacher target is
     shared by the student's global *and* local views of the same image;
  5. evaluate the cross-entropy-style :class:`DINOLoss` between the
     student's log-probabilities and the (sharpened, centered) teacher
     probabilities.

  Args:
      encoder: Backbone encoder.
      proj_dim: Output dimension of the projection head (the softmax
          dimension over which the self-distillation loss is computed).
      proj_hidden: Hidden dimension of the projection head.
      backbone_dim: Explicit backbone output dimension.  Inferred if None.
      teacher_temp: Teacher sharpening temperature (small value →
          peaky teacher distribution).
      student_temp: Student temperature (softmax temperature applied to
          the student logits).
      center_momentum: EMA momentum for center update.
  """

  def __init__(self,
               encoder,
               proj_dim=256,
               proj_hidden=2048,
               backbone_dim=None,
               teacher_temp=0.04,
               student_temp=0.1,
               center_momentum=0.9):
    super().__init__()
    self.student = _EncoderWithHead(encoder, proj_dim, proj_hidden, backbone_dim)
    self.teacher = copy.deepcopy(self.student)
    for p in self.teacher.parameters():
      p.requires_grad = False

    if backbone_dim is None:
      backbone_dim = self.student.backbone_dim
    self.loss = DINOLoss(proj_dim, teacher_temp, student_temp, center_momentum)

  @torch.no_grad()
  def update_momentum(self, momentum):
    """Blend the teacher weights toward the student weights.

    Performs the EMA update ``θ_teacher ← m·θ_teacher + (1−m)·θ_student``
    parameter-wise, with *momentum* typically scheduled from ~0.996 to
    1.0 over the course of pre-training (see
    :meth:`DINOMethod._current_momentum`).

    Args:
        momentum: EMA coefficient in ``[0, 1]``; values close to 1 keep
            the teacher nearly frozen.
    """
    for sp, tp in zip(self.student.parameters(), self.teacher.parameters()):
      tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)

  def forward(self, global_crops, local_crops):
    """Compute DINO loss.

    Args:
        global_crops: ``(B, C, H, W)`` — 2 global crops stacked.
        local_crops: ``(N, C, h, w)`` — N local crops stacked.

    Returns:
        (loss, info_dict)
    """
    n_global = global_crops.shape[0]

    s_global = self.student(global_crops)
    s_local = self.student(local_crops)

    with torch.no_grad():
      t_global = self.teacher(global_crops)
      self.loss.update_center(t_global)

    s_all = torch.cat([s_global, s_local], dim=0)
    t_all = t_global.repeat((s_all.shape[0] // n_global) + 1, 1)[:s_all.shape[0]]

    loss = self.loss(s_all, t_all)
    return loss, {"loss": loss.item()}


class _EncoderWithHead(nn.Module):
  """Backbone + projection head.

  Internal helper bundling the two modules DINO needs per tower: the
  backbone produces a pooled feature vector and the projection head maps
  it to the ``proj_dim``-dimensional space in which the self-distillation
  loss operates.  After pre-training the backbone is kept for downstream
  fine-tuning and the projection head is discarded.
  """

  def __init__(self, encoder, proj_dim, proj_hidden, backbone_dim=None):
    super().__init__()
    self.encoder = encoder
    if backbone_dim is None:
      backbone_dim = detect_backbone_dim(encoder)
    self.backbone_dim = backbone_dim
    self.projection = ProjectionHead(backbone_dim, proj_hidden, proj_dim)

  def encode(self, images):
    return encode_with_backbone(self.encoder, images)

  def forward(self, images):
    return self.projection(self.encode(images))
