"""DINO loss (Caron et al., ICCV 2021).

Cross-entropy between sharpened/centered teacher logits and student
logits across global crops.

Reference: Caron et al., *"Emerging Properties in Self-Supervised Vision
Transformers"*, ICCV 2021 — https://arxiv.org/abs/2104.14294
"""

import torch
import torch.nn as nn


class DINOLoss(nn.Module):
  """DINO self-distillation loss with exponential moving average centering.

  The loss is the cross-entropy ``H(p_teacher, p_student)`` between the
  teacher's output distribution and the student's, where:

  * the **teacher** distribution is *centered* (batch mean subtracted)
    and *sharpened* (divided by ``teacher_temp < 1``) before the softmax.
    Centering plus sharpening are the two mechanisms that prevent
    collapse: without them the loss can be minimised trivially by the
    teacher producing the same constant output for every input (center
    cancels any constant), while sharpening forces the teacher to commit
    to a peaky distribution rather than a flat one;
  * the **student** side is a plain ``log_softmax`` at temperature
    ``student_temp``, so minimising the loss raises the student's
    probability where the teacher's is high.

  The ``center`` buffer is EMA-updated once per training step from the
  teacher's outputs (see :meth:`update_center`) and is persisted in
  checkpoints by :meth:`DINOMethod.get_checkpoint_state`.

  Args:
      out_dim: Projection dimension.
      teacher_temp: Temperature for teacher sharpening.
      student_temp: Temperature for student logits.
      center_momentum: EMA momentum for the center.
  """

  def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
    super().__init__()
    self.teacher_temp = teacher_temp
    self.student_temp = student_temp
    self.center_momentum = center_momentum
    self.register_buffer("center", torch.zeros(1, out_dim))

  def forward(self, student_output, teacher_output):
    """Compute cross-entropy loss.

    Args:
      student_output: ``(B, D)`` student projections (rows are the
        student's views of every crop in the batch).
      teacher_output: ``(B, D)`` teacher projections with one row per
        student row (computed under ``torch.no_grad()``); centering and
        sharpening are applied inside this method.

    Returns:
      Scalar loss tensor.
    """
    teacher_out = (teacher_output - self.center) / self.teacher_temp
    teacher_prob = torch.softmax(teacher_out, dim=1).detach()

    student_out = student_output / self.student_temp
    student_log_prob = torch.log_softmax(student_out, dim=1)

    loss = -(teacher_prob * student_log_prob).sum(dim=1).mean()
    return loss

  @torch.no_grad()
  def update_center(self, teacher_output):
    """Update the center with EMA.

    Applies ``c ← m·c + (1−m)·mean(teacher_output)`` where *m* is
    ``center_momentum``.  Must be called once per training step with the
    teacher's raw outputs, *before* the next loss computation uses the
    center.
    """
    batch_center = teacher_output.mean(dim=0, keepdim=True)
    self.center.mul_(self.center_momentum).add_(batch_center,
                                                alpha=1.0 - self.center_momentum)
