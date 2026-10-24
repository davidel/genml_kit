"""Tests for the knowledge-distillation losses (§4-11).  Pure CPU."""

import pytest
import torch

from genml_kit.losses.distillation import (
    kd_attention_loss,
    kd_feature_loss,
    kd_relation_loss,
    kd_response_loss,
)


class TestKdResponseLoss:

  def test_zero_when_student_matches_teacher(self):
    teacher = torch.randn(4, 5)
    loss = kd_response_loss(teacher.clone(), teacher)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

  def test_positive_when_students_differ(self):
    teacher = torch.randn(4, 5)
    student = teacher + torch.randn(4, 5)
    assert kd_response_loss(student, teacher).item() > 0.0

  def test_temperature_squared_scaling(self):
    # A fixed teacher/student pair; the T^2 factor must keep the loss scale
    # comparable across temperatures, not shrink like 1/T.
    torch.manual_seed(0)
    teacher = torch.randn(8, 10)
    student = torch.randn(8, 10)
    l_t2 = kd_response_loss(student, teacher, temperature=2.0)
    l_t4 = kd_response_loss(student, teacher, temperature=4.0)
    # Both finite and of the same order of magnitude (the T^2 restores scale).
    assert torch.isfinite(l_t2) and torch.isfinite(l_t4)
    assert 0.01 < (l_t2.item() / l_t4.item()) < 100.0

  def test_t2_factor_affects_gradient(self):
    # §5/§7: the loss must carry T^2, not just the KL.  Recompute the KL alone
    # and check the shipped gradient is exactly T^2 times larger.
    teacher = torch.randn(4, 5)
    # Student 1: through the shipped loss (which includes T^2).
    s1 = torch.randn(4, 5, requires_grad=True)
    kd_response_loss(s1, teacher, temperature=4.0).backward()
    g_t2 = s1.grad.clone()
    # Student 2: the identical tempered-KL objective, *without* the T^2 factor.
    s2 = s1.detach().clone().requires_grad_(True)
    # (B, C) log-softmax of the tempered student; (B, C) softmax of the tempered
    # teacher; batchmean KL -> scalar.
    student_log_probs = torch.log_softmax(s2 / 4.0, dim=-1)
    teacher_probs = torch.softmax(teacher / 4.0, dim=-1)
    torch.nn.functional.kl_div(student_log_probs, teacher_probs,
                               reduction="batchmean").backward()
    g_no_t2 = s2.grad.clone()
    # The factor is exactly T^2 = 16; dropping it changes the gradient scale.
    assert torch.allclose(g_t2, g_no_t2 * 16.0, rtol=1e-5)

  def test_invalid_temperature_raises(self):
    with pytest.raises(ValueError, match="temperature > 0"):
      kd_response_loss(torch.randn(2, 3), torch.randn(2, 3), temperature=0.0)


class TestKdFeatureLoss:

  def test_zero_when_matching(self):
    feat = torch.randn(4, 8)
    assert kd_feature_loss(feat.clone(), feat).item() == pytest.approx(0.0)

  def test_projector_lifts_student(self):
    # FitNets r: the student (B, 4) is lifted to the teacher's (B, 6) before the
    # MSE; without it the shapes would not match.
    teacher = torch.randn(3, 6)
    student = torch.randn(3, 4)
    projector = torch.nn.Linear(4, 6)
    loss = kd_feature_loss(student, teacher, projector=projector)
    assert loss.item() >= 0.0

  def test_shape_mismatch_raises(self):
    with pytest.raises(ValueError, match="shapes must match"):
      kd_feature_loss(torch.randn(3, 4), torch.randn(3, 6))


class TestKdAttentionLoss:

  def test_zero_when_matching(self):
    feat = torch.randn(2, 3, 4, 4)
    assert kd_attention_loss(feat.clone(), feat).item() == pytest.approx(0.0)

  def test_channel_agnostic(self):
    # §10: the attention map sums |.|^p over channels, so a teacher that is a
    # channel-permuted copy of the student (B, 3, H, W) collapses to the same
    # (B, H, W) map and scores ~0 -- the loss is agnostic to channel order.
    student = torch.randn(1, 3, 4, 4)
    teacher = student.flip(1)
    assert kd_attention_loss(student, teacher).item() == pytest.approx(0.0, abs=1e-6)

  def test_token_maps_3d(self):
    student = torch.randn(2, 5, 4)
    teacher = torch.randn(2, 5, 4)
    assert kd_attention_loss(student, teacher).item() >= 0.0

  def test_shape_mismatch_raises(self):
    with pytest.raises(ValueError, match="shapes must match"):
      kd_attention_loss(torch.randn(2, 3, 4, 4), torch.randn(2, 3, 5, 5))


class TestKdRelationLoss:

  def test_zero_when_matching(self):
    feat = torch.randn(6, 4)
    assert kd_relation_loss(feat.clone(), feat).item() == pytest.approx(0.0, abs=1e-6)

  def test_positive_when_students_differ(self):
    teacher = torch.randn(6, 4)
    student = torch.randn(6, 4)
    assert kd_relation_loss(student, teacher).item() > 0.0

  def test_is_scale_invariant(self):
    # §11: pairwise distances are normalised by their batch mean, so scaling the
    # student features by 10x does not change the objective (structure, not
    # magnitude, is what is transferred).
    teacher = torch.randn(6, 4)
    student = teacher + 0.1 * torch.randn(6, 4)
    a = kd_relation_loss(student, teacher).item()
    b = kd_relation_loss(student * 10.0, teacher).item()
    assert a == pytest.approx(b, rel=1e-3)

  def test_shape_mismatch_raises(self):
    with pytest.raises(ValueError, match="shapes must match"):
      kd_relation_loss(torch.randn(6, 4), torch.randn(6, 5))

  def test_gradients_flow(self):
    teacher = torch.randn(6, 4)
    student = torch.randn(6, 4, requires_grad=True)
    kd_relation_loss(student, teacher).backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()


class TestRegistryEntries:

  def test_kd_losses_registered(self):
    from genml_kit.losses import LOSSES

    for name in ("kd_response", "kd_feature", "kd_attention", "kd_relation"):
      assert LOSSES.contains(name)

  def test_registered_builders_return_callables(self):
    from genml_kit.losses import LOSSES

    assert callable(LOSSES.build("kd_response"))
    assert callable(LOSSES.build("kd_feature"))
    assert callable(LOSSES.build("kd_attention"))
    assert callable(LOSSES.build("kd_relation"))
