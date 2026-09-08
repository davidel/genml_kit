"""Tests for supervised contrastive loss."""

import pytest
import torch

from genml_kit.pretrain.losses.contrastive import supcon_loss


class TestSupConLoss:

  def test_same_class_pulls_together(self):
    """Two same-class pairs should produce lower loss than distant pairs."""
    torch.manual_seed(0)
    # 4 samples: class 0 pair close together, class 1 pair far apart.
    feat = torch.nn.functional.normalize(
        torch.tensor([
            [1.0, 0.0],  # class 0
            [0.95, 0.31],  # class 0, close
            [1.0, 0.0],  # class 1
            [-1.0, 0.0],  # class 1, far
        ]),
        dim=1,
    )
    loss_close = supcon_loss(feat, torch.tensor([0, 0, 1, 1]), temperature=0.1)

    feat2 = torch.nn.functional.normalize(
        torch.tensor([
            [1.0, 0.0],  # class 0
            [-1.0, 0.0],  # class 0, far
            [1.0, 0.0],  # class 1
            [-1.0, 0.0],  # class 1, far
        ]),
        dim=1,
    )
    loss_far = supcon_loss(feat2, torch.tensor([0, 0, 1, 1]), temperature=0.1)
    assert loss_close.item() < loss_far.item()

  def test_loss_is_finite(self):
    torch.manual_seed(42)
    features = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    labels = torch.randint(0, 5, (16,))
    loss = supcon_loss(features, labels, temperature=0.07)
    assert torch.isfinite(loss)

  def test_loss_non_negative(self):
    torch.manual_seed(42)
    features = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    labels = torch.randint(0, 5, (16,))
    loss = supcon_loss(features, labels, temperature=0.07)
    assert loss.item() >= 0.0

  def test_loss_with_two_classes(self):
    torch.manual_seed(123)
    features = torch.nn.functional.normalize(torch.randn(20, 64), dim=1)
    labels = torch.tensor([0] * 10 + [1] * 10)
    loss = supcon_loss(features, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

  def test_backward(self):
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(
        torch.randn(8, 16),
        dim=1,
    ).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(features, labels, temperature=0.07)
    loss.backward()
    assert features.grad is not None
    assert features.grad.abs().sum() > 0

  def test_single_positive_per_anchor(self):
    """Each anchor has exactly one same-class peer."""
    torch.manual_seed(7)
    features = torch.nn.functional.normalize(torch.randn(6, 32), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(features, labels, temperature=0.07)
    assert torch.isfinite(loss)

  def test_no_positives_returns_zero(self):
    """All-different classes should return zero loss."""
    features = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    loss = supcon_loss(features, labels, temperature=0.07)
    assert loss.item() == 0.0

  def test_small_temperature_is_finite(self):
    """Very small temperature must not produce NaN or Inf."""
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supcon_loss(features, labels, temperature=0.001)
    assert torch.isfinite(loss)

  def test_orthogonal_features_are_finite(self):
    """Nearly-orthogonal features (small off-diagonal logits) must not
    underflow to -inf in the logsumexp denominator."""
    torch.manual_seed(42)
    d = 128
    n = 6
    # Build near-orthogonal features via QR so off-diagonal sims ≈ 0.
    q, _ = torch.linalg.qr(torch.randn(d, n))
    features = q.T.float()  # (6, 128), rows are orthonormal.
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    loss = supcon_loss(features, labels, temperature=0.07)
    assert torch.isfinite(loss)

  def test_gradient_flows_through_loss(self):
    """The loss must be differentiable w.r.t. the input features."""
    torch.manual_seed(7)
    raw = torch.randn(4, 16, requires_grad=True)
    features = torch.nn.functional.normalize(raw, dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    loss = supcon_loss(features, labels, temperature=0.07)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()

  def test_logsumexp_matches_manual(self):
    """Numerically verify that logsumexp-based loss matches the
    mathematically equivalent closed-form for a small controlled input."""
    torch.manual_seed(99)
    features = torch.nn.functional.normalize(torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
    ]),
                                             dim=1)
    labels = torch.tensor([0, 0])
    loss = supcon_loss(features, labels, temperature=1.0)
    # logits = [[0, 1], [1, 0]], diagonal masked to -1e9.
    # For row 0: log_prob[0,1] = 1 - log(exp(1) + exp(-1e9)) ≈ 1 - 1 = 0.
    # loss = -(0 + 0) / 2 = 0.
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
