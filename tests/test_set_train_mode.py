"""Tests for set_train_mode / _find_frozen_modules."""

import torch
import torch.nn as nn

from genml_kit.training.model_utils import _find_frozen_modules, set_train_mode


class _Backbone(nn.Module):
  """Mimics a frozen backbone with BatchNorm and Dropout."""

  def __init__(self):
    super().__init__()
    self.bn = nn.BatchNorm1d(8)
    self.dropout = nn.Dropout(p=1.0)  # p=1.0: drops everything
    self.fc = nn.Linear(8, 8)

  def forward(self, x):
    return self.dropout(self.bn(self.fc(x)))


class _Head(nn.Module):
  """Mimics a trainable classification head."""

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(8, 2)

  def forward(self, x):
    return self.fc(x)


class _Model(nn.Module):

  def __init__(self):
    super().__init__()
    self.backbone = _Backbone()
    self.head = _Head()

  def forward(self, x):
    return self.head(self.backbone(x))


def _freeze_all(module):
  for p in module.parameters():
    p.requires_grad = False


def _assert_all_modules(module, expected_mode):
  for m in module.modules():
    assert m.training is expected_mode, (
        f"Expected {m.__class__.__name__}.training={expected_mode}, "
        f"got {m.training}")


class TestFindFrozenModules:

  def test_nothing_frozen(self):
    model = _Model()
    assert _find_frozen_modules(model) == set()

  def test_entire_model_frozen(self):
    model = _Model()
    _freeze_all(model)
    frozen = _find_frozen_modules(model)
    # Every module in the tree qualifies as fully frozen
    # (bottom-up property — children + own params all frozen).
    assert all(m in frozen for m in model.modules())

  def test_backbone_only_frozen(self):
    model = _Model()
    _freeze_all(model.backbone)
    frozen = _find_frozen_modules(model)
    assert model.backbone in frozen
    assert model.head not in frozen
    assert model not in frozen

  def test_backbone_and_head_both_frozen(self):
    model = _Model()
    _freeze_all(model)
    frozen = _find_frozen_modules(model)
    # Head and backbone are both fully frozen, and root is too
    assert model in frozen

  def test_partial_backbone_frozen(self):
    """Only freezing bn+fc but not the head means backbone is frozen
    (all its params frozen), but head is not."""
    model = _Model()
    _freeze_all(model.backbone)
    # Un-freeze one param in backbone to make it non-frozen
    model.backbone.fc.weight.requires_grad = True
    frozen = _find_frozen_modules(model)
    assert model.backbone not in frozen
    assert model.head not in frozen
    assert model not in frozen


class TestSetTrainMode:

  def test_eval_sets_everything_to_eval(self):
    model = _Model()
    set_train_mode(model, 'eval')
    _assert_all_modules(model, expected_mode=False)

  def test_train_full_model(self):
    model = _Model()
    set_train_mode(model, 'train')
    _assert_all_modules(model, expected_mode=True)

  def test_train_respects_frozen_backbone(self):
    model = _Model()
    _freeze_all(model.backbone)
    set_train_mode(model, 'train')

    # Backbone must be in eval mode
    _assert_all_modules(model.backbone, expected_mode=False)
    # Head must be in train mode
    _assert_all_modules(model.head, expected_mode=True)

  def test_batchnorm_running_stats_not_updated(self):
    """Verify that frozen BatchNorm running_mean/var stay unchanged
    when the parent model is set to train mode."""
    model = _Model()
    _freeze_all(model.backbone)

    # Run a forward pass to set initial running stats
    x = torch.randn(4, 8)
    model.eval()
    with torch.no_grad():
      model(x)
    running_mean_before = model.backbone.bn.running_mean.clone()

    # Set to train mode and run another forward pass
    set_train_mode(model, 'train')
    with torch.no_grad():
      model(x)
    running_mean_after = model.backbone.bn.running_mean.clone()

    # Running stats should not have changed
    assert torch.equal(
        running_mean_before,
        running_mean_after), ("Frozen BatchNorm running stats were updated!")

  def test_dropout_not_active_in_frozen_backbone(self):
    """With p=1.0, dropout would zero out everything if active."""
    model = _Model()
    _freeze_all(model.backbone)
    set_train_mode(model, 'train')

    # Forward pass: dropout is p=1.0 but should be inactive (eval mode)
    x = torch.randn(4, 8)
    with torch.no_grad():
      out = model(x)
    # If dropout were active, the backbone output would be all zeros
    assert out.abs().sum() > 0, "Frozen backbone dropout is active!"

  def test_round_trip(self):
    """train → eval → train respects frozen sub-trees each time."""
    model = _Model()
    _freeze_all(model.backbone)

    set_train_mode(model, 'train')
    assert model.backbone.training is False
    assert model.head.training is True

    set_train_mode(model, 'eval')
    _assert_all_modules(model, expected_mode=False)

    set_train_mode(model, 'train')
    assert model.backbone.training is False
    assert model.head.training is True
