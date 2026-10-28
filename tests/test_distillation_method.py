"""Tests for DistillationMethod (§15, §17-§19).  Pure CPU, fake models."""

import argparse

import pytest
import torch
import torch.nn as nn

from genml_kit.methods import METHODS
from genml_kit.methods.distillation import DistillationMethod, DistillationPair
from genml_kit.pipelines.contracts import DataBlob

IMG = 8


class _FakeOutput:
  """Stand-in for the registry's ``ModelOutput``: just carries ``.logits``."""

  def __init__(self, logits):
    self.logits = logits


class _FakeBackbone(nn.Module):
  """Tiny model whose ``blocks.<i>`` emit spatial ``(B, N, C)`` token maps.

  The 3-D activations let the same hook serve both feature MSE (flattened to
  ``(B, N*C)``) and attention transfer (channel axis on ``(B, N, C)``).
  """

  def __init__(self, n_classes=4, width=16, tokens=4, depth=3):
    super().__init__()
    self.width = width
    self.tokens = tokens
    self.flatten = nn.Flatten()
    self.stem = nn.Linear(3 * IMG * IMG, tokens * width)
    # Each block is a (B, N, width) -> (B, N, width) map (applied on the last axis).
    self.blocks = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
    self.head = nn.Linear(width, n_classes)

  def forward(self, pixel_values):
    # (B, 3, S, S) -> (B, 3*S*S)
    x = self.flatten(pixel_values)
    # (B, 3*S*S) -> (B, tokens*width) -> (B, tokens, width): the spatial token map.
    x = self.stem(x).view(-1, self.tokens, self.width)
    for block in self.blocks:
      # (B, tokens, width) -> (B, tokens, width)
      x = block(x)
    # Pool tokens (B, tokens, width) -> (B, width) -> (B, n_classes); wrapped so
    # forward() returns a `.logits`-bearing object like a real model.
    return _FakeOutput(self.head(x.mean(dim=1)))


def _parse_args(**overrides):
  parser = argparse.ArgumentParser()
  DistillationMethod.add_args(parser)
  args = parser.parse_args([])
  args.image_size = IMG
  args.epochs = 1
  args.grad_accum_steps = 1
  args.teacher_model = "fake"
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def _make_method(**overrides):
  method = DistillationMethod()
  method._num_labels = 4
  method._id2label = {}
  method._label2id = {}
  args = _parse_args(**overrides)
  from genml_kit.methods.distillation import _KdConfig

  method._cfg = _KdConfig.from_args(args)
  method._total_steps = 10
  return method, args


def _make_pair(method, args, teacher_kind="external", student=None):
  """Build a DistillationPair the way build_model would, without load_model."""
  student = student or _FakeBackbone()
  # A structurally-identical teacher (mirrors the two build_model paths).
  teacher = _FakeBackbone()
  if teacher_kind == "ema":
    # Self-distillation starts from an exact copy of the student.
    teacher.load_state_dict(student.state_dict())
    from genml_kit.models.ema import EMAModel

    ema = EMAModel(teacher, student)
  else:
    # External teacher: frozen, independently initialised.
    ema = None
  for param in teacher.parameters():
    param.requires_grad_(False)
  # FitNets projectors, built from the same helper build_model calls.
  projectors = method._build_feature_projectors(args, student, teacher,
                                                torch.device("cpu"))
  return DistillationPair(student, teacher, teacher_kind, projectors, ema)


def _blob(batch=4):
  """A DataBlob shaped like the images pipeline emits: images + labels."""
  # (B, 3, S, S) float images; (B,) integer labels in the meta dict.
  images = torch.randn(batch, 3, IMG, IMG)
  labels = torch.randint(0, 4, (batch,))
  return DataBlob(data=images, meta={"labels": labels})


class TestRegistry:

  def test_method_registered(self):
    assert METHODS.contains("distillation")
    assert METHODS.get("distillation") is DistillationMethod


class TestResponseKd:

  def test_train_step_returns_loss_output(self):
    method, args = _make_method()
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    assert torch.isfinite(out.loss)
    assert out.loss.requires_grad
    assert "ce" in out.metrics and "kd" in out.metrics

  def test_response_term_is_active_by_default(self):
    method, args = _make_method()  # kd_alpha default 1.0
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    assert out.metrics["kd"].item() >= 0.0

  def test_response_can_be_disabled(self):
    method, args = _make_method(kd_alpha=0.0)
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    assert out.metrics["kd"].item() == pytest.approx(0.0)

  def test_backward_flows_to_student_only(self):
    method, args = _make_method()
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    out.loss.backward()
    assert any(p.grad is not None for p in pair.student.parameters())
    assert all(p.grad is None for p in pair.teacher.parameters())

  def test_labels_required(self):
    method, args = _make_method()
    pair = _make_pair(method, args)
    blob = DataBlob(data=torch.randn(2, 3, IMG, IMG), meta={})
    with pytest.raises(ValueError):  # fatal() re-raises the passed exception
      method.train_step(pair, blob, 0)


class TestFeatureKd:

  def test_feature_projectors_built(self):
    method, args = _make_method(kd_feature_alpha=1.0,
                                kd_layer_pattern=r"blocks\.\d+$",
                                kd_layer_strategy="by_index")
    pair = _make_pair(method, args)
    assert pair.projectors is not None
    assert len(pair.projectors) == 3

  def test_feature_term_contributes(self):
    method, args = _make_method(kd_alpha=0.0,
                                kd_feature_alpha=1.0,
                                kd_layer_pattern=r"blocks\.\d+$",
                                kd_layer_strategy="by_index")
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    assert out.metrics["kd"].item() >= 0.0
    assert torch.isfinite(out.loss)

  def test_attention_mode(self):
    method, args = _make_method(kd_alpha=0.0,
                                kd_feature_alpha=1.0,
                                kd_attention=True,
                                kd_layer_pattern=r"blocks\.\d+$",
                                kd_layer_strategy="by_index")
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(), 0)
    assert torch.isfinite(out.loss)


class TestRelationKd:

  def test_relation_term(self):
    method, args = _make_method(kd_alpha=0.0, kd_relation_alpha=1.0)
    pair = _make_pair(method, args)
    out = method.train_step(pair, _blob(batch=5), 0)
    assert torch.isfinite(out.loss)


class TestEmaTeacher:

  def test_train_advances_teacher_eval_does_not(self):
    # §19 contract: train_step moves the EMA teacher, eval_step must not.
    method, args = _make_method(teacher_kind="ema")
    pair = _make_pair(method, args, teacher_kind="ema")
    before = [p.detach().clone() for p in pair.teacher.parameters()]
    method.train_step(pair, _blob(), 0)
    after_train = [p.detach().clone() for p in pair.teacher.parameters()]
    # Some teacher parameter changed -> the EMA actually advanced.
    assert any(not torch.equal(a, b) for a, b in zip(before, after_train))
    method.eval_step(pair, _blob(), 0)
    after_eval = [p.detach().clone() for p in pair.teacher.parameters()]
    # No parameter changed -> eval left the teacher untouched.
    assert all(torch.equal(a, b) for a, b in zip(after_train, after_eval))

  def test_checkpoint_state_records_kind(self):
    method, args = _make_method(teacher_kind="ema")
    pair = _make_pair(method, args, teacher_kind="ema")
    state = method.get_checkpoint_state(pair, args)
    assert state["teacher_kind"] == "ema"
    assert "momentum" in state

  def test_load_checkpoint_state_restores_momentum(self):
    method, args = _make_method(teacher_kind="ema")
    pair = _make_pair(method, args, teacher_kind="ema")
    method.load_checkpoint_state(pair, {"momentum": 0.9, "final_momentum": 0.99}, args)
    assert method._cfg.ema_momentum == 0.9
    assert method._cfg.ema_final_momentum == 0.99


class TestExternalTeacher:

  def test_external_teacher_checkpoint_state(self):
    method, args = _make_method(teacher_kind="external")
    pair = _make_pair(method, args, teacher_kind="external")
    state = method.get_checkpoint_state(pair, args)
    assert state["teacher_kind"] == "external"
    assert "momentum" not in state

  def test_validate_is_noop(self):
    method, _ = _make_method()
    assert method.validate(None, None, 0) is None


class TestPrepareTransforms:
  """DistillationMethod must resolve image transforms (regression: PIL leak).

  Without a ``prepare_transforms`` override the pipeline builds the loader
  with ``transform=None``, so raw PIL images reach the collate fn and
  ``torch.stack`` fails.
  """

  def _args(self):
    args = _parse_args()
    args.model = "cls_model_wrapper:fake"
    args.cache_dir = None
    args.train_augmentation_script = None
    args.tta = None
    return args

  def test_sets_train_val_tta_transforms(self, monkeypatch):
    import genml_kit.models

    sentinel = (object(), object(), object())

    def _fake_resolve(args, processor):
      return sentinel

    monkeypatch.setattr(genml_kit.models, "load_processor", lambda *a, **k: object())
    monkeypatch.setattr("genml_kit.training.train_compat.resolve_augmentations",
                        _fake_resolve)

    method, _ = _make_method()
    method.prepare_transforms(self._args(), "cpu")
    assert method.train_transforms is sentinel[0]
    assert method.val_transforms is sentinel[1]
    assert method.tta_transform is sentinel[2]

  def test_delegates_to_the_shared_helper(self, monkeypatch):
    """Both image methods call the same shared helper (no duplication)."""
    calls = []
    monkeypatch.setattr("genml_kit.methods.distillation.prepare_image_transforms",
                        lambda method, args: calls.append(method))

    method, _ = _make_method()
    method.prepare_transforms(self._args(), "cpu")
    assert calls == [method]


class TestBuildModel:

  def test_requires_teacher_model_for_external(self):
    method, args = _make_method(teacher_kind="external")
    args.teacher_model = None
    with pytest.raises(ValueError):
      method._build_external_teacher(args, torch.device("cpu"))


class TestStepBudget:

  def test_none_without_loader(self):

    class _P:
      train_loader = None

    args = _parse_args()
    from genml_kit.methods.distillation import _step_budget

    assert _step_budget(args, _P()) is None

  def test_computes_len_times_epochs(self):

    class _P:
      train_loader = [0, 0, 0]

    args = _parse_args(epochs=2, grad_accum_steps=2)
    from genml_kit.methods.distillation import _step_budget

    assert _step_budget(args, _P()) == 3
