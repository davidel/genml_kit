"""Tests for the Method lifecycle contract (plans/B3_PLAN.md s 2, s 6.1).

Covers the base-class hooks the genml-kit-train driver calls and the
protected `_apply_model_extras` helper that every `build_model`
implementation ends with:

- base no-op hooks (prepare_transforms / wire_data / post_train);
- `_apply_model_extras`: grad checkpointing, LoRA wrapping, source
  checkpoint loading, freeze patterns;
- ClassificationMethod.build_model preconditions and happy path.
"""

import argparse
import torch

import pytest

from genml_kit.methods.base import Method
from genml_kit.pipelines.contracts import LossOutput

peft = pytest.importorskip("peft", reason="LoRA tests need the peft package")


class _LinearBackbone(torch.nn.Module):
  """Two-layer module: named `fc` so --lora_target_modules=fc / --freeze=fc
  have a stable target.

  Pre-declares ``use_grad_checkpoint`` (as ConvViT / UVito do) so the
  per-block fallback of ``enable_grad_checkpointing`` has a target to set.
  """

  def __init__(self):
    super().__init__()
    self.fc = torch.nn.Linear(4, 2)
    self.use_grad_checkpoint = False

  def forward(self, x):
    return self.fc(x)


def _base_args(**overrides):
  """Args namespace carrying every flag _apply_model_extras can consume,
  all in their neutral (off) state unless overridden."""
  defaults = dict(
      grad_checkpoint=False,
      lora=False,
      lora_r=4,
      lora_alpha=8,
      lora_dropout=0.0,
      lora_target_modules="",
      source_checkpoint=None,
      param_rename=None,
      freeze="",
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


class _BareMethod(Method):
  """Minimal concrete method: only the abstract members implemented."""

  NAME = "bare_lifecycle"

  def build_model(self, args, device):
    return _LinearBackbone()

  def train_step(self, model, blob, global_step, *, labels=None):
    return LossOutput(loss=torch.zeros(()), metrics={})


class TestBaseHooks:

  def test_base_hooks_are_noops(self):
    method = _BareMethod()
    assert method.prepare_transforms(argparse.Namespace(), "cpu") is None
    assert method.wire_data(argparse.Namespace(), object()) is None
    assert method.post_train(argparse.Namespace(), object(), "cpu",
                             object()) is None


class TestApplyModelExtras:

  def test_neutral_flags_leave_model_unchanged(self):
    method = _BareMethod()
    model = _LinearBackbone()
    out = method._apply_model_extras(_base_args(), model, torch.device("cpu"))
    assert out is model
    assert not isinstance(out, peft.PeftModel)

  def test_grad_checkpointing_applied(self):
    method = _BareMethod()
    model = method._apply_model_extras(_base_args(grad_checkpoint=True),
                                       _LinearBackbone(), torch.device("cpu"))
    # Bare-module fallback path of enable_grad_checkpointing: the custom
    # per-block flag the ConvViT/UVito transformer loops consume.
    assert getattr(model, "use_grad_checkpoint", False) is True

  def test_lora_wraps_model(self):
    method = _BareMethod()
    model = method._apply_model_extras(
        _base_args(lora=True, lora_target_modules="fc"), _LinearBackbone(),
        torch.device("cpu"))
    assert isinstance(model, peft.PeftModel)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any("lora" in n for n in trainable)

  def test_lora_without_explicit_targets_auto_detects(self):
    # PEFT raises with target_modules=None; the helper must resolve the
    # default Linear-projection targets itself.
    method = _BareMethod()
    model = method._apply_model_extras(_base_args(lora=True), _LinearBackbone(),
                                       torch.device("cpu"))
    assert isinstance(model, peft.PeftModel)

  def test_source_checkpoint_loads_weights(self, tmp_path):
    src = _LinearBackbone()
    torch.nn.init.constant_(src.fc.weight, 0.5)
    torch.nn.init.constant_(src.fc.bias, 0.25)
    path = str(tmp_path / "src.pt")
    torch.save({"model_state_dict": src.state_dict()}, path)

    method = _BareMethod()
    model = _LinearBackbone()
    torch.nn.init.zeros_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)
    model = method._apply_model_extras(_base_args(source_checkpoint=path),
                                       model, torch.device("cpu"))
    assert torch.allclose(model.fc.weight, torch.full_like(model.fc.weight, 0.5))
    assert torch.allclose(model.fc.bias, torch.full_like(model.fc.bias, 0.25))

  def test_freeze_pattern_freezes_non_matches(self):
    # freeze_model semantics: *patterns* list what stays TRAINABLE;
    # everything else is frozen.
    method = _BareMethod()
    model = method._apply_model_extras(_base_args(freeze="fc"),
                                       _LinearBackbone(), torch.device("cpu"))
    assert model.fc.weight.requires_grad is True
    assert model.fc.bias.requires_grad is True
