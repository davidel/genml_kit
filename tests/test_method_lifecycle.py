"""Tests for the Method lifecycle contract.

Covers the base-class hooks the genml-kit-train driver calls and the
protected `_apply_model_extras` helper that every `build_model`
implementation ends with:

- base no-op hooks (prepare_transforms / wire_data / post_train);
- `_apply_model_extras`: grad checkpointing, LoRA wrapping, source
  checkpoint loading, freeze patterns;
- ClassificationMethod.build_model preconditions and happy path.
"""

import argparse
from unittest.mock import patch

import torch

import pytest

from genml_kit.methods import get_method
from genml_kit.methods.base import Method
from genml_kit.models.contrastive import ContrastiveEncoder
from genml_kit.pipelines.contracts import LossOutput

peft = pytest.importorskip("peft", reason="LoRA tests need the peft package")


class _LinearBackbone(torch.nn.Module):
  """Two-layer module: named `fc` so --lora_target_modules=fc / --freeze=fc
  have a stable target.

  Pre-declares ``use_grad_checkpoint`` (as ConvViT / UVito do) so the
  per-block fallback of ``enable_grad_checkpointing`` has a target to set,
  and ``num_features`` so ``detect_backbone_dim`` can infer the output dim
  (required by the contrastive wrappers).
  """

  def __init__(self, out_features=2):
    super().__init__()
    self.fc = torch.nn.Linear(4, out_features)
    self.use_grad_checkpoint = False
    self.num_features = out_features

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
    assert method.post_train(argparse.Namespace(), object(), "cpu", object()) is None


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
    model = method._apply_model_extras(_base_args(lora=True, lora_target_modules="fc"),
                                       _LinearBackbone(), torch.device("cpu"))
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
    model = method._apply_model_extras(_base_args(source_checkpoint=path), model,
                                       torch.device("cpu"))
    assert torch.allclose(model.fc.weight, torch.full_like(model.fc.weight, 0.5))
    assert torch.allclose(model.fc.bias, torch.full_like(model.fc.bias, 0.25))

  def test_freeze_pattern_freezes_non_matches(self):
    # freeze_model semantics: *patterns* list what stays TRAINABLE;
    # everything else is frozen.
    method = _BareMethod()
    model = method._apply_model_extras(_base_args(freeze="fc"), _LinearBackbone(),
                                       torch.device("cpu"))
    assert model.fc.weight.requires_grad is True
    assert model.fc.bias.requires_grad is True

  def test_self_supervised_build_model_lora_path(self):
    # End-to-end through a real self-supervised method's build_model:
    # SupCon wraps (backbone + projection head) and the extras must reach
    # the backbone through the wrapper.
    method = get_method("supcon")()
    parser = argparse.ArgumentParser()
    method.add_args(parser)
    args = parser.parse_args([
        "--proj_dim",
        "8",
        "--proj_hidden",
        "8",
        "--temperature",
        "0.07",
    ])
    args.model = "fc-tiny"
    args.image_size = 16
    args.lora = True
    args.lora_r = 2
    args.lora_alpha = 4
    args.lora_dropout = 0.0
    args.lora_target_modules = ""
    args.model_arg = {}
    args.cache_dir = None
    args.source_checkpoint = None
    args.param_rename = None
    args.freeze = ""
    args.grad_checkpoint = False
    with patch.dict("genml_kit.models.registry._MODEL_REGISTRY",
                    {"fc-tiny": lambda **kwargs: _LinearBackbone(out_features=8)}):
      model = method.build_model(args, torch.device("cpu"))
    # The wrapper itself is now the PeftModel base (extras wrap the whole
    # ContrastiveEncoder so the backbone Linear is adapted through it).
    assert isinstance(model, peft.PeftModel)
    assert isinstance(model.get_base_model(), ContrastiveEncoder)
    backbone_linears = [
        n for n, m in model.named_modules() if isinstance(m, peft.tuners.lora.Linear)
    ]
    assert backbone_linears, "LoRA adapters must reach the backbone"
    # The adapter's target Linear is the backbone's `fc`; the LoRA wrapper
    # module itself carries `lora_` child names.  Assert the backbone fc is
    # adapted.
    assert any(".fc" in n for n in backbone_linears)
    # And inspect the actual adapted weight locations: adapter params live
    # under lora_A/lora_B children of the wrapped fc layer.
    adapted = [n for n, p in model.named_parameters() if "lora_" in n]
    assert any(".fc." in n for n in adapted)


class TestClassificationLifecycle:
  """wire_data -> build_model -> post_train contract (B3 step 4)."""

  def _method(self):
    return get_method("classification")()

  def _pipeline(self, num_labels=3):
    pipeline = argparse.Namespace()
    pipeline.num_labels = num_labels
    if num_labels is not None:
      pipeline.id2label = {i: f"c{i}" for i in range(num_labels)}
      pipeline.label2id = {f"c{i}": i for i in range(num_labels)}
    else:
      pipeline.id2label = None
      pipeline.label2id = None
    pipeline.class_weights = None
    return pipeline

  def _args(self):
    return argparse.Namespace(
        model="dummy/tiny-model",
        image_size=16,
        cache_dir=None,
        model_arg={},
        mixup_alpha=0.0,
        focal_gamma=0.0,
        label_smoothing=0.0,
        grad_checkpoint=False,
        lora=False,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules="fc",
        source_checkpoint=None,
        param_rename=None,
        freeze="",
    )

  def test_build_model_requires_wire_data(self):
    with pytest.raises(RuntimeError, match="wire_data"):
      self._method().build_model(self._args(), torch.device("cpu"))

  def test_wire_data_builds_criterion_and_label_space(self):
    method = self._method()
    method.wire_data(self._args(), self._pipeline(num_labels=3))
    assert method._num_labels == 3
    assert method._criterion is not None

  def test_wire_data_rejects_unlabeled_pipeline(self):
    with pytest.raises(ValueError, match="labeled"):
      self._method().wire_data(self._args(), self._pipeline(num_labels=None))

  def test_build_model_happy_path(self):
    from collections import namedtuple

    LogitsOutput = namedtuple("LogitsOutput", ["logits"])

    class _Classifier(torch.nn.Module):

      def __init__(self, num_labels):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 16 * 16, num_labels)

      def forward(self, pixel_values, **kwargs):
        return LogitsOutput(logits=self.fc(pixel_values.flatten(1)))

    method = self._method()
    method.wire_data(self._args(), self._pipeline(num_labels=3))
    with patch("genml_kit.models.load_model", return_value=_Classifier(3)):
      model = method.build_model(self._args(), torch.device("cpu"))
    logits = model(pixel_values=torch.zeros(2, 3, 16, 16)).logits
    assert logits.shape == (2, 3)

  def test_build_model_lora_path(self):
    from collections import namedtuple

    LogitsOutput = namedtuple("LogitsOutput", ["logits"])

    class _Classifier(torch.nn.Module):

      def __init__(self, num_labels):
        super().__init__()
        self.fc = torch.nn.Linear(3 * 16 * 16, num_labels)

      def forward(self, pixel_values, **kwargs):
        return LogitsOutput(logits=self.fc(pixel_values.flatten(1)))

    method = self._method()
    method.wire_data(self._args(), self._pipeline(num_labels=3))
    args = self._args()
    args.lora = True
    with patch("genml_kit.models.load_model", return_value=_Classifier(3)):
      model = method.build_model(args, torch.device("cpu"))
    assert isinstance(model, peft.PeftModel)

  def test_post_train_delegates_to_maybe_train_xgboost(self, monkeypatch):
    # The hook unconditionally forwards; the --xgboost_model flag check is
    # owned by maybe_train_xgboost (its own guard, covered by its contract).
    calls = []
    monkeypatch.setattr("genml_kit.training.train_compat.maybe_train_xgboost",
                        lambda *a, **k: calls.append(a))
    method = self._method()
    args = self._args()
    args.xgboost_model = "some/path"
    pipeline = self._pipeline()
    method.post_train(args, pipeline, "cpu", None)
    assert len(calls) == 1
    assert calls[0][0] is args
    assert calls[0][1] is pipeline


class TestDriverIsBranchless:
  """The structural guarantee of B3: main() dispatches through the
  lifecycle hooks only, never by inspecting concrete method types."""

  def test_main_has_no_method_type_sniffing(self):
    import inspect

    from genml_kit.training import train

    source = inspect.getsource(train.main)
    assert "isinstance(method" not in source
    assert 'get_method("classification")' not in source
    assert "is_classification" not in source
    # And the hooks are actually invoked.
    for hook in ("prepare_transforms", "wire_data", "build_model", "post_train"):
      assert f"method.{hook}" in source

  def test_driver_module_has_no_method_type_sniffing(self):
    import inspect

    from genml_kit.training import train

    source = inspect.getsource(train)
    assert 'get_method("classification")' not in source
    assert "is_classification" not in source
    # The old driver helpers are gone.
    for gone in ("_wire_classification", "_build_model", "_load_classification_model",
                 "_resolve_classification_transforms"):
      assert f"def {gone}" not in source
