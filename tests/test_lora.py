"""Tests for LoRA / PEFT support."""

import re

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from genml_kit.io.checkpointing import (
    checkpoint_dict,
    deserialize_lora_state,
    restore_training_state,
    resume_checkpoint,
    serialize_lora_state,
)
from genml_kit.training.model_utils import apply_lora, extract_lora_params, freeze_model


class _TinyModel(nn.Module):

  def __init__(self, in_features=16, out_features=4):
    super().__init__()
    self.fc = nn.Linear(in_features, out_features)

  def forward(self, x):
    return self.fc(x)


def _make_linear_model(in_features=16, out_features=4):
  """Return a tiny model wrapped with LoRA."""
  model = _TinyModel(in_features, out_features)
  config = LoraConfig(
      r=4,
      lora_alpha=8,
      target_modules=["fc"],
      bias="none",
  )
  return get_peft_model(model, config)


class TestSerializeDeserialize:

  def test_round_trip(self):
    model = _make_linear_model()
    blob = serialize_lora_state(model)
    assert isinstance(blob, bytes)
    assert len(blob) > 0

    new_model = _TinyModel()
    restored, loaded_keys = deserialize_lora_state(new_model, blob)
    assert isinstance(restored, PeftModel)
    assert len(loaded_keys) > 0

    orig_sd = model.state_dict()
    restored_sd = restored.state_dict()
    for key in orig_sd:
      if "lora_" in key:
        assert key in restored_sd, f"Missing adapter key after restore: {key}"
        assert torch.equal(orig_sd[key],
                           restored_sd[key]), (f"Adapter mismatch for {key}")

  def test_serialize_non_peftmodel_raises(self):
    model = nn.Linear(10, 5)
    with pytest.raises(TypeError, match="Expected a PeftModel"):
      serialize_lora_state(model)


class TestApplyLora:

  def test_returns_peft_model(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    assert isinstance(wrapped, PeftModel)

  def test_freezes_base_params(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    for name, p in wrapped.named_parameters():
      if "lora_" not in name:
        assert not p.requires_grad, (f"Non-LoRA param {name} should be frozen")

  def test_lora_params_are_trainable(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_params = [p for n, p in wrapped.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0
    assert all(p.requires_grad for p in lora_params)


class _TinyModelWithClassifier(nn.Module):
  """Tiny model with a LoRA-target and a separate classifier head."""

  def __init__(self, in_features=16, hidden=8, num_classes=4):
    super().__init__()
    self.backbone = nn.Linear(in_features, hidden)
    self.head = nn.Linear(hidden, num_classes)

  def forward(self, x):
    return self.head(self.backbone(x))


def _make_classifier_model(in_features=16, hidden=8, num_classes=4):
  """Return a two-layer model with LoRA on backbone, fresh head."""
  model = _TinyModelWithClassifier(in_features, hidden, num_classes)
  config = LoraConfig(
      r=4,
      lora_alpha=8,
      target_modules=["backbone"],
      bias="none",
  )
  return get_peft_model(model, config)


class TestCheckpointDictLoraStripping:

  def _unfreeze_all(self, model):
    """Unfreeze all params (simulates --freeze matching everything)."""
    for p in model.parameters():
      p.requires_grad = True

  def test_trainable_non_lora_weights_saved_when_blob_present(self):
    """With LoRA + unfrozen head, checkpoint must include head weights."""
    model = _make_classifier_model()
    self._unfreeze_all(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Write known values into the classifier head
    head_w = torch.randn_like(model.head.weight)
    head_b = torch.randn_like(model.head.bias)
    model.head.weight.data.copy_(head_w)
    model.head.bias.data.copy_(head_b)

    blob = serialize_lora_state(model)

    # save_frozen=False (CLI default) must save trainable non-LoRA weights.
    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        save_frozen=False,
        lora_state_blob=blob,
    )

    sd = d["model_state_dict"]
    assert len(sd) > 0, "model_state_dict must not be empty when head is trainable"

    # LoRA keys must be absent (they live in the blob).
    lora_keys = [k for k in sd if "lora_" in k]
    assert lora_keys == [], f"LoRA keys leaked into model_state_dict: {lora_keys}"

    head_w_key = "base_model.model.head.weight"
    head_b_key = "base_model.model.head.bias"
    assert head_w_key in sd, f"Missing classifier weight key: {head_w_key}"
    assert head_b_key in sd, f"Missing classifier bias key: {head_b_key}"
    assert torch.equal(sd[head_w_key], head_w)
    assert torch.equal(sd[head_b_key], head_b)

  def test_frozen_non_lora_weights_excluded_when_blob_present(self):
    """With LoRA only (all non-LoRA frozen by PEFT), save_frozen=False → empty."""
    model = _make_classifier_model()
    # Do NOT unfreeze — PEFT has frozen all non-LoRA params.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    blob = serialize_lora_state(model)
    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        save_frozen=False,
        lora_state_blob=blob,
    )
    sd = d["model_state_dict"]
    non_lora = [k for k in sd if "lora_" not in k]
    assert non_lora == [], (
        f"Frozen non-LoRA weights leaked into checkpoint: {non_lora}")
    assert d["lora_state_blob"] == blob

  def test_frozen_non_lora_weights_saved_when_save_frozen_true(self):
    """save_frozen=True forces ALL non-LoRA weights into the checkpoint."""
    model = _make_classifier_model()
    # Do NOT unfreeze — PEFT has frozen all non-LoRA params.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    blob = serialize_lora_state(model)
    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        save_frozen=True,
        lora_state_blob=blob,
    )
    sd = d["model_state_dict"]
    full_non_lora = {k: v for k, v in model.state_dict().items() if "lora_" not in k}
    assert len(sd) == len(full_non_lora)
    assert d["lora_state_blob"] == blob

  def test_lora_keys_stripped_when_blob_present(self):
    model = _make_linear_model()
    self._unfreeze_all(model)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    blob = b"fake-blob"

    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        save_frozen=False,
        lora_state_blob=blob,
    )

    for key in d["model_state_dict"]:
      assert "lora_" not in key, (
          f"LoRA key {key} should be stripped from model_state_dict")

  def test_lora_keys_kept_when_no_blob(self):
    model = _make_linear_model()
    optimizer = torch.optim.Adam(model.parameters())

    d = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
    )

    lora_keys = [k for k in d["model_state_dict"] if "lora_" in k]
    assert len(lora_keys) > 0, "LoRA keys should be present without blob"


class TestResumeCheckpointWithLoRA:
  """End-to-end: save checkpoint with LoRA blob, resume, verify weights."""

  def test_resume_restores_non_lora_weights(self, tmp_path):
    """Classifier weights must survive checkpoint-resume with LoRA."""
    from genml_kit.io.checkpointing import resume_checkpoint

    model = _make_classifier_model()
    # Unfreeze classifier head (simulates --freeze matching the head).
    for p in model.parameters():
      p.requires_grad = True
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Set known values in the classifier head
    head_w = torch.randn_like(model.head.weight)
    head_b = torch.randn_like(model.head.bias)
    model.head.weight.data.copy_(head_w)
    model.head.bias.data.copy_(head_b)

    blob = serialize_lora_state(model)
    ckpt = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=3,
        states_to_save=set(),
        save_frozen=False,
        best_macro_f1=0.42,
        lora_state_blob=blob,
    )
    path = str(tmp_path / "ckpt.pt")
    torch.save(ckpt, path)

    # Fresh model (random init) + fresh LoRA
    fresh = _TinyModelWithClassifier()
    fresh = apply_lora(fresh, r=4, alpha=8, target_modules=["backbone"])

    restored, epoch, metric, _extra = resume_checkpoint(
        path,
        path,
        fresh,
        device=torch.device("cpu"),
    )

    assert epoch == 4
    assert metric == 0.42
    assert isinstance(restored, PeftModel)

    restored_sd = restored.state_dict()
    head_w_key = "base_model.model.head.weight"
    head_b_key = "base_model.model.head.bias"
    assert head_w_key in restored_sd
    assert head_b_key in restored_sd
    assert torch.equal(restored_sd[head_w_key],
                       head_w), ("Classifier head weight not restored after resume")
    assert torch.equal(restored_sd[head_b_key],
                       head_b), ("Classifier head bias not restored after resume")

  def test_resume_restores_lora_weights(self, tmp_path):
    model = _make_linear_model()
    optimizer = torch.optim.Adam(model.parameters())
    blob = serialize_lora_state(model)

    ckpt = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=5,
        states_to_save=set(),
        best_macro_f1=0.9,
        lora_state_blob=blob,
    )
    path = str(tmp_path / "ckpt.pt")
    torch.save(ckpt, path)

    # Fresh model + fresh LoRA
    fresh = _TinyModel()
    fresh = apply_lora(fresh, r=4, alpha=8, target_modules=["fc"])

    restored, epoch, metric, _extra = resume_checkpoint(
        path,
        path,
        fresh,
        device=torch.device("cpu"),
    )

    assert epoch == 6
    assert metric == 0.9
    assert isinstance(restored, PeftModel)

    # Adapter weights must match the original
    orig_sd = model.state_dict()
    restored_sd = restored.state_dict()
    for key in orig_sd:
      if "lora_" in key:
        assert key in restored_sd, f"Missing adapter key: {key}"
        assert torch.equal(orig_sd[key],
                           restored_sd[key]), (f"Adapter mismatch for {key}")

  def test_trainability_and_optimizer_follow_lora_resume(self, tmp_path):
    """The production ordering keeps classifier and LoRA params trainable."""
    source = _make_classifier_model()
    source_optimizer = torch.optim.Adam(source.parameters())
    checkpoint = checkpoint_dict(
        source,
        source_optimizer,
        scheduler=None,
        epoch=0,
        states_to_save=set(),
        lora_state_blob=serialize_lora_state(source),
    )
    path = str(tmp_path / "ckpt.pt")
    torch.save(checkpoint, path)

    model = apply_lora(_TinyModelWithClassifier(),
                       r=4,
                       alpha=8,
                       target_modules=["backbone"])
    model, _, _, extra = resume_checkpoint(path,
                                           path,
                                           model,
                                           device=torch.device("cpu"))

    lora_names = extract_lora_params(model)
    freeze_model(model, (r"head",) + tuple(re.escape(name) for name in lora_names))
    classifier_params = [p for n, p in model.named_parameters() if "head" in n]
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    backbone_params = [
        p for n, p in model.named_parameters() if "backbone" in n and "lora_" not in n
    ]

    assert classifier_params and all(p.requires_grad for p in classifier_params)
    assert lora_params and all(p.requires_grad for p in lora_params)
    assert backbone_params and all(not p.requires_grad for p in backbone_params)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for p in model.parameters() if p.requires_grad}
    del extra

  def test_classifier_and_lora_update_in_one_step(self):
    """A resumed-style trainable model updates both parameter groups."""
    model = apply_lora(_TinyModelWithClassifier(),
                       r=4,
                       alpha=8,
                       target_modules=["backbone"])
    lora_names = extract_lora_params(model)
    freeze_model(model, (r"head",) + tuple(re.escape(name) for name in lora_names))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.1)
    classifier = next(p for n, p in model.named_parameters() if "head.weight" in n)
    lora = next(p for n, p in model.named_parameters() if "lora_B" in n)
    classifier_before = classifier.detach().clone()
    lora_before = lora.detach().clone()

    loss = model(torch.randn(8, 16)).pow(2).mean()
    loss.backward()
    assert classifier.grad is not None
    assert lora.grad is not None
    assert torch.isfinite(classifier.grad).all()
    assert torch.isfinite(lora.grad).all()
    optimizer.step()

    assert not torch.equal(classifier, classifier_before)
    assert not torch.equal(lora, lora_before)

  def test_restore_training_state_restores_optimizer_and_scheduler(self):
    """Auxiliary states load after the new optimizer is created."""
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss = model(torch.randn(4, 16)).pow(2).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    checkpoint = checkpoint_dict(
        model,
        optimizer,
        scheduler=scheduler,
        epoch=0,
        states_to_save={"opt", "sched"},
    )

    restored_optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer,
                                                         step_size=1)
    restore_training_state(checkpoint,
                           restored_optimizer,
                           restored_scheduler,
                           scaler=None,
                           states_to_load={"opt", "sched"})

    assert restored_optimizer.state
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

  def test_restore_training_state_skips_incompatible_model(self):
    """Partial model loads must not restore incompatible optimizer state."""
    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters())
    loss = model(torch.randn(4, 16)).pow(2).mean()
    loss.backward()
    optimizer.step()
    checkpoint = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt"},
    )
    checkpoint["_model_state_skipped"] = True
    restored_optimizer = torch.optim.Adam(model.parameters())

    restore_training_state(checkpoint,
                           restored_optimizer,
                           scheduler=None,
                           scaler=None,
                           states_to_load={"opt"})

    assert not restored_optimizer.state


class TestExtractLoraParams:
  """Tests for extract_lora_params()."""

  def test_returns_correct_keys(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    expected = {
        "base_model.model.fc.lora_A.default.weight",
        "base_model.model.fc.lora_B.default.weight",
    }
    assert params == expected

  def test_excludes_base_layer(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    for p in params:
      assert "base_layer" not in p

  def test_excludes_non_adapter_params(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    params = extract_lora_params(wrapped)
    all_names = {n for n, _ in wrapped.named_parameters()}
    assert params.issubset(all_names)
    non_lora = {n for n in all_names if n not in params}
    assert non_lora.isdisjoint(params)

  def test_multiple_target_modules(self):

    class TwoLayer(nn.Module):

      def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 4)

      def forward(self, x):
        return self.fc2(self.fc1(x))

    wrapped = apply_lora(TwoLayer(), r=4, alpha=8, target_modules=["fc1", "fc2"])
    params = extract_lora_params(wrapped)
    assert len(params) == 4  # lora_A + lora_B for each of fc1, fc2
    for p in params:
      assert "fc1" in p or "fc2" in p


class TestFreezeWithLoRA:
  """Tests for the full freeze_model + extract_lora_params flow."""

  def test_lora_params_thawed_after_freeze(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if name in lora_names:
        assert p.requires_grad, f"LoRA param {name} should be trainable"
      else:
        assert not p.requires_grad, (f"Non-LoRA param {name} should be frozen")

  def test_freeze_with_user_pattern_and_lora(self):
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    # User pattern to also thaw the classifier head.
    patterns.append("fc")
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if name in lora_names or "fc" in name:
        assert p.requires_grad, (f"Expected {name} to be trainable")
      else:
        assert not p.requires_grad, (f"Expected {name} to be frozen")

  def test_only_lora_thawed_without_user_pattern(self):
    """When --lora is set but --freeze is not, only LoRA params are thawed."""
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    lora_names = extract_lora_params(wrapped)
    import re
    patterns = [re.escape(k) for k in lora_names]
    freeze_model(wrapped, tuple(patterns))

    for name, p in wrapped.named_parameters():
      if "lora_" in name:
        assert p.requires_grad, f"LoRA param {name} should be trainable"
      else:
        assert not p.requires_grad, (f"Param {name} should be frozen")


class TestCheckpointDictFreezeCombinations:
  """Verify checkpoint_dict saves the right weights for every --lora/--freeze combo.

  The goal: checkpoints should never contain frozen backbone weights when
  ``save_frozen=False`` (the default), regardless of whether LoRA is enabled.
  """

  def _count_non_lora_keys(self, sd):
    """Return the number of keys in *sd* that are NOT LoRA adapter keys."""
    return sum(1 for k in sd if "lora_" not in k)

  def _count_trainable_non_lora_keys(self, model):
    """Count state-dict keys that are both trainable and non-LoRA."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return sum(1 for k in model.state_dict() if k in trainable and "lora_" not in k)

  def test_no_lora_no_freeze_save_frozen_false(self):
    """Plain model, save_frozen=False → all params saved (all are trainable)."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sd = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=False,
    )
    ckpt_sd = sd["model_state_dict"]
    assert len(ckpt_sd) == len(model.state_dict())
    assert self._count_non_lora_keys(ckpt_sd) == len(model.state_dict())

  def test_no_lora_no_freeze_save_frozen_true(self):
    """Plain model, save_frozen=True → all params saved."""
    model = _TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    sd = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=True,
    )
    assert len(sd["model_state_dict"]) == len(model.state_dict())

  def test_no_lora_with_freeze_save_frozen_false(self):
    """Freeze backbone, save_frozen=False → only head weights saved."""
    model = _TinyModel()
    freeze_model(model, ("fc",))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=False,
    )
    ckpt_sd = sd["model_state_dict"]
    expected = self._count_trainable_non_lora_keys(model)
    assert len(ckpt_sd) == expected
    # Head weights should be present
    assert "fc.weight" in ckpt_sd
    assert "fc.bias" in ckpt_sd

  def test_no_lora_with_freeze_save_frozen_true(self):
    """Freeze backbone, save_frozen=True → ALL weights saved."""
    model = _TinyModel()
    freeze_model(model, ("fc",))
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        model,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=True,
    )
    assert len(sd["model_state_dict"]) == len(model.state_dict())

  def test_lora_no_freeze_save_frozen_false(self):
    """LoRA without --freeze, save_frozen=False → no non-lora weights saved.

    Base-model weights are frozen and will be reloaded from the original
    source on resume; the LoRA blob carries the adapters.  The checkpoint
    model_state_dict should be empty after stripping lora_ keys.
    """
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    blob = serialize_lora_state(wrapped)
    optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        wrapped,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=False,
        lora_state_blob=blob,
    )
    ckpt_sd = sd["model_state_dict"]
    # No non-LoRA keys should remain
    assert self._count_non_lora_keys(ckpt_sd) == 0
    # But the blob should be present
    assert sd["lora_state_blob"] == blob

  def test_lora_no_freeze_save_frozen_true(self):
    """LoRA without --freeze, save_frozen=True → full base weights saved."""
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    blob = serialize_lora_state(wrapped)
    optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        wrapped,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=True,
        lora_state_blob=blob,
    )
    ckpt_sd = sd["model_state_dict"]
    # All non-lora keys from the full state dict should be present
    full_sd = wrapped.state_dict()
    full_non_lora = {k: v for k, v in full_sd.items() if "lora_" not in k}
    assert len(ckpt_sd) == len(full_non_lora)
    assert sd["lora_state_blob"] == blob

  def test_lora_with_freeze_save_frozen_false(self):
    """LoRA + freeze, save_frozen=False → only head weights saved (no backbone).

    This is the case that produced 1.4 GB checkpoints before the fix.
    """
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    # Simulate --freeze: thaw lora params + head-like params
    for p in wrapped.parameters():
      p.requires_grad = False
    for name, p in wrapped.named_parameters():
      if "lora_" in name or "fc" in name:
        p.requires_grad = True

    blob = serialize_lora_state(wrapped)
    optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        wrapped,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=False,
        lora_state_blob=blob,
    )
    ckpt_sd = sd["model_state_dict"]
    expected = self._count_trainable_non_lora_keys(wrapped)
    assert len(ckpt_sd) == expected
    # fc weights should be present (trainable non-lora)
    fc_keys = [k for k in ckpt_sd if "fc" in k and "lora_" not in k]
    assert len(fc_keys) > 0, "Head weights missing from checkpoint"
    assert sd["lora_state_blob"] == blob

  def test_lora_with_freeze_save_frozen_true(self):
    """LoRA + freeze, save_frozen=True → all non-lora weights saved."""
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    for name, p in wrapped.named_parameters():
      if "lora_" not in name:
        p.requires_grad = False

    blob = serialize_lora_state(wrapped)
    optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad],
                                lr=0.01)
    sd = checkpoint_dict(
        wrapped,
        optimizer,
        scheduler=None,
        epoch=0,
        states_to_save={"opt", "sched"},
        save_frozen=True,
        lora_state_blob=blob,
    )
    ckpt_sd = sd["model_state_dict"]
    full_sd = wrapped.state_dict()
    full_non_lora = {k: v for k, v in full_sd.items() if "lora_" not in k}
    assert len(ckpt_sd) == len(full_non_lora)
    assert sd["lora_state_blob"] == blob

  def test_lora_keys_never_in_model_state_dict(self):
    """lora_ keys must never leak into model_state_dict."""
    model = _TinyModel()
    wrapped = apply_lora(model, r=4, alpha=8, target_modules=["fc"])
    blob = serialize_lora_state(wrapped)
    optimizer = torch.optim.SGD([p for p in wrapped.parameters() if p.requires_grad],
                                lr=0.01)
    for save_frozen in (True, False):
      sd = checkpoint_dict(
          wrapped,
          optimizer,
          scheduler=None,
          epoch=0,
          states_to_save={"opt", "sched"},
          save_frozen=save_frozen,
          lora_state_blob=blob,
      )
      lora_keys = [k for k in sd["model_state_dict"] if "lora_" in k]
      assert lora_keys == [], (
          f"LoRA keys leaked into model_state_dict with save_frozen={save_frozen}")
