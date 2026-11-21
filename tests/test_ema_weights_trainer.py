"""Trainer-integration tests for the eval-time weights EMA."""

import pytest
import torch

from genml_kit.pipelines.contracts import ValidationResult
from tests.test_ema_weights import _EmaMethod, _make_ema_trainer
from tests.test_trainer import FakeMethod


class _WeightProbeMethod(_EmaMethod):
  """EMA method whose validation metric reports the live weight value.

  ``evaluate`` returns ``fc.weight[0][0]`` as the metric, so a test can
  assert exactly which weights validation observed (raw vs. swapped).
  """

  def evaluate(self, model, loader, device, to_device):
    weight_value = model.fc.weight[0][0].item()
    return ValidationResult({"macro_f1": weight_value, "loss": weight_value}, 4.0)


def test_no_ema_method_leaves_trainer_untouched(tmp_path):
  """Methods without the mixin (e.g. RL/fixture methods) get ``ema is None``."""
  trainer, method, model = _make_ema_trainer(tmp_path, method=FakeMethod())
  assert trainer.ema is None
  result = trainer.run()
  assert result.completed_epoch == 1


def test_ema_decay_without_capability_warns(tmp_path, caplog):
  """``--ema_decay`` on a method without the mixin warns instead of no-op."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=FakeMethod(),
                                             args_attrs={"ema_decay": 0.9})
  assert trainer.ema is None
  assert any("does not support" in record.message for record in caplog.records)
  # The run itself is unaffected.
  assert trainer.run().completed_epoch == 1


def test_wire_ema_called_once_with_args(tmp_path):
  """The trainer wires the EMA once, passing the args, model and pipeline."""
  trainer, method, model = _make_ema_trainer(tmp_path)
  trainer.run()
  assert method.wire_calls == 1
  args, wired_model, pipeline = method.wire_call_args
  assert wired_model is model
  assert pipeline is trainer.pipeline


def test_validation_runs_on_swapped_weights(tmp_path):
  """With eval enabled, validation sees the EMA weights, not the raw ones."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(decay=1.0))
  # decay=1.0 keeps the shadow bit-identical to what it was seeded from, so
  # the probe can tell raw (4.0) from shadow (0.0) without any update drift.
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
    trainer.ema.target.fc.weight.fill_(0.0)
  trainer.run()
  # metric == weight value: the shadow (0.0) was in the model during
  # validation, and the raw weights (4.0) were restored afterwards.
  assert trainer.best_metric == pytest.approx(0.0, abs=1e-4)
  assert model.fc.weight[0][0].item() == pytest.approx(4.0, abs=1e-4)


def test_validation_runs_on_raw_weights_when_eval_disabled(tmp_path):
  """``eval_enabled=False`` keeps validation on the raw trajectory."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(
                                                 decay=1.0, eval_enabled=False))
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
    trainer.ema.target.fc.weight.fill_(0.0)
  trainer.run()
  assert trainer.best_metric == pytest.approx(4.0, abs=1e-4)
  assert model.fc.weight[0][0].item() == pytest.approx(4.0, abs=1e-4)


def test_ema_updates_per_optimizer_step(tmp_path):
  """The shadow advances once per optimizer step (grad-accum folded in)."""
  trainer, method, model = _make_ema_trainer(tmp_path)
  # _Args sets epochs=2, so the run performs 2 optimizer steps.
  trainer.run()
  assert trainer.ema.step == 2


def test_best_checkpoint_carries_ema_weights(tmp_path):
  """Inside the window, best.pt records EMA weights + the pre-swap shadow."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(decay=0.5))
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
    trainer.ema.target.fc.weight.fill_(0.0)
  trainer.run()
  checkpoint = torch.load(str(tmp_path / "ckpt_best.pt"), weights_only=False)
  # The model slot holds the EMA shadow, not the raw 4.0.  With decay=0.5
  # the shadow (seeded at 0.0, live at 4.0) is 2.0 after the first update
  # and 3.0 after the second, so best.pt records 3.0.
  weight_value = checkpoint["model_state_dict"]["fc.weight"][0][0].item()
  assert weight_value == pytest.approx(3.0, abs=1e-4)
  # ema_state rides every save: the pre-swap shadow snapshot, for an exact
  # resume (here identical to the weights, since the shadow was captured
  # just before the swap).
  assert "ema_state" in checkpoint
  assert checkpoint["ema_state"]["step"] == trainer.ema.step
  shadow_weight = checkpoint["ema_state"]["state"]["fc.weight"][0][0].item()
  assert shadow_weight == pytest.approx(3.0, abs=1e-4)


def test_latest_checkpoint_keeps_raw_weights(tmp_path):
  """latest.pt (exit path, post-swap-out) carries raw weights + the shadow."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(decay=0.5))
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
    trainer.ema.target.fc.weight.fill_(0.0)
  trainer.run()
  checkpoint = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
  weight_value = checkpoint["model_state_dict"]["fc.weight"][0][0].item()
  assert weight_value == pytest.approx(4.0, abs=1e-4)
  assert "ema_state" in checkpoint


def test_resume_from_pre_ema_checkpoint(tmp_path):
  """A checkpoint without ``ema_state`` resumes with a fresh shadow."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(decay=1.0))
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
  # Simulate a checkpoint written before the feature existed: no ema_state.
  trainer2, method2, model2 = _make_ema_trainer(tmp_path,
                                                method=_WeightProbeMethod(decay=1.0),
                                                ema_state=None)
  assert trainer2.ema is not None
  # Fresh shadow == live weights: it has not been updated yet.
  assert torch.allclose(trainer2.ema.target.fc.weight, model2.fc.weight)


def test_resume_restores_shadow_from_ema_state(tmp_path):
  """``ema_state`` in a checkpoint restores the shadow and schedule step."""
  trainer, method, model = _make_ema_trainer(tmp_path,
                                             method=_WeightProbeMethod(decay=1.0))
  with torch.no_grad():
    model.fc.weight.fill_(4.0)
    trainer.ema.target.fc.weight.fill_(0.0)
  trainer.ema.step = 7
  payload = trainer.ema.state_dict()
  trainer2, method2, model2 = _make_ema_trainer(tmp_path,
                                                method=_WeightProbeMethod(decay=1.0),
                                                ema_state=payload)
  assert trainer2.ema.step == 7
  assert torch.allclose(trainer2.ema.target.fc.weight, trainer.ema.target.fc.weight)
