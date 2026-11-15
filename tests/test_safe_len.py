"""Resilience of step-budget bookkeeping to length-less (streaming) loaders.

A streaming corpus reaches the trainer as an ``IterableDataset``, and
``DataLoader.__len__`` delegates to ``len(self.dataset)`` -- so ``len(loader)``
raises ``TypeError`` rather than simply reporting an unknown length.  The
dataset layer already declares this: ``TextProxy.__len__`` and
``HFImageProxy.__len__`` deliberately raise ``TypeError`` to mean "streaming",
and :func:`genml_kit.datasets.base.num_samples` tolerates it.  The training
driver once did not, which made ``--dataset_arg streaming=True`` crash in
``genml_kit/cli/train.py`` *after* the model was built and the optimizer created
-- an expensive way to fail on a GPU box.

These tests pin the contract from both ends: the ``safe_len`` API itself, the
``compute_step_budget`` consumer that already guarded inline, and the
``cli.train`` call site that actually caused the crash.
"""

import importlib
import types

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from genml_kit.datasets.base import num_samples
from genml_kit.datasets.iterable import safe_len
from genml_kit.datasets.stream_batching import (StreamingLengthBucketBatch,
                                                StreamingPackingBatch)
from genml_kit.models.ema import compute_step_budget


class _MapDataset(TensorDataset):

  def __init__(self, count=8):
    super().__init__(torch.arange(count))


class _CountingStream(IterableDataset):
  """A stream of *total* items, split evenly across dataloader workers."""

  def __init__(self, total=8):
    self.total = total

  def __iter__(self):
    info = torch.utils.data.get_worker_info()
    workers = info.num_workers if info is not None else 1
    worker = info.id if info is not None else 0
    for index in range(self.total):
      if index % workers == worker:
        yield torch.tensor([index])


class _RaisesOnLen(IterableDataset):

  def __len__(self):
    raise ValueError("genuine bug, not a streaming source")

  def __iter__(self):
    yield torch.tensor([0])


def _args(epochs=2, grad_accum_steps=2):
  return types.SimpleNamespace(epochs=epochs, grad_accum_steps=grad_accum_steps)


def _pipeline(loader):
  return types.SimpleNamespace(train_loader=loader)


class TestSafeLen:

  def test_map_style_loader_reports_real_count(self):
    loader = DataLoader(_MapDataset(8), batch_size=4)
    assert safe_len(loader) == 2

  def test_streaming_loader_returns_default_not_typeerror(self):
    loader = DataLoader(_CountingStream(), batch_size=4)
    # Precondition: this is exactly what used to blow up the trainer.
    with pytest.raises(TypeError):
      len(loader)
    assert safe_len(loader) is None
    assert safe_len(loader, default=0) == 0

  def test_plain_iterable_dataset_object(self):
    assert safe_len(_CountingStream()) is None

  def test_none_loader_returns_default(self):
    # ``len(None)`` also raises ``TypeError``; the caller supplies the fallback.
    assert safe_len(None, 0) == 0
    assert safe_len(None) is None

  def test_broken_dataset_is_not_mistaken_for_a_stream(self):
    # Only "no length" (TypeError) is swallowed -- a genuine bug still surfaces.
    with pytest.raises(ValueError):
      safe_len(DataLoader(_RaisesOnLen(), batch_size=2))

  @pytest.mark.parametrize("pipe_factory", [
      lambda: StreamingPackingBatch(_CountingStream(8), 4, 4, 1),
      lambda: StreamingLengthBucketBatch(_CountingStream(8), 4),
  ])
  def test_batch_pipes_have_no_length_by_design(self, pipe_factory):
    pipe = pipe_factory()
    assert not hasattr(pipe, "__len__")
    assert safe_len(DataLoader(pipe, batch_size=None), default=0) == 0


class TestNumSamplesUnchanged:

  def test_stream_estimate_absent(self):
    assert num_samples(_CountingStream()) is None

  def test_map_estimate_is_exact(self):
    assert num_samples(_MapDataset(8)) == 8


class TestComputeStepBudget:

  def test_streaming_loader_disables_ramp(self):
    # Regression: this was an inline try/except before it moved into safe_len.
    loader = DataLoader(_CountingStream(), batch_size=4)
    assert compute_step_budget(_args(), _pipeline(loader)) is None

  def test_map_style_loader_computes_exact_budget(self):
    # 8 items / batch 4 = 2 micro-batches; * 2 epochs // 2 accumulation = 2.
    loader = DataLoader(_MapDataset(8), batch_size=4)
    assert compute_step_budget(_args(), _pipeline(loader)) == 2

  def test_missing_loader(self):
    assert compute_step_budget(_args(), _pipeline(None)) is None


class _FakePipeline:
  """Streaming-shaped pipeline: every loader hook exists, none has a length."""

  NAME = "text"

  def __init__(self):
    self.train_loader = DataLoader(_CountingStream(), batch_size=4)
    self.val_loader = None
    self.sharded = False

  def build_loader(self, args, mode="train", method=None):
    pass

  def shard_train_loader(self, args):
    self.sharded = True

  def shard_val_loader(self, args):
    pass

  def init_env(self, args):
    pass

  def load_checkpoint_state(self, state):
    pass


class _FakeMethod:
  """Just enough of the Method contract for the driver to reach the trainer."""

  NAME = "causal_lm"
  METRIC_KEY = "loss"
  initial_best_metric = float("inf")

  def prepare_transforms(self, args, device):
    pass

  def wire_data(self, args, pipeline):
    pass

  def build_model(self, args, device):
    return torch.nn.Linear(1, 1)

  def load_checkpoint_state(self, model, state, args):
    pass

  def wrap_model(self, model, device, args):
    return model

  def get_trainer_class(self):
    return _RecordingTrainer

  def post_train(self, args, pipeline, device, result):
    pass


class _RecordingTrainer:
  """Captures the driver's computed bookkeeping instead of training."""

  captured = {}

  def __init__(self, **kwargs):
    type(self).captured = kwargs

  def run(self):
    return types.SimpleNamespace(interrupt_signals=[], result_path=None)


class _NullLogger:

  def add_scalar(self, *a, **k):
    pass

  def close(self):
    pass


class TestTrainDriverStreamingLoader:
  """End-to-end: the driver must survive a length-less train loader.

  This reproduces the field failure directly -- ``cli.train._run`` with a
  streaming pipeline used to die in the ``global_step`` derivation, after the
  model and optimizer had already been built.  Only the collaborators around
  that line are stubbed, so the test asserts the *driver's* bookkeeping rather
  than re-testing the trainer.
  """

  def test_streaming_train_loader_does_not_crash_driver(self, tmp_path, monkeypatch):
    train = importlib.import_module("genml_kit.cli.train")
    pipeline = _FakePipeline()
    # Precondition: the loader really is hostile to ``len()``.
    with pytest.raises(TypeError):
      len(pipeline.train_loader)

    monkeypatch.setattr(train, "build_pipeline", lambda args: pipeline)
    monkeypatch.setattr(train, "build_method", lambda args: _FakeMethod())
    monkeypatch.setattr(train, "resolve_device", lambda spec: torch.device("cpu"))
    monkeypatch.setattr(train, "build_optimization", lambda *a, **k: None)
    monkeypatch.setattr(train, "build_metrics_logger", lambda args: _NullLogger())
    monkeypatch.setattr(train, "broadcast_model_state", lambda model: None)
    monkeypatch.setattr(train, "warn_world_size_mismatch", lambda *a: None)
    monkeypatch.setattr(train, "setup_logging", lambda *a, **k: None)
    # Resume *into* epoch 2 with no batch count available: this is the exact
    # arithmetic that raised TypeError before the fix.
    monkeypatch.setattr(
        train, "open_resume_context", lambda *a, **k: types.SimpleNamespace(
            model=a[1], start_epoch=2, best_metric=float("inf"), extra={}))

    args = types.SimpleNamespace(
        log_level="WARNING",
        log_targets="STDERR",
        seed=None,
        device=None,
        pipeline="text",
        method="causal_lm",
        state_load="none",
        epochs=3,
        grad_accum_steps=4,
        log_dir=str(tmp_path),
        checkpoint=str(tmp_path / "ckpt"),
    )

    train._run(args)

    # start_epoch=2 with an unknown batch count -> 2 * (0 // 4) == 0.
    assert _RecordingTrainer.captured["global_step"] == 0
    # The shard hook still ran, i.e. we failed at the bookkeeping, not earlier.
    assert pipeline.sharded is True
