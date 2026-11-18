"""Unit tests for the training-budget wrapper (``genml_kit.training.budget``).

Covers the two collaborating objects in isolation: ``TrainingBudget`` (run-scoped
stop conditions, adaptive time-check cadence, cap on the progress denominator)
and ``EpochBatches`` (the ``__iter__`` view that yields ``TrainingSample``).
"""

import genml_kit.training.budget as budget_mod
from genml_kit.training.budget import EpochBatches, TrainingBudget


class _Clock:
  """Deterministic, manually-advanced stand-in for ``time.monotonic``."""

  def __init__(self, start=0.0):
    self.t = start

  def __call__(self):
    return self.t

  def advance(self, seconds):
    self.t += seconds


def _drain(iterable):
  return list(iterable)


def test_yields_blobs_with_position_and_epoch():
  clock = _Clock()
  budget = TrainingBudget(clock=clock)
  samples = _drain(EpochBatches(["a", "b", "c"], epoch=2, budget=budget))

  assert [s.blob for s in samples] == ["a", "b", "c"]
  assert [s.batch_in_epoch for s in samples] == [0, 1, 2]
  assert all(s.epoch == 2 for s in samples)
  assert budget.batches_seen == 3


def test_start_batch_offsets_position_absolutely():
  # A resume skip must shift the *absolute* index, not re-base to zero, or the
  # grad-accum boundary would desynchronise from the checkpoint.
  budget = TrainingBudget(clock=_Clock())
  samples = _drain(EpochBatches(["a", "b"], epoch=0, budget=budget, start_batch=4))
  assert [s.batch_in_epoch for s in samples] == [4, 5]


def test_max_batches_truncates_each_epoch():
  budget = TrainingBudget(max_batches=2, clock=_Clock())
  first = _drain(EpochBatches(["a", "b", "c", "d"], epoch=0, budget=budget))
  second = _drain(EpochBatches(["a", "b", "c", "d"], epoch=1, budget=budget))

  # Per-epoch: the cap applies independently, so the next epoch is not starved.
  assert [s.batch_in_epoch for s in first] == [0, 1]
  assert [s.batch_in_epoch for s in second] == [0, 1]
  assert budget.last_epoch_capped is True
  assert budget.stop_reason() == "max_batches"


def test_tail_flush_signal_is_a_normal_exit():
  # A capped epoch must look like an exhausted iterator (a clean ``for`` exit),
  # so the caller's post-loop tail flush still runs -- never an exception.
  budget = TrainingBudget(max_batches=1, clock=_Clock())
  out = []
  for sample in EpochBatches(["a", "b"], epoch=0, budget=budget):
    out.append(sample.blob)
  assert out == ["a"]


def test_cap_epoch_batches_applies_to_denominator():
  assert TrainingBudget(max_batches=3).cap_epoch_batches(10) == 3
  assert TrainingBudget(max_batches=3).cap_epoch_batches(2) == 2
  # No cap, or an unknown (stream) estimate, passes through untouched.
  assert TrainingBudget().cap_epoch_batches(10) == 10
  assert TrainingBudget(max_batches=3).cap_epoch_batches(None) is None


def test_no_time_budget_means_no_time_collective(monkeypatch):
  called = []
  monkeypatch.setattr(budget_mod, "is_distributed", lambda: True)
  monkeypatch.setattr(budget_mod, "all_reduce_max", lambda v: called.append(v) or v)

  budget = TrainingBudget(clock=_Clock())
  samples = _drain(EpochBatches(["a", "b", "c"], epoch=0, budget=budget))

  assert len(samples) == 3
  assert called == []


def test_time_budget_stops_mid_epoch():
  clock = _Clock()
  # 20s per batch -> the cadence rounds to one window (check every flush), so
  # the stop is crisp: elapsed crosses 50s at the fourth check.
  budget = TrainingBudget(max_train_time=50.0, grad_accum_steps=1, clock=clock)
  seen = []

  def source():
    while True:
      yield "b"

  for sample in EpochBatches(source(), epoch=0, budget=budget):
    seen.append(sample.batch_in_epoch)
    clock.advance(20.0)
    if len(seen) > 10:
      break  # safety

  assert budget.time_exhausted is True
  assert budget.stop_reason() == "max_time"
  assert seen == [0, 1, 2]


def test_adaptive_cadence_tunes_after_first_check():
  clock = _Clock()
  budget = TrainingBudget(max_train_time=10_000.0, grad_accum_steps=1, clock=clock)

  def source():
    while True:
      yield "b"

  # Each flush advances 20s, so the first check measures a 20s window and the
  # cadence settles at round(15/20) == 1 window.
  for sample in EpochBatches(source(), epoch=0, budget=budget):
    clock.advance(20.0)
    if sample.batch_in_epoch >= 2:
      break

  assert budget._cadence_set is True
  assert budget._windows_per_check == 1


def test_stop_reason_defaults_to_epochs():
  budget = TrainingBudget(clock=_Clock())
  budget.begin_run()
  _drain(EpochBatches(["a", "b"], epoch=0, budget=budget))
  assert budget.stop_reason() == "epochs"
