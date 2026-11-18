"""Training-run budgets: bound a supervised run by batches-per-epoch or wall time.

Two collaborating objects, split so each has one clear lifetime:

* :class:`TrainingBudget` -- one per *run*, owned by ``BaseTrainer``.  It holds
  the single wall-clock ``start`` timestamp, the per-epoch batch cap and the
  synchronized time-vote state, so "total training time" is measured from one
  fixed origin rather than resetting every epoch.
* :class:`EpochBatches` -- one per *epoch*, an ``__iter__``-bearing view over a
  loader that yields :class:`TrainingSample` records (blob + position + timings).
  It delegates every run-scoped question back to the budget, so a fresh iterator
  each epoch still shares one accumulating clock.
"""

import time
from collections import namedtuple

import torch

from genml_kit.utils.distributed import all_reduce_max, is_distributed

#: One yielded batch: the payload plus the position and timing it was read at.
#:
#: ``batch_in_epoch`` is the *absolute* micro-batch index (post-resume-skip), so
#: it stays ``grad_accum_steps``-aligned with the accumulation boundary and the
#: tail flush exactly like the ``enumerate(source, start=skipped)`` counter it
#: replaces.
TrainingSample = namedtuple("TrainingSample", [
    "blob",
    "epoch",
    "batch_in_epoch",
    "elapsed_epoch",
    "elapsed_total",
])

#: Target wall-clock spacing between synchronized time checks, in seconds.  The
#: checker adapts how many flush windows it skips so consecutive checks land
#: roughly this far apart, without ever checking more often than every window.
TIME_CHECK_INTERVAL = 15.0


class TrainingBudget:
  """Run-scoped stop conditions: ``max_batches`` (per epoch) and ``max_train_time``.

  ``max_batches`` truncates each epoch's batch loop independently (the outer
  epoch loop is untouched).  ``max_train_time`` is a *run-global* wall-clock
  budget measured from a single origin captured once at :meth:`begin_run`.

  The time check is a DDP synchronization point: ranks poll their own clock but
  agree on the verdict through one collective, so they stop on the same batch
  index and the following gradient all-reduce stays aligned.  When no time
  budget is set -- or the run is single-process -- there is no collective at all.

  Args:
      max_batches: Per-epoch cap on micro-batches, or ``None``.
      max_train_time: Run-global wall-clock budget in seconds, or ``None``.
      grad_accum_steps: Accumulation window size; the check cadence is a whole
          number of windows, so checks land on flush boundaries.
      clock: Monotonic time source; injectable for tests.
  """

  def __init__(self,
               *,
               max_batches=None,
               max_train_time=None,
               grad_accum_steps=1,
               clock=None):
    self.max_batches = max_batches
    self.max_train_time = max_train_time
    self._accum = max(1, int(grad_accum_steps or 1))
    self._clock = clock if clock is not None else time.monotonic
    self._start = None
    self._epoch_start = None
    #: Flush windows between time checks (adaptive; fixed after the first check).
    self._windows_per_check = 1
    self._windows_since_check = 0
    self._last_check_time = None
    self._cadence_set = False
    #: Set once the time vote says stop; read by the outer epoch loop.
    self.time_exhausted = False
    #: Whether the most recent epoch stopped at the ``max_batches`` cap.
    self.last_epoch_capped = False
    #: Total micro-batches actually yielded across the whole run.
    self.batches_seen = 0

  @property
  def has_time_budget(self):
    """Whether a positive ``max_train_time`` is in force."""
    return bool(self.max_train_time and self.max_train_time > 0)

  @property
  def start(self):
    """The single run-start timestamp (set by :meth:`begin_run`)."""
    return self._start

  def now(self):
    """Current reading of the injected clock."""
    return self._clock()

  def begin_run(self):
    """Capture the single run-start timestamp (idempotent)."""
    if self._start is None:
      self._start = self.now()
      self._last_check_time = self._start

  def begin_epoch(self):
    """Mark the start of an epoch (the per-epoch timing origin).

    Also re-bases the cadence timer on the epoch origin, so the first check
    measures one clean window rather than the (setup-inflated) span from run
    start -- all ranks do this at the same epoch boundary, so the collective
    stays in lockstep.
    """
    self._epoch_start = self.now()
    self._last_check_time = self._epoch_start
    self._windows_since_check = 0

  def at_flush(self, batch_in_epoch):
    """Whether *batch_in_epoch* closes an accumulation window."""
    return (batch_in_epoch + 1) % self._accum == 0

  def cap_epoch_batches(self, estimate):
    """Apply the per-epoch batch cap to a reporter's progress denominator."""
    if estimate is None or self.max_batches is None:
      return estimate
    return min(estimate, self.max_batches)

  def note_flush(self, now):
    """Record a flush boundary; return ``True`` when a time check is due."""
    if not self.has_time_budget:
      return False
    self._windows_since_check += 1
    return self._windows_since_check >= self._windows_per_check

  def time_check(self, now, *, tune=False):
    """Synchronized time vote; return ``True`` to stop.

    Packs the observed per-window duration and the local expiry flag into a
    single max-reduced tensor, so every rank leaves with the same verdict in one
    collective (and, on the first tuned check, the same agreed cadence).
    """
    per_window = (now - self._last_check_time) / max(1, self._windows_since_check)
    expired = (now - self._start) >= self.max_train_time
    if is_distributed():
      packed = torch.tensor([per_window, 1.0 if expired else 0.0])
      packed = all_reduce_max(packed)
      per_window = float(packed[0].item())
      expired = bool(packed[1].item() > 0)
    self._last_check_time = now
    self._windows_since_check = 0
    if tune and not self._cadence_set and per_window > 0:
      self._windows_per_check = max(1, int(round(TIME_CHECK_INTERVAL / per_window)))
      self._cadence_set = True
    if expired:
      self.time_exhausted = True
    return expired

  def stop_reason(self):
    """Why the run ended: ``"max_time"``, ``"max_batches"`` or ``"epochs"``."""
    if self.time_exhausted:
      return "max_time"
    if self.max_batches is not None and self.last_epoch_capped:
      return "max_batches"
    return "epochs"


class EpochBatches:
  """Iterate one epoch's loader, yielding :class:`TrainingSample` records.

  Stops on source exhaustion, the per-epoch ``max_batches`` cap, or a
  ``max_train_time`` vote.  Every stop is a normal loop exit (``StopIteration``),
  never an exception, so the caller's post-loop tail flush still runs.
  """

  def __init__(self, source, *, epoch, budget, start_batch=0):
    self._source = source
    self._epoch = epoch
    self._budget = budget
    self._start_batch = start_batch

  def __iter__(self):
    budget = self._budget
    # Idempotent: fixes the run origin when a caller skips ``begin_run`` (e.g.
    # ``train_epoch`` driven directly by a test).
    budget.begin_run()
    budget.begin_epoch()
    epoch_start = budget.now()
    limit = budget.max_batches
    capped = False
    for offset, blob in enumerate(self._source):
      if limit is not None and offset >= limit:
        capped = True
        break
      batch_in_epoch = self._start_batch + offset
      # Flush boundaries are where the gradient all-reduce already rendezvous, so
      # the time vote piggybacks one instead of adding a sync point.
      if budget.at_flush(batch_in_epoch):
        now = budget.now()
        if budget.note_flush(now) and budget.time_check(now, tune=True):
          break
      now = budget.now()
      budget.batches_seen += 1
      yield TrainingSample(blob=blob,
                           epoch=self._epoch,
                           batch_in_epoch=batch_in_epoch,
                           elapsed_epoch=now - epoch_start,
                           elapsed_total=now - budget.start)
    budget.last_epoch_capped = capped
