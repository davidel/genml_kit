"""Multi-process DDP test for the training-time budget (the STOP vote).

The ``max_train_time`` budget stops a run through one collective: every rank
polls its own clock, but a single max-reduce over (per-window duration, expired
flag) makes every rank leave with the *same* verdict.  This file proves the two
consequences that matter under DDP:

* ranks with *different* clock speeds still stop on the same batch index -- so
  the following gradient all-reduce stays aligned and nothing hangs;
* the adaptive check cadence is derived from the max-reduced window time, so
  ranks agree on *when* to check (not just *whether* to stop).

Skips when ``torch.distributed`` is unavailable.
"""

import os
import socket

import pytest
import torch

from genml_kit.training.budget import EpochBatches, TrainingBudget
from genml_kit.utils import distributed


def _gloo_available():
  return distributed.dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(),
                                   reason="torch.distributed unavailable")


def _free_port():
  """Return a currently-free localhost TCP port for the rendezvous.

  Binding port 0 and reading back the kernel-assigned port sidesteps the fixed
  port numbers the rest of the DDP suite hardcodes: two parallel CI shards (or a
  previous run's socket still in ``TIME_WAIT``) can no longer collide on the same
  number.  This is the same bind-then-close trick torch's own test suite uses;
  the tiny gap before ``gloo`` re-binds is far safer than a shared constant.
  """
  for _ in range(10):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
      try:
        probe.bind(("127.0.0.1", 0))
      except OSError:
        continue
      return probe.getsockname()[1]
  raise RuntimeError("no free localhost port available for rendezvous")


def _init(rank, world, port):
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world)
  os.environ["LOCAL_RANK"] = str(rank)
  distributed.dist.init_process_group("gloo", rank=rank, world_size=world)


class _StepClock:
  """Monotonic clock that advances a fixed *step* on every reading.

  Giving each rank a different step models the real hazard: without the vote,
  the slow rank would run far longer than the fast one, and their collectives
  would desynchronise.
  """

  def __init__(self, step):
    self.step = step
    self.t = 0.0

  def __call__(self):
    self.t += self.step
    return self.t


def _infinite_source():
  while True:
    yield torch.zeros(1)


def _time_budget_worker(rank, world, port, tmpdir):
  """Stop an infinite stream on a time budget; record where each rank stopped."""
  _init(rank, world, port)
  try:
    # Rank 0's clock races (1.0s/reading); rank 1's crawls (0.05s/reading).  The
    # fast rank expires the 10s budget long before the slow one ever would.
    step = 1.0 if rank == 0 else 0.05
    budget = TrainingBudget(max_train_time=10.0,
                            grad_accum_steps=1,
                            clock=_StepClock(step))

    seen = 0
    # A safety cap so a broken (unsynchronized) design fails loudly instead of
    # hanging the suite; correct behaviour stops far below it.
    for _sample in EpochBatches(_infinite_source(), epoch=0, budget=budget):
      seen += 1
      if seen >= 2000:
        break

    torch.save(
        {
            "seen": seen,
            "exhausted": budget.time_exhausted,
            "cadence": budget._windows_per_check,
        },
        os.path.join(tmpdir, f"budget{rank}.pt"),
    )
  finally:
    distributed.destroy_distributed()


def _max_batches_worker(rank, world, port, tmpdir):
  """``max_batches`` is deterministic, so every rank yields exactly the cap."""
  _init(rank, world, port)
  try:
    budget = TrainingBudget(max_batches=3, clock=lambda: 0.0)
    seen = sum(1 for _ in EpochBatches(_infinite_source(), epoch=0, budget=budget))
    torch.save(seen, os.path.join(tmpdir, f"cap{rank}.pt"))
  finally:
    distributed.destroy_distributed()


def _spawn(fn, tmp_path, port, world=2):
  # ``spawn`` pickles the target, so ``fn`` must be module level.
  torch.multiprocessing.spawn(fn,
                              args=(world, port, str(tmp_path)),
                              nprocs=world,
                              join=True)


@requires_gloo
def test_time_vote_stops_all_ranks_on_the_same_batch(tmp_path):
  """Divergent clocks still stop on one shared index (no hang, no skew)."""
  _spawn(_time_budget_worker, tmp_path, port=_free_port())
  r0 = torch.load(tmp_path / "budget0.pt", weights_only=False)
  r1 = torch.load(tmp_path / "budget1.pt", weights_only=False)

  # The vote actually ended the run -- both ranks hit the time budget.
  assert r0["exhausted"] is True
  assert r1["exhausted"] is True
  # ... and they stopped at the *same* batch, despite the 20x clock difference.
  assert r0["seen"] == r1["seen"]
  assert 0 < r0["seen"] < 2000
  # The cadence is agreed (max-reduced), so both ranks will next check together.
  assert r0["cadence"] == r1["cadence"]


@requires_gloo
def test_max_batches_cap_is_identical_on_every_rank(tmp_path):
  """A per-epoch deterministic cap needs no collective and stays in lockstep."""
  _spawn(_max_batches_worker, tmp_path, port=_free_port())
  c0 = torch.load(tmp_path / "cap0.pt", weights_only=False)
  c1 = torch.load(tmp_path / "cap1.pt", weights_only=False)
  assert c0 == c1 == 3
