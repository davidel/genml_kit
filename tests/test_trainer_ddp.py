"""Multi-process regression test for DDP grad-accumulation sync (torchrun).

Two files in one: this module is both a pytest file and the torchrun driver
it launches (``__main__`` guard).  The driver runs the real ``BaseTrainer``
on 2 ranks with ``grad_accum_steps=2`` over 3 rank-diverse micro-batches and
writes a per-rank marker file carrying

* the number of DDP bucket reductions observed (via a counting comm hook);
* a sha256 of the final parameter values.

The parent test asserts every rank observed exactly ``ceil(3 / 2) == 2``
reductions -- one per accumulation window.  Before the ``no_sync`` fix the
same run performed 3 reductions (one per micro-batch: DDP arms its reducer
at forward time, so wrapping only the backward in ``no_sync`` deferred
nothing), which is what this file guards against.

Skipped when ``torchrun`` or ``torch.distributed`` is unavailable.
"""
import hashlib
import os
import pathlib
import subprocess
import shutil
import sys

import pytest

torch = pytest.importorskip("torch")

_NPROC = 2
_ACCUM = 2
_N_BATCHES = 3

_DRIVER = pathlib.Path(__file__).resolve()


class _FakeLogger:

  def log_step(self, tag, value, *, step, resolution=None, kind=None):
    pass

  def log_epoch(self, tag, value, *, epoch, kind=None):
    pass

  def log_artifact(self, tag, payload_kind, payload, *, step=None):
    pass

  def add_scalar(self, tag, value, step):
    pass

  def close(self):
    pass


class _WrappedLoader:

  def __init__(self, batches):
    self.batches = batches

  def __len__(self):
    return len(self.batches)

  def __iter__(self):
    return iter(self.batches)


class _Pipeline:

  def __init__(self, batches):
    self.train_loader = _WrappedLoader(batches)
    self.val_loader = None

  @staticmethod
  def to_device(blob, device):
    return blob


class _Method:
  METRIC_KEY = "macro_f1"
  NAME = "ddp-accum-regression"

  def train_step(self, model, blob, global_step):
    from genml_kit.pipelines.contracts import LossOutput

    out = model(blob.data)
    target = float(os.environ["LOCAL_RANK"]) + 1.0
    loss = (out - target).pow(2).sum()
    return LossOutput(loss=loss, metrics={})

  def evaluate(self, model, loader, device, to_device):
    raise AssertionError("validation must not run (val_loader is None)")

  def has_metric_improved(self, new, best):
    return new > best

  def on_epoch_end(self, model, epoch, logger):
    pass

  def get_checkpoint_state(self, model, args):
    return {}

  def load_checkpoint_state(self, model, state, args):
    pass

  @classmethod
  def add_args(cls, parser):
    pass


def _run_driver(tmp_path):
  from genml_kit.pipelines.contracts import DataBlob
  from genml_kit.training.trainer import BaseTrainer
  from genml_kit.utils.distributed import init_distributed, wrap_model

  import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default_hooks

  init_distributed()
  rank = int(os.environ["RANK"])
  world = int(os.environ["WORLD_SIZE"])

  import tests.test_trainer as tt
  args = tt._Args()
  args.epochs = 1
  args.grad_accum_steps = _ACCUM
  args.grad_clip = 0.0
  args.amp_dtype = None
  args.save_every = 0
  args.checkpoint = str(tmp_path / "ckpt")
  args.log_every = 1

  torch.manual_seed(1234)
  raw = torch.nn.Linear(4, 2, bias=False)
  model = wrap_model(raw, torch.device("cpu"))

  counter = {"n": 0}

  def hook(state, bucket):
    # Count the reduction, then perform DDP's real (averaging) allreduce --
    # a bare Future would swallow the gradient exchange entirely.
    counter["n"] += 1
    return default_hooks.allreduce_hook(state, bucket)

  model._forward_module.register_comm_hook(state=None, hook=hook)

  scale = float(rank) + 1.0
  batches = [
      DataBlob(data=torch.ones(1, 4) * scale, meta={}, size=1)
      for _ in range(_N_BATCHES)
  ]

  optimization = tt._Optimization()
  optimization.optimizer = torch.optim.SGD(raw.parameters(), lr=0.01)

  trainer = BaseTrainer(args=args,
                        model=model,
                        method=_Method(),
                        pipeline=_Pipeline(batches),
                        optimization=optimization,
                        device=torch.device("cpu"),
                        logger=_FakeLogger(),
                        start_epoch=0,
                        best_metric=None,
                        global_step=0)
  trainer.run()

  flat = torch.cat([q.detach().flatten() for q in raw.parameters()])
  digest = hashlib.sha256(flat.numpy().tobytes()).hexdigest()
  (tmp_path / f"rank{rank}.marker"
  ).write_text(f"reductions={counter['n']} digest={digest} world={world}")


def _torchrun_available():
  return shutil.which("torchrun") is not None


def _run_torchrun(tmp_path):
  cmd = ["torchrun", "--standalone", f"--nproc_per_node={_NPROC}", str(_DRIVER)]
  env = dict(os.environ)
  env["PYTHONPATH"] = str(_DRIVER.parents[1])
  env["DIST_E2E_TMPDIR"] = str(tmp_path)
  return subprocess.run(cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False)


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_grad_accum_reduces_once_per_window_under_ddp(tmp_path):
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")
  proc = _run_torchrun(tmp_path)
  if proc.returncode != 0:
    logs = sorted(tmp_path.glob("error.rank*.log"))
    detail = "\n".join(p.name for p in logs)
    pytest.fail(f"torchrun failed (rc={proc.returncode})\n{proc.stdout}\n"
                f"{proc.stderr}\n{detail}")
  markers = sorted(tmp_path.glob("rank*.marker"))
  assert len(markers) == 2
  counts = []
  digests = []
  for p in markers:
    for part in p.read_text().split():
      if part.startswith("reductions="):
        counts.append(int(part.split("=")[1]))
      if part.startswith("digest="):
        digests.append(part.split("=")[1])
  assert counts == [2, 2]
  assert len(set(digests)) == 1


if __name__ == "__main__":
  try:
    _run_driver(pathlib.Path(os.environ["DIST_E2E_TMPDIR"]))
  except BaseException:
    import traceback
    tb = traceback.format_exc()
    with open(
        os.path.join(os.environ["DIST_E2E_TMPDIR"],
                     f"error.rank{os.environ.get('RANK', '?')}.log"), "w") as fh:
      fh.write(tb)
    sys.stderr.write(tb)
    raise
