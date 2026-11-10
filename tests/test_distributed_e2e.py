"""End-to-end multi-GPU test: real ``torchrun``, 2 ranks, gloo/CPU.

Runs the actual ``genml_kit.cli.train.main()`` in two processes (via
``torchrun --standalone``) on a tiny synthetic dataset with a mock model, then
asserts the distributed path really engaged:

* both ranks joined the group (``world=2``);
* rank 0 alone wrote the checkpoint;
* the checkpoint keys carry no ``module.`` prefix (portable single<->multi).

Skipped when ``torch.distributed`` or the ``torchrun`` launcher is unavailable,
or when the environment cannot spawn processes (e.g. restricted sandboxes).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_DRIVER = Path(__file__).parent / "dist_e2e_driver.py"


def _torchrun_available():
  return shutil.which("torchrun") is not None


def _run_torchrun(tmp_path, nproc=2, env_extra=None):
  env = dict(os.environ)
  env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
  env["DIST_E2E_TMPDIR"] = str(tmp_path)
  if env_extra:
    env.update(env_extra)
  cmd = [
      "torchrun",
      "--nproc_per_node",
      str(nproc),
      "--master_port",
      "29777",
      "--standalone",
      str(_DRIVER),
  ]
  return subprocess.run(cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False)


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_two_rank_torchrun_training(tmp_path):
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  proc = _run_torchrun(tmp_path)
  if proc.returncode != 0:
    # Surface rank tracebacks (the driver writes one per rank) for debugging.
    logs = sorted(tmp_path.glob("error*.log"))
    detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
    pytest.fail(f"torchrun failed (rc={proc.returncode})\n{proc.stdout}\n"
                f"{proc.stderr}\n{detail}")

  # Both ranks must have joined the group.
  markers = sorted(tmp_path.glob("rank*.marker"))
  assert len(markers) == 2, f"expected 2 rank markers, got {markers}"
  contents = {p.name: p.read_text() for p in markers}
  assert contents["rank0.marker"] == "rank=0 world=2"
  assert contents["rank1.marker"] == "rank=1 world=2"

  # Exactly one writer: rank 0's checkpoint exists.
  ckpt = tmp_path / "ckpts" / "model_latest.pt"
  assert ckpt.exists(), "rank 0 did not write a checkpoint"

  # Portable keys: the facade must keep the semantic model's own key layout.
  state = torch.load(ckpt, map_location="cpu", weights_only=False)
  keys = list(state["model_state_dict"])
  assert keys, "empty model_state_dict"
  assert not any(k.startswith("module.") for k in keys), keys


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_single_rank_torchrun_matches_plain_run(tmp_path):
  """A 1-process torchrun run must behave like the plain single-GPU path."""
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  proc = _run_torchrun(tmp_path, nproc=1)
  if proc.returncode != 0:
    logs = sorted(tmp_path.glob("error*.log"))
    detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
    pytest.fail(f"torchrun failed (rc={proc.returncode})\n{proc.stdout}\n"
                f"{proc.stderr}\n{detail}")

  assert (tmp_path / "rank0.marker").read_text() == "rank=0 world=1"
  assert (tmp_path / "ckpts" / "model_latest.pt").exists()


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
@pytest.mark.parametrize("method_name", ["simmim", "byol"])
def test_two_rank_ssl_pretraining(method_name, tmp_path):
  """A real SSL method must pre-train end to end under ``torchrun``.

    Unlike the classification e2e above (mocked model/processor), this drives
    the *real* registered ``convvit`` backbone and the *real* images
    pre-training pipeline through ``genml_kit.cli.train.main()``.  That makes
    it the proof that method-declared ``arg_defaults`` (e.g. BYOL's
    ``find_unused_parameters``) are applied by production argument parsing --
    without them the second iteration aborts with "parameters that were not
    used in producing loss".

    Two methods are covered: ``simmim`` (label-free, no EMA tower) and ``byol``
    (EMA teacher, the case that *requires* the DDP knob).
    """
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  proc = _run_torchrun(tmp_path, env_extra={"DIST_E2E_SSL_METHOD": method_name})
  if proc.returncode != 0:
    logs = sorted(tmp_path.glob("error*.log"))
    detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
    pytest.fail(f"torchrun ssl ({method_name}) failed (rc={proc.returncode})\n"
                f"{proc.stdout}\n{proc.stderr}\n{detail}")

  # Both ranks must have joined the group.
  markers = sorted(tmp_path.glob("rank*.marker"))
  assert len(markers) == 2, f"expected 2 rank markers, got {markers}"
  contents = {p.name: p.read_text() for p in markers}
  assert contents["rank0.marker"] == "rank=0 world=2"
  assert contents["rank1.marker"] == "rank=1 world=2"

  # Exactly one writer: rank 0's checkpoint exists, with portable keys.
  ckpt = tmp_path / "ckpts" / "model_latest.pt"
  assert ckpt.exists(), "rank 0 did not write a checkpoint"
  state = torch.load(ckpt, map_location="cpu", weights_only=False)
  keys = list(state["model_state_dict"])
  assert keys, "empty model_state_dict"
  assert not any(k.startswith("module.") for k in keys), keys


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_two_rank_mid_epoch_resume(tmp_path):
  """A 2-rank run interrupted mid-epoch resumes without deadlocking.

  Guards the DDP lockstep invariant: on resume every rank derives the *same*
  skip from the single rank-0-written checkpoint, so they hit the same
  all-reduce windows.  A per-rank skip would skew the windows and hang.
  """
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  # Phase 1: run with periodic saving, interrupting during the *training*
  # forwards (each rank sees 2 training batches of the 12-sample/bs4 dataset;
  # forwards 3+ belong to validation, after the epoch marker is cleared).
  interrupted = _run_torchrun(tmp_path,
                              env_extra={
                                  "DIST_E2E_SAVE_EVERY": "1",
                                  "DIST_E2E_INTERRUPT_AT": "2",
                              })
  # An interrupt may surface as a non-zero rc; what matters is the marker.
  ckpt = tmp_path / "ckpts" / "model_latest.pt"
  if not ckpt.exists():
    logs = sorted(tmp_path.glob("error*.log"))
    detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
    pytest.fail(f"no checkpoint after interrupt (rc={interrupted.returncode})\n"
                f"{interrupted.stdout}\n{interrupted.stderr}\n{detail}")

  state = torch.load(ckpt, map_location="cpu", weights_only=False)
  # The rank count is recorded so a later resume can detect a mismatch.
  assert state.get("world_size") == 2
  assert state.get("in_progress", {}).get("epoch") == 0

  # Phase 2: resume under a fresh (2-rank) group; it must not hang.
  resumed = _run_torchrun(tmp_path)
  if resumed.returncode != 0:
    logs = sorted(tmp_path.glob("error*.log"))
    detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
    pytest.fail(f"resume torchrun failed (rc={resumed.returncode})\n"
                f"{resumed.stdout}\n{resumed.stderr}\n{detail}")
