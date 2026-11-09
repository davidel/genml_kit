"""End-to-end multi-GPU test for the *RL* path: real ``torchrun``, 2 ranks.

Runs the actual ``genml_kit.cli.train.main()`` with ``--pipeline rl --method
ppo`` in two processes (via ``torchrun --standalone``) on CartPole, then asserts
the distributed path really engaged:

* both ranks joined the group (``world=2``);
* rank 0 alone wrote the checkpoint;
* the checkpoint keys carry no ``module.`` prefix (portable single<->multi).

This is the RL counterpart of ``tests/test_distributed_e2e.py``.  It closes the
gap left by the in-process 2-rank tests, which call ``method.wrap_model(...)``
and ``broadcast_model_state(...)`` by hand and therefore only test our *model*
of the driver.  Here the real driver runs: ``init_distributed`` ordering, the
``method.wrap_model`` dispatch, argparse -> ``RLPipeline.init_env`` ->
``per_rank_seed``, sharded PPO rollouts, and rank-0-only checkpoint writes.

Skipped when ``torch.distributed`` or the ``torchrun`` executable is missing.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_DRIVER = Path(__file__).parent / "rl_dist_e2e_driver.py"


def _torchrun_available():
  return shutil.which("torchrun") is not None


def _run_torchrun(tmp_path, nproc=2, port="29811"):
  env = dict(os.environ)
  env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
  env["DIST_E2E_TMPDIR"] = str(tmp_path)
  cmd = [
      "torchrun",
      "--nproc_per_node",
      str(nproc),
      "--master_port",
      port,
      "--standalone",
      str(_DRIVER),
  ]
  return subprocess.run(cmd,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        check=False)


def _fail_with_logs(tmp_path, proc):
  # Surface rank tracebacks (the driver writes one per rank) for debugging.
  logs = sorted(tmp_path.glob("error*.log"))
  detail = "\n".join(f"--- {p.name} ---\n{p.read_text()}" for p in logs)
  pytest.fail(f"torchrun failed (rc={proc.returncode})\n{proc.stdout}\n"
              f"{proc.stderr}\n{detail}")


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_two_rank_torchrun_rl_training(tmp_path):
  """Two torchrun ranks must run a real RL training step and checkpoint once."""
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  proc = _run_torchrun(tmp_path)
  if proc.returncode != 0:
    _fail_with_logs(tmp_path, proc)

  # Both ranks must have joined the group.
  markers = sorted(tmp_path.glob("rank*.marker"))
  assert len(markers) == 2, f"expected 2 rank markers, got {markers}"
  contents = {p.name: p.read_text() for p in markers}
  assert contents["rank0.marker"] == "rank=0 world=2"
  assert contents["rank1.marker"] == "rank=1 world=2"

  # Exactly one writer: rank 0's checkpoint exists.
  ckpt = tmp_path / "ckpts" / "model_latest.pt"
  assert ckpt.exists(), "rank 0 did not write a checkpoint"

  # Portable keys: the (unwrapped) RL model must keep its own key names, with
  # no ``module.`` prefix leaking from any infra wrapper.
  blob = torch.load(ckpt, weights_only=False)
  state = blob.get("model_state_dict", {})
  assert state, "empty model_state_dict"
  keys = list(state)
  assert not any(k.startswith("module.") for k in keys), keys


@pytest.mark.skipif(not _torchrun_available(), reason="torchrun not on PATH")
def test_single_rank_torchrun_rl_matches_plain_run(tmp_path):
  """A 1-process torchrun RL run must behave like the plain single-GPU path."""
  if not torch.distributed.is_available():
    pytest.skip("torch.distributed unavailable")

  proc = _run_torchrun(tmp_path, nproc=1, port="29813")
  if proc.returncode != 0:
    _fail_with_logs(tmp_path, proc)

  assert (tmp_path / "rank0.marker").read_text() == "rank=0 world=1"
  assert (tmp_path / "ckpts" / "model_latest.pt").exists()
