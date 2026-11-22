"""Subprocess runner for the README CLI example suite.

Executes one marked README fence: builds argv, synthesizes referenced
fixtures, applies environment overrides, and runs each command with a
timeout and process-tree teardown.
"""

import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from tests import services
from tests.readme_cli_fixtures import _prepare_fixtures
from tests.readme_cli_harness import _EXTRA_IMPORTS, build_argv
from tests.threads import wide_blas_env

REPO_ROOT = Path(__file__).resolve().parents[1]


def _failure_message(spec, argv, proc=None, stderr=None):
  lines = [
      f"README CLI command failed: {spec.md}:{spec.line} ({spec.heading})",
      f"argv: {' '.join(argv)}",
  ]
  if proc is not None:
    lines.append(f"return code: {proc.returncode}")
    out = (proc.stdout or "").splitlines()[-40:]
    err = (proc.stderr or "").splitlines()[-40:]
    lines.append("--- stdout (tail) ---")
    lines.extend(out)
    lines.append("--- stderr (tail) ---")
    lines.extend(err)
  if stderr is not None:
    lines.append(stderr)
  return "\n".join(lines)


def _launcher_available(spec):
  """True when the fence's launcher can actually run here.

  ``torchrun`` (``torch.distributed.run``) ships with PyTorch and needs a
  working ``torch.distributed``; when either is missing the DDP example is
  skipped rather than failed, mirroring ``tests/test_distributed_e2e.py``.
  """
  if spec.launcher is None:
    return True
  if spec.launcher != "torchrun":
    return False
  try:
    import torch.distributed
  except ImportError:
    return False
  return torch.distributed.is_available()


def _process_tree(root_pid):
  """Return *root_pid* and every live descendant, via ``/proc``.

  A ``launcher:`` fence forks a supervisor (``torchrun``) that puts *each*
  worker (rank) in its own session, so a single ``killpg`` on our direct
  child would miss the ranks and their DataLoader workers.  Walking the
  ``ppid`` links gathers the whole tree so it can be reaped as one unit.
  """
  children = {}
  for name in os.listdir("/proc"):
    if not name.isdigit():
      continue
    try:
      with open(f"/proc/{name}/stat", "rb") as fh:
        data = fh.read()
    except OSError:
      continue
    rparen = data.rfind(b")")
    if rparen == -1:
      continue
    fields = data[rparen + 2:].split()
    if len(fields) < 2:
      continue
    children.setdefault(int(fields[1]), []).append(int(name))
  tree, stack = [], [root_pid]
  while stack:
    pid = stack.pop()
    tree.append(pid)
    stack.extend(children.get(pid, ()))
  return tree


def _signal_pids(pids, sig):
  for pid in pids:
    with contextlib.suppress(OSError):
      os.kill(pid, sig)


def _kill_process_tree(proc, grace=5.0):
  """SIGTERM then SIGKILL *proc* and all its descendants.

  Snapshot the descendant set *before* signalling: once the supervisor is
  dead its children are reparented and the ``ppid`` links that tie them to
  the tree are gone.
  """
  tree = _process_tree(proc.pid)
  _signal_pids(tree, signal.SIGTERM)
  deadline = time.monotonic() + grace
  while time.monotonic() < deadline:
    if all(not _pid_alive(pid) for pid in tree):
      break
    time.sleep(0.1)
  _signal_pids(_process_tree(proc.pid) + tree, signal.SIGKILL)
  with contextlib.suppress(OSError, subprocess.SubprocessError):
    proc.wait(timeout=grace)


def _pid_alive(pid):
  try:
    os.kill(pid, 0)
    return True
  except OSError:
    return False


def _run_command(argv, cwd, env, timeout):
  """Run *argv* in its own session, reaping the whole tree on exit.

  Returns the completed :class:`subprocess.CompletedProcess`.  Unlike
  ``subprocess.run``, a timeout (or an unexpected error) tears down every
  process the command spawned -- ``torchrun`` ranks live in their own
  sessions and would otherwise leak, pinning a CPU and holding the captured
  stdout/stderr pipes open, which can wedge the enclosing test runner.
  """
  proc = subprocess.Popen(argv,
                          cwd=cwd,
                          env=env,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True,
                          start_new_session=True)
  try:
    stdout, stderr = proc.communicate(timeout=timeout)
  except subprocess.TimeoutExpired:
    _kill_process_tree(proc)
    raise
  except BaseException:
    _kill_process_tree(proc)
    raise
  return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def run_spec(spec, tmp_path):
  run_root = Path(tmp_path) / "readme_cli"
  run_root.mkdir(parents=True, exist_ok=True)
  try:
    if not _launcher_available(spec):
      pytest.skip(f"launcher {spec.launcher!r} unavailable")
    for extra in spec.extras:
      module = _EXTRA_IMPORTS.get(extra)
      if module is None:
        raise ValueError(f"unknown extras key: {extra!r}")
      pytest.importorskip(module, reason=f"needs [{extra}]")
    # A gated fence is remapped to a local fixture, so it needs no service.
    if spec.service and not spec.gated:
      services.require(spec.service, os.environ)

    _prepare_fixtures(run_root, spec)

    env = os.environ.copy()
    # The session pins BLAS/OpenMP to one thread; widen it for the spawned
    # CLI so the heavy README fences are not slowed several-fold.
    env.update(wide_blas_env())
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT),
                                         env.get("PYTHONPATH", "")]).strip(os.pathsep)
    for key, value in spec.env.items():
      env[key] = str(value).format(tmp=run_root)
    cwd = spec.cwd.format(tmp=run_root)

    for index, command in enumerate(spec.commands):
      argv = build_argv(spec, command, run_root, cmd_index=index)
      try:
        proc = _run_command(argv, cwd, env, spec.timeout)
      except subprocess.TimeoutExpired:
        pytest.fail(
            _failure_message(spec, argv, stderr=f"timed out after {spec.timeout}s"))
      if proc.returncode != 0:
        pytest.fail(_failure_message(spec, argv, proc))
  finally:
    shutil.rmtree(run_root, ignore_errors=True)
