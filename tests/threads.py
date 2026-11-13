"""Thread-width policy shared by the test session and its subprocesses.

``conftest.py`` pins the BLAS/OpenMP pools to a single thread for the whole
session.  That keeps XGBoost and the forked DataLoader workers inside the CI
thread budget, but it makes the heavy convergence/e2e tests several times
slower: their cost is real convolution/attention math, not the remote fetch.

Those heavy tests opt back into a *moderate* width here.  Widening to every
core at runtime is deliberately avoided: the runtime pools oversubscribe the
host and are measurably slower than the pinned default, while a small width
recovers most of the throughput.
"""

import os

# Backend variables honoured by OpenMP, OpenBLAS, MKL and NumExpr.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS")

# Width granted to the opt-in tests.  Capped rather than "all cores": a
# full-width runtime pool oversubscribes the host and is slower than the
# single-threaded default (measured), while a small width is near-optimal.
DEFAULT_WIDE_THREADS = 8


def wide_threads():
  """Thread count for a slow/network test, clamped to the CPU count.

  Overridable via ``GENML_TEST_BLAS_THREADS`` so CI can tune the cap without
  touching the suite.
  """
  requested = int(os.environ.get("GENML_TEST_BLAS_THREADS", DEFAULT_WIDE_THREADS))
  return max(1, min(requested, os.cpu_count() or 1))


def wide_blas_env():
  """Environment overlay that widens the BLAS/OpenMP pools for a child.

  The README CLI harness spawns ``genml-kit-*`` in a subprocess that inherits
  the session-pinned environment, so an in-process fixture cannot reach it;
  this overlay is merged into the child's ``env`` instead.
  """
  width = str(wide_threads())
  return {name: width for name in THREAD_ENV_VARS}
