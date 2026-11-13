"""Session-wide test configuration.

Pin the BLAS/OpenMP thread pools to a single thread for the whole test
session.  These environment variables must be set before NumPy (and thus its
BLAS backend) is first imported, which is why this lives in ``conftest.py``
rather than in an individual test module.

Why this is needed
------------------
XGBoost, NumPy and friends default to one worker thread per CPU core.  A
single ``XGBClassifier.fit`` on a 32-core host can therefore push the process
toward ~95 OS threads.  In a full-suite run that is compounded by the forked
``DataLoader`` workers the streaming/DDP tests spawn, and on a loaded CI
runner the per-user process/thread budget can run out mid-run.  The thread
creations then fail with ``pthread_create ... Resource temporarily
unavailable`` (EAGAIN), which surfaces as an intermittent
``tests/test_xgb_pipeline.py`` failure that passes fine in isolation.

Opting back into parallelism
----------------------------
This sets the *default* pool size; it is not a hard ceiling.  A test that
genuinely needs multi-threaded BLAS can raise it back for its own scope::

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=8):
      ...

``threadpoolctl`` ships transitively with scikit-learn; add it to the ``dev``
extras if a test ever needs to rely on it directly.  The variables use
``setdefault`` so an explicit override in the environment still wins.

Opting heavy tests back in
--------------------------
The pin above is a *session default*, not a blanket rule.  The slow/network
tests spend their time in convolution/attention math, so single-threaded
kernels make them several times slower.  Those tests opt back in through the
``_wide_blas`` fixture below, which widens the torch intra-op pool to a
moderate width (see :mod:`tests.threads`) for the duration of the test.

The width is deliberately capped rather than set to the CPU count: widening
to every core *at runtime* oversubscribes the host and is measurably slower
than the pinned default, while a small width recovers most of the throughput.
The cap is overridable via ``GENML_TEST_BLAS_THREADS``.
"""

import os

# Set these before importing anything that may pull in NumPy/PyTorch: the BLAS
# backend reads its thread pool size once, at first import, so a later
# assignment would be ignored.  (Ruff's E402 allows os.environ setup between
# imports for exactly this reason.)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest

from tests.threads import wide_threads


@pytest.fixture(autouse=True)
def _wide_blas(request):
  """Widen BLAS/OpenMP for heavy slow/network tests, then restore.

  The session default pins the pools to one thread (see the module
  docstring); that keeps XGBoost and the forked DataLoader workers inside the
  CI thread budget, but it makes the convergence/e2e tests several times
  slower.  Those opt back in here -- to a *moderate* width, because widening
  to every core at runtime oversubscribes the host and is slower still.

  Tests without a ``slow``/``network``/``service`` marker (notably the XGB
  suite the pin exists for) are left untouched.
  """
  marks = ("slow", "network", "service")
  if not any(request.node.get_closest_marker(mark) for mark in marks):
    yield
    return

  import torch

  previous = torch.get_num_threads()
  torch.set_num_threads(wide_threads())
  try:
    yield
  finally:
    torch.set_num_threads(previous)
