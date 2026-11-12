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
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
