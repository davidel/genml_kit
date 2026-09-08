"""Reproducibility helpers: global seeding and deterministic execution.

Seeding covers Python's ``random``, NumPy, and PyTorch (CPU and all CUDA
devices).  The ``deterministic`` option additionally trades throughput for
bit-exact reproducibility where CUDA kernels allow it; ops without a
deterministic kernel log a warning instead of failing (``warn_only``), so
long-running jobs are not aborted by a single unsupported op.

Example::

    from genml_kit.utils.seed import seed_everything, seed_worker

    info = seed_everything(args.seed, args.deterministic)
    loader = DataLoader(
        dataset,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
        ...
    )
"""

import logging
import os
import random

import numpy as np
import torch


def seed_everything(seed, deterministic=False):
  """Seed all RNG sources and optionally enable deterministic algorithms.

  Args:
    seed: Integer seed applied to ``random``, ``numpy``, and ``torch``
      (CPU and all CUDA devices).
    deterministic: If True, additionally configure cuDNN and
      ``torch.use_deterministic_algorithms`` for bit-exact reproducible
      execution.  Costs throughput and some ops are unsupported (they
      warn instead of raising).

  Returns:
    Dict describing the applied settings, for logging.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

  settings = {
      "seed": seed,
      "deterministic": bool(deterministic),
      "cudnn_deterministic": False,
      "cudnn_benchmark": torch.backends.cudnn.benchmark,
  }

  if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only so ops without a deterministic CUDA kernel log instead of
    # crashing long-running training jobs.
    torch.use_deterministic_algorithms(True, warn_only=True)
    settings["cudnn_deterministic"] = True
    settings["cudnn_benchmark"] = False
    # Required by some deterministic CUDA kernels (e.g. index_add) to
    # guarantee reproducible results across runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

  logging.info(
      "Seeded RNGs: seed=%d, deterministic=%s, cudnn.deterministic=%s, "
      "cudnn.benchmark=%s", seed, settings["deterministic"],
      settings["cudnn_deterministic"], settings["cudnn_benchmark"])
  return settings


def seed_worker(worker_id):
  """DataLoader ``worker_init_fn``: give each worker a deterministic seed.

  Without this, PyTorch seeds workers from OS entropy, so augmentation
  randomness differs between runs even when the main process is seeded.
  Derived from the base torch seed so workers are reproducible and
  distinct.
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)
