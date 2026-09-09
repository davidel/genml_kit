"""Reproducibility helpers: global seeding and deterministic execution.

Seeding covers Python's ``random``, NumPy, and PyTorch (CPU and all CUDA
devices).  There are two levels of determinism:

* **RNG seeding** (always on): an internal seed drives data shuffling,
  augmentation randomness, mixup, and dropout.  It comes from the user's
  ``--seed`` when given, otherwise from the ``GENML_KIT_SEED`` environment
  variable (default 42) — the user not caring about determinism does not
  mean the toolkit runs on unseeded entropy.
* **Deterministic kernels** (``--seed`` given): additionally enables cuDNN
  deterministic mode, disables ``cudnn.benchmark``, and switches
  ``torch.use_deterministic_algorithms`` on, trading throughput for
  bit-exact repeatability.  Ops without a deterministic kernel log a
  warning instead of failing (``warn_only``), so long-running jobs are
  not aborted by a single unsupported op.

Example::

    from genml_kit.utils.seed import seed_everything, resolve_seed, seed_worker

    seed = resolve_seed(args.seed)
    info = seed_everything(seed)
    loader = DataLoader(
        dataset,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
        ...
    )
"""

import logging
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def resolve_seed(seed=None):
  """Return the effective internal RNG seed.

  An explicit *seed* wins; otherwise the ``GENML_KIT_SEED`` environment
  variable is consulted, falling back to :data:`DEFAULT_SEED`.  This
  keeps every run seeded by default (the user omitting ``--seed``
  expresses "I do not care", not "run on entropy") while letting
  deployments pin or vary the default without code changes.
  """
  if seed is not None:
    return int(seed)
  return int(os.getenv("GENML_KIT_SEED", DEFAULT_SEED))


def seed_everything(seed, deterministic=None):
  """Seed all RNG sources and optionally enable deterministic algorithms.

  Args:
    seed: Integer seed applied to ``random``, ``numpy``, and ``torch``
      (CPU and all CUDA devices).  Use :func:`resolve_seed` to derive it
      from ``--seed`` / ``GENML_KIT_SEED``.
    deterministic: When None (default) it is derived from *seed*: an
      explicitly chosen seed asks for deterministic kernels, while an
      internally resolved one keeps ``cudnn.benchmark`` on for
      throughput.  Pass True/False to override the derivation.

  Returns:
    Dict describing the applied settings, for logging.
  """
  if deterministic is None:
    deterministic = seed is not None

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

  if not deterministic:
    torch.backends.cudnn.benchmark = True
    settings["cudnn_benchmark"] = True
  else:
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
