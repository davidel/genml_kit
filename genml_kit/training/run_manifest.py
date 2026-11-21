"""Run manifest: a reproducibility snapshot written at the start of a run.

A plain ``genml-kit-train`` run used to persist no record of *how* it was
launched: the resolved arguments existed only in the process, and the
``metrics.jsonl`` sink carried a handful of run facts.  This module dumps the
complete picture to ``<log_dir>/run.json`` once, before any training work:

* the fully resolved argparse namespace (``vars(args)``),
* the git SHA of the working tree (``None`` when not a git checkout),
* the versions of Python / torch / torchvision / transformers,
* the CUDA device names, seeds, hostname and UTC start timestamp.

The same mapping is also emitted through
:meth:`~genml_kit.training.metrics_sinks.MetricsLogger.log_run_fact` so
JSONL consumers see it alongside the metric series without reading a second
file.
"""

import datetime
import json
import logging
import os
import platform
import subprocess
import sys

import torch
import torchvision
import transformers

MANIFEST_NAME = "run.json"


def git_sha(repo_root=None):
  """Return the HEAD commit SHA, or ``None`` outside a git checkout.

  Failures (no git binary, not a repository) are swallowed: the manifest is a
  best-effort record, never a hard dependency of training.
  """
  try:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        cwd=repo_root,
        check=True,
    )
    return out.stdout.strip() or None
  except (OSError, subprocess.SubprocessError):
    return None


def collect_run_manifest(args):
  """Build the manifest mapping from the resolved *args* namespace."""
  manifest = {
      "argv_resolved": {
          k: _jsonable(v) for k, v in vars(args).items()
      },
      "git_sha": git_sha(),
      "versions": {
          "python": sys.version.split()[0],
          "torch": torch.__version__,
          "torchvision": torchvision.__version__,
          "transformers": transformers.__version__,
      },
      "cuda_devices": _cuda_device_names(),
      "seeds": {
          "seed": args.seed,
          "deterministic": args.seed is not None,
      },
      "hostname": platform.node(),
      "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  }
  return manifest


def write_manifest(args, log_dir):
  """Write ``<log_dir>/run.json`` and return the manifest mapping.

  Called early in the driver -- before dataset/model construction -- so the
  record exists even when the expensive steps crash.  The logger does not
  exist yet at that point; call :func:`log_manifest_facts` once it does.
  """
  manifest = collect_run_manifest(args)
  try:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(manifest, f, indent=2, sort_keys=True)
      f.write("\n")
    logging.info("Run manifest written to %s", path)
  except OSError as e:
    # A read-only log dir must not kill a training run that was fine
    # before this feature existed.
    logging.warning("Could not write run manifest to %s: %s", log_dir, e)
  return manifest


def log_manifest_facts(logger, manifest):
  """Emit an already-collected *manifest* through every sink of *logger*."""
  for key, value in manifest.items():
    if key == "argv_resolved":
      for arg_name, arg_value in value.items():
        logger.log_run_fact(f"args/{arg_name}", arg_value)
    else:
      logger.log_run_fact(f"manifest/{key}", value)


def _cuda_device_names():
  """Device names for logging; empty when CUDA is unavailable."""
  if not torch.cuda.is_available():
    return []
  return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def _jsonable(value):
  """Best-effort conversion of an argparse value to a JSON-safe object."""
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, dict):
    return {str(k): _jsonable(v) for k, v in value.items()}
  return repr(value)
