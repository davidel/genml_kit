"""GPU statistics helpers (copied from conv_vit)."""

import contextlib
import logging
import os

import torch


def resolve_device(device_arg):
  """Resolve a CLI ``--device`` argument to a ``torch.device``.

  Under ``torchrun`` (``LOCAL_RANK`` set) an explicit ``--device`` is ignored
  in favour of ``cuda:{LOCAL_RANK}``: every process must own a distinct GPU,
  and the default ``cuda:0`` would otherwise be claimed by all of them.

  Args:
      device_arg: The raw ``--device`` string (``cpu``, ``cuda`` or
          ``cuda:INDEX``), or ``None`` for auto-detection.

  Returns:
      ``torch.device(device_arg)`` if given, otherwise the best available
      device (CUDA if present, else CPU).  Logs the resolved device.
  """
  local = _local_rank_env()
  if local is not None and torch.cuda.is_available():
    device = torch.device("cuda", local)
  elif device_arg:
    device = torch.device(device_arg)
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")
  return device


def _local_rank_env():
  """Return ``LOCAL_RANK`` as an int, or ``None`` when not launched distributed."""
  raw = os.environ.get("LOCAL_RANK")
  try:
    return int(raw) if raw not in (None, "") else None
  except ValueError:
    return None


def gpu_stats_str(device):
  """Return a human-readable GPU stats string, or empty string if not CUDA.

  Fields (all memory numbers are MiB, *util* is the only percentage)::

      mem=allocated/totalMiB (pct) res=reservedMiB util=smPct

  ``mem`` counts live tensors, ``res`` is what the caching allocator
  holds for reuse (>= ``mem``), and ``util`` is the SM utilization
  sampled at log time (a single instant, not an average).  See the
  "Low GPU utilization" note in the README when ``util`` stays near 0.
  """
  if device.type != "cuda":
    return ""
  mem_used = torch.cuda.memory_allocated(device) / 1024**2
  mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
  mem_total = torch.cuda.get_device_properties(device).total_memory / 1024**2
  mem_pct = (mem_used / mem_total * 100.0) if mem_total else 0.0
  msg = f"GPU: mem={mem_used:.0f}/{mem_total:.0f}MiB ({mem_pct:.0f}%)"
  msg += f" res={mem_reserved:.0f}MiB"
  if hasattr(torch.cuda, "utilization"):
    # NVML/pynvml may be missing or unavailable; a stats string must never
    # be able to kill a training run.
    with contextlib.suppress(Exception):
      msg += f" util={torch.cuda.utilization(device):.0f}%"
  return msg
