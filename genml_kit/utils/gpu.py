"""GPU statistics helpers (copied from conv_vit)."""

import contextlib
import logging

import torch


def resolve_device(device_arg):
  """Resolve a CLI ``--device`` argument to a ``torch.device``.

  Args:
      device_arg: The raw ``--device`` string (``cpu``, ``cuda`` or
          ``cuda:INDEX``), or ``None`` for auto-detection.

  Returns:
      ``torch.device(device_arg)`` if given, otherwise the best available
      device (CUDA if present, else CPU).  Logs the resolved device.
  """
  if device_arg:
    device = torch.device(device_arg)
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")
  return device


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
