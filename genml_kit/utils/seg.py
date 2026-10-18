"""Shared segmentation helpers.

Kept free of ``genml_kit.models`` imports so that light modules (the metrics
accumulator, the training method) can use them without pulling in the whole
model package.
"""

import json
import os

import numpy as np
import torch.nn.functional as F

from genml_kit.utils.logging import fatal

# Fixed qualitative palette used to colorize class maps for TensorBoard
# overlays.  20 distinct hues (plus black for the ignore/void class) keep
# adjacent labels separable without a matplotlib dependency.
_PALETTE = np.array(
    [
        [0, 0, 0],  # 0  void / background
        [255, 0, 0],  # 1
        [0, 255, 0],  # 2
        [0, 0, 255],  # 3
        [255, 255, 0],  # 4
        [0, 255, 255],  # 5
        [255, 0, 255],  # 6
        [192, 192, 192],  # 7
        [128, 0, 0],  # 8
        [128, 128, 0],  # 9
        [0, 128, 0],  # 10
        [128, 0, 128],  # 11
        [0, 128, 128],  # 12
        [0, 0, 128],  # 13
        [255, 128, 0],  # 14
        [255, 0, 128],  # 15
        [128, 255, 0],  # 16
        [0, 255, 128],  # 17
        [128, 0, 255],  # 18
        [0, 128, 255],  # 19
    ],
    dtype=np.uint8)


def class_mask_to_rgb(mask, ignore_index=None):
  """Colorize a compact ``(H, W)`` class-id map as an ``(H, W, 3)`` uint8 RGB image.

  Used to render ground-truth / predicted masks in TensorBoard overlays.
  ids are looked up in a fixed 20-entry palette; ids at or beyond it wrap,
  and the ignore marker (e.g. ``255``) is drawn black (void).

  Parameters
  ----------
  mask : numpy.ndarray
    ``(H, W)`` array of dense ``[0, C)`` class ids.
  ignore_index : int or None
    Value marking void pixels (drawn black).  ``None`` disables.

  Returns
  -------
  numpy.ndarray
    ``(H, W, 3)`` ``uint8`` RGB.
  """
  import numpy as np

  arr = np.asarray(mask)
  if ignore_index is not None:
    arr = arr.copy()
    arr[arr == ignore_index] = 0
  return _PALETTE[arr % len(_PALETTE)]


def upsample_to(logits, size):
  """Bilinearly resize *logits* to ``(H, W) = size`` when they differ.

  Segmentation decoders (and HF models) commonly emit low-resolution logits
  such as ``(B, C, H/4, W/4)``; the loss and metric need full-resolution maps
  to match the mask grid.  Returns *logits* unchanged when already aligned.

  Parameters
  ----------
  logits : torch.Tensor
    ``(B, C, H, W)`` raw scores.
  size : tuple[int, int]
    Target ``(H, W)``.

  Returns
  -------
  torch.Tensor
    ``(B, C, H', W')`` logits at *size*.
  """
  if logits.shape[-2:] == tuple(size):
    return logits
  return F.interpolate(logits, size=tuple(size), mode="bilinear", align_corners=False)


def apply_mask_map(arr, mask_map, ignore_index=None):
  """Remap raw mask pixel values / RGB triplets to compact class ids.

  Parameters
  ----------
  arr : numpy.ndarray
    ``(H, W)`` single-channel or ``(H, W, 3)`` RGB mask array.
  mask_map : dict
    ``{raw_key: compact_id}`` where *raw_key* is either an ``int``
    (single-channel) or a 3-tuple (RGB triplet).
  ignore_index : int or None
    Label value (or, for RGB masks, any channel value) marking a void
    pixel.  The class-map build excludes it, so it is passed through
    unchanged; unknown values (e.g. a validation-only id) raise.

  Returns
  -------
  numpy.ndarray
    Same shape as *arr* with every pixel remapped to dense ``[0, C)``
    ids (or the untouched *ignore_index*).
  """
  import numpy as np

  # A negative ignore index disables the pass-through entirely (the
  # original proxy semantics, preserved here).
  ignore_index = ignore_index if (ignore_index is not None and
                                  ignore_index >= 0) else None

  if arr.ndim == 3:
    h, w, _ = arr.shape
    arr = arr.reshape(-1, 3)
    out = np.empty(arr.shape[0], dtype=np.int64)
    for i, px in enumerate(arr):
      key = tuple(int(v) for v in px)
      if key in mask_map:
        out[i] = mask_map[key]
      elif ignore_index is not None and ignore_index in key:
        out[i] = ignore_index
      else:
        fatal(f"Mask value {key} not in the class map (val-only id?)", ValueError)
    return out.reshape(h, w)

  ignore = ignore_index
  out = np.empty(arr.size, dtype=np.int64)
  flat = arr.ravel()
  for i in range(flat.size):
    raw = int(flat[i])
    if raw in mask_map:
      out[i] = mask_map[raw]
    elif ignore is not None and raw == ignore:
      out[i] = ignore
    else:
      fatal(f"Mask value {raw} not in the class map (val-only id?)", ValueError)
  return out.reshape(arr.shape)


def build_mask_map(masks, ignore_index=None):
  """Compact the raw ids appearing in *masks* into dense ``[0, C)``.

  Parameters
  ----------
  masks : iterable of numpy.ndarray
    Mask arrays, each ``(H, W)`` (int class ids) or ``(H, W, 3)``
    (RGB triplets).  The whole corpus is scanned so sparse / 1-based ids
    and late-appearing classes are all captured.
  ignore_index : int or None
    Value excluded from the map (void pixels); passed through unchanged
    by :func:`apply_mask_map`.

  Returns
  -------
  tuple[dict, str]
    ``(mask_map, key_type)`` -- the ``{raw_key: compact_id}`` mapping and
    ``\"rgb\"`` / ``\"int\"`` for cache round-tripping.
  """
  raw_keys = set()
  rgb = False
  for arr in masks:
    arr = np.asarray(arr)
    if arr.ndim == 3:
      rgb = True
      flat = arr.reshape(-1, 3)
      for j in range(flat.shape[0]):
        key = tuple(int(v) for v in flat[j])
        if any(v == ignore_index for v in key):
          continue
        raw_keys.add(key)
    else:
      for v in np.unique(arr):
        if int(v) == ignore_index:
          continue
        raw_keys.add(int(v))
  raw_keys = sorted(raw_keys)
  if rgb:
    return {key: idx for idx, key in enumerate(raw_keys)}, "rgb"
  return {int(key): idx for idx, key in enumerate(raw_keys)}, "int"


def load_mask_map_cache(cache_path, total_masks):
  """Return the cached ``{raw: compact}`` map, or ``None`` when absent/stale.

  A cache is stale (and therefore ignored, not fatal) when ``total_masks``
  differs, e.g. an older train-only scan once validation ids were added.
  """
  import json
  import logging

  if not os.path.exists(cache_path):
    return None
  try:
    with open(cache_path) as f:
      payload = json.load(f)
    key_type = payload.get("key_type", "int")
    mapping = {}
    for k, v in payload["raw_to_compact"].items():
      if key_type == "rgb":
        key = tuple(int(x) for x in k.strip("()").split(","))
      else:
        key = int(k)
      mapping[key] = int(v)
    if payload.get("total_masks") != total_masks:
      logging.info(
          "seg: cached class map is stale (%s masks, expected %s); "
          "rescanning", payload.get("total_masks"), total_masks)
      return None
    return mapping
  except (OSError, ValueError, KeyError, AttributeError):
    logging.warning("seg: cached class map unreadable; rescanning")
    return None


def save_mask_map_cache(cache_path, payload):
  """Best-effort persist *payload*; cache failures only warn."""
  import logging

  try:
    with open(cache_path, "w") as f:
      json.dump(payload, f, sort_keys=True)
    logging.info("seg: cached class map to %s", cache_path)
  except OSError as exc:
    logging.warning("seg: could not cache class map (%s)", exc)
