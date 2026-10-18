"""Segmentation pipeline: paired image/mask data for semantic segmentation.

Data side only.  Owns the loaders, the ``DataBlob`` contract for
segmentation (``data=image tensor``, ``meta={"mask": long tensor}``) and
device transfer; the objective belongs to
:class:`~genml_kit.methods.segmentation.SegmentationMethod`.

Sources are resolved through :func:`parse_dataset_specs`, which understands:

* ``segfolder/<path>`` -- a local ``images/`` + ``masks/`` tree;
* a HuggingFace dataset name plus ``--mask_column`` for sets that ship a
  mask feature.  When ``--mask_column`` is omitted, the mask column is
  auto-detected (``mask``, ``label``, ``annotation``, ...).  Mask ids are
  compacted into a dense ``[0, C)`` set via a full scan, cached on disk
  under ``--cache_dir`` (or ``~/.cache/genml_kit`` when unset) so
  re-runs skip the scan.

Transforms follow the standard two-phase order: the method builds a
:class:`~genml_kit.datasets.transforms.PairedSegTransform` for train and
val (geometric ops to image + mask, photometric/normalize to the image
only) in ``prepare_transforms`` and stores it as ``train_transforms`` /
``val_transforms``; this pipeline applies it.  When no method transform is
present, a geometric-only default keeps image and mask aligned.
"""

import hashlib
import json
import logging
import os
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from genml_kit.datasets.factory import parse_dataset_specs
from genml_kit.datasets.hf_proxy import HFDatasetProxy
from genml_kit.datasets.seg_folder import load_segfolder
from genml_kit.datasets.transforms import (
    PairedSegTransform,
    build_seg_geometric,
)
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker


class _SegTransformAdapter(Dataset):
  """Apply a callable transform to a dataset of ``{"image", "mask"}`` dicts."""

  def __init__(self, dataset, transform):
    self._dataset = dataset
    self._transform = transform

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    row = self._dataset[idx]
    if isinstance(row, (tuple, list)):
      image, mask = row
      row = {"image": image, "mask": mask}
    if self._transform is None:
      return dict(row)
    return self._transform(dict(row))


@PIPELINES.register
class SegmentationPipeline(DataPipeline):
  """Paired image/mask pipeline for semantic segmentation."""

  NAME = "segmentation"

  def __init__(self):
    self.train_loader = None
    self.val_loader = None
    self.mask_map = None

  @classmethod
  def add_args(cls, parser):
    # Only the segmentation-specific flag is declared here: ``--dataset``,
    # ``--image_column``, ``--val_split`` and ``--cache_dir`` are owned by
    # the always-registered ``images`` pipeline and must not be re-declared
    # (argparse option strings are global to the single parser).
    group = parser.add_argument_group("segmentation data")
    group.add_argument("--mask_column",
                       type=str,
                       default=None,
                       help="HuggingFace dataset column holding the mask. "
                       "Optional: when omitted the mask column is "
                       "auto-detected (mask/label/annotation/...). "
                       "'segfolder/<path>' sources do not need it.")

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    """Build (or return cached) train/val loader."""
    if self.train_loader is None:
      self._build(args, method=kwargs.get("method"))
    return self.val_loader if mode == "val" else self.train_loader

  def build_val_loader(self, args, **kwargs):
    return self.build_loader(args, mode="val", **kwargs)

  def _build(self, args, method=None):
    if not getattr(args, "dataset", None):
      fatal(
          "SegmentationPipeline requires --dataset "
          "(e.g. segfolder:/data/voc2012 or a HF name + --mask_column)", ValueError)

    train_ds, val_ds = self._load_datasets(args)

    train_tf = getattr(method, "train_transforms", None) or self._default_transform(
        args, is_train=True)
    val_tf = getattr(method, "val_transforms", None) or self._default_transform(
        args, is_train=False)

    train_ds = _SegTransformAdapter(train_ds, train_tf)
    val_ds = _SegTransformAdapter(val_ds, val_tf)

    self.train_loader = self._make_loader(train_ds, args, shuffle=True)
    self.val_loader = self._make_loader(val_ds, args, shuffle=False)
    logging.info("SegmentationPipeline: %d train / %d val samples", len(train_ds),
                 len(val_ds))

  @staticmethod
  def _default_transform(args, *, is_train):
    """Fallback when a method supplies no transform.

    Geometric ops keep image + mask aligned; a tensor-producing normalize
    converts the PIL image to a ``(C, H, W)`` float tensor so the collate
    can stack batches (used when no method transform is present).
    """
    import torch
    import torchvision.transforms.v2 as v2

    normalize = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return PairedSegTransform(geometric=build_seg_geometric(getattr(
        args, "image_size", 224),
                                                            is_train=is_train),
                              normalize=normalize)

  def _load_datasets(self, args):
    """Resolve train/val datasets from segfolder or HF sources."""
    specs = parse_dataset_specs(
        [args.dataset],
        image_column=getattr(args, "image_column", "image"),
        split="train",
    )
    spec = specs[0]
    if spec["source"] == "segfolder":
      return self._load_segfolder(spec["name"], args)
    return self._load_hf(args, spec["name"])

  def _load_segfolder(self, root, args):
    splits = load_segfolder(root)
    if "val" in splits:
      return splits["train"], splits["val"]
    # No explicit val split: derive one from the train set.
    train_full = splits["train"]
    n_val = max(1, int(len(train_full) * getattr(args, "val_split", 0.2)))
    n_train = len(train_full) - n_val
    if n_train <= 0:
      fatal("SegmentationPipeline: val_split leaves no training samples", ValueError)
    logging.info("SegmentationPipeline: deriving %d val / %d train samples", n_val,
                 n_train)
    generator = torch.Generator().manual_seed(getattr(args, "seed", 0) or 0)
    return random_split(train_full, [n_train, n_val], generator=generator)

  def _load_hf(self, args, name):
    import datasets as _datasets

    raw = _datasets.load_dataset(name, cache_dir=getattr(args, "cache_dir", None))
    train_raw, val_raw = self._split_hf(raw, args)

    # Image column: an explicit --image_column wins; otherwise auto-detect
    # (pixel_values / image / ...) instead of forcing a literal "image".
    image_col = getattr(args, "image_column", None)
    if image_col is None or image_col not in train_raw.column_names:
      image_col = (HFDatasetProxy.detect_image_column(train_raw) or "image")

    # Mask column: an explicit --mask_column wins; otherwise auto-detect
    # (mask / label / annotation / ...).  No longer required.
    mask_col = getattr(args, "mask_column", None)
    if mask_col is None or mask_col not in train_raw.column_names:
      mask_col = HFDatasetProxy.detect_mask_column(train_raw)
    if mask_col is None:
      fatal(
          "SegmentationPipeline: no mask column found in dataset "
          f"{name!r} (columns {train_raw.column_names}). Pass --mask_column.",
          ValueError)

    ignore_index = getattr(args, "seg_ignore_index", 255)
    mask_map = self._load_or_build_mask_map(args, name, image_col, mask_col,
                                            ignore_index, raw)

    return (HFDatasetProxy(train_raw,
                           image_column=image_col,
                           mask_column=mask_col,
                           mask_map=mask_map,
                           mask_ignore_index=ignore_index),
            HFDatasetProxy(val_raw,
                           image_column=image_col,
                           mask_column=mask_col,
                           mask_map=mask_map,
                           mask_ignore_index=ignore_index))

  @staticmethod
  def _class_map_cache_dir(args):
    """Resolve the class-map cache directory.

    Priority: ``--cache_dir`` when given, else ``$HOME/.cache/genml_kit``
    when writeable, else a temp directory.
    """
    if getattr(args, "cache_dir", None):
      d = os.path.join(args.cache_dir, "seg_class_maps")
    else:
      home = os.path.expanduser("~")
      cand = os.path.join(home, ".cache", "genml_kit", "seg_class_maps")
      try:
        os.makedirs(cand, exist_ok=True)
        probe = os.path.join(cand, ".write_probe")
        with open(probe, "w") as f:
          f.write("ok")
        os.remove(probe)
        return cand
      except OSError:
        d = os.path.join(tempfile.gettempdir(), "genml_kit", "seg_class_maps")
    os.makedirs(d, exist_ok=True)
    return d

  @staticmethod
  def _class_map_cache_path(args, name, image_col, mask_col):
    """Return the cache file path for a dataset spec."""
    ignore_index = getattr(args, "seg_ignore_index", 255)
    spec = "|".join([name, image_col, mask_col, str(ignore_index)])
    digest = hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]
    return os.path.join(SegmentationPipeline._class_map_cache_dir(args),
                        f"segclassmap__{digest}.json")

  def _load_or_build_mask_map(self, args, name, image_col, mask_col, ignore_index, raw):
    """Return (and cache) the ``{raw: compact}`` mask map.

    A full scan over **all** masks (train + val) builds the class-id
    universe so sparse ids and RGB triplets compact into ``[0, C)``, and
    val-only ids (a rare class split entirely into validation) are still
    covered.  The map is cached to disk (spec-keyed) so re-runs skip the
    scan.
    """
    import numpy as np

    cache_path = self._class_map_cache_path(args, name, image_col, mask_col)
    splits = raw.values() if hasattr(raw, "values") else [raw]
    n = sum(len(s) for s in splits)
    if os.path.exists(cache_path):
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
        # Reject caches built from a different mask count (e.g. an older
        # train-only scan when val ids were added to the spec).  A stale
        # cache is not an error: it is simply rebuilt below.
        if payload.get("total_masks") != n:
          logging.info(
              "SegmentationPipeline: cached class map is stale "
              "(%s masks, expected %s); rescanning", payload.get("total_masks"), n)
          mapping = None
        else:
          logging.info(
              "SegmentationPipeline: using cached class map (%d classes) "
              "from %s", len(mapping), cache_path)
          self.mask_map = mapping
          return mapping
      except (OSError, ValueError, KeyError, AttributeError) as exc:
        logging.warning(
            "SegmentationPipeline: cached class map unreadable (%s); "
            "rescanning", exc)
    logging.info(
        "SegmentationPipeline: scanning %d masks to build the "
        "class-id map (one-time, may take a moment)...", n)

    raw_keys = set()
    rgb = False
    seen = 0
    import time
    t0 = time.time()
    for split in splits:
      for i in range(len(split)):
        row = split[i]
        mask = row[mask_col]
        arr = np.asarray(mask)
        if arr.ndim == 3:
          rgb = True
          h, w, _ = arr.shape
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
        seen += 1

    raw_keys = sorted(raw_keys)
    if rgb:
      mask_map = {key: idx for idx, key in enumerate(raw_keys)}
      logging.info(
          "SegmentationPipeline: RGB mask detected; mapped %d unique "
          "triplets to classes in %.1fs", len(mask_map),
          time.time() - t0)
    else:
      mask_map = {int(key): idx for idx, key in enumerate(raw_keys)}
      logging.info("SegmentationPipeline: %d classes from %d masks in %.1fs",
                   len(mask_map), seen,
                   time.time() - t0)

    self.mask_map = mask_map
    try:
      payload = {
          "dataset": name,
          "image_column": image_col,
          "mask_column": mask_col,
          "ignore_index": ignore_index,
          "key_type": "rgb" if rgb else "int",
          "total_masks": n,
          "raw_to_compact": {
              str(k): int(v) for k, v in mask_map.items()
          },
      }
      with open(cache_path, "w") as f:
        json.dump(payload, f, sort_keys=True)
      logging.info("SegmentationPipeline: cached class map to %s", cache_path)
    except OSError as exc:
      logging.warning("SegmentationPipeline: could not cache class map (%s)", exc)
    return mask_map

  @staticmethod
  def _split_hf(raw, args):
    """Return ``(train_split, val_split)`` from an HF Dataset / DatasetDict."""
    import datasets as _datasets

    val_frac = getattr(args, "val_split", 0.2)
    seed = getattr(args, "seed", 0) or 0
    if isinstance(raw, _datasets.Dataset):
      split = raw.train_test_split(test_size=val_frac, seed=seed)
      return split["train"], split["test"]

    keys = list(raw.keys())
    if "validation" in keys:
      return raw["train"], raw["validation"]
    if "test" in keys:
      return raw["train"], raw["test"]
    train_key = "train" if "train" in keys else keys[0]
    split = raw[train_key].train_test_split(test_size=val_frac, seed=seed)
    return split["train"], split["test"]

  def _make_loader(self, dataset, args, *, shuffle):
    generator = None
    if shuffle and getattr(args, "seed", None) is not None:
      generator = torch.Generator()
      generator.manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=False,
        drop_last=shuffle,
        collate_fn=_seg_collate,
    )

  def to_device(self, blob, device):
    data = blob.data.to(device, non_blocking=True)
    mask = blob.meta.get("mask")
    meta = {"mask": mask.to(device, non_blocking=True) if mask is not None else None}
    return DataBlob(data=data, meta=meta)


def _seg_collate(batch):
  """Collate ``{"image": tensor, "mask": long tensor}`` dicts into a DataBlob.

  ``blob.data`` is the stacked ``(B, C, H, W)`` image; the mask becomes a
  ``(B, H, W)`` long tensor in ``meta["mask"]``.
  """
  images = torch.stack([b["image"] for b in batch])
  masks = torch.stack([b["mask"].to(torch.long) for b in batch])
  return DataBlob(data=images, meta={"mask": masks})
