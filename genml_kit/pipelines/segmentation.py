"""Segmentation pipeline: paired image/mask data for semantic segmentation.

Data side only.  Owns the loaders, the ``DataBlob`` contract for
segmentation (``data=image tensor``, ``meta={"mask": long tensor}``) and
device transfer; the objective belongs to
:class:`~genml_kit.methods.segmentation.SegmentationMethod`.

Sources are resolved through :func:`parse_dataset_specs`, which understands:

* ``segfolder/<path>`` -- a local ``images/`` + ``masks/`` tree;
* a HuggingFace dataset name plus ``--mask_column`` for sets that ship a
  mask feature.

Transforms follow the standard two-phase order: the method builds a
:class:`~genml_kit.datasets.transforms.PairedSegTransform` for train and
val (geometric ops to image + mask, photometric/normalize to the image
only) in ``prepare_transforms`` and stores it as ``train_transforms`` /
``val_transforms``; this pipeline applies it.  When no method transform is
present, a geometric-only default keeps image and mask aligned.
"""

import logging

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
                       help="HuggingFace dataset column holding the mask "
                       "(required for HF segmentation sets; "
                       "'segfolder/<path>' sources do not need it).")

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
    """Geometric-only fallback when a method supplies no transform."""
    return PairedSegTransform(geometric=build_seg_geometric(
        getattr(args, "image_size", 224), is_train=is_train))

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
    if not getattr(args, "mask_column", None):
      fatal(
          "SegmentationPipeline: HuggingFace sources require "
          "--mask_column (the mask feature name).", ValueError)

    import datasets as _datasets

    raw = _datasets.load_dataset(name, cache_dir=getattr(args, "cache_dir", None))
    train_raw, val_raw = self._split_hf(raw, args)
    image_col = (getattr(args, "image_column", None) or
                 HFDatasetProxy.detect_image_column(train_raw) or "image")
    mask_col = args.mask_column
    return (HFDatasetProxy(train_raw, image_column=image_col, mask_column=mask_col),
            HFDatasetProxy(val_raw, image_column=image_col, mask_column=mask_col))

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
