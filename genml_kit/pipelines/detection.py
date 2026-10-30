"""Detection pipeline: COCO-JSON image + boxes/labels data.

Data side only.  Owns the loaders, the ``DataBlob`` contract for detection
(``data=(B, 3, H, W) images``, ``meta={"boxes": list[(N_i, 4)], "labels":
list[(N_i,)]}``) and device transfer; the objective belongs to
:class:`~genml_kit.methods.detection.DetectionMethod`.

Sources are resolved through :func:`parse_dataset_specs`, which understands:

* ``detfolder/<path>`` -- a local ``images/`` + ``annotations.json`` tree;
* a HuggingFace name -- deferred (COCO-JSON is the supported path for now).

The blob keeps boxes as a *list* (one ``(N_i, 4)`` tensor per image) rather
than a stacked tensor, because the number of boxes varies per image; stacking
would require padding and a mask, which every consumer would then have to
undo.  The collate therefore builds the list explicitly.
"""

import logging

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from genml_kit.datasets.det_folder import load_detfolder
from genml_kit.datasets.det_transforms import build_det_transform
from genml_kit.datasets.factory import parse_dataset_specs
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker


class _DetTransformAdapter(Dataset):
  """Apply a detection transform to each item of a source dataset.

  Mirrors the segmentation adapter: the transform is a plain callable over the
  item dict, so the pipeline stays agnostic to how it was built (default,
  method-supplied, or a user script).
  """

  def __init__(self, dataset, transform):
    self._dataset = dataset
    self._transform = transform

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    row = self._dataset[idx]
    if self._transform is not None:
      row = self._transform(row)
    return row


@PIPELINES.register
class DetectionPipeline(DataPipeline):
  """COCO-JSON image + boxes pipeline for object detection."""

  NAME = "detection"
  THROUGHPUT_UNIT = "img"

  def __init__(self):
    self.train_loader = None
    self.val_loader = None
    self.num_classes = None

  @classmethod
  def add_args(cls, parser):
    # Only detection-specific flags are declared here: ``--dataset``,
    # ``--image_column``, ``--val_split`` and ``--cache_dir`` are globally
    # shared (genml_kit.utils.shared_args) and registered once by the harness,
    # so they are read directly as ``args.<flag>`` rather than re-declared.
    group = parser.add_argument_group("detection data")
    group.add_argument("--det_annotation_file",
                       type=str,
                       default="annotations.json",
                       help="COCO-JSON annotation file name inside each split.")
    group.add_argument("--det_image_subdir",
                       type=str,
                       default="images",
                       help="Sub-directory holding the image files.")

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    """Build (or return cached) train/val loader."""
    if self.train_loader is None:
      self._build(args, method=kwargs.get("method"))
    return self.val_loader if mode == "val" else self.train_loader

  def build_val_loader(self, args, **kwargs):
    return self.build_loader(args, mode="val", **kwargs)

  def _build(self, args, method=None):
    if not args.dataset:
      fatal("DetectionPipeline requires --dataset "
            "(e.g. detfolder//data/coco2017)", ValueError)

    train_ds, val_ds = self._load_datasets(args)
    # The class count comes from the dataset's compacted label space; the
    # method reads it during wire_data to size the head.  Train and val share
    # the map (they come from the same corpus), so either is authoritative.
    self.num_classes = train_ds.num_classes if hasattr(train_ds,
                                                       "num_classes") else None

    train_tf = getattr(method, "train_transforms", None) or self._default_transform(
        args, is_train=True)
    val_tf = getattr(method, "val_transforms", None) or self._default_transform(
        args, is_train=False)

    train_ds = _DetTransformAdapter(train_ds, train_tf)
    val_ds = _DetTransformAdapter(val_ds, val_tf)

    self.train_loader = self._make_loader(train_ds, args, shuffle=True)
    self.val_loader = self._make_loader(val_ds, args, shuffle=False)
    logging.info("DetectionPipeline: %d train / %d val samples", len(train_ds),
                 len(val_ds))

  @staticmethod
  def _default_transform(args, *, is_train):
    """Fallback when a method supplies no transform.

    Letterboxing keeps the aspect ratio so boxes stay rectangular; the
    flip (train only) and tensor conversion are box-aware (see
    :mod:`genml_kit.datasets.det_transforms`).
    """
    return build_det_transform(args.image_size or 224, is_train=is_train)

  def _load_datasets(self, args):
    """Resolve train/val datasets from a detfolder source."""
    specs = parse_dataset_specs(
        [args.dataset],
        image_column=args.image_column,
        split="train",
    )
    spec = specs[0]
    if spec["source"] != "detfolder":
      fatal(
          "DetectionPipeline currently supports only 'detfolder/<path>' "
          f"datasets (got '{spec['source']}': {spec['name']}); COCO-JSON is "
          "the supported format for this first cut.", ValueError)
    return self._load_detfolder(spec["name"], args)

  def _load_detfolder(self, root, args):
    splits = load_detfolder(
        root,
        annotation_file=args.det_annotation_file or "annotations.json",
    )
    if "val" in splits:
      return splits["train"], splits["val"]

    # No explicit val split: derive one from the train set.  The datasets
    # implement __getitem__/__len__ but not Subset-friendly bookkeeping, so
    # torch's random_split works directly on the base dataset.
    train_full = splits["train"]
    n_val = max(1, int(len(train_full) * args.val_split))
    n_train = len(train_full) - n_val
    if n_train <= 0:
      fatal("DetectionPipeline: val_split leaves no training samples", ValueError)
    logging.info("DetectionPipeline: deriving %d val / %d train samples", n_val,
                 n_train)
    generator = torch.Generator().manual_seed(args.seed or 0)
    train_subset, val_subset = random_split(train_full, [n_train, n_val],
                                            generator=generator)
    return _SubsetWithMeta(train_subset,
                           train_full), _SubsetWithMeta(val_subset, train_full)

  def _make_loader(self, dataset, args, *, shuffle):
    generator = None
    if shuffle and args.seed is not None:
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
        collate_fn=_det_collate,
    )

  def to_device(self, blob, device):
    """Move the image batch and every per-image box/label tensor to *device*."""
    data = blob.data.to(device, non_blocking=True)
    boxes = [b.to(device, non_blocking=True) for b in blob.meta.get("boxes", [])]
    labels = [lab.to(device, non_blocking=True) for lab in blob.meta.get("labels", [])]
    return blob._replace(data=data, meta={"boxes": boxes, "labels": labels})


class _SubsetWithMeta:
  """Wrap a ``random_split`` Subset so it re-exposes the base dataset's meta.

  ``random_split`` returns a ``Subset`` that only forwards ``__getitem__`` /
  ``__len__``; the method's ``wire_data`` needs ``num_classes`` / ``id2label``,
  which live on the underlying :class:`DetFolderDataset`.  This thin proxy
  restores that access without changing the split semantics.
  """

  def __init__(self, subset, base):
    self._subset = subset
    self.num_classes = getattr(base, "num_classes", None)
    self.id2label = getattr(base, "id2label", {})

  def __len__(self):
    return len(self._subset)

  def __getitem__(self, idx):
    return self._subset[idx]


def _det_collate(batch):
  """Collate detection item dicts into a DataBlob.

  ``blob.data`` is the stacked ``(B, 3, H, W)`` image tensor.  Boxes and labels
  stay as *lists* of per-image tensors (``[(N_i, 4)]`` / ``[(N_i,)]``) because
  ``N_i`` varies per image; a padded tensor would need a mask that every
  consumer (loss, metric, NMS) would have to re-derive.
  """
  images = torch.stack([item["image"] for item in batch])
  boxes = [item["boxes"] for item in batch]
  labels = [item["labels"] for item in batch]
  return DataBlob(data=images, meta={"boxes": boxes, "labels": labels})
