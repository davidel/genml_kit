"""Single-image pipeline: HF datasets / ImageFolder / ensemble.

Merges the classification data path (``--dataset`` with train/val split,
HF proxies, weighted sampling) and the pre-training ensemble path
(``--datasets`` with :class:`DatasetEnsemble`) into one data-only
pipeline per the v4.2 plan (s 6.1).  Owns the loaders, the DataBlob
contract (``data=images``, ``meta={\"labels\": ...}``) and device
transfer; never builds models or losses.
"""

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.factory import parse_dataset_specs
from genml_kit.datasets.field_dataset import FieldSectorDataset
from genml_kit.datasets.transforms import DictFieldTransform
from genml_kit.datasets.weighted_sampler import build_weighted_sampler
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import register_pipeline
from genml_kit.training.model_utils import model_mode
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker


class _ProxyAdapter(Dataset):
  """Adapt HFDatasetProxy (tuple items ``(image, label)``) to dict items."""

  def __init__(self, proxy):
    self._proxy = proxy

  def __len__(self):
    return len(self._proxy)

  def __getitem__(self, idx):
    image, label = self._proxy[idx]
    return {"image": image, "label": label}


class _DictAdapter(Dataset):
  """Adapt dict-returning datasets to the uniform ``{"image", "label"}`` key."""

  def __init__(self, dataset, image_column="image"):
    self._ds = dataset
    self._image_column = image_column

  def __len__(self):
    return len(self._ds)

  def __getitem__(self, idx):
    item = self._ds[idx]
    return {
        "image": item[self._image_column],
        "label": item.get("label", None),
    }


@register_pipeline("images")
class ImagesPipeline(DataPipeline):
  """Image data pipeline: single dataset (train) or ensemble (pretrain)."""

  NAME = "images"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.train_dataset = None
    self.val_dataset = None
    self.train_proxy = None
    self.val_proxy = None
    self.ensemble = None
    self.num_labels = None
    self.id2label = None
    self.label2id = None
    self.class_weights = None
    self.class_multipliers = None
    self.data_generator = None
    self.tta_transform = None

  # --- Arg surface --------------------------------------------------------

  def add_args(self, parser):
    group = parser.add_argument_group("images pipeline")
    group.add_argument("--dataset",
                       type=str,
                       default=None,
                       help="Single dataset (classification): HF name/path, or "
                       "'imagefolder/<dir>'.")
    group.add_argument("--datasets",
                       nargs="+",
                       default=None,
                       help="Dataset names or local paths for the ensemble "
                       "(pre-training).")
    group.add_argument("--label_column",
                       type=str,
                       default=None,
                       help="Dataset column holding the class label.")
    group.add_argument("--image_column",
                       type=str,
                       default="image",
                       help="Dataset column holding the image.")
    group.add_argument("--split",
                       type=str,
                       default="train",
                       help="HF dataset split to load.")
    group.add_argument("--cache_dir",
                       type=str,
                       default=None,
                       help="HuggingFace cache directory.")
    group.add_argument("--sampler",
                       type=str,
                       default="none",
                       choices=["none", "weighted"],
                       help="Training sampler (classification).")
    group.add_argument("--class_multipliers",
                       type=str,
                       default="",
                       help="Comma-separated NAME=VALUE per-class priority "
                       "multipliers (classification).")
    group.add_argument("--val_split",
                       type=float,
                       default=0.2,
                       help="Fraction held out as validation "
                       "(classification, --dataset).")

  # --- Loader construction ------------------------------------------------

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    """Build (and cache) the loader for *mode* (train/val).

    Dispatches on whether ``--dataset`` (classification) or ``--datasets``
    (ensemble) was provided; the two builder paths are moved verbatim from
    ``training/train.py::build_data`` and ``pretrain/cli.py`` (v4.2 s 6.1).
    ``method`` (optional) supplies the objective-level augmentation via
    ``method.build_transform(args, image_size)``.

    Args:
        needs_labels: If True, the pipeline must ensure labels are present in
            the data. If False, labels are stripped. If None (default), the
            pipeline decides based on its own logic (backward compat).
    """
    method = kwargs.get("method")
    # Use method's NEEDS_LABELS if not explicitly provided
    if needs_labels is None and method is not None:
      needs_labels = getattr(method, "NEEDS_LABELS", False)
    if getattr(args, "dataset", None):
      if self.train_loader is None or self.val_loader is None:
        self._build_classification(args, method=method, needs_labels=needs_labels)
      return self.val_loader if mode == "val" else self.train_loader
    if self.train_loader is None:
      self._build_ensemble(args, method=method, needs_labels=needs_labels)
    return self.train_loader

  def build_val_loader(self, args, **kwargs):
    return self.build_loader(args, mode="val", **kwargs)

  # --- Classification path (train.py::build_data + load_and_split_dataset) --

  def _build_classification(self, args, method=None, needs_labels=True):
    # Imported lazily to avoid a circular import (train -> pipelines.images).
    from genml_kit.training.train import load_and_split_dataset

    train_proxy, val_proxy = load_and_split_dataset(
        args.dataset,
        cache_dir=args.cache_dir,
        test_size=args.val_split,
        train_transform=getattr(args, "train_transforms", None),
        val_transform=getattr(args, "val_transforms", None),
        image_column=args.image_column,
        label_column=args.label_column,
        seed=args.seed,
    )
    self.train_proxy = train_proxy
    self.val_proxy = val_proxy
    self.num_labels = train_proxy.num_labels
    self.id2label = {int(k): v for k, v in train_proxy.id2label.items()}
    self.label2id = {k: int(v) for k, v in train_proxy.label2id.items()}

    logging.info("num_labels: %s", self.num_labels)
    if train_proxy.label_column:
      self.class_weights = compute_class_weights(train_proxy.dataset, self.num_labels,
                                                 train_proxy.label_column)
      logging.info("Inverse-frequency weights: %s", fmt_weights(self.class_weights))
      label2id_int = {k: int(v) for k, v in train_proxy.label2id.items()}
      self.class_multipliers = parse_class_multipliers(args.class_multipliers,
                                                       self.num_labels, label2id_int)

    self.train_dataset = _ProxyAdapter(train_proxy)
    self.val_dataset = _ProxyAdapter(val_proxy) if val_proxy is not None else None
    self.tta_transform = getattr(args, "tta_transform", None)

    data_generator = torch.Generator()
    if args.seed is not None:
      data_generator.manual_seed(args.seed)
    self.data_generator = data_generator

    sampler = None
    if args.sampler == "weighted" and train_proxy.label_column:
      sampler = build_weighted_sampler(train_proxy.dataset,
                                       self.num_labels,
                                       train_proxy.label_column,
                                       args.sampler_weights,
                                       multipliers=self.class_multipliers)

    self.train_loader = DataLoader(
        self.train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=False,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=_images_collate,
    )
    self.val_loader = DataLoader(
        self.val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=False,
        shuffle=False,
        collate_fn=_images_collate,
    )

  # --- Ensemble path (pretrain/cli.py::build_pretrain_*) -------------------

  def _build_ensemble(self, args, method=None, needs_labels=False):
    """Build the ensemble loader (pre-training path)."""
    configs = parse_dataset_specs(
        args.datasets,
        image_column=getattr(args, "image_column", "image"),
        label_column=getattr(args, "label_column", None),
    )
    ensemble = DatasetEnsemble(
        configs,
        cache_dir=args.cache_dir,
        hf_token=getattr(args, "hf_token", None),
        strict=getattr(args, "strict_datasets", False),
    )
    self.ensemble = ensemble
    self.num_labels = ensemble.num_labels if ensemble.has_labels else None
    if needs_labels:
      ensemble.ensure_label_space()
      if ensemble.unlabeled_datasets:
        fatal(
            f"Datasets {[d.name for d in ensemble.unlabeled_datasets]} "
            f"provide no labels but a label-requiring method "
            f"({getattr(args, 'method', '?')}) was selected.",
            ValueError,
        )

    if method is not None and hasattr(method, "build_transform"):
      transform = method.build_transform(args, getattr(args, "image_size", 224))
    else:
      transform = getattr(args, "train_transforms", None)
    if transform is None:
      # No method/CLI transform: fall back to the historic default
      # pre-training augmentation (v4.2 s 6.1).
      transform = build_pretrain_transform(getattr(args, "image_size", 224))
    dataset = DictFieldTransform(ensemble, transform, fields=(ensemble.image_column,))
    if not needs_labels:
      # Methods that ignore labels get image-only items: a mixed ensemble
      # (some sources labeled, some not) would otherwise produce batches
      # with inconsistent keys that default_collate cannot handle.
      dataset = FieldSectorDataset(
          dataset, fields={ensemble.image_column: ensemble.image_column})

    self.train_dataset = _DictAdapter(dataset, image_column=ensemble.image_column)
    self.data_generator = torch.Generator()
    if args.seed is not None:
      self.data_generator.manual_seed(args.seed)
    self.train_loader = DataLoader(
        self.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=self.data_generator,
        pin_memory=False,
        drop_last=True,
        collate_fn=_images_collate,
    )
    logging.info(ensemble.summary())

  # --- Helpers ------------------------------------------------------------

  def _make_loader(self, dataset, batch_size, **kwargs):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      **kwargs,
                      collate_fn=_images_collate)

  def to_device(self, blob, device):
    if isinstance(blob.data, (tuple, list)):
      data = tuple(x.to(device, non_blocking=True) for x in blob.data)
    else:
      data = blob.data.to(device, non_blocking=True)
    meta = {}
    for k, v in blob.meta.items():
      if hasattr(v, "to"):
        meta[k] = v.to(device, non_blocking=True)
      elif isinstance(v, (tuple, list)):
        meta[k] = [
            x.to(device, non_blocking=True) if hasattr(x, "to") else x for x in v
        ]
      else:
        meta[k] = v
    return DataBlob(data=data, meta=meta)

  # --- Collation ------------------------------------------------------------

  @staticmethod
  def _collate(batch):
    """Collate a list of uniform dict items into a DataBlob.

    Each item is ``{"image": tensor|tuple, "label": int|None}``.  A plain
    tensor stacks to ``(B, C, H, W)``; a tuple (e.g. BYOL dual-view,
    DINO multi-crop) is collated per-position so ``blob.data`` keeps the
    view structure.  Labels (if present) become ``(B,)`` in ``meta``.
    """
    image0 = batch[0]["image"] if batch else None
    if isinstance(image0, (tuple, list)):
      images = tuple(
          torch.stack([b["image"][i] for b in batch]) for i in range(len(image0)))
    else:
      images = torch.stack([b["image"] for b in batch])
    labels = None
    if batch and batch[0].get("label") is not None:
      labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return DataBlob(data=images, meta={"labels": labels})


_images_collate = ImagesPipeline._collate

# --- Moved verbatim from pretrain/cli.py (v4.2 s 12.10) --------------------


def build_pretrain_transform(image_size=448):
  """Default augmentations for pre-training.

  Resize to *image_size* (shortest side), center crop, then horizontal
  and vertical flips.  Used by any method without a ``build_transform``
  override (e.g. I-JEPA).
  """
  return v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])


def build_pretrain_dataset(args, needs_labels=False, transform=None):
  """Build the DatasetEnsemble + transform pipeline.

  If *transform* is ``None``, the default pretrain transform is used.
  """
  configs = parse_dataset_specs(
      args.datasets,
      image_column=getattr(args, "image_column", "image"),
      label_column=getattr(args, "label_column", None),
  )

  ensemble = DatasetEnsemble(
      configs,
      cache_dir=args.cache_dir,
      hf_token=args.hf_token,
      strict=args.strict_datasets,
  )
  if needs_labels:
    ensemble.ensure_label_space()
    if ensemble.unlabeled_datasets:
      fatal(
          "These datasets provide no labels, but this pre-training method "
          f"requires them: {', '.join(ensemble.unlabeled_datasets)}.  Pass "
          f"only labeled datasets (or drop the unlabeled ones), or switch "
          f"to a label-free method (e.g. --method simmim).", ValueError)
  if transform is None:
    transform = build_pretrain_transform(args.image_size)
  dataset = DictFieldTransform(ensemble, transform, fields=(ensemble.image_column,))
  if not needs_labels:
    # Methods that ignore labels get image-only items: a mixed ensemble
    # (some sources labeled, some not) would otherwise produce batches
    # with inconsistent keys that default_collate cannot handle.
    dataset = FieldSectorDataset(dataset,
                                 fields={ensemble.image_column: ensemble.image_column})
  logging.info(ensemble.summary())
  return dataset, ensemble


def log_validation_images(method,
                          model,
                          loader,
                          writer,
                          global_step,
                          device,
                          image_column,
                          num_samples=8):
  """Log method-specific validation images to TensorBoard.

  Pulls a single batch from *loader*, extracts the images stored under
  ``batch[image_column]`` (the loader yields dicts), slices the first
  *num_samples* and hands them to ``method.validate()``, which returns
  optional reconstruction images.  If the method returns ``None``
  (methods without pixel-space visualisations, e.g. I-JEPA), nothing
  is logged.

  The model is restored to train mode even when nothing is logged.
  """
  with model_mode(model, "eval"):
    batch = next(iter(loader))
    raw = batch[image_column]
    if isinstance(raw, (tuple, list)):
      images = type(raw)(v[:num_samples].to(device) for v in raw)
    else:
      images = raw[:num_samples].to(device)
    with torch.no_grad():
      recon = method.validate(model, images, num_samples)
    if recon is None:
      return
    # recon is (N, C, H, W) -- log first sample.
    writer.add_image("recon/original", images[0], global_step)
    writer.add_image("recon/reconstructed", recon[0].clamp(0, 1), global_step)


# --- Moved verbatim from training/train.py (v4.2 s 6.1) --------------------


def parse_class_multipliers(s, num_labels, label2id):
  """Parse a ``--class_multipliers`` string into a ``[num_labels]`` tensor.

  *s* is a comma-separated string of ``NAME=VALUE`` pairs where *NAME* is
  a label string or an integer label index and *VALUE* is a float
  multiplier.  Unspecified classes default to ``1.0``.

  Returns a ``torch.Tensor`` of shape ``[num_labels]``.
  """
  m = torch.ones(num_labels)
  if not s or not s.strip():
    return m
  for pair in s.split(","):
    pair = pair.strip()
    if not pair:
      continue
    if "=" not in pair:
      fatal(
          f"Invalid --class_multipliers entry: '{pair}'. "
          "Expected NAME=VALUE (e.g. cat=4.0).",
          ValueError,
      )
    name, val = pair.split("=", 1)
    name, val = name.strip(), val.strip()
    if name.isdigit():
      idx = int(name)
    else:
      if name not in label2id:
        fatal(
            f"Unknown class name '{name}' in --class_multipliers. "
            f"Available: {list(label2id.keys())}",
            ValueError,
        )
      idx = label2id[name]
    if not (0 <= idx < num_labels):
      fatal(f"Label index {idx} out of range [0, {num_labels})", ValueError)
    m[idx] = float(val)
  return m


def compute_class_weights(train_dataset, num_labels, label_column="label"):
  """Compute inverse-frequency class weights as a CPU tensor."""
  labels = np.asarray([row[label_column] for row in train_dataset], dtype=np.int64)
  actual_labels = set(labels.tolist())
  if len(actual_labels) > num_labels:
    fatal(
        f"Dataset has {len(actual_labels)} unique labels but "
        f"num_labels={num_labels}",
        ValueError,
    )
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  weights = 1.0 / counts
  weights = weights / weights.sum() * num_labels
  return torch.tensor(weights, dtype=torch.float32)


def fmt_weights(weights, decimals=3):
  """Format a 1-D tensor as a human-readable list string."""
  return "[" + ", ".join(f"{v:.{decimals}f}" for v in weights.tolist()) + "]"
