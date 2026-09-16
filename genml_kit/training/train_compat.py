"""Classification helpers moved out of the unified train.py (v4.2).

The v4.2 harness (plans/GENERIC_PIPELINE.md s 5) reduced ``train.py`` to
the pipeline/method wiring.  The classification-specific pieces that used
to live there (loss construction, transforms/TTA, metrics logging, class
weights, the XGBoost post-stage) are re-homed here and re-exported from
``training.train`` for backward compatibility.
"""

import gc
import logging
import re

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from genml_kit.losses.focal import CombinedFocalLoss
from genml_kit.training.eval import evaluate_performance
from genml_kit.training.model_utils import (
    apply_lora,
    enable_grad_checkpointing,
    freeze_model,
)
from genml_kit.utils.logging import fatal
from genml_kit.utils.script import load_extern

__all__ = [
    "CombinedFocalLoss",
    "apply_freeze_patterns",
    "apply_lora",
    "build_transforms",
    "compute_class_weights",
    "enable_grad_checkpointing",
    "evaluate_performance",
    "fmt_weights",
    "load_augmentation_script",
    "maybe_train_xgboost",
    "mixup_data",
    "parse_class_multipliers",
    "resolve_augmentations",
]


def load_augmentation_script(path_or_url):
  """Load a Python script and return its ``create_train_transform`` callable."""
  return load_extern(path_or_url, "create_train_transform")


def build_transforms(processor, image_size, train_aug_fn=None):
  """Create train / val augmentation pipelines."""
  tail = [
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(mean=processor.image_mean, std=processor.image_std),
  ]
  if train_aug_fn is not None:
    user_transforms = train_aug_fn(image_size)
    if not isinstance(user_transforms, list):
      fatal(
          "create_train_transform() must return a list of transforms, "
          f"got {type(user_transforms).__name__}",
          TypeError,
      )
    train_augmentations = v2.Compose(user_transforms + tail)
  else:
    train_augmentations = v2.Compose([
        v2.RandomResizedCrop(size=(image_size, image_size),
                             scale=(0.2, 1.0),
                             antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=360),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        v2.ElasticTransform(alpha=50.0, sigma=5.0),
        *tail,
    ])
  val_augmentations = v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      *tail,
  ])
  return train_augmentations, val_augmentations


def fmt_weights(weights, decimals=3):
  """Format a 1-D tensor as a human-readable list string."""
  return "[" + ", ".join(f"{v:.{decimals}f}" for v in weights.tolist()) + "]"


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


def mixup_data(x, y, alpha=0.2):
  """Apply Mixup to a batch: returns mixed images, and two label sets +
  lambda."""
  if alpha <= 0:
    return x, y, y, 1.0
  lam = np.random.beta(alpha, alpha)
  # Fold into lam so y_a is always the dominant label.
  lam = max(lam, 1.0 - lam)
  batch_size = x.size(0)
  index = torch.randperm(batch_size, device=x.device)
  mixed_x = lam * x + (1.0 - lam) * x[index]
  return mixed_x, y, y[index], lam


def resolve_augmentations(args, processor):
  """Resolve the train/val transforms and the optional TTA transform."""
  train_aug_fn = None
  if args.train_augmentation_script:
    train_aug_fn = load_augmentation_script(args.train_augmentation_script)

  tta_transform = None
  if args.tta:
    if args.tta == "default":
      from genml_kit.training.tta import create_default_tta_transform
      tta_transform = create_default_tta_transform(image_size=args.image_size)
    else:
      from genml_kit.training.tta import load_tta_transform
      tta_transform = load_tta_transform(args.tta, image_size=args.image_size)
      logging.info(f"Loaded TTA transform from {args.tta}")

  train_transforms, val_transforms = build_transforms(processor, args.image_size,
                                                      train_aug_fn)
  logging.info(f"Image size: {args.image_size}")
  logging.info(f"Train transforms: {train_transforms}")
  logging.info(f"Val transforms:   {val_transforms}")
  return train_transforms, val_transforms, tta_transform


def apply_freeze_patterns(args, model):
  """Freeze parameters matching the user/LORA patterns, when requested."""
  from genml_kit.training.model_utils import extract_lora_params
  if args.freeze or args.lora:
    patterns = list(args.freeze.split(",")) if args.freeze else []
    if args.lora:
      patterns.extend(re.escape(k) for k in extract_lora_params(model))
    freeze_model(model, tuple(patterns))


def maybe_train_xgboost(args, pipeline, device):
  """Free training VRAM, then optionally train the XGBoost head."""
  if not getattr(args, "xgboost_model", None):
    return
  from genml_kit.training.xgb_pipeline import train_xgboost_on_backbone

  gc.collect()
  if hasattr(torch, "cuda") and torch.cuda.is_available():
    torch.cuda.empty_cache()

  train_ds = pipeline.train_proxy.dataset
  val_ds = pipeline.val_proxy.dataset
  logging.info("Training XGBoost head on backbone features...")
  train_xgboost_on_backbone(
      train_ds=train_ds,
      val_ds=val_ds,
      train_loader=pipeline.train_loader,
      val_loader=pipeline.val_loader,
      num_labels=pipeline.num_labels,
      id2label=pipeline.id2label,
      device=device,
      xgb_model=args.xgboost_model,
      xgb_use_gpu=args.xgb_use_gpu,
      max_depth=args.xgb_max_depth,
      n_estimators=args.xgb_n_estimators,
      learning_rate=args.xgb_learning_rate,
      subsample=args.xgb_subsample,
      colsample_bytree=args.xgb_colsample_bytree,
      min_child_weight=args.xgb_min_child_weight,
      gamma=args.xgb_gamma,
      reg_alpha=args.xgb_reg_alpha,
      random_state=args.seed,
  )
