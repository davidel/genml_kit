"""Fine-tune a HuggingFace image-classification model."""

import argparse
import gc
import logging
import os
import re

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import v2
from torchvision.transforms.v2 import InterpolationMode

from scdiag.checkpointing import (
    CheckpointSaver,
    create_model_report,
    fetch_remote_checkpoint,
    parse_state_flags,
    restore_training_state,
    resume_checkpoint,
    serialize_lora_state,
)
from scdiag.cli_args import (
    add_checkpoint_args,
    add_logging_args,
    add_optimization_args,
    add_source_checkpoint_args,
    add_training_state_args,
)
from scdiag.cli_utils import KVPairAction
from scdiag.datasets.hf_proxy import HFDatasetProxy
from scdiag.datasets.weighted_sampler import build_weighted_sampler
from scdiag.eval import evaluate_performance
from scdiag.grad_monitor import GradMonitor
from scdiag.logging_utils import fatal, setup_logging
from scdiag.losses.focal import CombinedFocalLoss
from scdiag.metrics import confusion_row_strings
from scdiag.model_utils import (
    apply_lora,
    enable_grad_checkpointing,
    extract_lora_params,
    freeze_model,
    set_train_mode,
)
from scdiag.models import load_model, load_processor
from scdiag.optim_factory import (
    build_param_groups,
    build_param_groups_llrd,
    create_optimizer,
    create_scheduler,
)
from scdiag.script_utils import load_extern
from scdiag.seed_utils import seed_everything, seed_worker
from scdiag.train_reporting import TrainReporting
from scdiag.tta import create_default_tta_transform, load_tta_transform
from scdiag.xgb_pipeline import train_xgboost_on_backbone

__all__ = [
    "CombinedFocalLoss",
    "evaluate_performance",
    "train_xgboost_on_backbone",
]


def parse_class_multipliers(s, num_labels, label2id):
  """Parse a ``--class_multipliers`` string into a ``[num_labels]`` tensor.

    *s* is a comma-separated string of ``NAME=VALUE`` pairs where *NAME* is
    a label string (e.g. ``melanoma``) or an integer label index and *VALUE*
    is a float multiplier.  Unspecified classes default to ``1.0``.

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
          "Expected NAME=VALUE (e.g. melanoma=4.0).",
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


def load_augmentation_script(path_or_url):
  """Load a Python script and return its ``create_train_transform`` callable.

    The script must define a ``create_train_transform(image_size, **kwargs)``
    function that returns a list of ``torchvision.transforms.v2`` transforms.

    Args:
        path_or_url: Local file path or HTTP/HTTPS URL to the script.

    Returns:
        The ``create_train_transform`` callable.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If the script does not define a callable
            ``create_train_transform``.
    """
  return load_extern(path_or_url, "create_train_transform")


def build_transforms(processor, image_size, train_aug_fn=None):
  """Create train / val augmentation pipelines.

    If *train_aug_fn* is ``None`` the default (hand-picked) augmentation
    list is used.  Otherwise *train_aug_fn(image_size)* must return a
    list of ``v2`` transforms; the fixed preprocessing tail is appended
    automatically.

    Includes the processor's normalization (mean / std) so images arrive
    at the model ready for inference.
    """
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


def load_and_split_dataset(
    dataset_name,
    cache_dir=None,
    test_size=0.2,
    seed=42,
    train_transform=None,
    val_transform=None,
    image_column=None,
    label_column=None,
):
  """Load a HuggingFace dataset, return ``(train_proxy, val_proxy)``."""
  if dataset_name.startswith("imagefolder/"):
    data_dir = dataset_name.split("/", 1)[1]
    raw = load_dataset("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
  else:
    raw = load_dataset(dataset_name, cache_dir=cache_dir)

  # Single split: validate, split and wrap.
  if isinstance(raw, datasets.Dataset):
    detected_image_column = image_column or HFDatasetProxy.detect_image_column(raw)
    if detected_image_column is None:
      fatal(
          f"No image column detected in {dataset_name}. "
          f"Columns: {list(raw.features.keys())}",
          ValueError,
      )
    split = raw.train_test_split(test_size=test_size, seed=seed)
    return (
        HFDatasetProxy(
            split["train"],
            transform=train_transform,
            image_column=image_column,
            label_column=label_column,
        ),
        HFDatasetProxy(
            split["test"],
            transform=val_transform,
            image_column=image_column,
            label_column=label_column,
        ),
    )

  # DatasetDict: validate and ensure train/test splits exist.
  for split_name in raw:
    detected_image_column = image_column or HFDatasetProxy.detect_image_column(
        raw[split_name])
    if detected_image_column is None:
      fatal(
          f"No image column in split '{split_name}' of {dataset_name}. "
          f"Columns: {list(raw[split_name].features.keys())}",
          ValueError,
      )

  splits = set(raw.keys())
  if "train" not in splits or "test" not in splits:
    if "train" in splits:
      split = raw["train"].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    elif len(splits) == 1:
      only = next(iter(splits))
      split = raw[only].train_test_split(test_size=test_size, seed=seed)
      raw = datasets.DatasetDict(split)
    else:
      names = list(raw.keys())
      raw = datasets.DatasetDict({"train": raw[names[0]], "test": raw[names[1]]})

  logging.info(
      "Using image column: %s%s",
      image_column or "auto-detected",
      " (explicit)" if image_column else "",
  )
  logging.info(
      "Using label column: %s%s",
      label_column or "auto-detected",
      " (explicit)" if label_column else "",
  )
  return (
      HFDatasetProxy(
          raw["train"],
          transform=train_transform,
          image_column=image_column,
          label_column=label_column,
      ),
      HFDatasetProxy(
          raw["test"],
          transform=val_transform,
          image_column=image_column,
          label_column=label_column,
      ),
  )


def fmt_weights(weights, decimals=3):
  """Format a 1-D tensor as a human-readable list string."""
  return "[" + ", ".join(f"{v:.{decimals}f}" for v in weights.tolist()) + "]"


def compute_class_weights(train_dataset, num_labels, label_column="label"):
  """Compute inverse-frequency class weights as a CPU tensor.

    *train_dataset* is a raw HF ``Dataset`` (before ``set_transform`` is
    applied) so that column access does not trigger any registered
    transforms.
    """
  feat = train_dataset.features[label_column]
  raw_labels = train_dataset[label_column]
  if isinstance(feat, datasets.ClassLabel):
    labels = np.array(
        feat.str2int(raw_labels) if isinstance(raw_labels[0], str) else raw_labels,
        dtype=np.int64,
    )
  else:
    labels = np.array(raw_labels, dtype=np.int64)

  actual_labels = np.unique(labels)
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


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Fine-tune an image-classification model (HuggingFace or timm).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  parser.add_argument(
      "--model",
      type=str,
      default="google/vit-base-patch16-224",
      help="HuggingFace model name/path, or timm model "
      "(e.g. 'timm:eva02_base_patch14_224.mim_in22k').",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default="marmal88/skin_cancer",
      help=
      "HuggingFace dataset name, or 'imagefolder/PATH' for local ImageFolder datasets.",
  )
  parser.add_argument(
      "--image_column",
      type=str,
      help="Image column name; auto-detected when omitted.",
  )
  parser.add_argument(
      "--label_column",
      type=str,
      help="Label column name; auto-detected when omitted.",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=448,
      help="Resize images to this size.",
  )
  parser.add_argument(
      "--train_augmentation_script",
      type=str,
      help="Path or URL to a Python script defining "
      "create_train_transform(image_size, **kwargs) -> list of v2 "
      "transforms. The fixed tail (ToImage, ToDtype, Normalize) is "
      "appended automatically.",
  )
  parser.add_argument(
      "--tta",
      type=str,
      help="Test-Time Augmentation. 'default' uses the built-in "
      "4-view transform (identity + flips). A path/URL loads an "
      "external script defining create_tta_transform. "
      "Omit to disable TTA.",
  )
  parser.add_argument(
      "--epochs",
      type=int,
      default=5,
      help="Number of training epochs.",
  )
  parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
  parser.add_argument(
      "--val_split",
      type=float,
      default=0.2,
      help="Fraction of data to hold out for validation.",
  )
  parser.add_argument(
      "--lr",
      type=float,
      default=3e-5,
      help="Peak learning rate.",
  )
  parser.add_argument(
      "--weight_decay",
      type=float,
      default=0.01,
      help="Weight decay.",
  )
  parser.add_argument(
      "--lr_group",
      nargs="+",
      metavar="REGEX=LR",
      help="Per-parameter-group learning rates. "
      "E.g. --lr_group 'backbone.*=1e-5' 'classifier.*=1e-3'. "
      "Regexes matched against named_parameters(); first match wins. "
      "Unmatched trainable params use --lr.",
  )
  parser.add_argument(
      "--llrd_decay",
      type=float,
      metavar="FACTOR",
      help="Layer-wise learning rate decay factor. "
      "When set, learning rates decay by this factor per depth level "
      "(shallow layers get lower LR). E.g. --llrd_decay 0.85.",
  )

  parser.add_argument(
      "--label_smoothing",
      type=float,
      default=0.0,
      help="Label smoothing.",
  )
  parser.add_argument(
      "--focal_gamma",
      type=float,
      default=0.0,
      help="Focal loss gamma. 0.0 disables focal modulation "
      "(standard weighted CE).",
  )
  parser.add_argument(
      "--class_multipliers",
      type=str,
      default="",
      help="Comma-separated NAME=VALUE pairs to override per-class "
      "clinical severity multipliers (M_c). NAME is a label string "
      "(e.g. melanoma) or integer label index. VALUE is a float. "
      "Unspecified classes default to 1.0. Example: "
      "'melanoma=4.0,melanocytic_Nevi=0.5'.",
  )
  parser.add_argument(
      "--sampler",
      default="none",
      choices=["none", "weighted"],
      help="Training sampler. 'weighted' uses WeightedRandomSampler to "
      "upsample rare classes.",
  )
  parser.add_argument(
      "--sampler_weights",
      default="frequency",
      choices=["frequency", "multipliers", "combined"],
      help="How to compute per-sample weights for --sampler weighted: "
      "'frequency' (inverse-freq), 'multipliers' (--class_multipliers), "
      "or 'combined' (freq x multipliers).",
  )

  add_optimization_args(parser)

  parser.add_argument(
      "--classifier",
      type=str,
      help="Classifier head spec: a registered name (e.g. 'mlp') or a "
      "path to a .py file defining a Classifier class. Only used with "
      "--model cls_model_wrapper:<hf_name>.",
  )
  parser.add_argument(
      "--classifier_args",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra classifier kwargs (repeatable). "
      "Example: --classifier_args hidden=512 dropout=0.3",
  )

  parser.add_argument(
      "--freeze",
      type=str,
      help="Comma-separated list of regex patterns (re.match) for "
      "parameter names to keep trainable. All other parameters are "
      "frozen. Each pattern is anchored at the start of the name. "
      "Examples: 'head' (matches names starting with 'head'), "
      "'.*pool.*' (matches any name containing 'pool'), "
      "'classifier\\.(head|pool)' (matches classifier.head or "
      "classifier.pool). If omitted, all parameters are trainable.",
  )

  add_checkpoint_args(parser, checkpoint_default="scdiag")

  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="RNG seed for data shuffling, split, mixup, dropout, and "
      "the XGBoost stage. 42 by default; pass the same value to "
      "reproduce a run.",
  )
  parser.add_argument(
      "--deterministic",
      action="store_true",
      help="Enable deterministic algorithms (cuDNN deterministic mode, "
      "benchmark off). Costs throughput; ops without a deterministic "
      "CUDA kernel log a warning instead of failing.",
  )
  parser.add_argument(
      "--device",
      type=str,
      help="Device: cpu, cuda, or cuda:INDEX (default: auto-detect).",
  )

  parser.add_argument(
      "--log_dir",
      type=str,
      help="TensorBoard log directory. Defaults to "
      "<dir_of_latest_ckpt>/logs.",
  )
  parser.add_argument(
      "--log_every",
      type=int,
      default=20,
      help="Log every N steps.",
  )
  parser.add_argument(
      "--grad_monitor",
      type=int,
      default=-1,
      help="Log gradient stats every N steps. -1 (default) = disabled.",
  )
  parser.add_argument(
      "--norm_history",
      type=int,
      default=0,
      help="Keep last N norm snapshots per param for trend analysis. "
      "0 (default) = disabled. Requires --grad_monitor.",
  )
  parser.add_argument(
      "--trend_top_n",
      type=int,
      default=10,
      help="Show top N params in trend table by abs change %. "
      "0 = show all. Requires --norm_history.",
  )
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N optimizer steps. 0 disables.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=2,
      help="DataLoader worker processes.",
  )
  add_logging_args(parser)

  add_training_state_args(parser,
                          state_save="opt,sched,amp",
                          state_load="opt,sched,amp")
  parser.add_argument(
      "--save_frozen",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Include frozen (non-trainable) parameters in checkpoints. "
      "When disabled (default), only trainable parameters are saved, "
      "greatly reducing checkpoint size for fine-tuning runs.",
  )

  lora_group = parser.add_argument_group("lora")
  lora_group.add_argument(
      "--lora",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable LoRA (Low-Rank Adaptation) via PEFT. "
      "Requires: pip install scdiag[lora].",
  )
  lora_group.add_argument(
      "--lora_r",
      type=int,
      default=8,
      help="LoRA rank.",
  )
  lora_group.add_argument(
      "--lora_alpha",
      type=int,
      default=16,
      help="LoRA alpha (scaling factor = alpha / r).",
  )
  lora_group.add_argument(
      "--lora_dropout",
      type=float,
      default=0.0,
      help="Dropout probability for LoRA layers.",
  )
  lora_group.add_argument(
      "--lora_target_modules",
      type=str,
      help="Comma-separated module names to apply LoRA to "
      "(e.g. 'query,key,value'). Auto-detect if omitted.",
  )

  parser.add_argument(
      "--mixup_alpha",
      type=float,
      default=0.0,
      help="Mixup alpha. Recommended: 0.2 for skin lesion classification.",
  )

  parser.add_argument(
      "--cache_dir",
      type=str,
      help="Cache directory for downloaded datasets.",
  )
  add_source_checkpoint_args(parser)

  xgb_group = parser.add_argument_group("xgboost")
  xgb_group.add_argument(
      "--xgboost_model",
      help="Output path for XGBoost model. If set, train XGBoost on "
      "backbone features after training completes.",
  )
  xgb_group.add_argument(
      "--xgb_max_depth",
      type=int,
      default=6,
      help="XGBoost max tree depth.",
  )
  xgb_group.add_argument(
      "--xgb_n_estimators",
      type=int,
      default=200,
      help="XGBoost number of trees.",
  )
  xgb_group.add_argument(
      "--xgb_learning_rate",
      type=float,
      default=0.1,
      help="XGBoost learning rate.",
  )
  xgb_group.add_argument(
      "--xgb_subsample",
      type=float,
      default=0.8,
      help="XGBoost row sampling ratio.",
  )
  xgb_group.add_argument(
      "--xgb_colsample_bytree",
      type=float,
      default=0.8,
      help="XGBoost column sampling ratio.",
  )
  xgb_group.add_argument(
      "--xgb_min_child_weight",
      type=int,
      default=1,
      help="XGBoost min child weight.",
  )
  xgb_group.add_argument(
      "--xgb_gamma",
      type=float,
      default=0.0,
      help="XGBoost min split loss.",
  )
  xgb_group.add_argument(
      "--xgb_reg_alpha",
      type=float,
      default=0.0,
      help="XGBoost L1 regularization.",
  )
  xgb_group.add_argument(
      "--xgb_use_gpu",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Use GPU for XGBoost training (requires xgboost with CUDA support).",
  )

  parser.add_argument(
      "--model_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override model configuration (repeatable). "
      "Example: --model_arg depth=6 num_heads=8",
  )
  parser.add_argument(
      "--proc_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override processor configuration (repeatable).",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="AdamW",
      help="torch.optim optimizer class name (case-sensitive), "
      "or a path to a custom .py script. "
      "Examples: AdamW (default), Adam, SGD.",
  )
  parser.add_argument(
      "--opt_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra optimizer kwargs (repeatable). "
      "Example: --opt_arg betas=0.9,0.999 momentum=0.9",
  )
  parser.add_argument(
      "--scheduler",
      type=str,
      help="torch.optim.lr_scheduler class name (case-sensitive), "
      "or a path to a custom .py script. "
      "Examples: CosineAnnealingLR, StepLR. "
      "Default: None (no scheduler).",
  )
  parser.add_argument(
      "--sched_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra scheduler kwargs (repeatable). "
      "Example: --sched_arg T_max=50 eta_min=1e-6",
  )

  parser.add_argument(
      "--grad_checkpoint",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable gradient checkpointing to reduce VRAM usage "
      "(~40-50% less activation memory, ~25-35% more compute per step). "
      "Enables larger batch sizes. Net throughput gain depends on "
      "whether the GPU is memory-bound or compute-saturated.",
  )

  return parser.parse_args(argv)


def mixup_data(x, y, alpha=0.2):
  """Apply Mixup to a batch: returns mixed images, and two label sets + lambda.

    Returns ``(mixed_x, y_a, y_b, lam)`` where ``lam`` is the interpolation
    coefficient sampled from ``Beta(alpha, alpha)``.  When ``alpha <= 0`` the
    function is a no-op and returns the originals unchanged.
    """
  if alpha <= 0:
    return x, y, y, 1.0
  lam = np.random.beta(alpha, alpha)
  # Fold into lam so y_a is always the dominant label and callers can
  # use y_a alone for hard-label metrics; the y_b contribution survives
  # only through the 1-lam weight in the soft-target loss.
  lam = max(lam, 1.0 - lam)
  batch_size = x.size(0)
  index = torch.randperm(batch_size, device=x.device)
  mixed_x = lam * x + (1.0 - lam) * x[index]
  return mixed_x, y, y[index], lam


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scaler,
    scheduler,
    device,
    amp_dtype,
    epoch,
    args,
    writer=None,
    monitor=None,
    global_step=0,
    *,
    saver,
    best_macro_f1=0.0,
):
  """Train for one epoch.

    If ``args.grad_accum_steps > 1``, gradients are accumulated over that many
    micro-batches before stepping the optimizer.

    *global_step* is the running count of actual optimizer steps across all
    epochs (used for the gradient monitor so it receives contiguous step
    numbers).  The updated value is returned.

    When *saver* has a positive ``save_every``, a periodic checkpoint is
    written to ``<root>_latest.pt`` every N optimizer steps (recording
    ``epoch - 1`` as the last fully completed epoch).
    """
  set_train_mode(model, "train")
  total_batches = len(dataloader)
  reporter = TrainReporting(
      total_batches=total_batches,
      log_every=args.log_every,
      writer=writer,
      device=device,
      optimizer=optimizer,
  )

  for batch_idx, (images, targets) in enumerate(dataloader):
    images, targets = images.to(device), targets.to(device)

    use_mixup = args.mixup_alpha > 0 and images.size(0) >= 2
    if use_mixup:
      images, targets_a, targets_b, lam = mixup_data(images,
                                                     targets,
                                                     alpha=args.mixup_alpha)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      outputs = model(pixel_values=images)
      logits = outputs.logits
      if use_mixup:
        soft_targets = (lam * F.one_hot(targets_a, logits.size(-1)).float() +
                        (1.0 - lam) * F.one_hot(targets_b, logits.size(-1)).float())
        loss = criterion(logits, soft_targets) / args.grad_accum_steps
      else:
        loss = criterion(logits, targets) / args.grad_accum_steps

    if amp_dtype == torch.float16 and scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == total_batches:
      if amp_dtype == torch.float16 and scaler is not None:
        scaler.unscale_(optimizer)
        # Gradient monitor must read AFTER unscale_() so it sees the true
        # gradient magnitudes, not the scaled values.
        if monitor is not None:
          monitor.step(global_step)
        if args.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
      else:
        if monitor is not None:
          monitor.step(global_step)
        if args.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      global_step += 1

      # Mid-epoch periodic checkpoint. Records *epoch - 1* (the last fully
      # completed epoch): on resume the interrupted epoch restarts instead
      # of being skipped.
      if saver.should_save(global_step):
        saver.save_latest(
            epoch - 1,
            global_step=global_step,
            best_macro_f1=best_macro_f1,
            lora_state_blob=serialize_lora_state(model) if args.lora else None,
        )

    with torch.no_grad():
      # Report metrics against the dominant label only (targets_a always
      # holds it, see the clamp in mixup_data).  The loss above uses the
      # full lam / (1-lam) mixture; only these hard-label metrics are
      # conservative on blended inputs.
      orig_targets = targets_a if use_mixup else targets

    batch_size = orig_targets.size(0)
    report_now = (batch_idx + 1) == total_batches
    reporter.step(
        batch_idx=batch_idx,
        batch_size=batch_size,
        loss_value=loss.item() * batch_size * args.grad_accum_steps,
        logits=logits,
        targets=orig_targets,
        global_step=global_step,
        report_now=report_now,
    )

  avg_loss, top1 = reporter.summary()
  return avg_loss, top1, global_step


def main():
  args = parse_args()
  setup_logging(args.log_level, args.log_targets)
  seed_everything(args.seed, args.deterministic)

  # Convert string amp_dtype to torch.dtype.
  args.amp_dtype = getattr(torch, args.amp_dtype, None) if args.amp_dtype else None

  states_to_save = parse_state_flags(args.state_save)
  states_to_load = parse_state_flags(args.state_load)

  if args.device:
    device = torch.device(args.device)
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  log_dir = args.log_dir or os.path.join(
      os.path.dirname(args.checkpoint) or ".", "logs")
  os.makedirs(log_dir, exist_ok=True)
  writer = SummaryWriter(log_dir=log_dir)

  # Load processor — unified registry dispatches to HF or custom.
  processor = load_processor(
      args.model,
      image_size=args.image_size,
      cache_dir=args.cache_dir,
      **args.proc_arg,
  )

  # Resolve custom augmentation script to a callable, if provided.
  train_aug_fn = None
  if args.train_augmentation_script:
    train_aug_fn = load_augmentation_script(args.train_augmentation_script)
    logging.info(f"Using custom augmentation script: "
                 f"{args.train_augmentation_script}")

  # Resolve TTA transform.
  tta_transform = None
  if args.tta == "default":
    tta_transform = create_default_tta_transform(image_size=args.image_size)
    logging.info("TTA enabled: default (identity + 3 flips, N=4)")
  elif args.tta is not None:
    tta_transform = load_tta_transform(args.tta, image_size=args.image_size)
    logging.info(f"TTA enabled: loaded from {args.tta}")

  train_transforms, val_transforms = build_transforms(processor, args.image_size,
                                                      train_aug_fn)

  logging.info(f"Image size: {args.image_size}")
  logging.info(f"Train transforms: {train_transforms}")
  logging.info(f"Val transforms:   {val_transforms}")

  train_proxy, val_proxy = load_and_split_dataset(
      args.dataset,
      cache_dir=args.cache_dir,
      test_size=args.val_split,
      train_transform=train_transforms,
      val_transform=val_transforms,
      image_column=args.image_column,
      label_column=args.label_column,
      seed=args.seed,
  )

  num_labels = train_proxy.num_labels
  logging.info(f"num_labels: {num_labels}")

  w_freq = compute_class_weights(train_proxy.dataset, num_labels,
                                 train_proxy.label_column)
  logging.info(f"Inverse-frequency weights: {fmt_weights(w_freq)}")

  # Convert proxy label2id (str values) to int values for parse_class_multipliers.
  label2id_int = {k: int(v) for k, v in train_proxy.label2id.items()}

  # Apply clinical severity multipliers from --class_multipliers.
  clinical_m = parse_class_multipliers(
      args.class_multipliers,
      num_labels,
      label2id_int,
  )
  logging.info(f"Clinical multipliers (M_c): {fmt_weights(clinical_m)}")

  if args.sampler == "weighted":
    # When using a weighted sampler, class representation is already
    # balanced in each batch, so we skip w_freq to avoid double-
    # compensating for class imbalance.
    class_weights = clinical_m.to(device)
    logging.info(f"Final class weights (M_c only, sampler handles frequency): "
                 f"{fmt_weights(class_weights)}")
  else:
    class_weights = (w_freq * clinical_m).to(device)
    logging.info(f"Final class weights (W_freq x M_c): {fmt_weights(class_weights)}")

  # Seeded generator for DataLoader shuffling; shared with the val loader
  # (which never shuffles) so worker ordering is reproducible too.
  data_generator = torch.Generator().manual_seed(args.seed)

  if len(train_proxy) < args.batch_size:
    fatal(
        f"Training set ({len(train_proxy)} samples) is smaller than "
        f"batch_size ({args.batch_size}). Reduce --batch_size.",
        ValueError,
    )
  if args.sampler == "weighted":
    sampler = build_weighted_sampler(
        train_proxy.dataset,
        num_labels,
        train_proxy.label_column,
        args.sampler_weights,
        multipliers=clinical_m,
    )
    train_loader = DataLoader(
        train_proxy,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
  else:
    train_loader = DataLoader(
        train_proxy,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
  val_loader = DataLoader(
      val_proxy,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      worker_init_fn=seed_worker,
      generator=data_generator,
      pin_memory=(device.type == "cuda"),
  )

  model = load_model(
      args.model,
      num_labels=num_labels,
      id2label=train_proxy.id2label,
      label2id=train_proxy.label2id,
      image_size=args.image_size,
      device=device,
      checkpoint_path=args.checkpoint,
      cache_dir=args.cache_dir,
      classifier=args.classifier,
      classifier_args=args.classifier_args,
      **args.model_arg,
  )

  if args.grad_checkpoint:
    enable_grad_checkpointing(model)

  # Optionally load weights from a source checkpoint
  if args.source_checkpoint:
    from scdiag.checkpointing import load_checkpoint_weights

    logging.info(f"Loading source checkpoint: {args.source_checkpoint}")
    if args.param_rename:
      logging.info(f"  Key renames: {args.param_rename}")
    load_checkpoint_weights(
        args.source_checkpoint,
        model,
        device=device,
        param_rename=args.param_rename,
    )

  if args.focal_gamma > 0 and args.label_smoothing > 0:
    logging.warning(
        "Both --focal_gamma (%.1f) and --label_smoothing (%.2f) are > 0. "
        "Focal loss and label smoothing conflict. Proceeding anyway — "
        "monitor for instability.",
        args.focal_gamma,
        args.label_smoothing,
    )

  criterion = CombinedFocalLoss(
      weights=class_weights,
      gamma=args.focal_gamma,
      label_smoothing=args.label_smoothing,
  )

  scaler = (torch.amp.GradScaler(device)
            if args.amp_dtype == torch.float16 and device.type == "cuda" else None)

  if args.lora:
    target = (args.lora_target_modules.split(",") if args.lora_target_modules else None)
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target,
    )
  ckpt_latest = args.checkpoint + "_latest.pt"
  ckpt_best = args.checkpoint + "_best.pt"
  fetch_remote_checkpoint(args.remote_checkpoint, ckpt_latest, ckpt_best)
  model, start_epoch, best_macro_f1, ckpt_extra = resume_checkpoint(
      ckpt_latest,
      ckpt_best,
      model,
      device,
  )

  if args.freeze or args.lora:
    patterns = list(args.freeze.split(",")) if args.freeze else []
    if args.lora:
      patterns.extend(re.escape(k) for k in extract_lora_params(model))
    freeze_model(model, tuple(patterns))

  if args.llrd_decay is not None:
    param_groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        decay_factor=args.llrd_decay,
    )
  else:
    param_groups = build_param_groups(
        dict(model.named_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_groups=args.lr_group,
    )
  optimizer = create_optimizer(
      param_groups,
      name=args.optimizer,
      **args.opt_arg,
  )

  scheduler = create_scheduler(
      optimizer,
      name=args.scheduler,
      epochs=args.epochs,
      base_lr=args.lr,
      **args.sched_arg,
  )

  restore_training_state(
      ckpt_extra,
      optimizer,
      scheduler,
      scaler,
      states_to_load,
  )

  optimizer_global_step = ckpt_extra.get(
      "global_step",
      start_epoch * (len(train_loader) // args.grad_accum_steps),
  )
  del ckpt_extra

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  grad_monitor = None
  if args.grad_monitor >= 0:
    grad_monitor = GradMonitor(
        model,
        log_every=args.grad_monitor,
        norm_history=args.norm_history,
        trend_top_n=args.trend_top_n,
    )
    logging.info(f"Gradient monitoring enabled (every {args.grad_monitor} steps).")
    if args.norm_history > 0:
      logging.info(f"  Norm trend history: last {args.norm_history} snapshots")

  # Report the final model state after LoRA, freezing, optimizer setup, and
  # checkpoint restoration, immediately before training begins.
  logging.info(create_model_report(model))

  # All checkpoint writes go through one saver: state sources bound once,
  # per-save data (epoch, global_step, metrics) passed per call.
  saver = CheckpointSaver(
      model,
      optimizer,
      scheduler,
      root=args.checkpoint,
      states_to_save=states_to_save,
      scaler=scaler,
      save_frozen=args.save_frozen,
      remote_uri=args.remote_checkpoint,
      save_every=args.save_every,
  )

  try:
    for epoch in range(start_epoch, args.epochs):
      effective_batch = args.batch_size * args.grad_accum_steps
      logging.info(f"=== Epoch {epoch + 1}/{args.epochs} "
                   f"(eff_batch={effective_batch}) ===")

      train_loss, train_t1, optimizer_global_step = train_one_epoch(
          model,
          train_loader,
          criterion,
          optimizer,
          scaler,
          scheduler,
          device,
          args.amp_dtype,
          epoch,
          args,
          writer=writer,
          monitor=grad_monitor,
          global_step=optimizer_global_step,
          saver=saver,
          best_macro_f1=best_macro_f1,
      )

      if scheduler is not None:
        scheduler.step()
      writer.add_scalar("Epoch/Loss_Train", train_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Train_Top1", train_t1, epoch)

      (v_loss, v_t1, v_balanced_acc, v_macro_f1, v_weighted_f1, v_per_class_metrics,
       v_cm, v_original_metrics) = evaluate_performance(
           model,
           val_loader,
           criterion,
           device,
           args.amp_dtype,
           id2label=train_proxy.id2label,
           tta_transform=tta_transform,
       )
      writer.add_scalar("Epoch/Loss_Val", v_loss, epoch)
      writer.add_scalar("Epoch/Accuracy_Val_Top1", v_t1, epoch)
      writer.add_scalar("Epoch/Balanced_Accuracy_Val", v_balanced_acc, epoch)
      writer.add_scalar("Epoch/Macro_F1_Val", v_macro_f1, epoch)
      writer.add_scalar("Epoch/Weighted_F1_Val", v_weighted_f1, epoch)
      logging.info(f"Epoch {epoch + 1} Results -> "
                   f"Val Loss: {v_loss:.4f} | Top1: {v_t1:.2f}%"
                   f" | Balanced Acc: {v_balanced_acc:.2f}%"
                   f" | Macro F1: {v_macro_f1:.2f}%"
                   f" | Weighted F1: {v_weighted_f1:.2f}%")
      if v_original_metrics is not None:
        orig = v_original_metrics
        logging.info(
            f"  Original-view metrics (TTA comparison): Top1={orig['top1']:.2f}% "
            f"| Balanced Acc={orig['balanced_accuracy']:.2f}% "
            f"| Macro F1={orig['macro_f1']:.2f}% "
            f"| Weighted F1={orig['weighted_f1']:.2f}%")
        logging.info(
            f"  TTA delta: Top1={v_t1 - orig['top1']:+.2f}% "
            f"| Balanced Acc={v_balanced_acc - orig['balanced_accuracy']:+.2f}% "
            f"| Macro F1={v_macro_f1 - orig['macro_f1']:+.2f}% "
            f"| Weighted F1={v_weighted_f1 - orig['weighted_f1']:+.2f}%")
        writer.add_scalar("Epoch/Accuracy_Val_Original_Top1", orig["top1"], epoch)
        writer.add_scalar("Epoch/Balanced_Accuracy_Val_Original",
                          orig["balanced_accuracy"], epoch)
        writer.add_scalar("Epoch/Macro_F1_Val_Original", orig["macro_f1"], epoch)
        writer.add_scalar("Epoch/Weighted_F1_Val_Original", orig["weighted_f1"], epoch)
      logging.info("Confusion matrix:")
      for line in confusion_row_strings(v_cm, id2label=train_proxy.id2label):
        logging.info(f"  {line}")
      if v_per_class_metrics:
        logging.info("Class metrics:")
        for cls_name, metrics in v_per_class_metrics.items():
          writer.add_scalar(f"Epoch/F1_Val/{cls_name}", metrics["f1"], epoch)
          logging.info(f"  {cls_name}: precision={metrics['precision']:.2f}% "
                       f"recall={metrics['recall']:.2f}% F1={metrics['f1']:.2f}% "
                       f"support={metrics['support']}")

      if v_macro_f1 > best_macro_f1:
        best_macro_f1 = v_macro_f1
        saver.save_best(
            epoch,
            best_macro_f1=best_macro_f1,
            global_step=optimizer_global_step,
            lora_state_blob=serialize_lora_state(model) if args.lora else None,
        )
        logging.info(f"New best macro F1, checkpoint saved: {best_macro_f1:.2f}%")

      completed_epoch = epoch
  except KeyboardInterrupt:
    logging.warning("Interrupt detected!")
  finally:
    saver.save_latest(
        completed_epoch,
        best_macro_f1=best_macro_f1,
        global_step=optimizer_global_step,
        lora_state_blob=serialize_lora_state(model) if args.lora else None,
    )
    logging.info("Checkpoint saved on exit.")
    writer.close()

    # Free training model VRAM before XGBoost block.
    del saver
    del model
    del optimizer
    del scaler
    del scheduler
    del train_loader
    del val_loader
    gc.collect()
    torch.cuda.empty_cache()

    if args.xgboost_model:
      # Access raw HF datasets (before proxy wrapping) for XGBoost.
      train_xgboost_on_backbone(
          train_proxy.dataset,
          val_proxy.dataset,
          device,
          num_labels,
          checkpoint_dir=args.checkpoint,
          model_spec=args.model,
          cache_dir=args.cache_dir,
          proc_kwargs=args.proc_arg,
          image_size=args.image_size,
          output_path=args.xgboost_model,
          batch_size=args.batch_size,
          use_gpu=args.xgb_use_gpu,
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


if __name__ == "__main__":
  main()
