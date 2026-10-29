"""Dataset-loading helper and legacy compat re-exports for the trainer.

The ``genml-kit-train`` entry point itself lives in :mod:`genml_kit.cli.train`.
This module keeps the reusable data-loading helper
(:func:`load_and_split_dataset`) plus the historic ``training.train`` import
surface (re-exported below for backward compatibility).
"""

import logging

import datasets as _datasets

from genml_kit.datasets.capped import maybe_cap
from genml_kit.datasets.hf_proxy import HFDatasetProxy  # noqa: F401
from genml_kit.models import load_model, load_processor  # noqa: F401
from genml_kit.pipelines.images import (  # noqa: F401
    build_pretrain_dataset, build_pretrain_transform, compute_class_weights,
)
from genml_kit.training.train_compat import (
    CombinedFocalLoss,  # noqa: F401
    build_transforms,  # noqa: F401
    evaluate_performance,  # noqa: F401
    load_augmentation_script,  # noqa: F401
    mixup_data,  # noqa: F401
    parse_class_multipliers,  # noqa: F401
)
from genml_kit.training.trainer import (  # noqa: F401 (compat export)
    BaseTrainer, TrainingResult,
)
from genml_kit.utils.logging import fatal


def load_and_split_dataset(
    dataset_name,
    cache_dir=None,
    test_size=0.2,
    seed=None,
    train_transform=None,
    val_transform=None,
    image_column=None,
    label_column=None,
    loader=None,
    max_items=None,
):
  """Load a HuggingFace dataset, return ``(train_proxy, val_proxy)``.

    *loader* defaults to ``datasets.load_dataset`` and is a parameter (rather
    than a module-level alias) so tests can pass a stub directly.  The
    default is late-bound (``None`` resolved at call time) so tests can also
    simply ``patch("datasets.load_dataset")`` for call paths that go through
    ``main()`` / the pipeline without threading the loader explicitly.

    *max_items* caps each split to at most that many rows (a smoke-test
    convenience; the cap is applied to each split *after* the train/val
    split, so both halves get their own full budget).
    """
  if loader is None:
    loader = _datasets.load_dataset
  if dataset_name.startswith("imagefolder/"):
    data_dir = dataset_name.split("/", 1)[1]
    raw = loader("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
  else:
    raw = loader(dataset_name, cache_dir=cache_dir)

  # Single split: validate, split and wrap.
  if isinstance(raw, _datasets.Dataset):
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
            maybe_cap(split["train"], max_items),
            transform=train_transform,
            image_column=image_column,
            label_column=label_column,
        ),
        HFDatasetProxy(
            maybe_cap(split["test"], max_items),
            transform=val_transform,
            image_column=image_column,
            label_column=label_column,
        ),
    )

  # DatasetDict: validate and select existing train/validation splits, or
  # create a validation split from training data when none is supplied.
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
  if "train" in splits:
    train_split = raw["train"]
    if "validation" in splits:
      val_split = raw["validation"]
    elif "test" in splits:
      val_split = raw["test"]
    else:
      split = train_split.train_test_split(test_size=test_size, seed=seed)
      train_split, val_split = split["train"], split["test"]
  elif len(splits) == 1:
    only = next(iter(splits))
    split = raw[only].train_test_split(test_size=test_size, seed=seed)
    train_split, val_split = split["train"], split["test"]
  else:
    names = list(raw.keys())
    train_split, val_split = raw[names[0]], raw[names[1]]

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
          maybe_cap(train_split, max_items),
          transform=train_transform,
          image_column=image_column,
          label_column=label_column,
      ),
      HFDatasetProxy(
          maybe_cap(val_split, max_items),
          transform=val_transform,
          image_column=image_column,
          label_column=label_column,
      ),
  )
