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
    dataset_arg=None,
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

    *dataset_arg* (a ``KEY=VALUE`` dict from ``--dataset_arg``) is forwarded
    verbatim to ``datasets.load_dataset``, so ``streaming=True`` reaches the
    loader.  When the result is a streaming source the split is positional
    (:class:`~genml_kit.datasets.split.IterableDatasetFilter`) and the sources
    are wrapped in stream-safe :class:`HFDatasetProxy` instances.
    """
  if loader is None:
    loader = _datasets.load_dataset
  if dataset_name.startswith("imagefolder/"):
    data_dir = dataset_name.split("/", 1)[1]
    raw = loader("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
  else:
    raw = loader(dataset_name, cache_dir=cache_dir, **(dataset_arg or {}))

  # A streaming source (or a dict of them) has no index space: the map-style
  # split/wrap below cannot touch it.  Route it through the positional splitter.
  if isinstance(raw, _datasets.IterableDataset):
    return _load_streaming_splits(raw,
                                  test_size=test_size,
                                  image_column=image_column,
                                  label_column=label_column,
                                  train_transform=train_transform,
                                  val_transform=val_transform,
                                  max_items=max_items)
  if isinstance(raw, _datasets.IterableDatasetDict):
    return _load_streaming_splits(_streaming_base_split(raw),
                                  test_size=test_size,
                                  image_column=image_column,
                                  label_column=label_column,
                                  train_transform=train_transform,
                                  val_transform=val_transform,
                                  max_items=max_items)

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


def _streaming_base_split(raw):
  """Pick the single source to split positionally from an ``IterableDatasetDict``.

    The ``train`` split (or the first declared, when there is no ``train``) is
    the one source that :class:`IterableDatasetFilter` splits into train/val.

    Args:
      raw: A ``datasets.IterableDatasetDict``.

    Returns:
      The ``datasets.IterableDataset`` to split.
    """
  names = list(raw.keys())
  if "train" in raw:
    return raw["train"]
  return raw[names[0]]


def _load_streaming_splits(source,
                           test_size=0.2,
                           image_column=None,
                           label_column=None,
                           train_transform=None,
                           val_transform=None,
                           max_items=None):
  """Split a streaming source positionally and wrap both buckets.

    A stream has no index space, so ``train_test_split``/``select`` cannot be
    used; the split is a fixed positional pattern over the one source
    (:class:`~genml_kit.datasets.split.IterableDatasetFilter`), and both buckets
    are independent views that stream it in a single pass.

    Args:
      source: The ``datasets.IterableDataset`` to split.
      test_size: Fraction of items routed to the validation bucket.
      image_column: Optional explicit image column (forwarded to the proxies).
      label_column: Optional explicit label column (forwarded to the proxies).
      train_transform, val_transform: Per-split item transforms.
      max_items: Optional cap on the number of items per split (trims the
        stream iterator; ``maybe_cap`` needs random access, which a stream lacks).

    Returns:
      tuple: ``(train_proxy, val_proxy)``, both stream-safe ``HFDatasetProxy``.

    Raises:
      ValueError: If *test_size* leaves no training rows.
    """
  from genml_kit.datasets.split import IterableDatasetFilter

  if test_size is None or not 0.0 < test_size < 1.0:
    fatal(
        f"--val_split={test_size} must be strictly between 0 and 1 for a "
        f"streaming source.", ValueError)
  splitter = IterableDatasetFilter(source, {
      "train": 1.0 - test_size,
      "val": test_size,
  })
  train_stream = splitter.get_dataset("train")
  val_stream = splitter.get_dataset("val")
  if max_items is not None:
    train_stream = _take(train_stream, max_items)
    val_stream = _take(val_stream, max_items)

  logging.info("Streaming source: positional train/val split (val_split=%s).",
               test_size)
  return (
      HFDatasetProxy(train_stream,
                     transform=train_transform,
                     image_column=image_column,
                     label_column=label_column),
      HFDatasetProxy(val_stream,
                     transform=val_transform,
                     image_column=image_column,
                     label_column=label_column),
  )


def _take(stream, limit):
  """Yield at most *limit* items from *stream* (a stream-safe ``--max_items``)."""

  class _Limited:
    """Truncating view: yields the first *limit* items of a one-pass stream."""

    def __init__(self, inner, n):
      self._inner = inner
      self._n = n

    def __iter__(self):
      for index, item in enumerate(self._inner):
        if index >= self._n:
          return
        yield item

    def __getattr__(self, name):
      return getattr(self._inner, name)

  return _Limited(stream, limit)
