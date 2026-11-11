"""Unified image dataset for self-supervised pre-training.

Stitches together multiple HuggingFace datasets (or pre-downloaded image
directories) into a single :class:`torch.utils.data.Dataset` with flat
indexing.  All ``__getitem__`` calls return ``dict[str, Any]`` with named
fields (e.g. ``"image"``, ``"label"``).
"""

import bisect
import logging

import numpy as np
from PIL import Image

from genml_kit.datasets.base import is_random_access
from genml_kit.datasets.iterable import is_streaming_source
from genml_kit.datasets.capped import maybe_cap
from genml_kit.datasets.hf_proxy import HFDatasetProxy
from genml_kit.datasets.image_folder import ImageFolderDataset
from genml_kit.datasets.retry import getitem_retry
from genml_kit.utils.logging import fatal


class _HFDataset:
  """Wrapper around a HuggingFace ``datasets.Dataset``."""

  #: False for a random-access source; set in ``__init__`` once the loaded
  #: dataset is known to be a stream.  A class-level default keeps the attribute
  #: readable on instances built via ``__new__`` (as the ensemble tests do).
  streaming = False

  def __init__(self,
               name,
               split="train",
               image_column=None,
               label_column=None,
               cache_dir=None,
               hf_token=None,
               max_items=None,
               dataset_arg=None):
    self.name = name
    self._image_column = image_column
    self._label_col = None
    self._label_names = []
    self._label2id = {}
    self._column_map = {}
    self._dataset_arg = dataset_arg

    from datasets import load_dataset

    logging.info(f"Loading HF dataset '{name}' (split={split}) ...")
    ds = load_dataset(name,
                      split=split,
                      cache_dir=cache_dir,
                      token=hf_token,
                      **(dataset_arg or {}))
    # Detect the mode from what was actually loaded, not from the request: a
    # caller may hand back a streaming source for any reason.  A streaming
    # source has no index space, so it cannot be sized or column-scanned.
    self.streaming = is_streaming_source(ds) or not is_random_access(ds)
    if not self.streaming and len(ds) == 0:
      fatal(f"Dataset '{name}' (split={split}) has 0 samples", ValueError)
    col = self._image_column or HFDatasetProxy.detect_image_column(ds)
    if col is None:
      fatal(
          f"No image column detected in '{name}'. "
          f"Columns: {ds.column_names}. Set 'image_column' explicitly.",
          ValueError,
      )
    self._image_column = col
    ds = HFDatasetProxy.normalize_image_column(ds, self._image_column)
    self._ds = maybe_cap(ds, max_items)
    self._detect_labels(label_column)
    if self.streaming:
      logging.info(f"  '{name}': streaming samples "
                   f"(image column: '{self._image_column}')")
    else:
      logging.info(f"  '{name}': {len(self._ds):,} samples "
                   f"(image column: '{self._image_column}')")

  def _detect_labels(self, label_column=None):
    """Detect and normalize the label column, if any."""

    import datasets as ds_lib

    self._label_names = []
    self._label2id = {}
    self._label_col = label_column
    if self._label_col is None:
      self._label_col = HFDatasetProxy.detect_label_column(self._ds)
    self._build_column_map()
    if self._label_col is None:
      return
    if self._label_col not in self._ds.column_names:
      fatal(
          f"Label column '{self._label_col}' is not present. "
          f"Available columns: {self._ds.column_names}",
          ValueError,
      )
    if self.streaming:
      # A stream has no index space, so the column cannot be scanned to build
      # the label vocabulary.  Labels are therefore integer ids already (the
      # only case that needs no vocabulary); anything else cannot be mapped.
      feat = getattr(self._ds, "features", None)
      feat = feat[self._label_col] if feat else None
      if not isinstance(feat, ds_lib.ClassLabel):
        fatal(
            f"Dataset '{self.name}': a streaming source needs a ClassLabel "
            f"column to map labels (found {type(feat).__name__} for column "
            f"'{self._label_col}'). Pre-map the labels to a ClassLabel, or "
            f"drop --label_column to train without labels.", ValueError)
      self._label_names = feat.names
      self._label2id = {name: i for i, name in enumerate(self._label_names)}
      logging.info(f"  '{self.name}': {len(self._label_names)} classes "
                   f"(label column: '{self._label_col}', streaming)")
      return
    feat = self._ds.features[self._label_col]
    if isinstance(feat, ds_lib.ClassLabel):
      self._label_names = feat.names
      self._label2id = {name: i for i, name in enumerate(self._label_names)}
    else:
      self._label_names = sorted(set(self._ds[self._label_col]))
      self._label2id = {name: i for i, name in enumerate(self._label_names)}
    logging.info(f"  '{self.name}': {len(self._label_names)} classes "
                 f"(label column: '{self._label_col}')")

  @property
  def has_labels(self):
    return self._label_col is not None

  @property
  def num_labels(self):
    return len(self._label_names)

  @property
  def label_names(self):
    return list(self._label_names)

  @property
  def column_map(self):
    """``{SOURCE_COL: ENSEMBLE_COL}`` mapping applied by ``__getitem__``."""
    return dict(self._column_map)

  def _build_column_map(self):
    """Build the source-to-ensemble output column mapping.

        The ensemble unilaterally names its outputs ``"image"`` and
        ``"label"``; sub-datasets remap their own column names onto
        those keys.
        """
    column_map = {self._image_column: "image"}
    if self._label_col is not None:
      column_map[self._label_col] = "label"
    self._column_map = column_map

  @property
  def labels_array(self):
    """Numpy array of dataset-local integer labels for all samples."""
    if self._label_col is None:
      return np.array([], dtype=np.int64)
    raw = np.asarray(self._ds[self._label_col])
    mapped = []
    for value in raw:
      if isinstance(value, str):
        if value not in self._label2id:
          fatal(
              f"Dataset '{self.name}': unknown label value {value!r} in "
              f"column '{self._label_col}'.  Known labels: "
              f"{sorted(self._label2id)}", ValueError)
        mapped.append(self._label2id[value])
      else:
        label_id = int(value)
        if label_id < 0 or label_id >= len(self._label_names):
          fatal(
              f"Dataset '{self.name}': local label id {label_id} is out "
              f"of range for {len(self._label_names)} classes", ValueError)
        mapped.append(label_id)
    return np.array(mapped, dtype=np.int64)

  def __len__(self):
    if self.streaming:
      raise TypeError("streaming image dataset has no length")
    return len(self._ds)

  def _convert(self, row):
    """Build the canonical ``{image, label}`` item for one raw row."""
    image = row[self._image_column]
    if not isinstance(image, Image.Image):
      image = Image.open(image)
    image = image.convert("RGB")
    result = {self._image_column: image}
    if self._label_col is not None:
      raw_label = row[self._label_col]
      result[self._label_col] = self._label2id.get(raw_label, raw_label)
    return {self._column_map[k]: v for k, v in result.items()}

  def __getitem__(self, idx):
    # ``getitem_retry`` retries a transient read failure on another index; it
    # needs the length, which a streaming source does not have (and which never
    # reaches here: a stream is read through ``__iter__``).
    item, _ = getitem_retry(idx, lambda i: self._convert(self._ds[i]), len(self._ds))
    return item

  def __iter__(self):
    """Iterate the underlying source, converting one row at a time.

        Works for both a random-access dataset and a streaming source; the latter
        has no index, so this is the only way to read it.
        """
    for row in self._ds:
      yield self._convert(row)


def _first_sample_image_member(source):
  """Name of the first image-typed member in *source*'s first sample.

    webdataset names members by extension, so this discovers which one holds the
    picture without the caller having to name it.  Reading one sample is cheap
    and the source is re-iterable (each pass re-opens the shards), so this does
    not consume the stream.

    Args:
      source: An iterable of ``{member_name: value}`` samples.

    Returns:
      str: The image member's name.

    Raises:
      ValueError: If the first sample holds no image-typed member.
    """
  for member, value in next(iter(source)).items():
    if isinstance(value, Image.Image):
      return member
  raise fatal(
      "webdataset: the first sample holds no image member.  Pass "
      "--image_column naming the member that holds the image.", ValueError)


class DatasetEnsemble:
  """Concatenation of multiple image datasets.

  Images are returned as dicts with at least an ``"image"`` key
  (a :class:`PIL.Image.Image`).  When any dataset provides labels,
  a ``"label"`` key (integer) is also present.

  If every member is streaming, the ensemble is a stream too: it is iterated
  (concatenation order) and has no length.  Mixing the two kinds is rejected,
  since a stream has no index space for a flat random-access walk.
  """

  #: False for an all-random-access ensemble; set in ``__init__`` when any
  #: member streams.
  streaming = False

  def __init__(self,
               dataset_configs,
               cache_dir=None,
               hf_token=None,
               strict=False,
               max_items=None):
    self._datasets = []
    self._max_items = max_items
    self._offsets = None
    self._global_label_names = []
    self._global_label2id = {}

    for cfg in dataset_configs:
      name = cfg["name"]
      source = cfg.get("source", "hf")
      try:
        if source == "hf":
          hf_kwargs = {}
          if cfg.get("dataset_arg"):
            hf_kwargs["dataset_arg"] = cfg["dataset_arg"]
          ds = _HFDataset(
              name=name,
              split=cfg.get("split", "train"),
              image_column=cfg.get("image_column"),
              label_column=cfg.get("label_column"),
              cache_dir=cache_dir,
              hf_token=hf_token,
              max_items=self._max_items,
              **hf_kwargs,
          )
        elif source == "imagefolder":
          ds = ImageFolderDataset(root_dir=name)
        elif source == "webdataset":
          # The spec's payload is a shard list; ``WebDatasetSource`` yields whole
          # samples and streams, so this member makes the ensemble a stream.
          from genml_kit.datasets.field_selector import \
              IterableFieldSelectorDataset
          from genml_kit.datasets.webdataset_source import WebDatasetSource
          source_ds = WebDatasetSource(name,
                                       name=f"webdataset:{name}",
                                       image_format=cfg.get("image_format", "RGB"))
          # webdataset keys members by *extension* (``png``, ``jpg``, ...), so the
          # image member has to be renamed onto the ensemble's canonical key.
          # ``--image_column`` names that member explicitly; without it, the
          # first image-typed member of the first sample wins.
          image_member = (cfg.get("image_column") or
                          _first_sample_image_member(source_ds))
          ds = IterableFieldSelectorDataset(source_ds, {image_member: "image"})
        else:
          logging.warning(f"Unknown source '{source}' for dataset '{name}', skipping")
          continue
        if is_random_access(ds):
          n = len(ds)
          if n == 0:
            logging.warning(f"Dataset '{name}' has 0 images, skipping")
            continue
          logging.info(f"  + {name}: {n:,} images")
        else:
          # A stream cannot be sized, so a zero-length check is not possible (and
          # would consume the source to find out).
          logging.info(f"  + {name}: streaming (unknown size)")
        self._datasets.append(ds)
      except Exception:
        if strict:
          fatal(f"Failed to initialize dataset '{name}' ({source})", RuntimeError)
        logging.exception(
            "  Failed to initialize dataset '%s' (%s); skipping.",
            name,
            source,
        )

    if not self._datasets:
      fatal("No datasets loaded successfully", RuntimeError)

    # A source may be a one-pass stream, which has no index space.  The two
    # kinds cannot be concatenated: a stream has no length to build offsets
    # from, so it cannot precede a random-access member in a flat index.  If any
    # member streams, the whole ensemble streams (concatenation order), which is
    # the only mode that works without lengths.
    streamed = [ds for ds in self._datasets if not is_random_access(ds)]
    if streamed and len(streamed) != len(self._datasets):
      fatal(
          "Cannot mix streaming and random-access datasets in one ensemble: a "
          f"stream has no index space.  Streaming: "
          f"{[ds.name for ds in streamed]}.", ValueError)
    self.streaming = bool(streamed)

    if self.streaming:
      # No offsets and no global label space: both are built from lengths and
      # column scans, which a stream does not have.
      if self.has_labels:
        self._build_global_label_space()
      return

    self._build_offsets()

    if self.has_labels:
      self._build_global_label_space()

  def _build_offsets(self):
    """Compute cumulative offsets for flat indexing."""
    offsets = [0]
    for ds in self._datasets:
      offsets.append(offsets[-1] + len(ds))
    self._offsets = offsets

  @property
  def image_column(self):
    """Canonical image key emitted by every sub-dataset."""
    return "image"

  @property
  def label_column(self):
    """Canonical label key, or ``None`` when nothing is labeled."""
    return "label" if self.has_labels else None

  @property
  def unlabeled_datasets(self):
    """Names of sub-datasets that provide no labels."""
    return [ds.name for ds in self._datasets if not ds.has_labels]

  def _build_global_label_space(self):
    """Map per-dataset label names to a shared integer space.

        Idempotent: rebuilding is a no-op once the global space exists.
        """
    if not self._global_label_names:
      all_names = set()
      for ds in self._datasets:
        if ds.has_labels:
          all_names.update(ds.label_names)
      self._global_label_names = sorted(all_names)
      self._global_label2id = {
          name: i for i, name in enumerate(self._global_label_names)
      }

  def ensure_label_space(self):
    """Build the global label space across labeled sub-datasets.

        Mixed ensembles (some sources labeled, some not) are supported:
        unlabeled datasets are skipped with a warning.  Public entry
        point for callers that iterate a label-requiring dataset outside
        :meth:`__init__`.  Idempotent, so it is cheap to call
        defensively.
        """
    unlabeled = self.unlabeled_datasets
    if unlabeled:
      logging.warning(
          "ensure_label_space: %d of %d datasets provide no labels and "
          "will be skipped for label-based sampling: %s", len(unlabeled),
          len(self._datasets), ", ".join(unlabeled))
    self._build_global_label_space()

  def _remap_label(self, ds, local_label):
    """Convert a dataset-local label id to the global integer id."""
    if isinstance(local_label, bool) or not isinstance(local_label, (int, np.integer)):
      fatal(
          f"Dataset '{ds.name}': expected an integer label id, got "
          f"{local_label!r} ({type(local_label).__name__})", ValueError)
    label_id = int(local_label)
    if label_id < 0 or label_id >= len(ds.label_names):
      fatal(
          f"Dataset '{ds.name}': local label id {label_id} is out of "
          f"range for {len(ds.label_names)} classes", ValueError)
    return self._global_label2id[ds.label_names[label_id]]

  @property
  def has_labels(self):
    return any(ds.has_labels for ds in self._datasets)

  @property
  def num_labels(self):
    if not self.has_labels:
      return 0
    return len(self._global_label_names)

  @property
  def label_names(self):
    if not self.has_labels:
      return []
    return list(self._global_label_names)

  @property
  def label2id(self):
    """Mapping of global label name to global integer id."""
    return dict(self._global_label2id)

  @property
  def labels_array(self):
    """Global integer labels aligned with ``__getitem__`` positions.

        Requires every sub-dataset to provide labels; fatals otherwise
        (use :attr:`unlabeled_datasets` to inspect mixed ensembles).
        """
    unlabeled = self.unlabeled_datasets
    if unlabeled:
      fatal(
          "DatasetEnsemble.labels_array requires every dataset to provide "
          f"labels; these do not: {', '.join(unlabeled)}", ValueError)
    if self.streaming:
      fatal(
          "labels_array requires a materialisable index space; a streaming "
          "ensemble has none.", ValueError)
    parts = []
    for ds in self._datasets:
      local = ds.labels_array
      global_ids = np.array(
          [self._global_label2id[ds.label_names[int(lbl)]] for lbl in local],
          dtype=np.int64,
      )
      parts.append(global_ids)
    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

  def __len__(self):
    if self.streaming:
      raise TypeError("streaming ensemble has no length")
    return self._offsets[-1] if self._offsets else 0

  def __iter__(self):
    """Concatenate every member's items, in declaration order.

        The streaming analogue of ``__getitem__``'s flat index walk: each member
        is read to exhaustion before the next begins, which is the same order a
        random-access ensemble produces.  A per-member label remap is applied so
        the emitted items look identical either way.
        """
    for ds in self._datasets:
      for item in ds:
        if self.has_labels and "label" in item:
          item["label"] = self._remap_label(ds, item["label"])
        yield item

  def _get_item(self, idx):
    """Resolve a valid global index and load its image (and label)."""
    ds_idx = bisect.bisect_right(self._offsets, idx) - 1
    local_idx = idx - self._offsets[ds_idx]
    ds = self._datasets[ds_idx]
    item = ds[local_idx]
    if self.has_labels and "label" in item:
      item["label"] = self._remap_label(ds, item["label"])
    return item

  def __getitem__(self, idx):
    if idx < 0 or idx >= len(self):
      fatal(
          f"Index {idx} out of range for ensemble of length {len(self)}",
          IndexError,
      )
    return self._get_item(idx)

  def summary(self):
    """Return a human-readable summary of the ensemble."""
    if self.streaming:
      head = (f"DatasetEnsemble: streaming, "
              f"{len(self._datasets)} dataset(s) of unknown size")
    else:
      head = (f"DatasetEnsemble: {len(self):,} images from "
              f"{len(self._datasets)} dataset(s)")
    lines = [head]
    for i, ds in enumerate(self._datasets):
      tag = "streaming" if self.streaming else f"{len(ds):,} images"
      if ds.has_labels:
        tag += f", {ds.num_labels} classes"
      else:
        tag += ", NO LABELS"
      lines.append(f"  [{i}] {ds.name}: {tag}")
    if self.has_labels:
      lines.append(f"  Global label space: {self.num_labels} classes")
    return "\n".join(lines)
