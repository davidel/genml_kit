"""Bridge HuggingFace datasets to PyTorch DataLoader format."""

import logging

import datasets
import numpy as np
from PIL import Image

from genml_kit.utils.logging import fatal


def _is_array_feature(feat):
  """Return True for pre-decoded array image features.

  Covers ``datasets.Sequence(Value(\"uint8\"))`` (possibly nested) and
  ``datasets.Array3D``, which hold raw image arrays rather than file
  references.
  """
  if isinstance(feat, datasets.Array3D):
    return True
  if isinstance(feat, datasets.Sequence):
    inner = feat.feature
    return _is_array_feature(inner) if isinstance(inner, type(feat)) else isinstance(
        inner, datasets.Value)
  return False


class HFDatasetProxy:
  """Wrap a HF ``Dataset`` for use with PyTorch ``DataLoader``.

    Auto-detects image and label columns, normalizes labels to
    ``ClassLabel``, and converts filepath strings to decoded PIL images.
    """

  KNOWN_IMAGE_NAMES = frozenset({
      "image",
      "img",
      "pixels",
      "pixel_values",
      "image_array",
      "image_file",
      "image_url",
      "file_name",
      "path",
  })

  KNOWN_LABEL_NAMES = frozenset({
      "label",
      "labels",
      "class",
      "classes",
      "category",
      "categories",
      "species",
      "breed",
      "object",
  })

  IGNORE_LABEL_NAMES = frozenset({
      "image_path",
      "file_path",
      "image_id",
      "id",
      "filename",
      "file",
      "caption",
      "text",
      "bbox",
      "bounding_box",
      "segmentation",
      "mask",
      "width",
      "height",
      "channel",
  })

  MASK_COLUMN_NAMES = frozenset({
      "mask",
      "masks",
      "label",
      "labels",
      "annotation",
      "annotations",
      "segmentation",
      "segmentations",
      "seg_mask",
      "ground_truth",
      "gt",
      "target",
  })

  def __init__(self,
               hf_dataset,
               transform=None,
               image_column=None,
               label_column=None,
               mask_column=None,
               mask_map=None):
    self._image_col = image_column or self.detect_image_column(hf_dataset)
    self.dataset = self.normalize_image_column(hf_dataset, self._image_col)
    self._mask_col = mask_column
    self._label_col = None
    self._mask_map = mask_map

    if self._mask_col is not None:
      # A mask column makes this a segmentation source: normalise it as an
      # image feature and skip label detection entirely.
      self.dataset = self.normalize_image_column(self.dataset, self._mask_col)
    else:
      self._label_col = label_column or self.detect_label_column(self.dataset)
      self.dataset = self.normalize_labels(self.dataset, self._label_col)

    self._transform = transform

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item[self._image_col]
    if hasattr(image, "convert"):
      image = image.convert("RGB")

    if self._mask_col is not None:
      # Segmentation mode: return a dict so the paired transform can
      # co-process image and mask.  (The classification path keeps the
      # ``(image, label)`` tuple contract.)
      mask = item[self._mask_col]
      arr = mask if isinstance(mask, np.ndarray) else np.asarray(mask)
      # A mask_map is provided when the pipeline compacted a sparse or
      # RGB mask: remap raw pixel values / RGB triplets to dense ids.
      if self._mask_map is not None:
        arr = self._apply_mask_map(arr, self._mask_map)
      elif arr.ndim == 3:
        # No map (standalone proxy): collapse RGB to single-channel.
        arr = np.asarray(Image.fromarray(arr).convert("L"), dtype=np.int64)
      else:
        arr = arr.astype(np.int64)
      mask = arr
      if self._transform is not None:
        return self._transform({"image": image, "mask": mask})
      return {"image": image, "mask": mask}

    label = item[self._label_col]
    if self._transform is not None:
      image = self._transform(image)
    return image, label

  @staticmethod
  def _apply_mask_map(arr, mask_map):
    """Remap raw mask pixel values / RGB triplets to compact ids.

    Args:
        arr: ``(H, W)`` single-channel or ``(H, W, 3)`` RGB mask array.
        mask_map: ``{raw_key: compact_id}`` where *raw_key* is either an
            int (single-channel) or a 3-tuple (RGB triplet).

    Returns:
        ``(H, W)`` int64 array with compact ids.

    Raises:
        ValueError: if a raw value/triplet is missing from *mask_map*
            (a val-only id that never appeared in the training scan).
    """
    if arr.ndim == 3:
      h, w, _ = arr.shape
      flat_keys = arr.reshape(-1, 3)
      out = np.empty(flat_keys.shape[0], dtype=np.int64)
      for i in range(flat_keys.shape[0]):
        key = tuple(int(v) for v in flat_keys[i])
        if key not in mask_map:
          fatal(f"Mask value {key} not in the class map (val-only id?)", ValueError)
        out[i] = mask_map[key]
      return out.reshape(h, w)
    out = np.empty(arr.size, dtype=np.int64)
    flat = arr.ravel()
    for i in range(flat.size):
      raw = int(flat[i])
      if raw not in mask_map:
        fatal(f"Mask value {raw} not in the class map (val-only id?)", ValueError)
      out[i] = mask_map[raw]
    return out.reshape(arr.shape)

  @staticmethod
  def detect_mask_column(dataset):
    """Return the name of the mask column, or ``None``.

        Detection priority:
          1. Column named ``"mask"`` with ``datasets.Image`` feature.
          2. Any column with ``datasets.Image`` feature and a known mask
             name.
          3. Column matching a known mask name.
        """
    features = dataset.features

    if "mask" in features and isinstance(features["mask"], datasets.Image):
      return "mask"

    for name in features:
      if (name in HFDatasetProxy.MASK_COLUMN_NAMES and
          isinstance(features[name], datasets.Image)):
        return name

    for name in HFDatasetProxy.MASK_COLUMN_NAMES:
      if name in features:
        return name

  @property
  def mask_column(self):
    """Return the mask column name, or ``None`` for non-segmentation data."""
    return self._mask_col

  @property
  def has_labels(self):
    """Return True when a label column was detected."""
    return self._label_col is not None

  @property
  def label_names(self):
    """Return the list of class names from the ``label`` feature."""
    if self._label_col is None:
      return []
    feat = self.dataset.features[self._label_col]
    if isinstance(feat, datasets.ClassLabel):
      return feat.names
    return sorted(set(self.dataset[self._label_col]))

  @property
  def num_labels(self):
    """Return the number of classes from the ``label`` feature."""
    return len(self.label_names)

  @property
  def label2id(self):
    """Return a ``{name: str(i)}`` mapping."""
    return {name: str(i) for i, name in enumerate(self.label_names)}

  @property
  def id2label(self):
    """Return a ``{str(i): name}`` mapping."""
    return {str(i): name for i, name in enumerate(self.label_names)}

  @property
  def label_column(self):
    """Return the name of the column used for labels."""
    return self._label_col

  @staticmethod
  def detect_image_column(dataset):
    """Return the name of the image column, or ``None``.

        Detection priority:
          1. Column with ``datasets.Image`` feature and a known name.
          2. Any column with ``datasets.Image`` feature.
          3. Column with a known image name and string feature (file paths).
        """
    features = dataset.features

    for name in HFDatasetProxy.KNOWN_IMAGE_NAMES:
      if name in features and isinstance(features[name], datasets.Image):
        return name

    for name, feat in features.items():
      if isinstance(feat, datasets.Image):
        return name

    for name in HFDatasetProxy.KNOWN_IMAGE_NAMES:
      if name in features:
        feat = features[name]
        if isinstance(feat, datasets.Value) and feat.dtype == "string":
          return name

    # Array features (e.g. pixel_values as Sequence(Value('uint8')) or
    # Array3D) are pre-decoded image tensors; a known image name is enough.
    for name in HFDatasetProxy.KNOWN_IMAGE_NAMES:
      if name in features:
        feat = features[name]
        if _is_array_feature(feat):
          return name

  @staticmethod
  def normalize_image_column(dataset, image_column=None):
    """Cast the selected image column to ``datasets.Image`` when needed.

    Args:
      dataset: Hugging Face dataset to normalize.
      image_column: Optional explicit image column. When omitted, the image
        column is detected automatically.
    """
    image_col = image_column or HFDatasetProxy.detect_image_column(dataset)
    if image_col is None:
      return dataset
    if image_col not in dataset.features:
      fatal(
          f"Image column '{image_col}' is not present. "
          f"Available columns: {dataset.column_names}", ValueError)
    feat = dataset.features[image_col]
    if isinstance(feat, datasets.Value) and feat.dtype == "string":
      dataset = dataset.cast_column(image_col, datasets.Image())
    elif _is_array_feature(feat):
      # Pre-decoded array (e.g. pixel_values): convert to Image so the
      # downstream code sees a PIL-decodable column (H,W[,C]).
      def _to_image(arr):
        arr = np.asarray(arr)
        if arr.dtype != np.uint8:
          arr = arr.astype(np.uint8)
        return Image.fromarray(arr)

      dataset = dataset.map(lambda row: {image_col: _to_image(row[image_col])})
    return dataset

  @staticmethod
  def normalize_labels(dataset, label_column=None):
    """Cast the selected label column to ClassLabel."""
    label_col = label_column or HFDatasetProxy.detect_label_column(dataset)
    if label_col is not None and label_col not in dataset.features:
      fatal(
          f"Label column '{label_col}' is not present. "
          f"Available columns: {dataset.column_names}", ValueError)
    if label_col is None:
      return dataset
    if not isinstance(dataset.features[label_col], datasets.ClassLabel):
      dataset = dataset.class_encode_column(label_col)
    return dataset

  @staticmethod
  def detect_label_column(dataset):
    """Return the name of the label column, or ``None``.

        Detection priority:
          1. Column named ``"label"`` with ``ClassLabel`` feature.
          2. Any column with ``ClassLabel`` feature (prefer known names).
          3. Column matching a known label name with string/int feature.
          4. First string/int column that is not an ignored name.

        .. note::

           Priority 4 is a heuristic fallback: when no known label name
           matches, the first non-ignored string/int64 column wins, which
           can silently pick a metadata column (e.g. ``patient_id``).  A
           warning is logged whenever this fallback triggers.
        """
    cols = dataset.column_names
    features = dataset.features

    if "label" in cols and isinstance(features["label"], datasets.ClassLabel):
      return "label"

    classlabel_cols = [
        name for name in cols if isinstance(features[name], datasets.ClassLabel)
    ]
    if classlabel_cols:
      for name in classlabel_cols:
        if name in HFDatasetProxy.KNOWN_LABEL_NAMES:
          return name
      logging.warning(
          "No known label column name found; using first ClassLabel "
          "column '%s' of %s. Pass label_column explicitly to override.",
          classlabel_cols[0],
          classlabel_cols,
      )
      return classlabel_cols[0]

    for name in cols:
      if (name in HFDatasetProxy.KNOWN_LABEL_NAMES and
          name not in HFDatasetProxy.IGNORE_LABEL_NAMES):
        feat = features[name]
        if isinstance(feat, datasets.Value) and feat.dtype in (
            "string",
            "int64",
        ):
          return name

    fallback = None
    for name in cols:
      if name in HFDatasetProxy.IGNORE_LABEL_NAMES:
        continue
      feat = features[name]
      if isinstance(feat, datasets.Value) and feat.dtype in ("string", "int64"):
        fallback = name
        break
    if fallback is not None:
      logging.warning(
          "No known label column detected in dataset with columns %s; "
          "falling back to first string/int64 column '%s'. If this is "
          "not the label (e.g. an ID or patient column), pass "
          "--label_column explicitly.",
          cols,
          fallback,
      )
    return fallback
