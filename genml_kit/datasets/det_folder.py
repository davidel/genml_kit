"""DetFolderDataset -- COCO-JSON object-detection corpus on disk.

Layout expected::

    <root>/images/<file_name>
    <root>/annotations.json

or, when the corpus has named splits::

    <root>/<split>/images/<file_name>
    <root>/<split>/annotations.json

``annotations.json`` follows the COCO object-detection schema: an
``images`` list (``id`` / ``file_name``), an ``annotations`` list
(``image_id`` / ``bbox`` / ``category_id``) and a ``categories`` list.
``bbox`` uses the COCO ``[x, y, width, height]`` top-left convention; the
dataset converts it to the ``xyxy`` convention the rest of the harness
(box transforms, IoU, NMS, mAP) uses, so *every* downstream box tensor is
``xyxy`` in absolute input-image pixels.

Raw category ids (COCO's are 1-based and sparse) are compacted to a dense
``[0, C)`` label space -- exactly as :class:`SegFolderDataset` compacts
mask ids -- so the model head's width equals the number of distinct
categories actually present, and the class-id -> name map stays
recoverable through :meth:`DetFolderDataset.id2label`.
"""

import json
import logging
from pathlib import Path

import torch
from PIL import Image

from genml_kit.datasets.retry import getitem_retry
from genml_kit.utils.logging import fatal

_SPLIT_NAMES = ("train", "val", "valid", "validation", "test")


class DetFolderDataset:
  """COCO-JSON object-detection dataset backed by a single JSON index.

  Parameters
  ----------
  root_dir : str | Path
      Root containing ``images/`` and ``annotations.json`` (or the split
      dir that contains them).
  annotation_file : str
      Name of the COCO JSON file inside *root_dir*.
  image_subdir : str
      Sub-directory holding the image files referenced by ``file_name``.

  Raises
  ------
  FileNotFoundError, ValueError
      Via :func:`fatal` when the directories or JSON are missing, or the
      JSON is malformed.
  """

  def __init__(self,
               root_dir,
               annotation_file="annotations.json",
               image_subdir="images"):
    self.name = str(root_dir)
    self._root = Path(root_dir)
    self._image_dir = self._root / image_subdir
    self._annotation_path = self._root / annotation_file

    if not self._root.is_dir():
      fatal(f"DetFolderDataset: '{self._root}' is not a directory", ValueError)
    if not self._image_dir.is_dir():
      fatal(
          f"DetFolderDataset: image dir '{self._image_dir}' does not exist",
          FileNotFoundError,
      )
    if not self._annotation_path.is_file():
      fatal(
          f"DetFolderDataset: annotation file '{self._annotation_path}' does "
          "not exist (expected a COCO-JSON index)",
          FileNotFoundError,
      )

    self._records, self._id2label = self._parse()

  def _parse(self):
    """Read the COCO JSON and build per-image box/label records.

    Returns
    -------
    (list, dict)
      ``records`` is a list of ``(image_path, boxes, labels)`` where
      ``boxes`` is ``(N, 4)`` float32 ``xyxy`` and ``labels`` is ``(N,)``
      int64 in the compacted ``[0, C)`` space.  ``id2label`` maps a compact
      label to its category name.
    """
    try:
      with open(self._annotation_path, encoding="utf-8") as handle:
        coco = json.load(handle)
    except json.JSONDecodeError as exc:
      fatal(
          f"DetFolderDataset: '{self._annotation_path}' is not valid JSON: {exc}",
          ValueError,
      )

    images = coco.get("images")
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not isinstance(images, list) or not images:
      fatal(
          f"DetFolderDataset: '{self._annotation_path}' has no 'images' list",
          ValueError,
      )

    # Build the {raw category id -> compact label} map from the categories
    # table when present, else from the ids actually used by annotations.
    # Sorting keeps the compaction deterministic across runs (and machines),
    # which matters because the checkpoint stores num_labels derived from it.
    raw_ids = [int(cat["id"]) for cat in categories if "id" in cat]
    if not raw_ids:
      raw_ids = sorted(
          {int(ann["category_id"]) for ann in annotations if "category_id" in ann})
    raw_ids = sorted(set(raw_ids))
    raw_to_compact = {raw: idx for idx, raw in enumerate(raw_ids)}
    cat_names = {
        int(cat["id"]): str(cat.get("name", f"class_{cat['id']}"))
        for cat in categories
        if "id" in cat
    }
    id2label = {
        idx: cat_names.get(raw, f"class_{raw}") for raw, idx in raw_to_compact.items()
    }

    # Group annotations by image_id so each image is parsed exactly once.
    # A missing image_id or category_id would otherwise raise a bare KeyError
    # deep in a worker; surface it as an actionable fatal() instead.
    by_image = {}
    for ann in annotations:
      if "image_id" not in ann or "category_id" not in ann or "bbox" not in ann:
        fatal(
            f"DetFolderDataset: annotation missing image_id/category_id/bbox "
            f"in '{self._annotation_path}'",
            ValueError,
        )
      by_image.setdefault(int(ann["image_id"]), []).append(ann)

    records = []
    for entry in images:
      if "id" not in entry or "file_name" not in entry:
        fatal(
            f"DetFolderDataset: image entry missing id/file_name in "
            f"'{self._annotation_path}'",
            ValueError,
        )
      image_path = self._image_dir / entry["file_name"]
      if not image_path.is_file():
        fatal(
            f"DetFolderDataset: image '{image_path}' referenced by the "
            "annotation file does not exist",
            FileNotFoundError,
        )
      boxes, labels = self._boxes_for(by_image.get(int(entry["id"]), []),
                                      raw_to_compact)
      records.append((image_path, boxes, labels))

    if not records:
      fatal(f"DetFolderDataset: no images found in '{self._annotation_path}'",
            ValueError)
    logging.info("DetFolderDataset: %d images, %d classes (%s)", len(records),
                 len(raw_to_compact), self.name)
    return records, id2label

  @staticmethod
  def _boxes_for(anns, raw_to_compact):
    """Convert one image's COCO annotations into ``xyxy`` boxes + labels.

    COCO ``bbox`` is ``[x, y, w, h]`` with a top-left origin; the harness
    wants ``[x1, y1, x2, y2]``, so ``x2 = x + w`` and ``y2 = y + h``.
    Degenerate boxes (``w`` or ``h`` <= 0) are dropped: they carry no area,
    so they can never be matched by IoU and would only pollute the loss.

    Returns
    -------
    (torch.Tensor, torch.Tensor)
      ``(N, 4)`` float32 ``xyxy`` boxes and ``(N,)`` int64 compact labels.
    """
    boxes = []
    labels = []
    for ann in anns:
      x, y, w, h = (float(v) for v in ann["bbox"])
      if w <= 0.0 or h <= 0.0:
        continue
      boxes.append([x, y, x + w, y + h])
      labels.append(raw_to_compact[int(ann["category_id"])])
    if not boxes:
      # An image with no (usable) boxes is legal -- it is a pure background
      # sample and still contributes negatives to the classification loss.
      return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,),
                                                                   dtype=torch.long)
    return (torch.tensor(boxes,
                         dtype=torch.float32), torch.tensor(labels, dtype=torch.long))

  @property
  def id2label(self):
    """Return the ``{compact label -> category name}`` map."""
    return dict(self._id2label)

  @property
  def num_classes(self):
    """Return the number of distinct (compacted) categories."""
    return len(self._id2label)

  def __len__(self):
    return len(self._records)

  def __getitem__(self, idx):
    n = len(self._records)
    if idx < 0:
      idx += n
    if idx < 0 or idx >= n:
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)

    def load(i):
      path, boxes, labels = self._records[i]
      # .convert("RGB") guarantees 3 channels even for grayscale / palette
      # JPEGs, so the encoder always receives a consistent (C, H, W) input.
      image = Image.open(path).convert("RGB")
      # Clone so a later in-place box transform cannot mutate the cached
      # record (transforms run in DataLoader workers over shared memory).
      return {"image": image, "boxes": boxes.clone(), "labels": labels.clone()}

    row, _ = getitem_retry(idx, load, n)
    return row


def _has_named_splits(root):
  """Return True when *root* holds ``train``/``val``/``test`` sub-dirs."""
  return any(child.is_dir() and child.name in _SPLIT_NAMES for child in root.iterdir())


def load_detfolder(root, annotation_file="annotations.json", split=None):
  """Load a COCO-JSON detection corpus, returning a dict of datasets.

  Two layouts are supported:

  - **Split layout** -- first-level sub-directories are ``train`` / ``val``
    / ``test``, each containing ``images/`` and ``annotations.json``.
  - **Flat layout** -- *root* itself contains ``images/`` and
    ``annotations.json``; returned as ``{"train": <dataset>}``.

  Args:
    root: Path to the dataset root directory.
    annotation_file: COCO JSON file name inside each split dir.
    split: If given, return only the dataset for that split.

  Returns:
    A dict mapping split name to :class:`DetFolderDataset`.
  """
  root = Path(root)
  if not root.is_dir():
    fatal(f"load_detfolder: '{root}' is not a directory", ValueError)

  if _has_named_splits(root):
    splits = {}
    for child in sorted(root.iterdir()):
      if child.is_dir() and child.name in _SPLIT_NAMES:
        # Normalise the "valid"/"validation" spelling to the canonical
        # "val" key the rest of the harness expects.
        key = "val" if child.name in ("valid", "validation") else child.name
        splits[key] = DetFolderDataset(child, annotation_file=annotation_file)
    if split is not None:
      if split not in splits:
        fatal(
            f"load_detfolder: split '{split}' not found in '{root}' "
            f"(available: {sorted(splits)})", ValueError)
      return {split: splits[split]}
    return splits

  ds = DetFolderDataset(root, annotation_file=annotation_file)
  if split is not None and split != "train":
    fatal(
        f"load_detfolder: split '{split}' not found in '{root}' "
        f"(available: ['train'])", ValueError)
  return {"train": ds}
