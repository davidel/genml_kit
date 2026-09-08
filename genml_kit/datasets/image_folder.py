"""ImageFolderDataset — fast local-image dataset backed by Path.rglob()."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from genml_kit.datasets.retry import getitem_retry
from genml_kit.utils.logging import fatal

_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".png",
    ".apng",
    ".gif",
    ".bmp",
    ".dib",
    ".tiff",
    ".tif",
    ".webp",
    ".avif",
    ".avifs",
    ".jp2",
    ".j2k",
    ".mpo",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".tga",
    ".icb",
    ".vda",
    ".vst",
    ".dds",
    ".pcx",
    ".ico",
    ".xbm",
    ".xpm",
    ".psd",
}

# First-level directory names recognised as split folders.
_SPLIT_NAMES = frozenset({"train", "val", "test"})


def _is_split_dir(root):
  """Return True if *root* contains split sub-directories."""
  if (root / ".splits").exists():
    return True
  return any(child.is_dir() and child.name in _SPLIT_NAMES for child in root.iterdir())


class ImageFolderDataset:
  """Fast local-image dataset backed by ``Path.rglob()``.

  Labels are auto-detected from the directory structure relative to
  *root_dir*.  The scanner considers the depth of each image and
  whether the first-level sub-directories are recognised splits:

  Depth 1 (``root/file.jpg``)
    No labels.  Flat layout.

  Depth 2 (``root/X/file.jpg``)
    If *X* is a recognised split name (``train`` / ``val`` / ``test``)
    **or** a ``.splits`` marker file exists in *root_dir*, the images
    are treated as a split folder with **no labels**.

    Otherwise the first-level directories are treated as class labels
    (``root/class/file.jpg``).

  Depth ≥ 3 (``root/split/class/file.jpg``)
    Always treated as split/label/file — the second path component is
    the class label.

  Mixed depths within a single root are not allowed and trigger a
  fatal error.  Non-existent paths and empty directories also trigger
  a fatal error.

  ``__getitem__`` always returns a dict:

  - With labels: ``{"image": PIL.Image, "label": int}``
  - Without labels: ``{"image": PIL.Image}``
  """

  def __init__(self, root_dir):
    self.name = str(root_dir)
    self._root_dir = Path(root_dir)
    self._label_names = []
    self._label2id = {}
    self._labels = None
    self._paths = []

    if not self._root_dir.exists():
      fatal(f"ImageFolderDataset: '{self._root_dir}' does not exist", FileNotFoundError)
    if not self._root_dir.is_dir():
      fatal(f"ImageFolderDataset: '{self._root_dir}' is not a directory", ValueError)

    self._scan()

  def _scan(self):
    all_paths = sorted(p for p in self._root_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    if not all_paths:
      fatal(
          f"No images found under '{self._root_dir}' "
          f"(extensions checked: {_IMAGE_EXTS})", FileNotFoundError)

    depth_map = {}  # depth → list[Path]
    for p in all_paths:
      depth = len(p.relative_to(self._root_dir).parts)
      depth_map.setdefault(depth, []).append(p)

    depths = sorted(depth_map.keys())

    if len(depths) > 1:
      fatal(
          f"Mixed layout: images found at depths {depths} relative to "
          f"'{self._root_dir}'.  All images must be at the same depth.", ValueError)

    depth = depths[0]
    self._paths = all_paths

    has_labels = False
    if depth == 1:
      # root/file.jpg → flat, no labels
      pass
    elif depth == 2:
      # Check whether the first-level dirs are split folders.
      if _is_split_dir(self._root_dir):
        # root/split/file.jpg → organisational, no labels
        pass
      else:
        # root/class/file.jpg → labels
        has_labels = True
    else:
      # depth ≥ 3: root/split/class/file.jpg → second component is label
      has_labels = True

    if has_labels:
      label_set = set()
      for p in all_paths:
        rel = p.relative_to(self._root_dir)
        label_name = rel.parts[0] if depth == 2 else rel.parts[1]
        label_set.add(label_name)
      label_set = sorted(label_set)

      self._label_names = label_set
      self._label2id = {name: i for i, name in enumerate(label_set)}

      labels = []
      for p in self._paths:
        rel = p.relative_to(self._root_dir)
        label_name = rel.parts[0] if depth == 2 else rel.parts[1]
        labels.append(self._label2id[label_name])
      self._labels = np.array(labels, dtype=np.int64)

    logging.info(f"ImageFolderDataset: found {len(self._paths):,} images " +
                 (f", {len(self._label_names)} classes " if self._label_names else "") +
                 f" in '{self._root_dir}'")
    for name in self._label_names:
      count = int((self._labels == self._label2id[name]).sum())
      logging.info(f"  class {name}: {count:,} images")

  @property
  def has_labels(self):
    return self._labels is not None

  @property
  def num_labels(self):
    return len(self._label_names)

  @property
  def label_names(self):
    return list(self._label_names)

  @property
  def column_map(self):
    """``{SOURCE_COL: ENSEMBLE_COL}`` mapping applied by ``__getitem__``."""
    column_map = {"image": "image"}
    if self.has_labels:
      column_map["label"] = "label"
    return column_map

  @property
  def labels_array(self):
    return self._labels

  def __len__(self):
    return len(self._paths)

  def __getitem__(self, idx):
    if idx >= len(self._paths):
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)

    def load(i):
      return Image.open(self._paths[i]).convert("RGB")

    image, gidx = getitem_retry(idx, load, len(self._paths))
    if self._labels is not None:
      return {"image": image, "label": int(self._labels[gidx])}
    return {"image": image}


def load_imagefolder(root, *, split=None):
  """Load images from a directory tree, returning a dict of datasets.

  The directory structure is auto-detected:

  - **Split layout** — first-level sub-directories are ``train`` /
    ``val`` / ``test`` (or a ``.splits`` marker exists).  Returns one
    :class:`ImageFolderDataset` per split.

  - **Class layout** — first-level sub-directories are class names.
    Returns ``{"train": <dataset>}``.

  - **Flat layout** — images live directly in *root*.
    Returns ``{"train": <dataset>}``.

  Args:
    root: Path to the dataset root directory.
    split: If given, return only the dataset for that split.

  Returns:
    dict[str, ImageFolderDataset]
  """
  root = Path(root)
  if not root.exists():
    fatal(f"load_imagefolder: '{root}' does not exist", FileNotFoundError)

  if _is_split_dir(root):
    # Split layout: create one dataset per split sub-directory.
    # When a .splits marker exists, ALL immediate sub-dirs are splits.
    # Otherwise only dirs named in _SPLIT_NAMES are splits.
    has_marker = (root / ".splits").exists()
    splits = {}
    for child in sorted(root.iterdir()):
      if not child.is_dir():
        continue
      if has_marker or child.name in _SPLIT_NAMES:
        splits[child.name] = ImageFolderDataset(child)
    if not splits:
      fatal(
          f"load_imagefolder: '{root}' has no split sub-directories "
          f"(looked for {_SPLIT_NAMES})", FileNotFoundError)
    if split is not None:
      if split not in splits:
        fatal(
            f"load_imagefolder: split '{split}' not found in '{root}' "
            f"(available: {sorted(splits.keys())})", ValueError)
      return {split: splits[split]}
    return splits

  # Non-split layout: single dataset keyed as "train".
  ds = ImageFolderDataset(root)
  if split is not None and split != "train":
    fatal(
        f"load_imagefolder: split '{split}' not found in '{root}' "
        f"(available: ['train'])", ValueError)
  return {"train": ds}
