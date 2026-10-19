"""SegFolderDataset -- paired-image segmentation corpus on disk.

Layout expected::

    <root>/images/<stem>.<ext>
    <root>/masks/<stem>.png

or, when the corpus has named splits::

    <root>/<split>/images/<stem>.<ext>
    <root>/<split>/masks/<stem>.png

Images are decoded as RGB PIL images; masks are decoded as ``int64``
``numpy.ndarray`` class ids.  Palette (``P``), grayscale (``L``) and 16-bit
(``I;16``) masks all yield an ``(H, W)`` array.  A true RGB mask is kept as
``(H, W, 3)`` so the pipeline can remap colour triplets via
:func:`genml_kit.utils.seg.apply_mask_map`; a mask with any other channel
count is rejected as ambiguous.  ``__getitem__`` returns a dict
``{"image": PIL.Image, "mask": np.ndarray}`` so the paired transform can
co-process the two fields.

The extension list is shared with :mod:`genml_kit.datasets.image_folder`
so both local loaders recognise the same image formats.
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from genml_kit.datasets.image_folder import _IMAGE_EXTS
from genml_kit.datasets.retry import getitem_retry
from genml_kit.utils.logging import fatal

_SPLIT_NAMES = frozenset({"train", "val", "test"})


class SegFolderDataset:
  """Paired image/mask dataset backed by ``Path.rglob()``.

  Parameters
  ----------
  root_dir : str | Path
      Root containing ``images/`` and ``masks/`` (or the split dir that
      contains them).
  image_subdir : str
      Sub-directory holding images.
  mask_subdir : str
      Sub-directory holding masks.

  Raises
  ------
  FileNotFoundError, ValueError
      Via :func:`fatal` when the directories are missing, empty, or the
      image and mask stems do not match.
  """

  def __init__(self, root_dir, image_subdir="images", mask_subdir="masks"):
    self.name = str(root_dir)
    self._root = Path(root_dir)
    self._image_dir = self._root / image_subdir
    self._mask_dir = self._root / mask_subdir

    if not self._root.is_dir():
      fatal(f"SegFolderDataset: '{self._root}' is not a directory", ValueError)
    if not self._image_dir.is_dir():
      fatal(
          f"SegFolderDataset: image dir '{self._image_dir}' does not exist",
          FileNotFoundError,
      )
    if not self._mask_dir.is_dir():
      fatal(
          f"SegFolderDataset: mask dir '{self._mask_dir}' does not exist",
          FileNotFoundError,
      )

    self._scan()

  def _scan(self):
    images = self._index_by_stem(self._image_dir, "image")
    masks = self._index_by_stem(self._mask_dir, "mask")

    # Match on the lowercase stem so case differences in the corpus
    # (e.g. ``IMG_1.JPG`` vs ``img_1.png``) do not break pairing.
    only_images = sorted(set(images) - set(masks))
    only_masks = sorted(set(masks) - set(images))
    if only_images or only_masks:
      fatal(
          f"SegFolderDataset: image/mask stems do not match under "
          f"'{self._root}'. Missing masks for {only_images[:5]} "
          f"(+{max(len(only_images) - 5, 0)} more); missing images for "
          f"{only_masks[:5]} (+{max(len(only_masks) - 5, 0)} more).",
          ValueError,
      )

    self._stems = sorted(images)
    self._image_paths = [images[s] for s in self._stems]
    self._mask_paths = [masks[s] for s in self._stems]
    logging.info("SegFolderDataset: found %d image/mask pairs in '%s'",
                 len(self._stems), self._root)

  @staticmethod
  def _index_by_stem(directory, label):
    index = {}
    for path in sorted(directory.rglob("*")):
      if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
        continue
      stem = path.stem.lower()
      if stem in index:
        fatal(
            f"SegFolderDataset: duplicate {label} stem '{path.stem}' in "
            f"'{directory}' ('{index[stem]}' and '{path}')",
            ValueError,
        )
      index[stem] = path
    if not index:
      fatal(
          f"SegFolderDataset: no {label} files found under '{directory}'",
          FileNotFoundError,
      )
    return index

  def __len__(self):
    return len(self._stems)

  def __getitem__(self, idx):
    n = len(self._stems)
    if idx < 0:
      idx += n
    if idx < 0 or idx >= n:
      fatal(f"Index {idx} out of range for '{self.name}'", IndexError)

    def load(i):
      image = Image.open(self._image_paths[i]).convert("RGB")
      with Image.open(self._mask_paths[i]) as handle:
        mask = np.array(handle, dtype=np.int64)
      return image, mask

    (image, mask), _ = getitem_retry(idx, load, n)
    # Palette ('P'), grayscale ('L') and 16-bit ('I;16') masks all decode to
    # (H, W) class-id maps and are returned as-is.  A true RGB mask stays
    # (H, W, 3) so ``apply_mask_map`` can remap colour triplets instead of
    # silently keeping channel 0 (which collapses distinct colours onto a
    # single id).  Any other channel count is ambiguous, so reject it rather
    # than guess.
    if mask.ndim == 3:
      if mask.shape[-1] == 1:
        mask = mask[..., 0]
      elif mask.shape[-1] != 3:
        fatal(
            f"SegFolderDataset: mask '{self._mask_paths[idx]}' has "
            f"{mask.shape[-1]} channels; expected a single-channel class-id "
            "map or a 3-channel RGB mask.", ValueError)
    return {"image": image, "mask": mask}


def _has_named_splits(root):
  """Return True when *root* holds ``train``/``val``/``test`` sub-dirs."""
  return any(child.is_dir() and child.name in _SPLIT_NAMES for child in root.iterdir())


def load_segfolder(root, *, split=None):
  """Load paired image/mask data, returning a dict of datasets.

  Two layouts are supported:

  - **Split layout** — first-level sub-directories are ``train`` / ``val``
    / ``test``, each containing ``images/`` and ``masks/``.
  - **Flat layout** — *root* itself contains ``images/`` and ``masks/``;
    returned as ``{"train": <dataset>}``.

  Args:
    root: Path to the dataset root directory.
    split: If given, return only the dataset for that split.

  Returns:
    dict[str, SegFolderDataset]
  """
  root = Path(root)
  if not root.is_dir():
    fatal(f"load_segfolder: '{root}' is not a directory", FileNotFoundError)

  if _has_named_splits(root):
    splits = {}
    for child in sorted(root.iterdir()):
      if child.is_dir() and child.name in _SPLIT_NAMES:
        splits[child.name] = SegFolderDataset(child)
    if not splits:
      fatal(f"load_segfolder: '{root}' has no split sub-directories", FileNotFoundError)
    if split is not None:
      if split not in splits:
        fatal(
            f"load_segfolder: split '{split}' not found in '{root}' "
            f"(available: {sorted(splits)})", ValueError)
      return {split: splits[split]}
    return splits

  ds = SegFolderDataset(root)
  if split is not None and split != "train":
    fatal(
        f"load_segfolder: split '{split}' not found in '{root}' "
        f"(available: ['train'])", ValueError)
  return {"train": ds}
