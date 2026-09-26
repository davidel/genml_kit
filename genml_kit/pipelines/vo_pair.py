"""VO pair pipeline: synthetic pairs from VOPairDataset."""

import torch
from torch.utils.data import DataLoader

from genml_kit.datasets.vo_pairs import VOPairDataset
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.seed import seed_worker


@PIPELINES.register
class VOPairPipeline(DataPipeline):
  """Synthetic VO pair pipeline (no external data).

  Owns the train/val ``VOPairDataset``s and their loaders.  A loader
  step (via :func:`default_collate`) is a flat dict::

      {
        "image_a": (B, C, H, W),
        "image_b": (B, C, H, W),
        "meta": VOPairMeta-list...
      }

  which becomes ``DataBlob(data=(image_a, image_b), meta=...)``.
  """

  NAME = "vo_pair"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.train_dataset = None
    self.val_dataset = None

  @staticmethod
  def _collate(batch):
    """Collate a list of VO pair dict items into a DataBlob.

    Each item is ``{"image_a", "image_b", "meta": VOPairMeta}``.  The two
    frames stack to ``(B, C, H, W)`` each; the VOPairMeta fields become
    batched tensors in ``blob.meta``: ``gt`` (dict of tensors),
    ``gt_residual``, ``terrain`` and ``range_bin``.
    """
    data_a = torch.stack([b["image_a"] for b in batch])
    data_b = torch.stack([b["image_b"] for b in batch])
    meta = {
        "gt": {},
        "gt_residual":
            torch.stack([b["meta"].gt_residual for b in batch]),
        "terrain": [b["meta"].terrain for b in batch],
        "range_bin":
            torch.tensor([b["meta"].range_bin for b in batch], dtype=torch.long),
    }
    for key in ("log_s", "theta", "t"):
      meta["gt"][key] = torch.stack([b["meta"].gt[key] for b in batch])
    return DataBlob(data=(data_a, data_b), meta=meta)

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("vo_pair pipeline")
    group.add_argument("--vo_length",
                       type=int,
                       default=20000,
                       help="Number of synthetic train pairs.")
    group.add_argument("--vo_val_length",
                       type=int,
                       default=2000,
                       help="Number of synthetic validation pairs.")
    group.add_argument("--pitch_deg",
                       type=float,
                       default=45.0,
                       help="Camera pitch for the ground-plane render.")
    group.add_argument("--terrain",
                       type=str,
                       default=None,
                       help="Optional terrain profile name (VOPairDataset).")
    group.add_argument("--seeded_pairs",
                       type=int,
                       default=0,
                       help="Seed the pair generator for reproducible pairs.")

  def build_loader(self, args, mode="train", **kwargs):
    """Build (and cache) the train/val loaders for *mode*."""
    seed = getattr(args, "seeded_pairs", 0)
    size = (getattr(args, "image_size", 64), getattr(args, "image_size", 64))
    pitch_deg = getattr(args, "pitch_deg", 45.0)

    if mode == "train":
      if self.train_loader is None:
        self.train_dataset = VOPairDataset(
            length=int(getattr(args, "vo_length", 20000)),
            size=size,
            pitch_deg=pitch_deg,
            terrain=getattr(args, "terrain", "field"),
            seed=int(seed),
        )
        self.train_loader = self._make_loader(args, self.train_dataset)
      return self.train_loader

    if self.val_loader is None:
      self.val_dataset = VOPairDataset(
          length=int(getattr(args, "vo_val_length", 2000)),
          size=size,
          pitch_deg=pitch_deg,
          terrain=getattr(args, "terrain", "field"),
          seed=int(seed),
      )
      self.val_loader = self._make_loader(args, self.val_dataset, shuffle=False)
    return self.val_loader

  def _make_loader(self, args, dataset, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=getattr(args, "batch_size", 32),
        shuffle=shuffle,
        num_workers=getattr(args, "num_workers", 0),
        worker_init_fn=seed_worker,
        pin_memory=False,
        drop_last=True,
        collate_fn=self._collate,
    )

  @staticmethod
  def _move_to_device(v, device):
    if hasattr(v, "to"):
      return v.to(device, non_blocking=True)
    if isinstance(v, dict):
      return {k: VOPairPipeline._move_to_device(x, device) for k, x in v.items()}
    # ints (range_bin), strings (terrain) stay on CPU
    return v

  def to_device(self, blob, device):
    moved_meta = {k: self._move_to_device(v, device) for k, v in blob.meta.items()}
    return DataBlob((blob.data[0].to(device, non_blocking=True), blob.data[1].to(
        device, non_blocking=True)), moved_meta)
