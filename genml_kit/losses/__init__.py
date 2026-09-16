"""Shared loss functions (classification + self-supervised objectives).

Moved out of ``genml_kit.pretrain`` (v4.2): with the unified
``genml-kit-train`` CLI there is no separate "pretrain" program, so the
loss library is top-level.  These losses are composed by the registered
``Method`` implementations.
"""

from genml_kit.losses.byol import byol_loss
from genml_kit.losses.contrastive import supcon_loss
from genml_kit.losses.dino import DINOLoss
from genml_kit.losses.focal import CombinedFocalLoss

__all__ = ["CombinedFocalLoss", "DINOLoss", "byol_loss", "supcon_loss"]
