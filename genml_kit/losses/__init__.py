"""Shared loss functions (classification + self-supervised + RL objectives).

Moved out of ``genml_kit.pretrain`` (v4.2): with the unified
``genml-kit-train`` CLI there is no separate "pretrain" program, so the
loss library is top-level.  These losses are composed by the registered
``Method`` implementations.
"""

from genml_kit.losses.byol import byol_loss
from genml_kit.losses.contrastive import supcon_loss
from genml_kit.losses.dino import DINOLoss
from genml_kit.losses.focal import CombinedFocalLoss
from genml_kit.losses.rl import (
    clipped_surrogate,
    entropy_bonus,
    gae,
    sac_alpha_loss,
    sac_policy_loss,
    sac_q_loss,
    td_loss,
    td_target,
    value_loss,
)

__all__ = [
    "CombinedFocalLoss",
    "DINOLoss",
    "byol_loss",
    "clipped_surrogate",
    "entropy_bonus",
    "gae",
    "sac_alpha_loss",
    "sac_policy_loss",
    "sac_q_loss",
    "supcon_loss",
    "td_loss",
    "td_target",
    "value_loss",
]
