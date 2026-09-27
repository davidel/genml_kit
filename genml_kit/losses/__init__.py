"""Shared loss functions (classification + self-supervised + RL objectives).

Moved out of ``genml_kit.pretrain`` (v4.2): with the unified
``genml-kit-train`` CLI there is no separate "pretrain" program, so the
loss library is top-level.  These losses are composed by the registered
``Method`` implementations.

The ``LOSSES`` registry (``genml_kit.losses.registry``) adds a second,
spec-driven layer: ``load_loss(spec, **kwargs)`` builds a loss from a
registered name or an external ``.py`` script defining
``build_loss(**kwargs)``.  See ``registry.py`` for the full protocol.
"""

# Import the builtins so their @LOSSES.register decorators run.  The
# imports are intentionally unused as names; importing the module performs
# the registration side effect.
from genml_kit.losses import builtin  # noqa: F401
from genml_kit.losses.byol import byol_loss
from genml_kit.losses.contrastive import supcon_loss
from genml_kit.losses.dino import DINOLoss
from genml_kit.losses.focal import CombinedFocalLoss
from genml_kit.losses.registry import DEFAULT_LOSS, LOSSES, load_loss
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
    "DEFAULT_LOSS",
    "DINOLoss",
    "LOSSES",
    "byol_loss",
    "clipped_surrogate",
    "entropy_bonus",
    "gae",
    "load_loss",
    "sac_alpha_loss",
    "sac_policy_loss",
    "sac_q_loss",
    "supcon_loss",
    "td_loss",
    "td_target",
    "value_loss",
]
