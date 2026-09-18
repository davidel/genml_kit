"""Dataset utilities.

Provides :class:`DatasetEnsemble` for stitching together multiple image
datasets into a unified pre-training corpus, :class:`HFDatasetProxy`
for bridging HuggingFace datasets to PyTorch, and
:class:`ReplayBufferDataset` for off-policy RL.
"""

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.hf_proxy import HFDatasetProxy
from genml_kit.datasets.replay_buffer import (
    ReplayBufferDataset,
    Transition,
)
from genml_kit.datasets.rollout_buffer import RolloutBuffer

__all__ = [
    "DatasetEnsemble",
    "HFDatasetProxy",
    "ReplayBufferDataset",
    "RolloutBuffer",
    "Transition",
]
