"""Dataset utilities.

Provides :class:`DatasetEnsemble` for stitching together multiple image
datasets into a unified pre-training corpus, and :class:`HFDatasetProxy`
for bridging HuggingFace datasets to PyTorch.
"""

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.hf_proxy import HFDatasetProxy

__all__ = ["DatasetEnsemble", "HFDatasetProxy"]
