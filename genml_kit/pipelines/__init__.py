"""Data pipelines: loaders + blob contract + device transfer.

Pipelines own the DATA side of training only.  The model/loss/metric
(objective side) belongs to the method registry -- ``PretrainMethod``
becomes a general "Method".  See plans/GENERIC_PIPELINE.md (v4.2).
"""

from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob, LossOutput
from genml_kit.pipelines.registry import PIPELINES, build_pipeline


def _register_builtins():
  from genml_kit.pipelines import images, rl, vo_pair  # noqa: F401


_register_builtins()

__all__ = [
    "PIPELINES",
    "DataBlob",
    "LossOutput",
    "DataPipeline",
    "build_pipeline",
]
