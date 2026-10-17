"""Segmentation model wrapper -- encoder backbone + pluggable decoder.

Importing this package registers the ``seg_wrapper`` model loader in the
``MODELS`` registry (import side effect).
"""

import genml_kit.models.seg_model_wrapper.loader  # noqa: F401
