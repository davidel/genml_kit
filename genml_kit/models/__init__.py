"""Model registry.

Each custom model registers itself via the ``MODELS.register`` decorator.
The registry maps model name strings to loader functions that return
``(model, processor)`` pairs conforming to the common protocol expected
by the training and inference harnesses.
"""

# Import bundled custom models so their MODELS.register /
# PROCESSORS.register decorators run.  The imports are intentionally
# unused as names; importing the modules performs the registration side
# effects required by model loading.
import genml_kit.models.cls_model_wrapper  # noqa: F401
import genml_kit.models.convvit  # noqa: F401
import genml_kit.models.rl  # noqa: F401
import genml_kit.models.timm  # noqa: F401
import genml_kit.models.uvito  # noqa: F401
import genml_kit.models.vo  # noqa: F401
from genml_kit.models.registry import (
    MODELS,
    PROCESSORS,
    ModelOutput,
    ParsedModelName,
    is_custom_model,
    load_model,
    load_processor,
    parse_model_name,
)

__all__ = [
    "MODELS",
    "PROCESSORS",
    "ModelOutput",
    "ParsedModelName",
    "is_custom_model",
    "load_model",
    "load_processor",
    "parse_model_name",
]
