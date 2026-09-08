"""Model registry.

Each custom model registers itself via the ``@register_model`` decorator.
The registry maps model name strings to loader functions that return
``(model, processor)`` pairs conforming to the common protocol expected
by the training and inference harnesses.
"""

# Import bundled custom models so their @register_model /
# @register_processor decorators run.  The imports are intentionally
# unused as names; importing the modules performs the registration side
# effects required by model loading.
import genml_kit.models.cls_model_wrapper
import genml_kit.models.convvit
import genml_kit.models.timm
import genml_kit.models.uvito  # noqa: F401
from genml_kit.models.registry import (
    ModelOutput,
    ParsedModelName,
    is_custom_model,
    load_model,
    load_processor,
    parse_model_name,
    register_model,
    register_processor,
)

__all__ = [
    "ModelOutput",
    "ParsedModelName",
    "is_custom_model",
    "load_model",
    "load_processor",
    "parse_model_name",
    "register_model",
    "register_processor",
]
