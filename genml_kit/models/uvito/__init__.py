"""UVito: SMP encoder + Transformer classification head."""

from genml_kit.models.uvito.loader import UVitoAdapter
from genml_kit.models.uvito.model import UVito
from genml_kit.models.uvito.processor import UVitoProcessor

__all__ = [
    "UVito",
    "UVitoAdapter",
    "UVitoProcessor",
]
