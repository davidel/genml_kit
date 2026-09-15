"""Method registry: objective side (model + loss + metric).

Reshapes ``PretrainMethod`` into a general ``Method`` per the v4.2 plan
(s 3): classification, VO (supervised/photometric) and the self-supervised
pre-training methods all register here.
"""

from genml_kit.methods.base import Method
from genml_kit.methods.registry import (
    build_method,
    get_method,
    list_methods,
    register_method,
)


def _register_builtins():
  # Import built-in methods to trigger registration.
  from genml_kit.methods import (  # noqa: F401
    byol,
    classification,
    dino,
    ijepa,
    simmim,
    supcon,
    vo_pair,
  )


_register_builtins()

__all__ = [
    "Method",
    "build_method",
    "get_method",
    "list_methods",
    "register_method",
]
