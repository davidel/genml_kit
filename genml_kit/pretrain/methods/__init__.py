"""Pre-training method registry."""
# Import built-in methods to trigger registration.
import genml_kit.pretrain.methods.byol
import genml_kit.pretrain.methods.dino
import genml_kit.pretrain.methods.ijepa
import genml_kit.pretrain.methods.simmim
import genml_kit.pretrain.methods.supcon  # noqa: F401
import genml_kit.pretrain.methods.vo_pair  # noqa: F401
from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import get_method, list_methods

__all__ = ["PretrainMethod", "get_method", "list_methods"]
