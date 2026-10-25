"""Detection model wrapper -- encoder backbone + pluggable detection head.

Importing this package registers the ``det_wrapper`` model loader in the
``MODELS`` registry (import side effect).
"""

import genml_kit.models.det_model_wrapper.loader  # noqa: F401
