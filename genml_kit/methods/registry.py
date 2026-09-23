"""General method registry (supervised + self-supervised).

Reshapes ``genml_kit/pretrain/methods/registry.py`` (v4.2 s 3.2): the
decorator semantics stay the same (``register_method(cls)`` reads
``cls.NAME``), and builds go through ``get_method`` + instantiation --
methods take NO constructor args (config flows through add_args/args).
"""

import logging

from genml_kit.utils.registry import Registry

_METHODS = Registry("method")

register_method = _METHODS.register
get_method = _METHODS.get
list_methods = _METHODS.list_names


def build_method(name):
  """Instantiate a method (no constructor args; config via CLI args)."""
  cls = _METHODS.get(name)
  logging.info("Built method '%s'", name)
  return cls()
