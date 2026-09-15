"""General method registry (supervised + self-supervised).

Reshapes ``genml_kit/pretrain/methods/registry.py`` (v4.2 s 3.2): the
decorator semantics stay the same (``register_method(cls)`` reads
``cls.NAME``), and builds go through ``get_method`` + instantiation --
methods take NO constructor args (config flows through add_args/args).
"""

import logging

from genml_kit.utils.logging import fatal

_METHODS = {}


def register_method(cls):
  """Class decorator that registers a training method."""
  name = cls.NAME
  if name in _METHODS:
    fatal(f"Duplicate method name '{name}'", RuntimeError)
  _METHODS[name] = cls
  logging.debug("Registered method '%s'", name)
  return cls


def get_method(name):
  """Look up a method class by name."""
  if name not in _METHODS:
    available = ", ".join(sorted(_METHODS)) or "(none)"
    fatal(f"Unknown method '{name}'. Available: {available}", ValueError)
  return _METHODS[name]


def build_method(name):
  """Instantiate a method (no constructor args; config via CLI args)."""
  cls = get_method(name)
  logging.info("Built method '%s'", name)
  return cls()


def list_methods():
  """Return sorted list of registered method names."""
  return sorted(_METHODS)
