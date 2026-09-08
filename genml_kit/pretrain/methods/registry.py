"""Pre-training method registry."""
import logging

from genml_kit.utils.logging import fatal

_METHODS = {}


def register_method(cls):
  """Class decorator that registers a pre-training method."""
  name = cls.NAME
  if name in _METHODS:
    fatal(f"Duplicate pre-training method name '{name}'", RuntimeError)
  _METHODS[name] = cls
  logging.debug("Registered pre-training method '%s'", name)
  return cls


def get_method(name):
  """Look up a pre-training method by name."""
  if name not in _METHODS:
    available = ", ".join(sorted(_METHODS)) or "(none)"
    fatal(
        f"Unknown pre-training method '{name}'. "
        f"Available: {available}",
        ValueError,
    )
  return _METHODS[name]


def list_methods():
  """Return sorted list of registered method names."""
  return sorted(_METHODS)
