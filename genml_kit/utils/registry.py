"""Generic class-based registry.

Provides a reusable ``Registry`` that reads ``cls.NAME`` and offers
``register`` / ``get`` / ``build`` / ``list_names`` helpers.  Used by
``genml_kit.methods.registry`` and ``genml_kit.pipelines.registry`` to
eliminate boilerplate.
"""

import logging

from genml_kit.utils.logging import fatal


class Registry:
  """Name-keyed registry for classes that expose a ``NAME`` attribute.

  Usage::

      _METHODS = Registry("method")
      register_method = _METHODS.register
  """

  def __init__(self, name):
    self._name = name
    self._entries = {}

  def register(self, cls):
    """Class decorator that reads ``cls.NAME``."""
    entry_name = cls.NAME
    if entry_name in self._entries:
      fatal(
          f"Duplicate {self._name} name '{entry_name}'",
          RuntimeError,
      )
    self._entries[entry_name] = cls
    logging.debug("Registered %s '%s'", self._name, entry_name)
    return cls

  def get(self, name):
    """Look up a registered class by *name*."""
    if name not in self._entries:
      available = ", ".join(sorted(self._entries)) or "(none)"
      fatal(
          f"Unknown {self._name} '{name}'. Available: {available}",
          ValueError,
      )
    return self._entries[name]

  def build(self, name, **kwargs):
    """Instantiate a registered class with *kwargs*."""
    cls = self.get(name)
    logging.info("Built %s '%s'", self._name, name)
    return cls(**kwargs)

  def list_names(self):
    """Return a sorted list of registered names."""
    return sorted(self._entries)
