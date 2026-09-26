"""Generic, decorator-aware registry.

A small reusable ``Registry`` keyed by name that supports both class and
function registration, with or without an explicit name:

- ``@registry.register``             name from ``NAME`` attribute or ``__name__``
- ``@registry.register("name")``     explicit name
- ``@registry.register(name="n")``   explicit name as keyword
- ``registry.register(obj, name="n")`` direct call

Entries are looked up with ``get`` / ``build`` / ``contains`` and removed
with ``unregister``.  Duplicate and unknown names raise ``ValueError`` with
stable messages listing the registered names.

This is the single registry abstraction used by the method, pipeline,
model and classifier registries.
"""

from genml_kit.utils.logging import fatal


class Registry:
  """Name-keyed registry for classes and functions.

  Args:
      name: Human-readable kind name used in errors and log messages
          (e.g. ``"method"``, ``"model"``, ``"classifier"``).
  """

  def __init__(self, name):
    self._name = name
    self._entries = {}

  # -- registration ------------------------------------------------------

  def register(self, obj=None, *, name=None):
    """Register an object, usable as a decorator or direct call.

    Supported forms::

        @registry.register
        class Foo: ...                        # name = Foo.NAME or "Foo"

        @registry.register("foo")
        class Foo: ...                        # name = "foo"

        @registry.register(name="foo")
        def make_foo(): ...                   # name = "foo"

        registry.register(make_foo, name="foo")  # direct call

    The registered object is returned unchanged so the decorator form
    preserves the original class/function.
    """
    if obj is None:
      # @registry.register or @registry.register(name=...):
      # return a decorator.
      def _decorator(o):
        return self._register(o, name)

      return _decorator
    if isinstance(obj, str) and name is None:
      # @registry.register("foo"): obj is the explicit name.
      def _decorator(o):
        return self._register(o, obj)

      return _decorator
    return self._register(obj, name)

  def _register(self, obj, name):
    key = name or getattr(obj, "NAME", None) or obj.__name__
    if key in self._entries:
      fatal(
          f"Duplicate {self._name} name '{key}'",
          ValueError,
      )
    self._entries[key] = obj
    return obj

  def unregister(self, name):
    """Remove *name* from the registry (no-op if absent)."""
    self._entries.pop(name, None)

  # -- lookup ------------------------------------------------------------

  def contains(self, name):
    """Return ``True`` if *name* is registered."""
    return name in self._entries

  def get(self, name):
    """Return the registered object for *name*.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in self._entries:
      available = ", ".join(sorted(self._entries)) or "(none)"
      fatal(
          f"Unknown {self._name} '{name}'. Available: {available}",
          ValueError,
      )
    return self._entries[name]

  def build(self, name, **kwargs):
    """Instantiate the registered class for *name* with *kwargs*."""
    cls = self.get(name)
    return cls(**kwargs)

  def list_names(self):
    """Return a sorted list of registered names."""
    return sorted(self._entries)
