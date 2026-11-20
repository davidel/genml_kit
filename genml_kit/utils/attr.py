"""Safe dotted-attribute access utilities.

Provides helpers for resolving, calling, and setting attributes on objects
using dotted path strings (e.g. ``'model.backbone.gradient_checkpointing_enable'``).

All public functions return a sentinel ``MISSING`` when any part of the
attribute path cannot be resolved, instead of raising ``AttributeError``.
A resolved attribute whose value is ``None`` is returned as ``None``, not
``MISSING``.  This makes them useful for dispatch-style code that must probe
multiple potential attribute locations.

Example
-------
>>> from genml_kit.utils.attr import MISSING, get_attribute
>>> class Wrapper:
...   model = None
>>> get_attribute(Wrapper(), 'model') is None
True
>>> get_attribute(Wrapper(), 'nonexistent') is MISSING
True
"""

import collections

MISSING = object()
"""Sentinel returned when an attribute path cannot be resolved."""

# What ``_getattr`` returns: the resolved value, its parent object, and the
# last attribute name (needed to set the attribute back on the parent).
AttrsResolved = collections.namedtuple("AttrsResolved", ["value", "parent", "attr"])


def _getattr(obj, attr):
  """Walk a dotted attribute path and return an :class:`AttrsResolved`.

  Args:
    obj: Root object to resolve against.
    attr: Dotted attribute path (e.g. ``'a.b.c'``).

  Returns:
    ``(value, parent, attr)`` on success, or
    ``(MISSING, MISSING, MISSING)`` if any segment is missing.
  """
  parent, parts = MISSING, attr.split('.')

  for name in parts:
    nobj = getattr(obj, name, MISSING)
    if nobj is MISSING:
      return AttrsResolved(value=MISSING, parent=MISSING, attr=MISSING)

    parent, obj = obj, nobj

  return AttrsResolved(value=obj, parent=parent, attr=parts[-1])


def get_attribute(obj, attr):
  """Retrieve a dotted attribute path, returning ``MISSING`` on failure.

  Args:
    obj: Root object.
    attr: Dotted attribute path (e.g. ``'model.backbone'``).

  Returns:
    The resolved attribute value, or ``MISSING`` if the path does not exist.
  """
  return _getattr(obj, attr).value


def maybe_call(obj, attr, *args, **kwargs):
  """Call a method at a dotted path if it exists, else return ``MISSING``.

  Args:
    obj: Root object.
    attr: Dotted path to a callable (e.g. ``'model.backbone.enable'``).
    *args: Forwarded to the callable if resolved.
    **kwargs: Forwarded to the callable if resolved.

  Returns:
    The return value of the callable, or ``MISSING`` if the path does not
    resolve or the target is not callable.
  """
  fn = _getattr(obj, attr).value

  if fn is MISSING:
    return MISSING

  if not callable(fn):
    return MISSING

  return fn(*args, **kwargs)


def maybe_setattr(obj, attr, value):
  """Set an attribute at a dotted path if it exists, else do nothing.

  Args:
    obj: Root object.
    attr: Dotted path whose **parent** must exist (e.g.
      ``'model.use_grad_checkpoint'`` sets
      ``parent.model.use_grad_checkpoint = value``).
    value: Value to assign.

  Returns:
    The previous value of the attribute, or ``MISSING`` if the path could not
    be resolved.
  """
  resolved = _getattr(obj, attr)

  if resolved.value is MISSING:
    return MISSING

  setattr(resolved.parent, resolved.attr, value)
  return resolved.value
