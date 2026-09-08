"""Safe dotted-attribute access utilities.

Provides helpers for resolving, calling, and setting attributes on objects
using dotted path strings (e.g. ``'model.backbone.gradient_checkpointing_enable'``).

All public functions return a sentinel ``MISSING`` when any part of the
attribute path cannot be resolved, instead of raising ``AttributeError``.
This makes them useful for dispatch-style code that must probe multiple
potential attribute locations.

Example
-------
>>> from genml_kit.utils.attr import maybe_call, maybe_setattr, MISSING
>>> class Wrapper:
...     def __init__(self):
...         self.model = nn.Module()
>>> maybe_call(Wrapper(), 'model.zero_grad')  # exists
<bound method Module.zero_grad ...>
>>> maybe_call(Wrapper(), 'model.nonexistent') is MISSING
True
"""

MISSING = object()
"""Sentinel returned when an attribute path cannot be resolved."""


def _getattr(obj, attr):
  """Walk a dotted attribute path and return (value, parent, last_attr).

  Parameters
  ----------
  obj : Any
      Root object to resolve against.
  attr : str
      Dotted attribute path (e.g. ``'a.b.c'``).

  Returns
  -------
  tuple[Any, Any, str]
      ``(value, parent_object, last_attribute_name)`` on success, or
      ``(MISSING, MISSING, MISSING)`` if any segment is missing.
  """
  parent, parts = MISSING, attr.split('.')

  for name in parts:
    nobj = getattr(obj, name, None)
    if nobj is None:
      return MISSING, MISSING, MISSING

    parent, obj = obj, nobj

  return obj, parent, parts[-1]


def get_attribute(obj, attr):
  """Retrieve a dotted attribute path, returning ``MISSING`` on failure.

  Parameters
  ----------
  obj : Any
      Root object.
  attr : str
      Dotted attribute path (e.g. ``'model.backbone'``).

  Returns
  -------
  Any
      The resolved attribute value, or ``MISSING`` if the path does not exist.

  Examples
  --------
  >>> get_attribute(model, 'use_grad_checkpoint') is MISSING if no such attr else True
  """
  value, *_ = _getattr(obj, attr)
  return value


def maybe_call(obj, attr, *args, **kwargs):
  """Call a method at a dotted path if it exists, else return ``MISSING``.

  Parameters
  ----------
  obj : Any
      Root object.
  attr : str
      Dotted path to a callable (e.g. ``'model.backbone.enable'``).
  *args, **kwargs
      Forwarded to the callable if resolved.

  Returns
  -------
  Any
      The return value of the callable, or ``MISSING`` if the path does
      not resolve or the target is not callable.

  Examples
  --------
  >>> maybe_call(model, 'model.set_grad_checkpointing', enable=True)
  >>> maybe_call(model, 'backbone.gradient_checkpointing_enable')
  """
  fn, *_ = _getattr(obj, attr)

  if fn is MISSING:
    return MISSING

  if not callable(fn):
    return MISSING

  return fn(*args, **kwargs)


def maybe_setattr(obj, attr, value):
  """Set an attribute at a dotted path if it exists, else do nothing.

  Parameters
  ----------
  obj : Any
      Root object.
  attr : str
      Dotted path whose **parent** must exist (e.g. ``'model.use_grad_checkpoint'``
      sets ``parent.model.use_grad_checkpoint = value``).
  value : Any
      Value to assign.

  Returns
  -------
  Any
      The previous value of the attribute, or ``MISSING`` if the path
      could not be resolved.

  Examples
  --------
  >>> maybe_setattr(model, 'model.use_grad_checkpoint', True)
  """
  avalue, parent, sattr = _getattr(obj, attr)

  if avalue is MISSING:
    return MISSING

  setattr(parent, sattr, value)
  return avalue
