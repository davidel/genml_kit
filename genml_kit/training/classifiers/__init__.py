"""Classifier registry — built-in and user-supplied classifier heads.

Usage::

    from genml_kit.training.classifiers import build_classifier

    model = build_classifier("mlp", num_labels=3, hidden_size=1024, hidden=256)

A classifier spec can be:

* A registered name (e.g. ``"mlp"``).
* A path/URL to a ``.py`` file defining a ``Classifier`` class.
* An inline spec ``"name:key=value,..."`` parsed by
  :func:`parse_classifier_spec`.
"""

import logging

from genml_kit.utils.cli import _split_list_items, parse_value
from genml_kit.utils.logging import fatal
from genml_kit.utils.script import load_extern

_CLASSIFIERS = {}


def parse_classifier_spec(spec):
  """Split an inline classifier spec into ``(name, kwargs)``.

  The CLI expresses classifier heads as a single string, e.g.::

      classifier=mlp:hidden=512,dropout=0.3

  The part before the first ``:`` is the classifier name (a registered
  name or a ``.py`` path); the remainder is a comma-separated
  ``key=value`` list converted with :func:`parse_value` (so ints,
  floats, booleans, and bracketed lists work).  Bracket-aware splitting
  keeps values such as ``cls_slice=(0, 1)`` intact.

  A spec without a colon is just a name with no kwargs.

  Parameters
  ----------
  spec : str
      The classifier specification.

  Returns
  -------
  (str, dict)
      The classifier name and its constructor kwargs.
  """
  spec = str(spec).strip()
  name, sep, body = spec.partition(":")
  if not sep or not body.strip():
    return name, {}
  kwargs = {}
  for item in _split_list_items(body.strip()):
    if "=" not in item:
      fatal(
          f"Expected key=value in classifier spec {spec!r}, got: {item!r}",
          ValueError,
      )
    key, val = item.split("=", 1)
    kwargs[key.strip()] = parse_value(val.strip())
  return name, kwargs


def register_classifier(name):
  """Decorator that registers a classifier class under *name*."""

  def wrapper(cls):
    if name in _CLASSIFIERS:
      fatal(f"Classifier {name!r} already registered", ValueError)
    _CLASSIFIERS[name] = cls
    return cls

  return wrapper


def build_classifier(spec, num_labels, hidden_size, **kwargs):
  """Instantiate a classifier head.

  Parameters
  ----------
  spec : str
      Registered classifier name or path to a ``.py`` file.
  num_labels : int
      Number of output classes.
  hidden_size : int
      Dimensionality of backbone hidden states (``D``).  Passed to the
      classifier so it can dimension its layers without holding a
      reference to the backbone.
  **kwargs
      Extra keyword arguments forwarded to the classifier constructor
      (e.g. ``hidden=512``, ``cls_slice=(0, 1)``).

  Returns
  -------
  torch.nn.Module
      A classifier with a ``forward(hidden_states)`` interface.
  """
  cls = None
  if spec in _CLASSIFIERS:
    cls = _CLASSIFIERS[spec]
    logging.info("Using registered classifier %r (%s)", spec, cls.__name__)
  elif spec.endswith(".py"):
    cls = load_extern(spec, interface="classifier")
    logging.info("Loaded external classifier from %s (%s)", spec, cls.__name__)
  else:
    available = sorted(_CLASSIFIERS.keys())
    fatal(
        f"Unknown classifier {spec!r}. Available: {', '.join(available)}. "
        f"Or provide a path to a .py file.",
        ValueError,
    )
  logging.info("Classifier kwargs: %s", kwargs)
  return cls(num_labels=num_labels, hidden_size=hidden_size, **kwargs)


def _register_builtins():
  """Import built-in classifiers so their ``@register_classifier`` fires."""
  from genml_kit.training.classifiers import cls_attention, mlp  # noqa: F401


_register_builtins()
