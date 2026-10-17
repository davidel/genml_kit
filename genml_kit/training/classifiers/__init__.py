"""Classifier registry -- built-in and user-supplied classifier heads.

Usage::

    from genml_kit.training.classifiers import build_classifier

    model = build_classifier("mlp", num_labels=3, hidden_size=1024, hidden=256)

A classifier spec can be:

* A registered name (e.g. ``"mlp"``).
* A path/URL to a ``.py`` file defining a ``create_classifier()`` function.
* An inline spec ``"name:key=value,..."`` parsed by
  :func:`parse_classifier_spec`.
"""

import logging

from genml_kit.utils.cli import parse_inline_spec
from genml_kit.utils.logging import fatal
from genml_kit.utils.registry import Registry
from genml_kit.utils.script import load_extern, is_script_spec

CLASSIFIERS = Registry("classifier")


def parse_classifier_spec(spec):
  """Split an inline classifier spec into ``(name, kwargs)``.

  The CLI expresses classifier heads as a single string, e.g.::

      classifier=mlp:hidden=512,dropout=0.3

  Delegates to :func:`~genml_kit.utils.cli.parse_inline_spec`.

  Parameters
  ----------
  spec : str
      Inline classifier specification.

  Returns
  -------
  (str, dict)
      The classifier name and its constructor kwargs.
  """
  return parse_inline_spec(spec, kind="classifier")


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
  cls_builder = None
  if CLASSIFIERS.contains(spec):
    cls_builder = CLASSIFIERS.get(spec)
    logging.info("Using registered classifier %r (%s)", spec, cls_builder.__name__)
  elif is_script_spec(spec):
    cls_builder = load_extern(spec, "create_classifier")
    logging.info("Loaded external classifier from %s", spec)
  else:
    available = ", ".join(CLASSIFIERS.list_names())
    fatal(
        f"Unknown classifier {spec!r}. Available: {available}. "
        f"Or provide a path to a .py file.",
        ValueError,
    )
  logging.info("Classifier kwargs: %s", kwargs)
  return cls_builder(num_labels=num_labels, hidden_size=hidden_size, **kwargs)


def _register_builtins():
  """Import built-in classifiers so their ``@CLASSIFIERS.register`` fires."""
  from genml_kit.training.classifiers import cls_attention, mlp  # noqa: F401


_register_builtins()
