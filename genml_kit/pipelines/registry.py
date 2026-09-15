"""Pipeline registry -- mirrors genml_kit/training/classifiers/__init__.py."""

import logging

from genml_kit.utils.logging import fatal

_PIPELINES = {}


def register_pipeline(name):

  def wrapper(cls):
    if name in _PIPELINES:
      fatal(f"Pipeline {name!r} already registered", ValueError)
    _PIPELINES[name] = cls
    return cls

  return wrapper


def get_pipeline(name):
  if name not in _PIPELINES:
    available = ", ".join(sorted(_PIPELINES)) or "(none)"
    fatal(f"Unknown pipeline '{name}'. Available: {available}", ValueError)
  return _PIPELINES[name]


def build_pipeline(name, **kwargs):
  cls = get_pipeline(name)
  logging.info("Pipeline kwargs: %s", kwargs)
  return cls(**kwargs)


def list_pipelines():
  return sorted(_PIPELINES)
