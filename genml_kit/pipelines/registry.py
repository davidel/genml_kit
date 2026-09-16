"""Pipeline registry -- mirrors genml_kit/methods/registry.py (v4.2 s 3.2)."""

import logging

from genml_kit.utils.logging import fatal

_PIPELINES = {}


def register_pipeline(cls):
  """Class decorator that registers a training pipeline."""
  name = cls.NAME
  if name in _PIPELINES:
    fatal(f"Duplicate pipeline name '{name}'", RuntimeError)
  _PIPELINES[name] = cls
  logging.debug("Registered pipeline '%s'", name)
  return cls


def get_pipeline(name):
  """Look up a pipeline class by name."""
  if name not in _PIPELINES:
    available = ", ".join(sorted(_PIPELINES)) or "(none)"
    fatal(f"Unknown pipeline '{name}'. Available: {available}", ValueError)
  return _PIPELINES[name]


def build_pipeline(name, **kwargs):
  """Instantiate a pipeline (config via CLI args)."""
  cls = get_pipeline(name)
  logging.info("Built pipeline '%s'", name)
  return cls(**kwargs)


def list_pipelines():
  """Return sorted list of registered pipeline names."""
  return sorted(_PIPELINES)
