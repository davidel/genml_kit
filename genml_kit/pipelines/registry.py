"""Pipeline registry -- mirrors genml_kit/methods/registry.py (v4.2 s 3.2)."""

import logging

from genml_kit.utils.registry import Registry

_PIPELINES = Registry("pipeline")

register_pipeline = _PIPELINES.register
get_pipeline = _PIPELINES.get
list_pipelines = _PIPELINES.list_names


def build_pipeline(name, **kwargs):
  """Instantiate a pipeline (config via CLI args)."""
  cls = _PIPELINES.get(name)
  logging.info("Built pipeline '%s'", name)
  return cls(**kwargs)
