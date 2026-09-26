"""Pipeline registry -- mirrors genml_kit/methods/registry.py (v4.2 s 3.2).

Pipeline classes register via ``@PIPELINES.register`` (name defaults to
``cls.NAME``) and are looked up with ``PIPELINES.get`` / ``PIPELINES.build``.
"""

import logging

from genml_kit.utils.registry import Registry

PIPELINES = Registry("pipeline")


def build_pipeline(name, **kwargs):
  """Instantiate a pipeline (config via CLI args)."""
  cls = PIPELINES.get(name)
  logging.info("Built pipeline '%s'", name)
  return cls(**kwargs)
