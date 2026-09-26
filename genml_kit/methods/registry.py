"""General method registry (supervised + self-supervised).

Reshapes ``genml_kit/pretrain/methods/registry.py`` (v4.2 s 3.2): method
classes register via ``@METHODS.register`` (name defaults to ``cls.NAME``)
and are looked up with ``METHODS.get`` / ``METHODS.build``.

Methods take NO constructor args -- config flows through add_args/args --
so ``build_method`` is a thin helper that logs the build.
"""

import logging

from genml_kit.utils.registry import Registry

METHODS = Registry("method")


def build_method(name):
  """Instantiate a method (no constructor args; config via CLI args)."""
  cls = METHODS.get(name)
  logging.info("Built method '%s'", name)
  return cls()
