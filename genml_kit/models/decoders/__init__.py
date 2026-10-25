"""Decoder registry -- built-in and user-supplied segmentation decoders.

Usage::

    from genml_kit.models.decoders import build_decoder

    decoder = build_decoder("unet:depth=4", in_channels=[64, 128, 256],
                            num_classes=21)

A decoder spec can be:

* A registered name (e.g. ``"unet"``, ``"fpn"``, ``"linear"``).
* A path/URL to a ``.py`` file defining a ``create_decoder()`` function.
* An inline spec ``"name:key=value,..."`` parsed by
  :func:`parse_decoder_spec`.

The same three forms (and the same parser) are used for classifier heads;
see :mod:`genml_kit.training.classifiers`.
"""

import logging

from genml_kit.utils.cli import parse_inline_spec
from genml_kit.utils.logging import fatal
from genml_kit.utils.registry import Registry
from genml_kit.utils.script import load_extern, is_script_spec

DECODERS = Registry("decoder")


def parse_decoder_spec(spec):
  """Split an inline decoder spec into ``(name, kwargs)``.

  The CLI expresses decoders as a single string, e.g.::

      decoder=unet:depth=4,decoder_channels=256

  Delegates to :func:`~genml_kit.utils.cli.parse_inline_spec`.

  Parameters
  ----------
  spec : str
      Inline decoder specification.

  Returns
  -------
  (str, dict)
      The decoder name and its constructor kwargs.
  """
  return parse_inline_spec(spec, kind="decoder")


def build_decoder(spec, in_channels, num_classes, **kwargs):
  """Instantiate a segmentation decoder.

  Parameters
  ----------
  spec : str
      Registered decoder name, an inline ``name:kwargs`` spec, or a path
      to a ``.py`` file exporting ``create_decoder()``.
  in_channels : list[int]
      Per-level encoder channel counts, finest level first.  Discovered by
      probing the encoder (see ``seg_model_wrapper``).
  num_classes : int
      Number of output classes.
  **kwargs
      Extra keyword arguments forwarded to the decoder constructor.

  Returns
  -------
  BaseDecoder
  """
  name, spec_kwargs = parse_decoder_spec(spec)
  spec_kwargs.update(kwargs)

  if is_script_spec(name):
    builder = load_extern(name, "create_decoder")
    decoder = builder(in_channels=in_channels, num_classes=num_classes, **spec_kwargs)
    logging.info("Built external decoder from %s", name)
    return decoder

  if not DECODERS.contains(name):
    available = ", ".join(DECODERS.list_names()) or "(none)"
    fatal(
        f"Unknown decoder {name!r}. Available: {available}. "
        f"Or provide a path to a .py file.", ValueError)
  logging.info("Decoder kwargs: %s", spec_kwargs)
  return DECODERS.build(name,
                        in_channels=in_channels,
                        num_classes=num_classes,
                        **spec_kwargs)


def _register_builtins():
  """Import built-in decoders so their ``@DECODERS.register`` fires."""
  from genml_kit.models.decoders import fcos_head, fpn, linear, unet  # noqa: F401


_register_builtins()
