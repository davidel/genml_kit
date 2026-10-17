"""seg_wrapper loader -- called by the model registry.

Registered as ``seg_wrapper``.  Invoked via::

    --model seg_wrapper:timm/resnet34 \\
    --model_arg decoder=unet:depth=4,decoder_channels=256

The text after ``:`` is the encoder backbone.  Two forms are accepted:

* a timm model name (optionally ``timm/``-prefixed), loaded with
  ``features_only=True`` so it yields a feature pyramid -- the canonical
  path for hierarchical encoders (ResNet / ConvNeXt / Swin / MiT);
* a registered custom encoder name (``MODELS`` registry).

The ``decoder`` kwarg is the inline decoder spec parsed by
:func:`~genml_kit.models.decoders.parse_decoder_spec`.
"""

import logging

from genml_kit.models.decoders import parse_decoder_spec
from genml_kit.models.registry import MODELS, PROCESSORS
from genml_kit.models.seg_model_wrapper.encoder import TimmEncoder
from genml_kit.models.seg_model_wrapper.model import build_seg_model
from genml_kit.utils.logging import fatal


def _build_timm_encoder(backbone, out_indices=None, pretrained=True, **kwargs):
  """Create a :class:`TimmEncoder` from a timm model string."""
  import timm

  name = _timm_name(backbone)
  model = timm.create_model(
      name,
      features_only=True,
      pretrained=pretrained,
      **kwargs,
  )
  return TimmEncoder(model, out_indices=out_indices)


def _timm_name(backbone):
  """Strip an optional ``timm/`` prefix so timm sees a bare model name."""
  return backbone[len("timm/"):] if backbone.startswith("timm/") else backbone


@MODELS.register("seg_wrapper")
def load_seg_wrapper(
    *,
    backbone=None,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    cache_dir=None,
    decoder="linear",
    encoder_out_indices=None,
    pretrained=True,
    **kwargs,
):
  """Build a ``SegModelWrapper`` (encoder + decoder) via the registry.

  Parameters
  ----------
  backbone : str
    Encoder backbone: a timm model string (optionally ``timm/``-prefixed)
    or a registered custom encoder name.
  num_labels : int
    Number of segmentation classes.  When ``0`` the number of classes is
    taken from ``id2label`` / ``label2id`` (HF convention) or defaults to
    1 (binary foreground).
  id2label, label2id : dict, optional
    Class maps (HF compatibility).  Used to recover ``num_labels`` when it
    is 0.
  image_size : int
    Input resolution; the encoder is probed at this size.
  device : str or torch.device
    Target device.
  decoder : str
    Decoder spec (registered name or inline ``name:kwargs``, or a ``.py``
    path defining ``create_decoder()``).
  encoder_out_indices : list[int] | str, optional
    Pyramid levels to expose.  Accepts a comma-separated string (CLI) or a
    list.  Default: all levels the encoder provides.
  pretrained : bool
    Whether to load pretrained encoder weights (timm only).
  **kwargs
    Extra arguments forwarded to the encoder factory.
  """
  if backbone is None:
    fatal("seg_wrapper requires a backbone, e.g. "
          "--model seg_wrapper:timm/resnet34", ValueError)

  num_labels = _resolve_num_labels(num_labels, id2label, label2id)
  out_indices = _parse_out_indices(encoder_out_indices)

  encoder = _build_encoder(backbone, out_indices, pretrained, kwargs)

  # Validate the decoder spec early so a typo fails before the probe.
  dec_name, _ = parse_decoder_spec(decoder)

  model = build_seg_model(
      encoder,
      decoder,
      num_classes=num_labels,
      image_size=image_size,
      device=device,
      id2label=id2label,
  )
  logging.info(
      "seg_wrapper ready - backbone=%s, decoder=%s, params=%.1fM",
      backbone,
      dec_name,
      sum(p.numel() for p in model.parameters()) / 1e6,
  )
  return model


def _resolve_num_labels(num_labels, id2label, label2id):
  """Recover the class count when the CLI passes 0 (HF convention)."""
  if num_labels:
    return num_labels
  if id2label:
    return len(id2label)
  if label2id:
    return len(label2id)
  logging.warning("seg_wrapper: num_labels=0 and no id2label/label2id; defaulting to "
                  "a single (binary foreground) class.")
  return 1


def _parse_out_indices(value):
  """Accept ``"0,1,2,3"`` or a list; return a list of ints or ``None``."""
  if value is None:
    return None
  if isinstance(value, str):
    if not value.strip():
      return None
    return [int(tok) for tok in value.split(",")]
  return list(value)


def _build_encoder(backbone, out_indices, pretrained, kwargs):
  """Build a timm encoder or a registered custom encoder."""
  # A registered custom encoder name wins over a timm string.
  custom = backbone[len("timm/"):] if backbone.startswith("timm/") else backbone
  if MODELS.contains(custom) and custom != "seg_wrapper":
    logging.info("seg_wrapper: loading custom encoder '%s'", custom)
    return MODELS.get(custom)(
        num_labels=0,
        image_size=kwargs.pop("image_size", 224),
        **kwargs,
    )
  return _build_timm_encoder(backbone,
                             out_indices=out_indices,
                             pretrained=pretrained,
                             **kwargs)


@PROCESSORS.register("seg_wrapper")
def load_seg_processor(*, backbone=None, image_size=224, **kwargs):
  """Return the encoder's processor (timm data config).

  Reuses the timm processor so segmentation shares the exact
  normalization / resize the encoder was trained with.
  """
  from genml_kit.models.timm.loader import load_timm_processor

  if backbone is None:
    fatal(
        "seg_wrapper processor requires a backbone "
        "(use --model seg_wrapper:<backbone>).", ValueError)
  return load_timm_processor(backbone=_timm_name(backbone), image_size=image_size)
