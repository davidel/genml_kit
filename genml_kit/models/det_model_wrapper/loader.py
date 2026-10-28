"""det_wrapper loader -- called by the model registry.

Registered as ``det_wrapper``.  Invoked via::

    --model det_wrapper:timm/resnet50 \\
    --model_arg head=fcos_head:num_convs=2

The text after ``:`` is the encoder backbone, resolved exactly as the
segmentation wrapper resolves it: a timm model name (optionally
``timm/``-prefixed) loaded with ``features_only=True`` so it yields a
feature pyramid.
"""

import logging

from genml_kit.models.decoders import parse_decoder_spec
from genml_kit.models.det_model_wrapper.model import build_det_model
from genml_kit.models.registry import MODELS, PROCESSORS
from genml_kit.models.seg_model_wrapper.encoder import TimmEncoder
from genml_kit.models.seg_model_wrapper.loader import _timm_name
from genml_kit.utils.logging import fatal


def _build_timm_encoder(backbone,
                        out_indices=None,
                        pretrained=True,
                        cache_dir=None,
                        **kwargs):
  """Create a :class:`TimmEncoder` from a timm model string."""
  import timm

  name = _timm_name(backbone)
  model = timm.create_model(
      name,
      features_only=True,
      pretrained=pretrained,
      cache_dir=cache_dir,
      **kwargs,
  )
  return TimmEncoder(model, out_indices=out_indices)


def _parse_out_indices(value):
  """Normalise ``encoder_out_indices`` (``"2,3,4"`` or a list) to a list."""
  if value is None:
    return None
  if isinstance(value, str):
    return [int(part) for part in value.split(",") if part.strip() != ""]
  return [int(part) for part in value]


@MODELS.register("det_wrapper")
def load_det_wrapper(
    *,
    backbone=None,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    cache_dir=None,
    head="fcos_head",
    encoder_out_indices=None,
    pretrained=True,
    **kwargs,
):
  """Build a ``DetModelWrapper`` (encoder + detection head) via the registry.

  Parameters
  ----------
  backbone : str
    Encoder backbone: a timm model string (optionally ``timm/``-prefixed).
  num_labels : int
    Number of foreground classes.  When ``0`` the count is recovered from
    ``id2label`` / ``label2id`` or defaults to 1.
  head : str
    Detection head spec (registered name, inline ``name:kwargs``, or a
    ``.py`` path defining ``create_decoder()``).  Defaults to ``fcos_head``.
  encoder_out_indices : list[int] | str, optional
    Pyramid levels to expose (comma-separated string or list).
  pretrained : bool
    Whether to load pretrained encoder weights (timm only).
  """
  if backbone is None:
    fatal("det_wrapper requires a backbone "
          "(use --model det_wrapper:<backbone>).", ValueError)

  num_labels = _resolve_num_labels(num_labels, id2label, label2id)
  out_indices = _parse_out_indices(encoder_out_indices)
  encoder = _build_timm_encoder(backbone,
                                out_indices=out_indices,
                                pretrained=pretrained,
                                cache_dir=cache_dir,
                                **kwargs)

  # Validate the head spec early so a typo fails before the probe.
  head_name, _ = parse_decoder_spec(head)

  # ``checkpoint_path`` is accepted for protocol parity but intentionally
  # unused here: the training harness restores weights itself (via
  # ``states_to_load``), exactly as it does for ``seg_wrapper``.
  del checkpoint_path
  model = build_det_model(
      encoder,
      head,
      num_classes=num_labels,
      image_size=image_size,
      device=device,
      id2label=id2label,
  )
  logging.info(
      "det_wrapper ready - backbone=%s, head=%s, params=%.1fM",
      backbone,
      head_name,
      sum(p.numel() for p in model.parameters()) / 1e6,
  )
  return model


def _resolve_num_labels(num_labels, id2label, label2id):
  """Recover the class count when the caller passes ``0`` (HF convention)."""
  if num_labels:
    return int(num_labels)
  if id2label:
    return len(id2label)
  if label2id:
    return len(label2id)
  return 1


@PROCESSORS.register("det_wrapper")
def load_det_processor(*, backbone=None, image_size=224, cache_dir=None, **kwargs):
  """Return the encoder's processor (timm data config).

  Reuses the timm processor so detection shares the exact normalization /
  resize the encoder was trained with.
  """
  from genml_kit.models.timm.loader import load_timm_processor

  if backbone is None:
    fatal(
        "det_wrapper processor requires a backbone "
        "(use --model det_wrapper:<backbone>).", ValueError)
  return load_timm_processor(backbone=_timm_name(backbone),
                             image_size=image_size,
                             cache_dir=cache_dir)
