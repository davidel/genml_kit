"""Model registry - unified loading for HF and custom models.

The registry exposes two entry points:

- ``load_model(name, ...)``  -- returns a ``torch.nn.Module``
- ``load_processor(name, ...)`` -- returns a processor object that
  exposes ``image_mean`` and ``image_std`` attributes.

Both functions transparently dispatch to the appropriate backend
(custom or HuggingFace) based on *name*.

Custom model/processor loaders register themselves via the
``MODELS`` / ``PROCESSORS`` registry instances::

    from genml_kit.models.registry import MODELS

    @MODELS.register("uvito")
    def load_uvito(backbone, cache_dir, **kwargs): ...
"""

import logging
from collections import namedtuple

from genml_kit.utils.registry import Registry
from genml_kit.utils.script import load_extern

MODELS = Registry("model")
PROCESSORS = Registry("processor")


def _is_script_spec(name):
  """Return *True* if *name* is a script path/URL rather than a model name."""
  return bool(name) and (name.endswith(".py") or name.startswith(
      ("http://", "https://")))


ParsedModelName = namedtuple(
    "ParsedModelName",
    ["model", "backbone", "processor"],
)


def parse_model_name(name):
  """Parse a fully-qualified model name string.

  The colon syntax ``"model_name:hf_name"`` (used for custom models
  that wrap a HuggingFace backbone) is split into its components.

  Parameters
  ----------
  name : str
      Fully-qualified model name, e.g.
      ``"cls_model_wrapper:google/vit-base-patch16-224"`` or
      ``"convvit"``.

  Returns
  -------
  ParsedModelName
      A namedtuple with fields:

      - **model** -- The registered custom model name, or the full
        *name* when there is no colon.
      - **backbone** -- The HuggingFace backbone identifier (the part
        after the colon), or ``None``.
      - **processor** -- The HuggingFace model identifier used to
        load the processor.  Equals *backbone* when a colon is
        present, otherwise equals *name*.
  """
  if ":" in name:
    model, backbone = name.split(":", 1)
    return ParsedModelName(
        model=model,
        backbone=backbone,
        processor=backbone,
    )
  return ParsedModelName(model=name, backbone=None, processor=name)


class ModelOutput:
  """Lightweight container matching the ``.logits`` interface of
    HuggingFace model outputs.

    Custom models whose ``forward()`` returns a raw tensor should wrap
    it before returning::

        return ModelOutput(logits)
    """

  def __init__(self, logits):
    self.logits = logits


def is_custom_model(model_name):
  """Return *True* if *model_name* maps to a registered custom model."""
  return MODELS.contains(model_name)


def load_model(
    model_name,
    *,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    cache_dir=None,
    **kwargs,
):
  """Load a model by *model_name* -- custom, script, or HuggingFace.

    Custom models are dispatched to the function registered via
    ``MODELS.register``.  HuggingFace models are loaded via
    ``AutoModelForImageClassification``; ``num_labels == 0`` follows the
    timm convention and loads the headless backbone via ``AutoModel``.

    External scripts: if *model_name* is a ``.py`` path or HTTP(S) URL
    the script must define ``create_model(num_labels, image_size,
    id2label, label2id, checkpoint_path, device, **kwargs)`` returning
    a ``torch.nn.Module``.  The script is fully responsible for weight
    loading: when *checkpoint_path* is truthy (inference) it should
    call ``genml_kit.io.checkpointing.load_checkpoint_weights`` inside
    ``create_model``, mirroring the ConvViT/UViTO registered loaders.
    In training *checkpoint_path* is ``None`` (random init) and the
    global ``--source_checkpoint`` / ``--resume`` paths apply after
    ``create_model`` returns.

    Returns:
        ``torch.nn.Module``
    """
  # External script protocol (checked before parse_model_name: script
  # specs are not registered names and may contain ':' on Windows).
  if _is_script_spec(model_name):
    create_model = load_extern(model_name, "create_model")
    logging.info("Loading external model from %s", model_name)
    return create_model(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        device=device,
        **kwargs,
    )

  # Keep this import local so importing the registry does not eagerly load
  # the Transformers dependency.
  from transformers import AutoModel, AutoModelForImageClassification

  parsed = parse_model_name(model_name)

  if MODELS.contains(parsed.model):
    logging.info("Loading custom model '%s' from registry.", parsed.model)
    if parsed.backbone is not None:
      kwargs["backbone"] = parsed.backbone
    return MODELS.get(parsed.model)(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        image_size=image_size,
        device=device,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )

  if num_labels == 0:
    # timm convention: num_labels == 0 -> load the headless backbone
    # (e.g. ViTModel instead of ViTForImageClassification); used by the
    # self-supervised pre-training paths.
    logging.info("Loading headless HuggingFace model '%s'.", parsed.model)
    model = AutoModel.from_pretrained(parsed.model, cache_dir=cache_dir)
  else:
    logging.info("Loading HuggingFace model '%s'.", parsed.model)
    model = AutoModelForImageClassification.from_pretrained(
        parsed.model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
    )
  model.to(device)
  return model


def load_processor(
    model_name,
    *,
    image_size=224,
    cache_dir=None,
    **kwargs,
):
  """Load an image processor by *model_name* -- custom, script, or HuggingFace.

    Custom processors are dispatched to the function registered via
    ``PROCESSORS.register``.  HuggingFace processors are loaded via
    ``AutoImageProcessor``.

    External scripts: if *model_name* is a ``.py`` path or HTTP(S) URL
    the script may define ``create_processor(image_size, **kwargs)``
    returning an object exposing ``image_mean`` / ``image_std``.  When
    the script omits ``create_processor`` a ``ValueError`` is raised
    (matching the ``load_extern`` contract).

    Supports the ``"model_name:hf_name"`` colon syntax used by custom
    models (e.g. ``"cls_model_wrapper:google/vit-base-patch16-224"``).
    The part before the colon is checked against the custom registry
    first; the part after the colon is always used as the HuggingFace
    model identifier for the processor.

    The returned object must expose ``image_mean`` and ``image_std``
    (list of floats) so that ``build_transforms`` can use them.

    Returns:
        processor object with ``image_mean`` / ``image_std`` attributes.
    """
  # External script protocol (checked before parse_model_name).
  if _is_script_spec(model_name):
    create_processor = load_extern(model_name, "create_processor")
    logging.info("Loading external processor from %s", model_name)
    return create_processor(image_size=image_size, **kwargs)

  # Local to avoid top-level import.
  from transformers import AutoImageProcessor

  parsed = parse_model_name(model_name)

  # Check if a custom processor is registered under the model name
  # (e.g. "convvit") or under the backbone/processor name.
  if parsed.backbone is not None:
    kwargs["backbone"] = parsed.backbone
  for name in dict.fromkeys([parsed.model, parsed.processor]):
    if PROCESSORS.contains(name):
      logging.info("Loading custom processor '%s' from registry.", name)
      return PROCESSORS.get(name)(
          image_size=image_size,
          **kwargs,
      )

  logging.info("Loading HuggingFace processor '%s'.", parsed.processor)
  return AutoImageProcessor.from_pretrained(parsed.processor, cache_dir=cache_dir)
