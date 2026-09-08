"""Model registry — unified loading for HF and custom models.

The registry exposes two entry points:

- ``load_model(name, ...)``  — returns a ``torch.nn.Module``
- ``load_processor(name, ...)`` — returns a processor object that
  exposes ``image_mean`` and ``image_std`` attributes.

Both functions transparently dispatch to the appropriate backend
(custom or HuggingFace) based on *name*.
"""

import logging
from collections import namedtuple

from genml_kit.utils.logging import fatal

_MODEL_REGISTRY = {}
_PROCESSOR_REGISTRY = {}

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

      - **model** – The registered custom model name, or the full
        *name* when there is no colon.
      - **backbone** – The HuggingFace backbone identifier (the part
        after the colon), or ``None``.
      - **processor** – The HuggingFace model identifier used to
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


def register_model(name):
  """Decorator to register a custom model loader under *name*.

    Usage::

        @register_model("convvit")
        def load_convvit(num_labels, id2label, label2id, image_size,
                         device, checkpoint_path, **kwargs):
            ...
            return model
    """

  def wrapper(fn):
    if name in _MODEL_REGISTRY:
      fatal(f"Model '{name}' is already registered.", ValueError)
    _MODEL_REGISTRY[name] = fn
    return fn

  return wrapper


def register_processor(name):
  """Decorator to register a custom processor loader under *name*.

    The loader must return an object with ``image_mean`` and ``image_std``

    attributes (list of floats).

    Usage::

        @register_processor("convvit")
        def load_convvit_processor(image_size, **kwargs):
            ...
            return processor
    """

  def wrapper(fn):
    if name in _PROCESSOR_REGISTRY:
      fatal(f"Processor '{name}' is already registered.", ValueError)
    _PROCESSOR_REGISTRY[name] = fn
    return fn

  return wrapper


def is_custom_model(model_name):
  """Return *True* if *model_name* maps to a registered custom model."""
  return model_name in _MODEL_REGISTRY


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
  """Load a model by *model_name* — custom or HuggingFace.

    Custom models are dispatched to the function registered via
    ``@register_model``.  HuggingFace models are loaded via
    ``AutoModelForImageClassification``; ``num_labels == 0`` follows the
    timm convention and loads the headless backbone via ``AutoModel``.

    Returns:
        ``torch.nn.Module``
    """
  # Keep this import local so importing the registry does not eagerly load
  # the Transformers dependency.
  from transformers import AutoModel, AutoModelForImageClassification

  parsed = parse_model_name(model_name)

  if parsed.model in _MODEL_REGISTRY:
    logging.info("Loading custom model '%s' from registry.", parsed.model)
    if parsed.backbone is not None:
      kwargs["backbone"] = parsed.backbone
    return _MODEL_REGISTRY[parsed.model](
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
  """Load an image processor by *model_name* — custom or HuggingFace.

    Custom processors are dispatched to the function registered via
    ``@register_processor``.  HuggingFace processors are loaded via
    ``AutoImageProcessor``.

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
  from transformers import AutoImageProcessor  # local to avoid top-level import

  parsed = parse_model_name(model_name)

  # Check if a custom processor is registered under the model name
  # (e.g. "convvit") or under the backbone/processor name.
  if parsed.backbone is not None:
    kwargs["backbone"] = parsed.backbone
  for name in dict.fromkeys([parsed.model, parsed.processor]):
    if name in _PROCESSOR_REGISTRY:
      logging.info("Loading custom processor '%s' from registry.", name)
      return _PROCESSOR_REGISTRY[name](
          image_size=image_size,
          **kwargs,
      )

  logging.info("Loading HuggingFace processor '%s'.", parsed.processor)
  return AutoImageProcessor.from_pretrained(parsed.processor, cache_dir=cache_dir)
