"""timm loader — called by the model registry.

Registers the ``"timm"`` model and processor names.  Invoked via::

    --model timm:eva02_base_patch14_224.mim_in22k
    --model timm:hf_hub:timm/eva02_base_patch14_224.mim_in22k
    --model timm:hf_hub:timm/eva02_base_patch14_224.mim_in22k \\
        --classifier mlp --classifier_args "hidden=512"

Everything after the ``timm:`` prefix is passed verbatim to
``timm.create_model()``.
"""

import logging
from types import SimpleNamespace

from genml_kit.models.registry import (
    register_model,
    register_processor,
)
from genml_kit.models.timm.model import TimmForClassification
from genml_kit.models.timm.processor import TimmProcessor


@register_model("timm")
def load_timm_model(
    *,
    backbone,
    num_labels,
    id2label=None,
    label2id=None,
    image_size=224,
    device="cpu",
    checkpoint_path=None,
    cache_dir=None,
    pretrained=True,
    classifier_args=None,
    **kwargs,
):
  """Instantiate a timm model wrapped for the genml_kit protocol.

  Parameters
  ----------
  backbone : str
    timm model string, e.g. ``"eva02_base_patch14_224.mim_in22k"`` or
    ``"hf_hub:timm/eva02_base_patch14_224.mim_in22k"``.
  num_labels : int
    Number of output classes.
  id2label : dict, optional
    Mapping ``{int: str}`` for class labels.
  label2id : dict, optional
    Mapping ``{str: int}`` for class labels.
  image_size : int
    Input resolution (pixels).
  device : str or torch.device
    Target device.
  checkpoint_path : str, optional
    Not used for initial creation (weights are loaded via
    ``source_checkpoint`` in ``train.py``).
  cache_dir : str, optional
    Directory to cache downloaded weights.
  **kwargs
    Extra arguments forwarded to ``timm.create_model``
    (e.g. ``drop_path_rate``, ``pretrained_cfg_overlay``).
  """
  import timm  # local import so the module is optional

  timm_model = timm.create_model(
      backbone,
      pretrained=pretrained,
      num_classes=num_labels,
      **kwargs,
  )

  config = SimpleNamespace(
      num_labels=num_labels,
      id2label=id2label or {},
      label2id=label2id or {},
  )

  wrapped = TimmForClassification(timm_model, config)
  wrapped.to(device)

  logging.info(
      "timm model ready — backbone=%s, params=%.1fM",
      backbone,
      sum(p.numel() for p in wrapped.parameters()) / 1e6,
  )
  return wrapped


@register_processor("timm")
def load_timm_processor(*, backbone, image_size=224, **kwargs):
  """Return a :class:`TimmProcessor` for the given *image_size*.

  Reads the model's ``pretrained_cfg`` to obtain the correct
  normalization and interpolation settings.
  """
  import timm  # local import so the module is optional

  timm_model = timm.create_model(backbone, pretrained=False)
  data_config = timm.data.resolve_data_config(timm_model.pretrained_cfg)

  logging.info(
      "timm processor — backbone=%s, mean=%s, std=%s",
      backbone,
      data_config["mean"],
      data_config["std"],
  )
  return TimmProcessor(data_config, image_size=image_size)
