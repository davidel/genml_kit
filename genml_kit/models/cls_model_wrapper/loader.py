"""ClsModelWrapper loader — called by the model registry.

Registered as ``cls_model_wrapper`` in the model registry.  Invoked via::

    --model cls_model_wrapper:google/vit-base-patch16-224 \\
    --model_arg classifier=mlp:hidden=512,dropout=0.3

The text after the model ``:`` is the HF model name/path (``backbone``).
The ``classifier`` kwarg is the inline head spec parsed by
:func:`~genml_kit.training.classifiers.parse_classifier_spec`.
"""

import logging

from genml_kit.models.cls_model_wrapper.model import ClsModelWrapper
from genml_kit.models.registry import register_model
from genml_kit.training.classifiers import parse_classifier_spec
from genml_kit.utils.logging import fatal


@register_model("cls_model_wrapper")
def load_cls_model_wrapper(*,
                           backbone,
                           num_labels,
                           id2label=None,
                           label2id=None,
                           image_size=224,
                           device="cpu",
                           **kwargs):
  """Instantiate :class:`ClsModelWrapper` and move to *device*."""
  if num_labels == 0:
    fatal(
        "cls_model_wrapper always requires a classification head "
        "(num_labels > 0). For headless encoders use the backbone model "
        "directly (e.g. --model timm:... or the HF name).", ValueError)

  classifier_spec = kwargs.pop("classifier", None)
  if not classifier_spec:
    fatal(
        "cls_model_wrapper requires a classifier head. Pass it via "
        "--model_arg, e.g. --model_arg classifier=mlp:hidden=512. "
        "Available built-ins: mlp, cls_attention. A .py path also works.", ValueError)
  classifier, classifier_args = parse_classifier_spec(classifier_spec)

  model = ClsModelWrapper(
      backbone_name=backbone,
      num_labels=num_labels,
      classifier=classifier,
      classifier_args=classifier_args,
  )
  model.to(device)

  logging.info(
      "ClsModelWrapper ready — classifier=%s, params=%.1fM",
      classifier,
      sum(p.numel() for p in model.parameters()) / 1e6,
  )
  return model
