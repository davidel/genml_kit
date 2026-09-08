"""TimmForClassification — wraps a timm model for the genml_kit protocol.

The wrapper adds the ``.config`` namespace (``num_labels``, ``id2label``,
``label2id``) and exposes ``.classifier`` so that the existing
``extract_backbone_features`` machinery works out of the box.

timm models also provide ``forward_features`` and ``forward_head`` natively,
which the wrapper leverages for clean feature extraction without hooks.
"""

import torch.nn as nn

from genml_kit.models.registry import ModelOutput


class TimmForClassification(nn.Module):
  """Thin wrapper that makes a timm model match the genml_kit / HF interface.

  * ``forward(pixel_values=)`` → ``ModelOutput`` with ``.logits``
  * ``config.id2label`` / ``config.label2id`` accessible
  * ``extract_backbone_features(pixel_values)`` for XGBoost
  """

  def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.config = config

  @property
  def classifier(self):
    """Alias for the classification head.

    Searches ``.fc``, ``.head``, and ``.classifier`` in that order,
    matching both timm and HuggingFace conventions.
    """
    for attr in ("fc", "head", "classifier"):
      mod = getattr(self.model, attr, None)
      if mod is not None and isinstance(mod, nn.Module):
        return mod
    return None

  def forward(self, pixel_values=None, **kwargs):
    logits = self.model(pixel_values)
    return ModelOutput(logits)

  def extract_backbone_features(self, pixel_values):
    """Extract pre-logit features using timm's native feature API.

    Uses ``forward_features`` → ``forward_head(pre_logits=True)`` which
    returns pooled features *before* the classification head — much
    cleaner than hook-based extraction.

    Returns
    -------
    torch.Tensor
      Shape ``(B, num_features)`` — ready for the classifier or XGBoost.
    """
    features = self.model.forward_features(pixel_values)
    return self.model.forward_head(features, pre_logits=True)
