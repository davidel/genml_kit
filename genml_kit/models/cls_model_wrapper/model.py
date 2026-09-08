"""ClsModelWrapper — HuggingFace backbone with a custom classifier head."""

import logging

import torch
import torch.nn as nn
from transformers import AutoModel

from genml_kit.models.registry import ModelOutput
from genml_kit.training.classifiers import build_classifier
from genml_kit.utils.logging import fatal


class ClsModelWrapper(nn.Module):
  """Wraps a HuggingFace backbone with a custom classifier head.

  The backbone lives here.  The classifier receives plain
  ``(B, N, D)`` tensors and does **not** hold a reference to the
  backbone.

  Parameters
  ----------
  backbone_name : str
      HuggingFace model name or path.
  num_labels : int
      Number of output classes.
  classifier : str
      Classifier spec (registered name or ``.py`` path).
  classifier_args : dict
      Extra kwargs forwarded to the classifier constructor.
  """

  def __init__(
      self,
      backbone_name,
      num_labels,
      classifier,
      classifier_args=None,
  ):
    super().__init__()
    base = AutoModel.from_pretrained(backbone_name)
    logging.info(
        "ClsModelWrapper: loaded backbone '%s' (%.1fM params)",
        backbone_name,
        sum(p.numel() for p in base.parameters()) / 1e6,
    )

    self.backbone = base
    hidden_size = getattr(base.config, "hidden_size", 768)
    self.classifier = build_classifier(
        spec=classifier,
        num_labels=num_labels,
        hidden_size=hidden_size,
        **(classifier_args or {}),
    )
    self.config = base.config

  @staticmethod
  def _extract_hidden_states(raw):
    """Translate any backbone output into a ``(B, N, D)`` tensor.

    Handles HuggingFace ``BaseModelOutput`` objects, plain tensors,
    and dict-style outputs.
    """
    if isinstance(raw, torch.Tensor):
      return raw
    if hasattr(raw, "last_hidden_state"):
      return raw.last_hidden_state
    if isinstance(raw, dict) and "last_hidden_state" in raw:
      return raw["last_hidden_state"]
    fatal(
        f"Cannot extract hidden states from {type(raw).__name__}. "
        "Expected a tensor, a BaseModelOutput, or a dict with "
        "'last_hidden_state' key.", ValueError)

  def forward(self, pixel_values):
    raw = self.backbone(pixel_values)
    hidden_states = self._extract_hidden_states(raw)
    return ModelOutput(logits=self.classifier(hidden_states))

  def extract_backbone_features(self, pixel_values):
    """Extract classifier-level features for XGBoost / external use.

    Runs the backbone, translates the output to a plain tensor, and
    delegates to the classifier's :meth:`extract_features` method.

    Parameters
    ----------
    pixel_values : Tensor
      Shape ``(B, C, H, W)`` — preprocessed input images.

    Returns
    -------
    Tensor
      Shape ``(B, F)`` — features that the classification head uses.
    """
    from genml_kit.training.model_utils import model_mode

    with model_mode(self, "eval"), torch.no_grad():
      raw = self.backbone(pixel_values)
      hidden_states = self._extract_hidden_states(raw)
      return self.classifier.extract_features(hidden_states)
