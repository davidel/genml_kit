"""CLS-guided attention pooling classifier head."""

import torch.nn as nn

from genml_kit.models.attention_pooling import CLSGuidedAttentionPooling
from genml_kit.training.classifiers import register_classifier
from genml_kit.training.classifiers.base import BaseClassifier
from genml_kit.utils.transformer import build_transformer_encoder


@register_classifier("cls_attention")
class Classifier(BaseClassifier):
  """Attention-pooled backbone features followed by a linear head.

  Uses the CLS token as a query to cross-attend over a subset of
  spatial tokens from the backbone, producing a single pooled
  representation that feeds a linear classifier.

  An optional ``nn.TransformerEncoder`` can be applied to the spatial
  tokens before the pooling step.  This is useful when the backbone is
  fully frozen and you want additional task-specific capacity without
  modifying the backbone weights.

  Parameters
  ----------
  num_labels : int
      Number of output classes.
  hidden_size : int
      Dimensionality of backbone hidden states (``D``).
  cls_slice : tuple[int, int]
      Start/end indices for the CLS token slice.  Default ``(0, 1)``.
  spc_slice : tuple[int, int | None]
      Start/end indices for the spatial token slice.  Tokens before
      the start index (e.g. register tokens) are dropped.
      Default ``(1, None)`` keeps all spatial tokens after CLS.
  num_heads : int
      Number of attention heads in the pooling layer.
  dropout : float
      Dropout rate inside the pooling layer.
  num_encoder_layers : int
      Number of ``TransformerEncoderLayer`` blocks applied to the
      spatial tokens before pooling.  ``0`` disables the encoder.
  encoder_dropout : float
      Dropout rate inside the optional encoder layers.
  """

  def __init__(
      self,
      num_labels,
      hidden_size,
      cls_slice=(0, 1),
      spc_slice=(1, None),
      num_heads=8,
      dropout=0.1,
      num_encoder_layers=0,
      encoder_dropout=0.1,
  ):
    super().__init__()
    self._cls_slice = slice(*cls_slice)
    self._spc_slice = slice(*spc_slice)

    self.encoder = None
    if num_encoder_layers > 0:
      encoder_layer = nn.TransformerEncoderLayer(
          d_model=hidden_size,
          nhead=num_heads,
          dim_feedforward=hidden_size * 4,
          dropout=encoder_dropout,
          batch_first=True,
      )
      self.encoder = build_transformer_encoder(
          encoder_layer,
          num_layers=num_encoder_layers,
      )

    self.pool = CLSGuidedAttentionPooling(
        embed_dim=hidden_size,
        num_heads=num_heads,
        dropout=dropout,
    )
    self.head = nn.Linear(hidden_size, num_labels)

  def forward(self, hidden_states):
    return self.head(self.extract_features(hidden_states))

  def extract_features(self, hidden_states):
    cls_out = hidden_states[:, self._cls_slice, :]
    spatial_out = hidden_states[:, self._spc_slice, :]
    if self.encoder is not None:
      spatial_out = self.encoder(spatial_out)
    return self.pool(cls_out, spatial_out)  # (B, D)
