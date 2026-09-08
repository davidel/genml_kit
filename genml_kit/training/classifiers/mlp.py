"""Two-layer MLP classifier head."""

import torch.nn as nn

from genml_kit.training.classifiers import register_classifier
from genml_kit.training.classifiers.base import BaseClassifier


@register_classifier("mlp")
class Classifier(BaseClassifier):
  """Two-layer MLP classification head.

  Receives backbone hidden states ``(B, N, D)`` and classifies the
  CLS token(s) selected by *cls_slice*.

  Parameters
  ----------
  num_labels : int
      Number of output classes.
  hidden_size : int
      Backbone hidden dimension ``D``.
  hidden : int
      Hidden layer width.
  dropout : float
      Dropout probability.
  cls_slice : tuple[int, int]
      ``(start, end)`` selecting which tokens to feed to the MLP.
      Default ``(0, 1)`` picks the first CLS token only.
      Multiple tokens are flattened into a ``(B, K*D)`` feature
      vector, where ``K = end - start``.
  """

  def __init__(self,
               num_labels,
               hidden_size,
               hidden=512,
               dropout=0.3,
               cls_slice=(0, 1)):
    super().__init__()
    self._cls_slice = slice(*cls_slice)
    feat_dim = (cls_slice[1] - cls_slice[0]) * hidden_size
    self.head = nn.Sequential(
        nn.Linear(feat_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, num_labels),
    )

  def forward(self, hidden_states):
    return self.head(self.extract_features(hidden_states))

  def extract_features(self, hidden_states):
    cls_tokens = hidden_states[:, self._cls_slice, :]  # (B, K, D)
    return cls_tokens.flatten(start_dim=1)  # (B, K*D)
