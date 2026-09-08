"""Base classifier interface."""

import abc

import torch
import torch.nn as nn


class BaseClassifier(nn.Module, abc.ABC):
  """Abstract base class for all genml_kit classifiers.

  A classifier receives pre-extracted backbone hidden states (a plain
  ``(B, N, D)`` tensor) and produces class logits.  It does **not**
  own or call the backbone — that responsibility belongs to
  ``ClsModelWrapper`` (or another top-level model).

  Subclasses must implement :meth:`forward` and :meth:`extract_features`.
  """

  @abc.abstractmethod
  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Classify *hidden_states* and return logits ``(B, num_labels)``.

    Parameters
    ----------
    hidden_states : Tensor
      Shape ``(B, N, D)`` — backbone output already normalised to a
      plain tensor (no ``BaseModelOutput`` wrapper).
    """

  @abc.abstractmethod
  def extract_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the feature vector fed to the classification head.

    Parameters
    ----------
    hidden_states : Tensor
      Shape ``(B, N, D)`` — same tensor passed to :meth:`forward`.

    Returns
    -------
    Tensor
      Shape ``(B, F)`` where ``F`` is the feature dimension consumed by
      the head (e.g. CLS token dim, pooled dim).
    """
