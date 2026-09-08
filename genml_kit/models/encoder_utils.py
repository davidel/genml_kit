"""Shared utilities for encoder feature extraction.

Several self-supervised model wrappers (ContrastiveEncoder, BYOL, DINO)
share two pieces of logic:

1. **Backbone dimension detection** — probing a list of known
   attribute paths to infer the encoder output dimension.

2. **Feature extraction** — trying ``extract_backbone_features`` first
   and falling back to a direct forward pass with HF output handling.

Extracted here to avoid triplicated code across model modules.
"""

import torch

from genml_kit.utils.attr import MISSING, get_attribute

# Attribute paths probed (in order) to infer the backbone output dimension.
_BACKBONE_DIM_PATHS = (
    "config.hidden_size",
    "config.d_model",
    "config.num_features",
    "model.num_features",
    "num_features",
    "head.in_features",
    "model.head.in_features",
    "classifier.in_features",
    "classifier.feat_dim",
)


def detect_backbone_dim(encoder):
  """Infer the backbone output feature dimension by probing known attributes.

  Args:
      encoder: A backbone encoder module (HF model, timm model, etc.).

  Returns:
      The inferred feature dimension (int).

  Raises:
      ValueError: If no known attribute path resolves to a value.
  """
  for path in _BACKBONE_DIM_PATHS:
    val = get_attribute(encoder, path)
    if val is not MISSING:
      return val
  raise ValueError("Cannot infer backbone output dimension.")


def encode_with_backbone(encoder, images):
  """Extract raw backbone features ``(B, D)`` from an encoder.

  Tries ``genml_kit.training.model_utils.extract_backbone_features`` first (which
  hooks into the model's classifier / pooling layer).  On failure falls
  back to a direct forward pass, handling HuggingFace model outputs
  that expose ``logits`` / ``pooler_output`` / ``last_hidden_state``.

  Args:
      encoder: Backbone encoder module.
      images: Input tensor ``(B, C, H, W)``.

  Returns:
      Feature tensor of shape ``(B, D)``.
  """
  from genml_kit.training.model_utils import extract_backbone_features
  try:
    return extract_backbone_features(encoder, images)
  except torch.cuda.OutOfMemoryError:
    # A genuine OOM must propagate: silently retrying a direct forward
    # pass would OOM again anyway and mask the real problem.
    raise
  except (ValueError, AttributeError, TypeError):
    # The hook-based extraction does not support this encoder; fall back
    # to a plain forward pass.
    pass
  raw = encoder(images)
  if hasattr(raw, "logits"):
    if raw.pooler_output is not None:
      return raw.pooler_output
    return raw.last_hidden_state.mean(dim=1)
  return raw
