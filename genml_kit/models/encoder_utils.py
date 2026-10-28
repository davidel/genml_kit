"""Shared utilities for encoder feature extraction.

Several self-supervised model wrappers (ContrastiveEncoder, BYOL, DINO)
share two pieces of logic:

1. **Backbone dimension detection** — probing a list of known
   attribute paths to infer the encoder output dimension.

2. **Feature extraction** — trying ``extract_backbone_features`` first
   and falling back to a direct forward pass with HF output handling.

Extracted here to avoid triplicated code across model modules.

**Output-shape contract.** ``encode_with_backbone`` is the single entry
point for turning an encoder + image batch into features, and it is
expected to return a pooled 2-D ``(B, D)`` matrix.  The
``extract_backbone_features`` happy path, the ``pooler_output`` branch and
the ``last_hidden_state.mean(dim=1)`` branch all satisfy this.  The bare
fallback, however, returns the encoder output **verbatim** — so an encoder
whose forward yields a 3-D/4-D or otherwise un-pooled tensor produces a
non-``(B, D)`` result.  Callers that combine features from *two* encoders
(a feature-distillation student/teacher pair, say) must therefore call
``require_matching_features`` so a mismatch fails loudly instead of a
broadcasting, silently-wrong loss.
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
    # Headless backbones (e.g. ConvViT pre-training with num_labels=0) have
    # no classifier/head to probe, so fall back to the config width.
    "config.embed_dim",
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
      Feature tensor, ``(B, D)`` for the supported paths.  The bare fallback
      returns ``encoder(images)`` verbatim, so an encoder that outputs an
      un-pooled tensor yields a non-``(B, D)`` result — see the module
      docstring's output-shape contract and ``require_matching_features``.
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
  logits = getattr(raw, "logits", None)
  if logits is not None:
    pooler = getattr(raw, "pooler_output", None)
    if pooler is not None:
      return pooler
    last_hidden = getattr(raw, "last_hidden_state", None)
    if last_hidden is not None:
      return last_hidden.mean(dim=1)
    # Any .logits-only container (registry.ModelOutput).
    return logits
  return raw


def require_matching_features(*features, names=None):
  """Assert that every feature tensor shares exactly one shape.

  Guards the output-shape contract of ``encode_with_backbone``: when two (or
  more) feature tensors feed a loss that broadcasts — e.g. the L2 of a
  feature-distillation student/teacher pair — a shape mismatch would
  otherwise be silently absorbed by broadcasting, yielding a
  plausible-but-wrong objective.  This makes that case fail loudly.

  Args:
      *features: Two or more ``torch.Tensor`` feature tensors to compare.
      names: Optional sequence of labels (same length as *features*) used to
          identify each tensor in the error message.

  Returns:
      The first feature tensor, for convenient in-line use.

  Raises:
      ValueError: If fewer than two features are given, or if their shapes
          are not all identical.
  """
  if len(features) < 2:
    raise ValueError("require_matching_features needs at least two tensors.")
  shapes = [tuple(f.shape) for f in features]
  if len(set(shapes)) != 1:
    if names is not None:
      labels = [f"{n}:{s}" for n, s in zip(names, shapes)]
    else:
      labels = [str(s) for s in shapes]
    raise ValueError("Feature shapes must match exactly to avoid broadcasting; got " +
                     ", ".join(labels))
  return features[0]
