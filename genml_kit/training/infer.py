"""Reusable inference helpers for image classification.

These helpers back the ``genml-kit-infer`` console script
(:mod:`genml_kit.cli.infer`); they live here because they are library
code, independent of the argparse/CLI surface.
"""

import numpy as np
import torch
from PIL import Image

from genml_kit.utils.logging import fatal


def open_image(source):
  """Open an image from a file path or URL."""
  if source.startswith(("http://", "https://")):
    import urllib.request
    with urllib.request.urlopen(source) as resp:
      return Image.open(resp).convert("RGB")
  return Image.open(source).convert("RGB")


def rank_indices(probs):
  """Return class indices ordered by descending probability.

  Ties break by class index (stable sort) so equal-probability classes
  always appear in the same, deterministic order — both for the
  torch tensor and numpy array code paths.
  """
  if isinstance(probs, torch.Tensor):
    return torch.sort(probs, descending=True, stable=True).indices
  return np.argsort(-np.asarray(probs), kind="stable")


def predict_single(model, transform, image, device):
  """Run inference on one image. Returns (logits, pixel_values) tensor."""
  pixel_values = transform(image).unsqueeze(0).to(device)
  with torch.no_grad():
    logits = model(pixel_values).logits
  probs = torch.softmax(logits, dim=-1)[0]
  return probs, pixel_values


def check_xgb_label_alignment(xgb_model, id2label):
  """Fail fast if an XGBoost model's class space mismatches the backbone.

  XGBoost class indices must align with the backbone model's label
  space; a mismatch would silently print wrong class names in the
  prediction output.

  Args:
      xgb_model: A fitted XGBClassifier (exposes ``n_classes_``).
      id2label: The backbone model's label mapping.

  Raises:
      ValueError: Via ``fatal`` when the class counts differ.
  """
  if getattr(xgb_model, "n_classes_", None) != len(id2label):
    fatal(
        f"XGBoost model has {getattr(xgb_model, 'n_classes_', '?')} classes "
        f"but the model config defines {len(id2label)} labels; the XGBoost "
        f"model does not match this checkpoint.",
        ValueError,
    )
