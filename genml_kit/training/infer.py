"""Inference script: predict labels for images using a fine-tuned model."""

import argparse
import json
import logging

import numpy as np
import torch
from PIL import Image

from genml_kit.training.model_utils import build_val_transform, load_model_for_inference
from genml_kit.utils.args import add_logging_args
from genml_kit.utils.cli import KVPairAction
from genml_kit.utils.label import get_label
from genml_kit.utils.logging import fatal, setup_logging


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Classify images with a fine-tuned model (HuggingFace or timm).")
  parser.add_argument(
      "--model",
      required=True,
      help="HuggingFace model name/path or timm model "
      "(e.g. 'timm:eva02_base_patch14_224.mim_in22k'). "
      "Defines architecture and processor.",
  )
  parser.add_argument(
      "--checkpoint",
      required=True,
      help="Path to .pt checkpoint (raw or wrapped state dictionary).",
  )
  parser.add_argument(
      "--image_size",
      type=int,
      default=448,
      help="Image resize target (default: 448).",
  )
  parser.add_argument(
      "--output",
      type=str,
      help="Path to write JSON results.",
  )
  parser.add_argument(
      "--top_k",
      type=int,
      help="Only show top-K predictions (default: all).",
  )
  parser.add_argument(
      "--device",
      type=str,
      help="Device: cuda or cpu (default: auto-detect).",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      help="HuggingFace cache directory.",
  )
  add_logging_args(parser)
  parser.add_argument(
      "--xgboost_model",
      type=str,
      help="Optional XGBoost model path. If provided, run XGBoost alongside "
      "the PyTorch classifier.",
  )
  parser.add_argument(
      "--model_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override model configuration (repeatable). "
      "Example: --model_arg depth=6 num_heads=8",
  )
  parser.add_argument(
      "--proc_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override processor configuration (repeatable).",
  )
  parser.add_argument(
      "images",
      nargs="+",
      help="Image file paths and/or URLs.",
  )
  return parser.parse_args(argv)


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


def main(argv=None):
  args = parse_args(argv)
  setup_logging(args.log_level, args.log_targets)

  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  logging.info(f"Using device: {device}")

  model, processor = load_model_for_inference(
      model_name=args.model,
      checkpoint_path=args.checkpoint,
      device=device,
      cache_dir=args.cache_dir,
      model_kwargs=args.model_arg,
      proc_kwargs=args.proc_arg,
      image_size=args.image_size,
  )
  transform = build_val_transform(processor, args.image_size)

  xgb_model = None
  if args.xgboost_model:
    from xgboost import XGBClassifier
    xgb_model = XGBClassifier()
    xgb_model.load_model(args.xgboost_model)
    # XGBoost class indices must align with the backbone model's label
    # space; a mismatch would silently print wrong class names.
    check_xgb_label_alignment(xgb_model, model.config.id2label)
    logging.info(f"Loaded XGBoost model: {args.xgboost_model}")

  results = []
  for source in args.images:
    image = open_image(source)
    probs, pixel_values = predict_single(model, transform, image, device)

    predictions = []
    for idx in rank_indices(probs).tolist():
      predictions.append({
          "label": get_label(model.config.id2label, idx),
          "probability": round(probs[idx].item(), 4),
      })
    if args.top_k:
      predictions = predictions[:args.top_k]

    logging.info(f"\n{source}")
    for p in predictions:
      logging.info(f"  {p['probability']:.1%}  {p['label']}")

    result = {
        "source": source,
        "predictions": predictions,
    }

    if xgb_model is not None:
      from genml_kit.training.model_utils import extract_backbone_features
      with torch.no_grad():
        features = extract_backbone_features(model, pixel_values)
      xgb_probs = xgb_model.predict_proba(features.cpu().numpy())[0]
      xgb_predictions = []
      for idx in rank_indices(xgb_probs).tolist():
        xgb_predictions.append({
            "label": get_label(model.config.id2label, idx),
            "probability": round(float(xgb_probs[idx]), 4),
        })
      if args.top_k:
        xgb_predictions = xgb_predictions[:args.top_k]
      result["xgboost_predictions"] = xgb_predictions
      logging.info("  XGBoost:")
      for p in xgb_predictions:
        logging.info(f"  {p['probability']:.1%}  {p['label']}")

    results.append(result)

  if args.output:
    with open(args.output, "w") as f:
      json.dump(results, f, indent=2)
    logging.info(f"Results written to {args.output}")


if __name__ == "__main__":
  main()
