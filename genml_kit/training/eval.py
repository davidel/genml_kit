"""Evaluation metrics and validation-set evaluation.

Extracted from ``train.py`` to reduce module size and allow reuse in
other training scripts.
"""

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from genml_kit.training.model_utils import model_mode
from genml_kit.utils.attr import MISSING, get_attribute


def _compute_classification_metrics(all_labels, all_preds, num_labels, id2label):
  """Compute aggregate and per-class classification metrics."""
  labels = list(range(num_labels))
  precisions, recalls, f1s, supports = precision_recall_fscore_support(
      all_labels,
      all_preds,
      labels=labels,
      average=None,
      zero_division=0,
  )
  per_class_metrics = {}
  for idx, (precision, recall, f1,
            support) in enumerate(zip(precisions, recalls, f1s, supports)):
    name = (id2label.get(str(idx), id2label.get(idx, f"Class {idx}"))
            if id2label else f"Class {idx}")
    per_class_metrics[name] = {
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
        "support": int(support),
    }

  return {
      "balanced_accuracy":
          recalls.mean() * 100.0,
      "macro_f1":
          f1_score(
              all_labels,
              all_preds,
              labels=labels,
              average="macro",
              zero_division=0,
          ) * 100.0,
      "weighted_f1":
          f1_score(
              all_labels,
              all_preds,
              labels=labels,
              average="weighted",
              zero_division=0,
          ) * 100.0,
      "per_class_metrics":
          per_class_metrics,
      "cm":
          confusion_matrix(all_labels, all_preds, labels=labels),
  }


def evaluate_performance(model,
                         dataloader,
                         criterion,
                         device,
                         amp_dtype,
                         id2label=None,
                         tta_transform=None):
  """Evaluate on a validation/test set.

  Returns ``(eval_loss, top1_acc_pct, balanced_accuracy, macro_f1,
  weighted_f1, per_class_metrics, cm, original_metrics)``.  The first seven
  values describe the predictions used for evaluation (TTA predictions when
  enabled).  *original_metrics* is ``None`` without TTA; otherwise it contains
  the corresponding metrics for original-view predictions.

  When *tta_transform* is provided predictions are averaged over N views
  (including the original) while the loss is always computed on the
  original image only.
  """
  amp_enabled = (amp_dtype is not None and device.type == "cuda")
  eval_loss, correct_top1, total_samples = 0.0, 0, 0
  all_preds = []
  all_orig_preds = []
  all_labels = []
  with model_mode(model, "eval"), torch.no_grad():
    for images, targets in dataloader:
      images, targets = images.to(device), targets.to(device)
      with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
        logits_orig = model(pixel_values=images).logits
        loss = criterion(logits_orig, targets)

        if tta_transform is not None:
          views = tta_transform(images)
          B, N = views.shape[:2]
          logits_aug = model(pixel_values=views.flatten(0, 1)).logits
          probs = (logits_orig.softmax(-1) +
                   logits_aug.softmax(-1).unflatten(0, (B, N)).sum(1)) / (N + 1)
          preds = probs.argmax(dim=1)
        else:
          preds = logits_orig.argmax(dim=1)

      eval_loss += loss.item() * images.size(0)
      total_samples += targets.size(0)
      correct_top1 += (preds == targets).sum().item()
      all_preds.extend(preds.cpu().tolist())
      all_orig_preds.extend(logits_orig.argmax(dim=1).cpu().tolist())
      all_labels.extend(targets.cpu().tolist())

  avg_loss = eval_loss / total_samples
  top1 = (correct_top1 / total_samples) * 100.0

  num_labels = get_attribute(model, "config.num_labels")
  if num_labels is MISSING:
    num_labels = max(all_labels + all_preds) + 1

  metrics = _compute_classification_metrics(all_labels, all_preds, num_labels, id2label)
  original_metrics = None
  if tta_transform is not None:
    original_metrics = _compute_classification_metrics(all_labels, all_orig_preds,
                                                       num_labels, id2label)
    original_metrics["top1"] = (
        sum(pred == target for pred, target in zip(all_orig_preds, all_labels)) /
        total_samples * 100.0)

  return (avg_loss, top1, metrics["balanced_accuracy"], metrics["macro_f1"],
          metrics["weighted_f1"], metrics["per_class_metrics"], metrics["cm"],
          original_metrics)
