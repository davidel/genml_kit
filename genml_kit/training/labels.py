"""Label-space training utilities shared by pipelines, methods and tests.

Single source of truth for the class-weight / class-multiplier helpers
and Mixup: ``pipelines.images``, ``training.train_compat`` and
``methods.classification`` import from here instead of each carrying a
copy.

Import direction is safe in both directions: this module only imports
``torch`` / ``numpy`` and ``utils.logging`` -- it never imports
``pipelines`` or the rest of ``training``, so pipelines, methods and the
compat re-export hub can all depend on it.
"""

import torch

from genml_kit.utils.logging import fatal


def fmt_weights(weights, decimals=3):
  """Format a 1-D tensor as a human-readable list string."""
  return "[" + ", ".join(f"{v:.{decimals}f}" for v in weights.tolist()) + "]"


def compute_class_weights(train_dataset, num_labels, label_column="label"):
  """Compute inverse-frequency class weights from a labeled dataset.

  Weight formula: ``w_c = N / (num_labels * n_c)`` normalized so the
  weights sum to ``num_labels`` (mean weight 1.0); empty classes get the
  weight of a single sample.
  """
  import numpy as np

  if label_column not in train_dataset.column_names:
    fatal(f"Label column '{label_column}' not in dataset; cannot compute "
          f"class weights (columns={train_dataset.column_names}).",
          ValueError)
  labels = train_dataset[label_column]
  if not labels:
    fatal(
        f"Cannot compute class weights from an empty dataset "
        f"(num_labels={num_labels})",
        ValueError,
    )
  counts = np.bincount(labels, minlength=num_labels).astype(np.float64)
  counts = np.maximum(counts, 1.0)
  weights = 1.0 / counts
  weights = weights / weights.sum() * num_labels
  return torch.tensor(weights, dtype=torch.float32)


def parse_class_multipliers(s, num_labels, label2id):
  """Parse a ``--class_multipliers`` string into a ``[num_labels]`` tensor.

  *s* is a comma-separated string of ``NAME=VALUE`` pairs where *NAME* is
  a label string or an integer label index and *VALUE* is a float
  multiplier.  Unspecified classes default to ``1.0``.

  Returns a ``torch.Tensor`` of shape ``[num_labels]``.
  """
  m = torch.ones(num_labels)
  if not s or not s.strip():
    return m
  for pair in s.split(","):
    pair = pair.strip()
    if not pair:
      continue
    if "=" not in pair:
      fatal(
          f"Invalid --class_multipliers entry: '{pair}'. "
          "Expected NAME=VALUE (e.g. cat=4.0).",
          ValueError,
      )
    name, val = pair.split("=", 1)
    name, val = name.strip(), val.strip()
    if name.isdigit():
      idx = int(name)
    else:
      if name not in label2id:
        fatal(
            f"Unknown class name '{name}' in --class_multipliers. "
            f"Known labels: {sorted(label2id)}",
            ValueError,
        )
      idx = label2id[name]
    if not (0 <= idx < num_labels):
      fatal(f"Label index {idx} out of range [0, {num_labels})", ValueError)
    m[idx] = float(val)
  return m


def mixup_data(x, y, alpha=0.2):
  """Apply Mixup to a batch: returns mixed images, and two label sets + lambda.

  Returns ``(mixed_x, y_a, y_b, lam)`` where ``lam`` is the interpolation
  coefficient sampled from ``Beta(alpha, alpha)``.  When ``alpha <= 0`` the
  function is a no-op and returns the originals unchanged.
  """
  import numpy as np

  if alpha <= 0:
    return x, y, y, 1.0
  lam = np.random.beta(alpha, alpha)
  # Fold into lam so y_a is always the dominant label and callers can
  # use y_a alone for hard-label metrics; the y_b contribution survives
  # only through the 1-lam weight in the soft-target loss.
  lam = max(lam, 1.0 - lam)
  batch_size = x.size(0)
  index = torch.randperm(batch_size, device=x.device)
  mixed_x = lam * x + (1.0 - lam) * x[index]
  return mixed_x, y, y[index], lam
