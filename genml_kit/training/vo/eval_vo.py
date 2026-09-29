"""Sliced VO evaluation: terrain x range-bin breakdowns.

Renders one text table per slice with the utils/table formatter (repo
style).  The model-side metrics come from
``train_vo.vo_metrics_from_output`` -- the same function the training
validation path uses -- so a number in a slice table and a number in the
training log are produced by identical code.

Two error columns are reported per slice:

* ``mce_px``    -- the *model's* corner reprojection error.
* ``gt_fit_px`` -- how far the ground-truth similarity fit itself moves
  the frame corners, i.e. the magnitude of the motion being estimated.
  Reported as a reference scale for the model column.
"""

import collections
import math

import torch

from genml_kit.geometry.similarity import (
    SimilarityParams,
    corner_residual,
)
from genml_kit.training.vo.train_vo import (VOMetrics, _gt_corners,
                                            vo_metrics_from_output)
from genml_kit.utils.table import format_table

EvalRow = collections.namedtuple(
    "EvalRow", ["slice", "n", "mce", "dlog_s", "dtheta", "conf_mae", "gt_fit_mce"])

# VOMetrics fields, accumulated in this order.
_FIELDS = ("mce", "dlog_s", "dtheta", "conf_mae")


def evaluate_sliced(dataset, predict, group_key="terrain", image_size=64):
  """Evaluate per slice of *group_key* and render a text table.

  Args:
      dataset: indexed VO dataset; each item carries ``meta`` with
          ``gt`` (a dict of 0-dim tensors: log_s, theta, t),
          ``gt_residual`` and *group_key*.
      predict: callable ``item -> VOModelOutput`` producing a
          single-item (leading batch of 1) prediction.  The caller is
          responsible for eval mode / no_grad; this function only
          groups and averages.
      group_key: metadata field to slice on, e.g. ``"terrain"`` or
          ``"range_bin"``.
      image_size: frame size in pixels; sets the corners used for the
          GT-fit reference column.

  Returns:
      (rows, table): one :class:`EvalRow` per slice, and the rendered
      text lines.  ``EvalRow.mce`` is model error, ``EvalRow.gt_fit_mce``
      the ground-truth-fit reference.

  Note:
      Previously ``(model, dataset, device, group_key)``.  The *model*
      and *device* arguments were never used: the error columns guarded
      on ``"pred" not in item``, which is never true for items from
      ``VOPairDataset``, so ``dlog_s`` / ``dtheta`` / ``conf_mae``
      always came back ``NaN``.  An explicit *predict* callable makes
      the prediction path visible and testable.
  """
  groups = collections.defaultdict(list)
  for index in range(len(dataset)):
    item = dataset[index]
    groups[getattr(item["meta"], group_key)].append(item)

  rows = []
  for key in sorted(groups):
    items = groups[key]
    accum = collections.defaultdict(list)
    for item in items:
      out = predict(item)
      src, dst = _gt_corners(item["meta"])
      metrics = vo_metrics_from_output(out, _batched_gt(item["meta"].gt),
                                       item["meta"].gt_residual.reshape(1),
                                       src.unsqueeze(0), dst.unsqueeze(0))
      for name in _FIELDS:
        accum[name].append(getattr(metrics, name))
    means = VOMetrics(**{
        name: sum(values) / len(values) for name, values in accum.items()
    })
    rows.append(
        EvalRow(key, len(items), means.mce, means.dlog_s, means.dtheta, means.conf_mae,
                _gt_fit_mce(items, image_size)))

  # format_table needs every cell as a str: the slice key may be an int
  # (group_key="range_bin").
  table = format_table(
      ["slice", "n", "mce_px", "gt_fit_px", "dlog_s", "dtheta_deg", "conf_mae"], [[
          str(row.slice),
          str(row.n), f"{row.mce:.2f}", f"{row.gt_fit_mce:.2f}", f"{row.dlog_s:.4f}",
          f"{row.dtheta * 180.0 / math.pi:.2f}", f"{row.conf_mae:.3f}"
      ] for row in rows])
  return rows, table


def _batched_gt(gt):
  """Turn a per-item ``gt`` dict into batch-of-1 tensors.

  ``log_s``/``theta`` are 0-dim per item, while ``t`` is already (1, 2);
  reshape only the 0-dim ones so ``t`` is not double-batched.
  """
  return {
      key: (value.reshape(1) if value.dim() == 0 else value)
      for key, value in gt.items()
  }


def _gt_fit_mce(items, image_size):
  """How far the GT similarity fit itself moves the frame corners.

  Uses the same corner reprojection as the model metric, but with the
  ground-truth parameters, and with identity as the target.  The corner
  layout is derived from *image_size* -- a hardcoded size would silently
  report a different quantity at any other resolution.
  """
  log_s = torch.stack([item["meta"].gt["log_s"] for item in items])
  theta = torch.stack([item["meta"].gt["theta"] for item in items])
  t = torch.stack([item["meta"].gt["t"] for item in items])
  side = float(image_size) - 1.0
  corners = torch.tensor([[0.0, 0.0], [side, 0.0], [side, side],
                          [0.0, side]]).unsqueeze(0).expand(len(items), -1, -1)
  params = SimilarityParams(log_s=log_s, theta=theta, t=t)
  return corner_residual(params, corners, corners).mean().item()
