"""Sliced VO evaluation: terrain x range-bin x augmentation x AGL config.

Renders one text table per slice with the utils/table formatter (repo
style).  Model-side batch metrics live in ``train_vo.evaluate_vo``; this
module groups item-level records by metadata slice.
"""

import collections
import math

import torch

from genml_kit.training.vo.train_vo import VOMetrics
from genml_kit.utils.table import format_table

EvalRow = collections.namedtuple("EvalRow",
                                 ["slice", "n", "mce", "dlog_s", "dtheta", "conf_mae"])


def evaluate_sliced(model, dataset, device, group_key="terrain"):
  """Evaluate per slice of ``group_key`` and render a text table.

  Args:
      model: the VO network.
      dataset: indexed VO dataset (meta with 'gt' present).
      device: torch device.
      group_key: metadata field to slice by ('terrain' or 'range_bin').

  Returns:
      (list of EvalRow, list of formatted table lines).
  """
  del device
  groups = collections.OrderedDict()
  for idx in range(len(dataset)):
    item = dataset[idx]
    key = str(getattr(item["meta"], group_key))
    groups.setdefault(key, []).append(item)
  rows = []
  for key, items in groups.items():
    metrics = VOMetrics(
        mce=_mean_gt_corner_error(items),
        dlog_s=_nan_mean([_dlog_s(item) for item in items]),
        dtheta=_nan_mean([_dtheta(item) for item in items]),
        conf_mae=_nan_mean([_conf_mae(item) for item in items]),
    )
    rows.append(
        EvalRow(key, len(items), metrics.mce, metrics.dlog_s, metrics.dtheta,
                metrics.conf_mae))
  table = format_table(["slice", "n", "mce_px", "dlog_s", "dtheta_deg", "conf_mae"], [[
      row.slice,
      str(row.n), f"{row.mce:.2f}", f"{row.dlog_s:.4f}",
      f"{row.dtheta * 180.0 / math.pi:.2f}", f"{row.conf_mae:.3f}"
  ] for row in rows])
  return rows, table


def _dlog_s(item):
  if "pred" not in item:
    return float("nan")
  return abs(item["pred"].log_s.item() - item["meta"].gt["log_s"].item())


def _dtheta(item):
  if "pred" not in item:
    return float("nan")
  delta = item["pred"].theta.item() - item["meta"].gt["theta"].item()
  return abs((delta + math.pi) % (2 * math.pi) - math.pi)


def _conf_mae(item):
  if "pred" not in item:
    return float("nan")
  return abs(item["pred"]["conf"].item() - item["meta"].gt_residual.item())


def _mean_gt_corner_error(items):
  """Mean corner reprojection error of the GT fit itself (sanity metric)."""
  total = 0.0
  for item in items:
    gt = item["meta"].gt
    mat = _build_matrix(gt["log_s"], gt["theta"], gt["t"])
    total += _corner_error_from_matrix(mat).item()
  return total / max(len(items), 1)


def _nan_mean(values):
  valid = [v for v in values if not math.isnan(v)]
  if not valid:
    return float("nan")
  return sum(valid) / len(valid)


def _build_matrix(log_s, theta, t):
  s = torch.exp(log_s)
  c, si = torch.cos(theta), torch.sin(theta)
  return torch.tensor([
      [s * c, -s * si, t[0]],
      [s * si, s * c, t[1]],
      [0.0, 0.0, 1.0],
  ])


def _corner_error_from_matrix(mat):
  corners = torch.tensor([[0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0]])
  ones = torch.ones(4, 1)
  pts = torch.cat([corners, ones], dim=1)
  proj = (mat @ pts.T).T[:, :2]
  return (proj - corners).norm(dim=1).mean()
