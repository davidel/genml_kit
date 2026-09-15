"""VO training losses and metrics (staged photometric + mce).

The loss schedule and the metric definitions follow ``vo/README.md``
section 4; the closed-form components come from
``genml_kit.geometry.similarity``.

In v4.2 the VO *objective* (model + loss + metric) is the
``vo_pair`` method (``genml_kit/methods/vo_pair.py``); this module keeps
the loss/eval building blocks as module-level functions so the method
and the unit tests share one implementation.
"""

import collections

import torch
import torch.nn.functional as f

from genml_kit.geometry.similarity import corner_residual, wrap_angle

VOMetrics = collections.namedtuple("VOMetrics", [
    "mce",
    "dlog_s",
    "dtheta",
    "conf_mae",
],
                                   defaults=[float("nan")] * 4)

STAGES = {"supervised": 0, "photometric": 1}


def vo_losses(pred, batch, cfg, stage):
  """Loss for one batch.

  Args:
      pred: dict from ``VOSimilarityNet.forward``.
      batch: dict from ``VOPairDataset`` (collated); needs 'image_a',
          'image_b' and 'meta' with 'gt' (log_s, theta, t) and
          'gt_residual'.
      cfg: loss weights (namedtuple with w_log_s, w_theta, w_conf,
          w_photo).
      stage: int loss stage (see ``STAGES``).

  Returns:
      (scalar loss tensor, dict of loss components for logging).
  """
  params = pred["params"]
  gt = batch["meta"].gt
  mce = corner_residual(params, _src(batch), _dst(batch)).mean()
  dlog_s = (params.log_s - gt["log_s"]).abs().mean()
  dtheta = wrap_angle(params.theta - gt["theta"]).abs().mean()
  conf = f.smooth_l1_loss(pred["conf"][:, 0], batch["meta"].gt_residual)
  total = mce + cfg.w_log_s * dlog_s + cfg.w_theta * dtheta \
      + cfg.w_conf * conf
  parts = {
      "mce": mce.detach(),
      "dlog_s": dlog_s.detach(),
      "dtheta": dtheta.detach(),
      "conf": conf.detach(),
  }
  if stage >= STAGES["photometric"]:
    photo = photometric_residual(batch["image_a"], batch["image_b"], params)
    total = total + cfg.w_photo * photo
    parts["photo"] = photo.detach()
  return total, parts


def _src(batch):
  """Reference corners from the collated batch (B, 4, 2)."""
  return batch["corners"]


def _dst(batch):
  """Target corners: the collated 'corners_dst' if present, else the
  network's own corner offsets (training-time semantics)."""
  if "corners_dst" in batch:
    return batch["corners_dst"]
  return batch["corners"] + batch["dc"]


def photometric_residual(image_a, image_b, params, eps=1e-6):
  """Masked zNCC residual between frame A and the warped frame B.

  A small differentiable photometric consistency term: the warp is the
  backward map of the predicted similarity, and the normalized
  cross-correlation over the valid (non-zero-padded) area measures how
  photometrically consistent the alignment is.  Used as a staged loss
  and, at inference, as the residual confidence gate.

  Args:
      image_a: (B, 1, H, W) reference frame.
      image_b: (B, 1, H, W) moving frame.
      params: SimilarityParams describing A -> B.
      eps: numerical floor for the normalization.

  Returns:
      (B,) residual in [0, 2]; 0 means perfect correlation.
  """
  from genml_kit.geometry.similarity import warp_similarity

  warped = warp_similarity(image_b, params, (image_a.shape[2], image_a.shape[3]))
  mask = ((image_a > 0) & (warped > 0)).float()
  count = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
  a = image_a * mask
  b = warped * mask
  a_mean = a.sum(dim=(1, 2, 3)) / count
  b_mean = b.sum(dim=(1, 2, 3)) / count
  a_c = (a - a_mean.view(-1, 1, 1, 1)) * mask
  b_c = (b - b_mean.view(-1, 1, 1, 1)) * mask
  num = (a_c * b_c).sum(dim=(1, 2, 3))
  den = torch.sqrt((a_c * a_c).sum(dim=(1, 2, 3)) * (b_c * b_c).sum(dim=(1, 2, 3)))
  zncc = num / den.clamp_min(eps)
  return 1.0 - zncc


@torch.no_grad()
def evaluate_vo(model, loader, device):
  """Mean metrics over one loader; corners come from the model output.

  Args:
      model: the VO network.
      loader: iterator of collated VO batches.
      device: torch device.

  Returns:
      VOMetrics with the means over the loader.
  """
  model.eval()
  sums = torch.zeros(4, dtype=torch.float64)
  count = 0
  for batch in loader:
    image_a = batch["image_a"].to(device)
    image_b = batch["image_b"].to(device)
    out = model(image_a, image_b)
    corners = out["corners"]
    mce = corner_residual(out["params"], corners, corners + out["dc"]).mean()
    gt = batch["meta"].gt
    dlog_s = (out["params"].log_s - gt["log_s"]).abs().mean()
    dtheta = wrap_angle(out["params"].theta - gt["theta"]).abs().mean()
    conf = f.smooth_l1_loss(out["conf"][:, 0], batch["meta"].gt_residual)
    sums += torch.tensor(
        [mce.item(), dlog_s.item(),
         dtheta.item(), conf.item()], dtype=torch.float64)
    count += 1
  if count == 0:
    return VOMetrics()
  sums /= count
  return VOMetrics(*sums.tolist())


class _LossCfg:
  """Default loss weights (overridable via ``args.vo_loss_cfg``)."""

  w_log_s = 1.0
  w_theta = 1.0
  w_conf = 0.5
  w_photo = 0.1
