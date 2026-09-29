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
from genml_kit.training.model_utils import model_mode

VOMetrics = collections.namedtuple("VOMetrics", [
    "mce",
    "dlog_s",
    "dtheta",
    "conf_mae",
],
                                   defaults=[float("nan")] * 4)

STAGES = {"supervised": 0, "photometric": 1}


def vo_metrics_from_output(out, gt, gt_residual, src, dst, *, gt_corners_dst=None):
  """Model-side VO metrics from a model output and its ground truth.

  The single implementation of the four VO metrics.  Both evaluation
  paths funnel through here: :meth:`VOPairMethod.evaluate` (DataBlob
  batches) and :func:`evaluate_vo` (raw dataset dicts) differ only in
  how they reach the tensors, not in what they compute.

  Args:
      out: ``VOModelOutput`` from ``VOSimilarityNet.forward``.
      gt: mapping with ``log_s`` / ``theta`` / ``t`` (batched tensors).
      gt_residual: (B,) irreducible residual of the GT fit.
      src: (B, 4, 2) reference corners.  **Required.**
      dst: (B, 4, 2) ground-truth target corners.  **Required.**
      gt_corners_dst: optional (B, 4, 2) *homography* destinations, used
          instead of ``dst`` when available.  vo/README.md s6 defines MCE
          against where the corners truly land, and the similarity family
          cannot absorb foreshortening, so the similarity-fit ``dst``
          carries a systematic bias that this removes.

  ``src`` and ``dst`` are deliberately positional and required.  They
  previously defaulted to ``out.corners`` / ``out.corners + out.dc`` --
  both *predicted* -- which silently produced a self-consistency
  residual with no ground truth in it, and a nearly constant ~0.03
  regardless of how wrong the prediction actually was.  Making them
  required means a caller cannot get that version by accident again.

  Returns:
      VOMetrics of batch means (mce in pixels, dtheta in radians).
  """
  if src is None or dst is None:
    raise ValueError("vo_metrics_from_output requires ground-truth corners: mce is "
                     "defined against the truth (vo/README.md s6), not against the "
                     "prediction's own corner offsets.")
  target = dst if gt_corners_dst is None else gt_corners_dst
  mce = corner_residual(out.params, src, target).mean()
  dlog_s = (out.params.log_s - gt["log_s"]).abs().mean()
  dtheta = wrap_angle(out.params.theta - gt["theta"]).abs().mean()
  conf = f.smooth_l1_loss(out.conf[:, 0], gt_residual)
  return VOMetrics(mce=mce.item(),
                   dlog_s=dlog_s.item(),
                   dtheta=dtheta.item(),
                   conf_mae=conf.item())


def vo_losses(pred, batch, cfg, stage):
  """Loss for one batch.

  Args:
      pred: ``VOModelOutput`` from ``VOSimilarityNet.forward``.
      batch: dict from ``VOPairDataset`` (collated); needs 'image_a',
          'image_b' and 'meta' with 'gt' (log_s, theta, t) and
          'gt_residual'.
      cfg: loss weights (namedtuple with w_log_s, w_theta, w_conf,
          w_photo).
      stage: int loss stage (see ``STAGES``).

  Returns:
      (scalar loss tensor, dict of loss components for logging).
  """
  params = pred.params
  gt = batch["meta"].gt
  mce = corner_residual(params, _src(batch), _dst(batch)).mean()
  dlog_s = (params.log_s - gt["log_s"]).abs().mean()
  dtheta = wrap_angle(params.theta - gt["theta"]).abs().mean()
  conf = f.smooth_l1_loss(pred.conf[:, 0], batch["meta"].gt_residual)
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


def _gt_corners(meta, device=None):
  """Return ``(src, dst)`` pixel corners from a meta object, batched.

  Both sides of the mean corner error of vo/README.md s6.  Raises when
  the meta predates the ``corners_src``/``corners_dst`` fields rather
  than silently falling back to the prediction -- a fallback here would
  restore exactly the bug these fields were added to fix.
  """
  src = getattr(meta, "corners_src", None)
  dst = getattr(meta, "corners_dst", None)
  if src is None or dst is None:
    raise ValueError("VOPairMeta is missing corners_src/corners_dst; mce cannot be "
                     "computed without ground-truth corners (vo/README.md s6).")
  if device is not None:
    src, dst = src.to(device), dst.to(device)
  return src, dst


def _src(batch):
  """Reference corners from the collated batch (B, 4, 2).

  Prefers the ground-truth pixel corners carried in the meta; falls back
  to the network's own reference corners only when a caller builds a
  batch without them (unit tests that construct minimal dicts).
  """
  meta = batch["meta"]
  corners_src = getattr(meta, "corners_src", None)
  if corners_src is not None:
    return corners_src
  return batch["corners"]


def _dst(batch):
  """Ground-truth target corners (B, 4, 2), in pixels.

  This is the ``dst`` side of the mean corner error of vo/README.md s6.
  It used to fall back to ``corners + dc`` -- the network's own
  prediction -- which made ``L_mce`` a self-consistency residual with no
  ground truth in it.
  """
  meta = batch["meta"]
  corners_dst = getattr(meta, "corners_dst", None)
  if corners_dst is not None:
    return corners_dst
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


def evaluate_vo(model, loader, device):
  """Mean metrics over one loader; corners come from the model output.

  Args:
      model: the VO network.
      loader: iterator of collated VO batches.
      device: torch device.

  Returns:
      VOMetrics with the means over the loader.
  """
  sums = torch.zeros(4, dtype=torch.float64)
  count = 0
  with model_mode(model, "eval"), torch.no_grad():
    for batch in loader:
      image_a = batch["image_a"].to(device)
      image_b = batch["image_b"].to(device)
      out = model(image_a, image_b)
      metrics = vo_metrics_from_output(out, batch["meta"].gt, batch["meta"].gt_residual,
                                       *_gt_corners(batch["meta"], device))
      sums += torch.tensor(
          [metrics.mce, metrics.dlog_s, metrics.dtheta, metrics.conf_mae],
          dtype=torch.float64)
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
