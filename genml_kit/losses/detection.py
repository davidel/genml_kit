"""Detection losses: FCOS target assignment, focal + GIoU + centerness.

The loss is a *three-headed* objective over the dense per-location predictions
produced by :class:`~genml_kit.models.decoders.fcos_head.FCOSHead`:

1. **Classification** -- sigmoid focal loss (Lin et al., 2017) over all
   locations, with the positive locations taken from the assignment below.
2. **Box regression** -- GIoU loss (Rezatofighi et al., 2019) over the
   positive locations only.
3. **Centerness** -- binary cross-entropy that teaches each positive location
   how central it is inside its target box, so the NMS stage can down-weight
   boxes predicted by peripheral locations.

The hard part is *assignment*: deciding, for each pyramid location, which
ground-truth box (if any) it is responsible for.  FCOS does this purely
geometrically, which is what makes it cheap and Hungarian-free:

* A location is a *candidate* for a box if it lies inside the box.
* It is *positive* if it lies inside the box **and** inside the box's
  centre region (centre-sampling: a box of side ``s`` has its centre square
  shrunk by ``radius``; the default 1.5 keeps a comfortable margin).
* When several boxes overlap a location, the smallest-area box wins -- the
  standard FCOS ambiguity resolution, so a small object is not starved of
  positives by the large object around it.

All box arithmetic is ``xyxy`` in absolute input-image pixels; the head
already scaled its distances by the stride, so no stride bookkeeping is
needed here.
"""

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from genml_kit.losses.registry import LOSSES


def _focal_loss(logits, targets, alpha=0.25, gamma=2.0):
  """Sigmoid focal loss over raw *logits* (BCE-with-logits formulation).

  ``logits`` and ``targets`` are both ``(N,)`` (or any broadcastable shape)
  with ``targets`` in ``[0, 1]`` (soft targets are allowed).  Returns the
  *summed* loss, which the caller normalises by the positive count -- the
  FCOS convention.
  """
  # Per-element BCE with logits, no reduction: (N,) -> (N,).
  bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
  prob = torch.sigmoid(logits)
  # p_t is the probability of the *target* class: (N,) -> (N,).
  p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
  # Focal modulation (1 - p_t)^gamma and class balancing alpha_t.
  alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
  # (N,) * (N,) -> (N,) -> scalar.
  return (alpha_t * (1.0 - p_t).pow(gamma) * bce).sum()


def _centerness_targets(ltrb):
  """Centerness from ``(l, t, r, b)`` distances, per the FCOS definition.

  ``centerness = sqrt(min(l,r) / max(l,r) * min(t,b) / max(t,b))``; it is 1
  at the box centre and decays towards the edges.  ``(N, 4) -> (N,)``.
  """
  left_right = torch.stack([ltrb[:, 0], ltrb[:, 2]], dim=1)
  top_bottom = torch.stack([ltrb[:, 1], ltrb[:, 3]], dim=1)
  lr_ratio = left_right.min(dim=1).values / left_right.max(dim=1).values.clamp(min=1e-6)
  tb_ratio = top_bottom.min(dim=1).values / top_bottom.max(dim=1).values.clamp(min=1e-6)
  return torch.sqrt(lr_ratio * tb_ratio)


def _location_grid(level_shape, stride, device):
  """Return the input-image ``(x, y)`` centre of every location on a level.

  A ``(H_i, W_i)`` feature map with ``stride`` maps location ``(row, col)``
  back to input pixel ``((col + 0.5) * stride, (row + 0.5) * stride)`` -- the
  ``+0.5`` places the point at the centre of the receptive-field cell rather
  than its top-left corner.

  Returns
  -------
  torch.Tensor
    ``(H_i * W_i, 2)`` float tensor of ``(x, y)`` points, in row-major
    order matching a flattened ``(H_i, W_i)`` feature map.
  """
  height, width = level_shape
  # 1-D grids of column / row indices: (W_i,) and (H_i,).
  xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * stride
  ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * stride
  # Meshgrid -> (H_i, W_i); stack -> (H_i, W_i, 2); flatten -> (H_i*W_i, 2).
  grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
  return torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)


def _assign_level(points, boxes, labels, radius, num_classes):
  """Assign a level's locations to ground-truth boxes.

  Parameters
  ----------
  points : torch.Tensor
      ``(P, 2)`` input-image centres of the level's locations.
  boxes : torch.Tensor
      ``(M, 4)`` ground-truth boxes in ``xyxy`` pixels.
  labels : torch.Tensor
      ``(M,)`` ground-truth compact class ids.
  radius : float
      Centre-sampling radius factor (FCOS default 1.5).
  num_classes : int
      Number of foreground classes.

  Returns
  -------
  (torch.Tensor, torch.Tensor, torch.Tensor)
    ``cls_target`` ``(P, num_classes)`` soft focal targets,
    ``pos_mask`` ``(P,)`` bool positives, and
    ``reg_target`` ``(P, 4)`` ``(l, t, r, b)`` distances for positives
    (garbage where ``pos_mask`` is False).
  """
  num_points = points.shape[0]
  # (P, num_classes): start every location as background (all-zero target).
  cls_target = points.new_zeros((num_points, num_classes))
  pos_mask = torch.zeros(num_points, dtype=torch.bool, device=points.device)
  reg_target = points.new_zeros((num_points, 4))
  if boxes.numel() == 0:
    return cls_target, pos_mask, reg_target

  # Distances from every point to every box edge: (P, 1, 2) - (1, M, 2) -> (P, M).
  px = points[:, 0].unsqueeze(1)
  py = points[:, 1].unsqueeze(1)
  left = px - boxes[:, 0].unsqueeze(0)
  top = py - boxes[:, 1].unsqueeze(0)
  right = boxes[:, 2].unsqueeze(0) - px
  bottom = boxes[:, 3].unsqueeze(0) - py
  # A point is inside the box when all four distances are positive: (P, M).
  inside = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)

  # Centre region: shrink each box towards its centre by ``radius``.  A point
  # must fall in this inner box to be a positive, which stops the ambiguous
  # ring of locations near a box edge from being trained on.
  cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
  cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
  box_w = boxes[:, 2] - boxes[:, 0]
  box_h = boxes[:, 3] - boxes[:, 1]
  half_w = box_w * radius / 2.0
  half_h = box_h * radius / 2.0
  in_center = ((px >= (cx - half_w).unsqueeze(0)) & (px <= (cx + half_w).unsqueeze(0)) &
               (py >= (cy - half_h).unsqueeze(0)) & (py <= (cy + half_h).unsqueeze(0)))
  # (P, M): candidate = inside the box AND inside its centre region.
  candidate = inside & in_center

  # Ambiguity resolution: for each point pick the *smallest-area* candidate
  # box.  Setting the area of non-candidates to +inf makes ``argmin`` pick
  # them last, so a point with no candidate gets a sentinel we can mask out.
  area = (box_w * box_h).unsqueeze(0).expand_as(candidate)
  masked_area = torch.where(candidate, area, torch.full_like(area, float("inf")))
  # (P,) best box index per point; (P,) whether any candidate existed.
  best_area, best_idx = masked_area.min(dim=1)
  has_candidate = torch.isfinite(best_area)

  if not has_candidate.any():
    return cls_target, pos_mask, reg_target

  pos_mask = has_candidate
  # Gather the chosen box's distances / label for each positive point.
  # (P,) index -> (P,) distances taken from the (P, M) tables.
  pos = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
  chosen = best_idx[pos]
  reg_target[pos, 0] = left[pos, chosen]
  reg_target[pos, 1] = top[pos, chosen]
  reg_target[pos, 2] = right[pos, chosen]
  reg_target[pos, 3] = bottom[pos, chosen]
  # Soft focal target = 1.0 on the true class at a positive location.
  cls_target[pos, labels[chosen]] = 1.0
  return cls_target, pos_mask, reg_target


class FCOSLoss(torch.nn.Module):
  """FCOS objective: focal classification + GIoU regression + centerness.

  Parameters
  ----------
  num_classes : int
    Number of foreground classes.
  center_sampling_radius : float
      Centre-sampling radius factor for assignment (FCOS default 1.5).
  focal_alpha, focal_gamma : float
      Focal-loss hyperparameters.
  box_weight, centerness_weight : float
      Relative weights of the regression and centerness terms.
  """

  def __init__(self,
               num_classes,
               center_sampling_radius=1.5,
               focal_alpha=0.25,
               focal_gamma=2.0,
               box_weight=1.0,
               centerness_weight=1.0):
    super().__init__()
    self.num_classes = int(num_classes)
    self.center_sampling_radius = float(center_sampling_radius)
    self.focal_alpha = float(focal_alpha)
    self.focal_gamma = float(focal_gamma)
    self.box_weight = float(box_weight)
    self.centerness_weight = float(centerness_weight)

  def forward(self, head_output, targets):
    """Return ``(loss, metrics)`` for a batch.

    Parameters
    ----------
    head_output : DetHeadOutput
      The head's per-level ``cls_logits`` / ``box_pred`` / ``centerness`` /
      ``strides``.
    targets : list[dict]
      One entry per image, each ``{"boxes": (M, 4) xyxy, "labels": (M,)}``.
      Boxes are in the *input* image frame, matching the level grids.

    Returns
    -------
    (torch.Tensor, dict)
      The scalar total loss and a dict of detached sub-losses for logging.
    """
    # Flatten every level's predictions and points into one long list, so the
    # assignment below treats all locations uniformly regardless of level.
    cls_logits, box_pred, centerness, points = [], [], [], []
    for level_idx, logits in enumerate(head_output.cls_logits):
      stride = head_output.strides[level_idx]
      height, width = logits.shape[-2:]
      # (B, C, H, W) -> (B, H*W, C): locations become the sequence axis.
      cls_logits.append(
          logits.permute(0, 2, 3, 1).reshape(logits.shape[0], -1, self.num_classes))
      # (B, 4, H, W) -> (B, H*W, 4).
      box_pred.append(head_output.box_pred[level_idx].permute(0, 2, 3, 1).reshape(
          logits.shape[0], -1, 4))
      # (B, 1, H, W) -> (B, H*W).
      centerness.append(head_output.centerness[level_idx].permute(0, 2, 3, 1).reshape(
          logits.shape[0], -1))
      # (H*W, 2) points are shared by the whole batch (same image geometry).
      points.append(_location_grid((height, width), stride, logits.device))
    # (B, P_total, C) / (B, P_total, 4) / (B, P_total); (P_total, 2).
    cls_logits = torch.cat(cls_logits, dim=1)
    box_pred = torch.cat(box_pred, dim=1)
    centerness = torch.cat(centerness, dim=1)
    points = torch.cat(points, dim=0)

    batch_size = cls_logits.shape[0]
    # Per-image assignment keeps the smallest-box rule local to one image.
    all_cls_t = []
    all_reg_t = []
    all_pos = []
    for img in range(batch_size):
      boxes = targets[img]["boxes"].to(cls_logits.device)
      labels = targets[img]["labels"].to(cls_logits.device)
      cls_t, pos_mask, reg_t = _assign_level(points, boxes, labels,
                                             self.center_sampling_radius,
                                             self.num_classes)
      all_cls_t.append(cls_t)
      all_reg_t.append(reg_t)
      all_pos.append(pos_mask)
    # (B, P_total, C) / (B, P_total, 4) / (B, P_total).
    cls_target = torch.stack(all_cls_t, dim=0)
    reg_target = torch.stack(all_reg_t, dim=0)
    pos_mask = torch.stack(all_pos, dim=0)

    # --- Classification: focal over all locations ---------------------------
    # (B, P, C) -> (B*P*C,) so the focal helper sees one flat vector.
    cls_loss = _focal_loss(cls_logits.reshape(-1), cls_target.reshape(-1),
                           self.focal_alpha, self.focal_gamma)
    # Normalise by the number of positives.  The *denominator* is clamped so
    # an image with no boxes (all-background batch) still yields a finite,
    # well-scaled loss; the *reported* count stays the true, unclamped value.
    num_pos = pos_mask.sum()
    num_pos_norm = num_pos.clamp(min=1.0)
    cls_loss = cls_loss / num_pos_norm

    if pos_mask.any():
      # (B, P, 4) -> (N_pos, 4): the positive locations' predicted distances.
      pred_ltrb = box_pred[pos_mask]
      tgt_ltrb = reg_target[pos_mask]
      # Convert distances back to xyxy: for a point (x, y),
      #   x1 = x - l, y1 = y - t, x2 = x + r, y2 = y + b.
      # pos_points is (N_pos, 2); broadcast over the 4 coordinates below.
      pos_points = points.unsqueeze(0).expand(batch_size, -1, -1)[pos_mask]
      pred_boxes = torch.stack([
          pos_points[:, 0] - pred_ltrb[:, 0],
          pos_points[:, 1] - pred_ltrb[:, 1],
          pos_points[:, 0] + pred_ltrb[:, 2],
          pos_points[:, 1] + pred_ltrb[:, 3],
      ],
                               dim=1)
      tgt_boxes = torch.stack([
          pos_points[:, 0] - tgt_ltrb[:, 0],
          pos_points[:, 1] - tgt_ltrb[:, 1],
          pos_points[:, 0] + tgt_ltrb[:, 2],
          pos_points[:, 1] + tgt_ltrb[:, 3],
      ],
                              dim=1)
      # (N_pos, 4) pairs -> scalar GIoU loss (already 1 - IoU averaged).
      box_loss = generalized_box_iou_loss(pred_boxes, tgt_boxes,
                                          reduction="sum") / num_pos_norm
      # Centerness target from the GT distances: (N_pos, 4) -> (N_pos,).
      ctr_target = _centerness_targets(tgt_ltrb)
      # Predicted centerness at the positive locations: (B, P) -> (N_pos,).
      ctr_pred = centerness[pos_mask]
      ctr_loss = F.binary_cross_entropy_with_logits(ctr_pred,
                                                    ctr_target,
                                                    reduction="sum")
      ctr_loss = ctr_loss / num_pos_norm
    else:
      # No positives in the batch: regression/centerness are undefined, so
      # contribute zero (with a graph-connected zero to keep autograd happy).
      zero = cls_logits.sum() * 0.0
      box_loss = zero
      ctr_loss = zero

    total = cls_loss + self.box_weight * box_loss + self.centerness_weight * ctr_loss
    metrics = {
        "cls_loss": cls_loss.detach(),
        "box_loss": box_loss.detach(),
        "centerness_loss": ctr_loss.detach(),
        "num_pos": num_pos.detach(),
    }
    return total, metrics


@LOSSES.register("fcos")
def fcos_criterion(*, num_classes, **kwargs):
  """FCOS criterion factory.

  ``num_classes`` is injected by the detection method (it is inferred from the
  dataset, not the CLI).  Remaining ``--loss_args`` keys are forwarded to
  :class:`FCOSLoss` (``center_sampling_radius``, ``focal_alpha``, ...).
  """
  return FCOSLoss(num_classes=num_classes, **kwargs)
