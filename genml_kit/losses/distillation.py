"""Knowledge-distillation losses.

Four pure objectives, one per family of knowledge (§2 of
``knowledge_distillation/README.md``):

- :func:`kd_response_loss`  -- match the teacher's tempered softmax (logits).
- :func:`kd_feature_loss`   -- match an intermediate feature (FitNets).
- :func:`kd_attention_loss` -- match a spatial attention summary (AT).
- :func:`kd_relation_loss`  -- match the batch's similarity structure (RKD).

Each is a small function, not an ``nn.Module``, so a method composes them with
its own weights.  ``kd_feature_loss`` and ``kd_attention_loss`` call the shared
shape guard from ``models/encoder_utils.py`` so a student/teacher whose tensors
do not line up fails loudly instead of a broadcasting, silently-wrong loss.

Shape-tracking convention (used throughout this module): every operation that
builds, reshapes or reduces a tensor is annotated on the line above with the
resulting shape, so the data flow through a function can be followed without a
debugger.  ``B`` is the batch size, ``C`` the class/channel count, ``D`` a
feature width and ``N`` a token count.
"""

import torch
import torch.nn.functional as F


def kd_response_loss(student_logits,
                     teacher_logits,
                     temperature=4.0,
                     reduction="batchmean"):
  """Hinton response distillation: ``T**2 * KL(p_t || p_s)``.

  The teacher distribution ``softmax(teacher_logits / T)`` is the target and
  the student is matched in log-space via ``KLDiv``.  The ``T**2`` factor keeps
  the gradient magnitude invariant to the temperature (see README §5): without
  it the distillation term would silently scale like ``1 / T``.

  Args:
      student_logits: ``(B, C)`` raw student logits.
      teacher_logits: ``(B, C)`` raw teacher logits.
      temperature: Softmax temperature ``T > 0``.
      reduction: ``torch.nn.functional.kl_div`` reduction.

  Returns:
      Scalar loss tensor.
  """
  if temperature <= 0:
    raise ValueError("kd_response_loss requires temperature > 0.")
  # (B, C) log-probabilities of the tempered student distribution.
  student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
  # (B, C) probabilities of the tempered teacher distribution (the target).
  teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
  # Per-sample forward KL(p_teacher || p_student); batchmean -> scalar.
  loss = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
  # Rescale by T**2 so the gradient is temperature-invariant (§5, §7).
  return loss * (temperature**2)


def kd_feature_loss(student_feat, teacher_feat, projector=None):
  """FitNets feature distillation: ``|| r(h_s) - h_t ||_2**2``.

  *projector* is the FitNets ``r`` (typically a
  :class:`~genml_kit.models.contrastive.ProjectionHead`) that lifts the student
  feature into the teacher's space; it is ``None`` when the two already match.
  The compared tensors must have identical shapes.

  Args:
      student_feat: ``(B, D_s)`` student features.
      teacher_feat: ``(B, D_t)`` teacher features.
      projector: Optional callable mapping ``student_feat`` to ``(B, D_t)``.

  Returns:
      Scalar MSE loss tensor.

  Raises:
      ValueError: If the (projected) student and teacher shapes differ.
  """
  from genml_kit.models.encoder_utils import require_matching_features

  # (B, D_s) -> (B, D_t) when a projector is given, else left as-is.
  projected = projector(student_feat) if projector is not None else student_feat
  # Guard the shapes before the L2: MSE would silently broadcast a (B, D_s) vs
  # (B, D_t) pair into a wrong-but-finite number otherwise (§25 pitfall).
  require_matching_features(projected, teacher_feat, names=("student", "teacher"))
  return F.mse_loss(projected, teacher_feat)


def kd_attention_loss(student_map, teacher_map, p=2, eps=1e-6):
  """Attention transfer: ``|| Â(h_s) - Â(h_t) ||_2**2``.

  ``A(h) = sum_c |h_c|^p`` collapses the channel axis and ``Â = A / ||vec A||_2``
  removes scale, so maps with a different number of channels are compared by
  *shape* alone (README §10).  Accepts ``(B, C, H, W)`` feature maps
  (channel axis 1) or ``(B, N, C)`` token maps (channel axis 2).

  Args:
      student_map: Student feature/token map.
      teacher_map: Teacher feature/token map, spatially aligned with the above.
      p: Attention exponent (``p=2`` reproduces the paper).
      eps: Norm floor guarding the normalisation.

  Returns:
      Scalar MSE loss tensor.

  Raises:
      ValueError: If the two attention maps do not have identical shapes.
  """
  from genml_kit.models.encoder_utils import require_matching_features

  # (B, C, H, W) -> (B, H, W)  or  (B, N, C) -> (B, N).
  att_s = _attention_map(student_map, p, eps)
  att_t = _attention_map(teacher_map, p, eps)
  # Collapsed maps must match exactly (same spatial/token layout).
  require_matching_features(att_s, att_t, names=("student", "teacher"))
  return F.mse_loss(att_s, att_t)


def kd_relation_loss(student_feat,
                     teacher_feat,
                     distance_weight=1.0,
                     angle_weight=2.0,
                     beta=1.0,
                     eps=1e-6):
  """Relational KD (RKD, Park et al. 2019): match the batch's similarity geometry.

  Two structures are transferred, each with a Huber (``smooth_l1``) penalty:
  the pairwise *distances* ``d(i, j)`` normalised by their batch mean, and the
  pairwise *angles* ``/_(j, i, k)`` between triplets ``(i, j, k)``.

  Args:
      student_feat: ``(B, D)`` student features.
      teacher_feat: ``(B, D)`` teacher features, identical shape.
      distance_weight: Weight of the distance-wise term.
      angle_weight: Weight of the angle-wise term.
      beta: Huber transition point.
      eps: Floor guarding the distance normalisation.

  Returns:
      Scalar loss tensor.

  Raises:
      ValueError: If the two feature tensors do not have identical shapes.
  """
  from genml_kit.models.encoder_utils import require_matching_features

  require_matching_features(student_feat, teacher_feat, names=("student", "teacher"))
  # Accumulator: scalar zero on the same device/dtype as the inputs.
  loss = student_feat.new_zeros(())
  if distance_weight:
    # Distance-wise term: scalar; weighted into the accumulator.
    loss = loss + distance_weight * _distance_wise_loss(student_feat, teacher_feat,
                                                        beta, eps)
  if angle_weight:
    # Angle-wise term: scalar; weighted into the accumulator.
    loss = loss + angle_weight * _angle_wise_loss(student_feat, teacher_feat, beta, eps)
  return loss


def _attention_map(feature_map, p, eps):
  """Collapse the channel axis and L2-normalise: ``(..., H, W)`` attention map."""
  # Pick the channel axis to collapse: axis 1 for (B, C, H, W) image maps,
  # axis 2 for (B, N, C) token maps.
  if feature_map.dim() == 4:
    channel_dim = 1
  elif feature_map.dim() == 3:
    channel_dim = 2
  else:
    raise ValueError("kd_attention_loss expects a 3-D or 4-D feature map; got "
                     f"{feature_map.dim()}-D.")
  # A(h) = sum_c |h_c|^p:  (B, C, H, W) -> (B, H, W)  or  (B, N, C) -> (B, N).
  att = feature_map.abs().pow(p).sum(dim=channel_dim)
  # L2 norm of each sample's flattened map: (B, *spatial) -> (B,).
  norm = att.flatten(1).norm(dim=1).clamp(min=eps)
  # Reshape (B,) -> (B, 1, 1...) so it broadcasts against the spatial axes, then
  # divide: (B, *spatial) stays (B, *spatial).
  return att / norm.view(-1, *([1] * (att.dim() - 1)))


def _pairwise_distances(feat, eps):
  """Normalised pairwise distances ``(B, B)`` (diagonal 0, batch-mean scaled)."""
  # Euclidean distance between every row pair of (B, D) -> (B, B).
  dist = torch.cdist(feat, feat, p=2)
  b = dist.shape[0]
  # Mean over the off-diagonal entries only (the b self-distances are 0).
  mean = dist.sum() / (b * (b - 1) + eps)
  # Scale to unit mean so the comparison is magnitude-invariant: (B, B).
  return dist / (mean + eps)


def _distance_wise_loss(student_feat, teacher_feat, beta, eps):
  """Huber penalty on the difference of normalised pairwise distances."""
  # (B, D) -> (B, B) normalised distance matrices for each side.
  d_s = _pairwise_distances(student_feat, eps)
  d_t = _pairwise_distances(teacher_feat, eps)
  b = d_s.shape[0]
  # Off-diagonal mask (B, B): False on the diagonal, True elsewhere.
  off_diag = ~torch.eye(b, dtype=torch.bool, device=d_s.device)
  # Both matrices indexed by the mask -> 1-D (B*(B-1),); Huber -> scalar.
  return F.smooth_l1_loss(d_s[off_diag], d_t[off_diag], beta=beta)


def _angle_wise_loss(student_feat, teacher_feat, beta, eps):
  """Huber penalty on the difference of pairwise cosine angles."""
  # (B, D) -> (B, B, B) cosine of /(j, i, k) for each side.
  cos_s = _angle_cosines(student_feat, eps)
  cos_t = _angle_cosines(teacher_feat, eps)
  b = cos_s.shape[0]
  # Boolean (B, B, B) mask of the triplets with three distinct indices.
  valid = _angle_mask(b, cos_s.device)
  # Both tensors indexed by the mask -> 1-D; Huber -> scalar.
  return F.smooth_l1_loss(cos_s[valid], cos_t[valid], beta=beta)


def _angle_cosines(feat, eps):
  """Cosine of the angle ``/(j, i, k)`` for every triplet: ``(B, B, B)``.

  Entry ``[i, j, k]`` is the cosine between the edges ``i->j`` and ``i->k``.
  """
  # edges[i, j] = f_j - f_i  ->  (B, B, D)
  edges = feat.unsqueeze(0) - feat.unsqueeze(1)
  # Edge lengths: (B, B).
  norms = edges.norm(dim=2).clamp(min=eps)
  # dot[i, j, k] = <edges[i, j], edges[i, k]>  ->  (B, B, B)
  dot = torch.einsum("ijd,ikd->ijk", edges, edges)
  # Outer product of the two edge lengths -> (B, B, B); / -> (B, B, B).
  denom = norms.unsqueeze(2) * norms.unsqueeze(1)
  return dot / denom.clamp(min=eps)


def _angle_mask(batch_size, device):
  """Boolean ``(B, B, B)`` mask of triplets with three distinct indices."""
  # (B,) index vector.
  idx = torch.arange(batch_size, device=device)
  # pairwise[i, j] == (i != j)  ->  (B, B).
  pairwise = idx.unsqueeze(0) != idx.unsqueeze(1)
  # valid[i, j, k] = (i != j) & (i != k) & (j != k), broadcasting the three
  # (B, B) views along the third axis -> (B, B, B).
  return pairwise.unsqueeze(2) & pairwise.unsqueeze(1) & pairwise.unsqueeze(0)
