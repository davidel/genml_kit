"""Knowledge-distillation method (§15, §17-§19).

Trains a **student** to mimic a **teacher**.  The teacher is either an
exponential moving average of the student's own weights (self-distillation,
§13) or a separately-loaded frozen model (§17); the two are selected with
``--teacher_kind``.

The student is built through the normal registry (``load_model``), so any
backbone/head works.  Distillation adds one or more of the response /
feature / attention / relation objectives on top of the student's ordinary
cross-entropy:

```text
L = CE(y, s) + a_resp * KD_resp + a_feat * KD_feat + a_rel * KD_rel
```

Feature distillation reuses the shared machinery rather than re-implementing
it (Part 8 contract): :func:`~genml_kit.models.layer_match.match_layers` pairs
teacher and student layers, the activations are captured with forward hooks,
and the paired tensors go through
:func:`~genml_kit.losses.distillation.kd_feature_loss` (or
``kd_attention_loss`` when ``--kd_attention`` is set), whose shape guard
(``require_matching_features``) fails loudly on a mismatch.  The FitNets
projection ``r`` is the shared
:class:`~genml_kit.models.contrastive.ProjectionHead`.

Shape-tracking convention: an intermediate tensor's shape is written on the
line above the operation that produces it.  ``B`` = batch, ``C`` = channels,
``H``/``W`` = image spatial size, ``N`` = token count, ``D`` = feature width.
"""

import argparse
import copy

import torch
import torch.nn as nn

from genml_kit.losses.distillation import (
    kd_attention_loss,
    kd_feature_loss,
    kd_relation_loss,
    kd_response_loss,
)
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.models import load_model
from genml_kit.models.contrastive import ProjectionHead
from genml_kit.models.ema import EMAModel, linear_momentum
from genml_kit.models.encoder_utils import detect_backbone_dim, encode_with_backbone
from genml_kit.models.layer_match import iter_named_layers, match_layers
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.utils.logging import fatal


def _forward_with_capture(module, images, capture_modules=None):
  """Run ``module(pixel_values=images)`` and return ``(logits, activations)``.

  When *capture_modules* is given, a forward hook records each module's output
  during the pass; *activations* is then a list parallel to *capture_modules*
  (``None`` for a module that did not fire, e.g. one the input never reaches).
  Hooks are always removed, so repeated calls do not leak them onto the graph.

  Args:
      module: Model whose ``forward`` accepts ``pixel_values``.
      images: Input batch ``(B, C, H, W)``.
      capture_modules: Optional list of submodules to hook.

  Returns:
      ``(logits, activations)``: logits ``(B, n_classes)`` (the model's
      ``.logits`` when it exposes one, else the raw output), and either ``None``
      (no capture requested) or a list parallel to *capture_modules*.
  """
  # Fast path: no hooks needed, so skip the bookkeeping entirely.
  if capture_modules is None:
    # (B, C, H, W) -> model output (or a ModelOutput holding .logits).
    out = module(pixel_values=images)
    return getattr(out, "logits", out), None

  # Captured outputs keyed by the hooking index, so order is independent of the
  # order in which the submodules actually fire.
  captured = {}

  def _make_hook(idx):
    """Build a hook that stores its module's output under *idx*."""

    def _hook(_mod, _inp, out):
      captured[idx] = out

    return _hook

  # Register one hook per captured submodule; keep handles to remove them.
  handles = [
      mod.register_forward_hook(_make_hook(i)) for i, mod in enumerate(capture_modules)
  ]
  try:
    # (B, C, H, W) -> forward pass; hooks record each captured submodule output.
    out = module(pixel_values=images)
  finally:
    # Always detach the hooks, even if the forward raised.
    for handle in handles:
      handle.remove()
  # (B, n_classes) logits (unwrapped from a ModelOutput when present).
  logits = getattr(out, "logits", out)
  # List aligned with capture_modules; entries are the recorded activations or
  # None when a given module never fired during this pass.
  activations = [captured.get(i) for i in range(len(capture_modules))]
  return logits, activations


class DistillationPair(nn.Module):
  """Student + teacher wrapper.

  ``student`` is the module the optimizer updates; ``teacher`` is a frozen
  external model or a copy of the student kept current by ``_ema`` (an
  :class:`~genml_kit.models.ema.EMAModel` held as a plain reference, so the
  student's ``state_dict`` keys stay intact — checkpoints keep loading).
  ``projectors`` are the FitNets ``r`` maps, one per matched layer pair; being
  submodules, they receive gradients and are optimised alongside the student.
  """

  def __init__(self, student, teacher, teacher_kind, projectors=None, ema=None):
    super().__init__()
    # The trained tower and the (frozen or EMA-updated) target tower.
    self.student = student
    self.teacher = teacher
    self.teacher_kind = teacher_kind
    # nn.ModuleList (never None) so it registers cleanly as a submodule; the
    # empty case means feature KD is off.
    self.projectors = projectors if projectors is not None else nn.ModuleList()
    # Plain-reference EMA updater (not an nn.Module).  None for external teacher.
    self._ema = ema


@METHODS.register
class DistillationMethod(Method):
  """Logit / feature / attention / relation distillation into a student."""

  NAME = "distillation"
  METRIC_KEY = "loss"
  METRIC_MINIMIZE = True
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    # Label space, filled in by wire_data() before build_model().
    self._num_labels = None
    self._id2label = None
    self._label2id = None
    # Resolved KD hyperparameters (see _KdConfig) and the EMA step budget.
    self._cfg = None
    self._total_steps = None
    # Matched (teacher, student) layer pairs for feature/attention KD.
    self._feature_pairs = None

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("distillation method")
    group.add_argument(
        "--teacher_model",
        type=str,
        help="Teacher model spec (registry name / script). Required for "
        "--teacher_kind external.",
    )
    group.add_argument(
        "--teacher_checkpoint",
        type=str,
        help="Checkpoint holding the external teacher's weights.",
    )
    group.add_argument(
        "--teacher_kind",
        type=str,
        default="external",
        choices=("external", "ema"),
        help="'external' loads a frozen teacher; 'ema' self-distils from an "
        "exponential moving average of the student.",
    )
    group.add_argument(
        "--kd_temperature",
        type=float,
        default=4.0,
        help="Response-KD softmax temperature.",
    )
    group.add_argument(
        "--kd_alpha",
        type=float,
        default=1.0,
        help="Weight of the response (logit) KD term (0 disables it).",
    )
    group.add_argument(
        "--kd_feature_alpha",
        type=float,
        default=0.0,
        help="Weight of the FitNets feature KD term (0 disables it).",
    )
    group.add_argument(
        "--kd_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use attention transfer on the matched layers instead of raw "
        "feature MSE (channel-agnostic; suits the ViT backbones).",
    )
    group.add_argument(
        "--kd_relation_alpha",
        type=float,
        default=0.0,
        help="Weight of the relational (RKD) KD term (0 disables it).",
    )
    group.add_argument(
        "--kd_layer_strategy",
        type=str,
        default="uniform",
        choices=("by_index", "uniform", "by_name"),
        help="How to pair teacher/student layers for feature KD.",
    )
    group.add_argument(
        "--kd_layer_pattern",
        type=str,
        help="Regex selecting the feature layers to pair (feature KD).",
    )
    group.add_argument(
        "--kd_proj_hidden",
        type=int,
        default=512,
        help="Hidden width of the FitNets projection r.",
    )
    group.add_argument(
        "--kd_ema_momentum",
        type=float,
        default=0.996,
        help="Initial EMA momentum for --teacher_kind ema.",
    )
    group.add_argument(
        "--kd_ema_final_momentum",
        type=float,
        default=1.0,
        help="Final EMA momentum for --teacher_kind ema.",
    )

  def prepare_transforms(self, args, device):
    """Resolve train/val/TTA transforms from the model processor.

    Distillation consumes labeled image data through the same
    classification pipeline path, so without this the loader is built with
    ``transform=None`` and raw PIL images reach the collate fn.
    """
    from genml_kit.models import load_processor
    from genml_kit.training.train_compat import resolve_augmentations

    processor = load_processor(
        args.model,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )
    self.train_transforms, self.val_transforms, self.tta_transform = (
        resolve_augmentations(args, processor))

  def set_label_space(self, num_labels, id2label, label2id):
    """Record the label space (mirrors the classification method)."""
    self._num_labels = num_labels
    self._id2label = id2label
    self._label2id = label2id

  def wire_data(self, args, pipeline):
    """Read the label space and size the EMA momentum ramp (§15).

    Lifecycle position 2: both loaders exist, so ``len(train_loader)`` is final
    and the optimizer-step budget can be computed.  Runs before build_model.
    """
    if pipeline.num_labels is None:
      fatal("--method distillation requires a labeled --dataset.", ValueError)
    self.set_label_space(pipeline.num_labels, pipeline.id2label, pipeline.label2id)
    self._cfg = _KdConfig.from_args(args)
    self._total_steps = _step_budget(args, pipeline)

  def build_model(self, args, device):
    """Construct the student, its teacher, and the FitNets projectors."""
    # Student through the standard registry: any backbone/head works.
    student = load_model(
        args.model,
        num_labels=self._num_labels,
        id2label=self._id2label,
        label2id=self._label2id,
        image_size=args.image_size,
        device=device,
        cache_dir=args.cache_dir,
    )
    # Apply --grad_checkpoint / --lora / --freeze / --source_checkpoint to the
    # student BEFORE the teacher copy is made, so the teacher mirrors the
    # student's trainable structure.
    student = self._apply_model_extras(args, student, device)

    if args.teacher_kind == "ema":
      # Self-distillation: the teacher is a frozen deep copy of the student,
      # advanced toward the student's weights by EMAModel each train step.
      teacher = copy.deepcopy(student)
      teacher.eval()
      for param in teacher.parameters():
        param.requires_grad_(False)
      ema = EMAModel(teacher, student)
    else:  # "external"
      teacher = self._build_external_teacher(args, device)
      ema = None

    # FitNets r maps (empty when feature KD is disabled).
    projectors = self._build_feature_projectors(args, student, teacher, device)
    return DistillationPair(student, teacher, args.teacher_kind, projectors,
                            ema).to(device)

  def _build_external_teacher(self, args, device):
    """Load and freeze the teacher for ``--teacher_kind external``."""
    if not args.teacher_model:
      fatal("--teacher_kind external requires --teacher_model.", ValueError)
    teacher = load_model(
        args.teacher_model,
        num_labels=self._num_labels,
        id2label=self._id2label,
        label2id=self._label2id,
        image_size=args.image_size,
        device=device,
        cache_dir=args.cache_dir,
        checkpoint_path=args.teacher_checkpoint,
    )
    # Frozen target: eval mode (no dropout/BN updates) + no gradients.
    teacher.eval()
    for param in teacher.parameters():
      param.requires_grad_(False)
    return teacher

  def _build_feature_projectors(self, args, student, teacher, device):
    """FitNets ``r`` maps, one per matched layer pair (``None`` when disabled).

    The layer pairing is delegated to the shared ``match_layers`` helper; the
    projector widths are inferred with a single dummy forward pass per side.
    """
    if not self._cfg.feature_alpha:
      return None
    # Candidate layers on each side, filtered by the optional regex.
    student_layers = iter_named_layers(student, self._cfg.layer_pattern)
    teacher_layers = iter_named_layers(teacher, self._cfg.layer_pattern)
    # Policy-driven pairing (by_index / uniform / by_name); skips+logs extras.
    self._feature_pairs = match_layers(teacher_layers,
                                       student_layers,
                                       strategy=self._cfg.layer_strategy,
                                       pattern=self._cfg.layer_pattern)
    # Flattened feature widths for each paired layer, per side.
    s_dims = _activation_dims(args, student, [p.student for p in self._feature_pairs],
                              device)
    t_dims = _activation_dims(args, teacher, [p.teacher for p in self._feature_pairs],
                              device)
    projectors = nn.ModuleList()
    for s_dim, t_dim in zip(s_dims, t_dims):
      if s_dim == t_dim:
        # Widths already agree: r is the identity (the loss applies None).
        projectors.append(nn.Identity())
      else:
        # Lift the student feature (B, s_dim) -> (B, t_dim), as in FitNets §9.
        projectors.append(ProjectionHead(s_dim, self._cfg.proj_hidden, t_dim))
    return projectors

  def train_step(self, model, blob, global_step, *, labels=None):
    """Student forward + teacher soft targets; weighted KD objective.

    Runs the student (gradients on), advances the EMA teacher when applicable,
    and runs the teacher under ``no_grad`` to produce the soft targets.
    """
    # (B, C, H, W) image batch and (B,) integer labels.
    images = blob.data
    labels = _resolve_labels(blob, labels)
    if labels is None:
      fatal("--method distillation needs labels in blob.meta['labels'].", ValueError)

    # Student logits (B, n_classes); s_feats is a list of (B, ...) activations
    # (or None when feature KD is off).
    logits, s_feats = _forward_with_capture(model.student, images,
                                            self._capture_modules(model))
    if model.teacher_kind == "ema":
      # Advance the teacher BEFORE producing its targets for this step.
      momentum = linear_momentum(global_step, self._total_steps, self._cfg.ema_momentum,
                                 self._cfg.ema_final_momentum)
      model._ema.update(momentum)

    # Teacher targets are constants: run under no_grad (no graph retained).
    with torch.no_grad():
      # (B, n_classes) teacher logits; t_feats parallel to s_feats.
      teacher_logits, t_feats = _forward_with_capture(model.teacher, images,
                                                      self._capture_targets(model))

    # Supervised anchor: scalar cross-entropy of the student logits.
    ce = torch.nn.functional.cross_entropy(logits, labels)
    # Weighted distillation terms (§15): scalar.
    kd = self._kd_terms(model, images, logits, teacher_logits, s_feats, t_feats)
    loss = ce + kd
    return LossOutput(loss=loss,
                      metrics={
                          "loss": loss.detach(),
                          "ce": ce.detach(),
                          "kd": kd.detach(),
                      })

  def eval_step(self, model, blob, global_step=0, *, labels=None):
    """Evaluation forward; never advances the EMA teacher.

    Mirrors ``train_step`` without the EMA update, so validating on a schedule
    does not perturb the teacher state (DINO/BYOL follow the same split).
    """
    images = blob.data
    labels = _resolve_labels(blob, labels)
    with torch.no_grad():
      logits, s_feats = _forward_with_capture(model.student, images,
                                              self._capture_modules(model))
      teacher_logits, t_feats = _forward_with_capture(model.teacher, images,
                                                      self._capture_targets(model))
      ce = torch.nn.functional.cross_entropy(logits, labels)
      kd = self._kd_terms(model, images, logits, teacher_logits, s_feats, t_feats)
    loss = ce + kd
    return LossOutput(loss=loss, metrics={"loss": loss.detach()})

  def _capture_modules(self, model):
    """Student modules to hook (parallel to the matched feature pairs)."""
    # None disables hooking entirely; an empty ModuleList means feature KD off.
    if not model.projectors:
      return None
    return [pair.student for pair in self._feature_pairs]

  def _capture_targets(self, model):
    """Teacher modules to hook (parallel to the matched feature pairs)."""
    if not model.projectors:
      return None
    return [pair.teacher for pair in self._feature_pairs]

  def _kd_terms(self, model, images, logits, teacher_logits, s_feats, t_feats):
    """Weighted sum of the enabled KD objectives (scalar).

    Each term is scalar; disabled terms cost nothing (their alpha is 0).
    """
    cfg = self._cfg
    # Scalar zero on the logits' device/dtype: start of the weighted sum.
    kd = logits.new_zeros(())
    if cfg.response_alpha:
      # Response KD: (B, n_classes) vs (B, n_classes) -> scalar.
      kd = kd + cfg.response_alpha * kd_response_loss(logits, teacher_logits,
                                                      cfg.temperature)
    if model.projectors:
      # Feature/attention KD over the matched layers -> scalar.
      kd = kd + cfg.feature_alpha * self._feature_term(model, s_feats, t_feats)
    if cfg.relation_alpha:
      # Relation KD on the pooled backbone features: (B, D) vs (B, D) -> scalar.
      s_repr = encode_with_backbone(model.student, images)
      t_repr = encode_with_backbone(model.teacher, images)
      kd = kd + cfg.relation_alpha * kd_relation_loss(s_repr, t_repr)
    return kd

  def _feature_term(self, model, s_feats, t_feats):
    """Feature/attention KD averaged over the matched layer pairs (scalar)."""
    # None until the first live pair is seen (a layer may not have fired).
    total = None
    for proj, s_feat, t_feat in zip(model.projectors, s_feats, t_feats):
      if s_feat is None or t_feat is None:
        # This paired layer never fired for the given input: skip it.
        continue
      if self._cfg.attention:
        # Attention transfer consumes the raw (B, ...) maps (channel-agnostic).
        term = kd_attention_loss(s_feat, t_feat)
      else:
        # Feature MSE compares flattened maps: (B, D_s) vs (B, D_t) -> scalar.
        term = kd_feature_loss(_flatten(s_feat),
                               _flatten(t_feat),
                               projector=_projector_fn(proj))
      total = term if total is None else total + term
    if total is None:
      # Every hooked layer missed (e.g. a degenerate dummy shape): return a
      # zero connected to the graph so training does not crash on an empty term.
      return _first_feature(s_feats).new_zeros(()) if s_feats else None
    # Mean over the layers that actually contributed (not the total slots).
    return total / max(sum(f is not None for f in s_feats), 1)

  def get_checkpoint_state(self, model, args):
    """Persist the teacher kind (and momentum schedule) for a faithful resume."""
    state = {
        "method": "distillation",
        # §19: an EMA teacher is reconstructible, an external one is not; this
        # key disambiguates the two on resume.
        "teacher_kind": args.teacher_kind,
    }
    if model.teacher_kind == "ema":
      state["momentum"] = args.kd_ema_momentum
      state["final_momentum"] = args.kd_ema_final_momentum
    return state

  def load_checkpoint_state(self, model, state, args):
    """Restore the EMA momentum schedule (§19: resume ambiguity)."""
    if self._cfg is not None and "momentum" in state:
      self._cfg.ema_momentum = state["momentum"]
      self._cfg.ema_final_momentum = state["final_momentum"]

  def validate(self, model, images, num_samples):
    # No pixel-space visualization for distillation.
    return None


class _KdConfig:
  """Resolved KD hyperparameters (kept off ``args`` for a stable signature)."""

  def __init__(self, **kw):
    self.__dict__.update(kw)

  @classmethod
  def from_args(cls, args):
    """Snapshot the ``--kd_*`` CLI flags into a plain config object."""
    return cls(
        temperature=args.kd_temperature,
        response_alpha=args.kd_alpha,
        feature_alpha=args.kd_feature_alpha,
        attention=args.kd_attention,
        relation_alpha=args.kd_relation_alpha,
        layer_strategy=args.kd_layer_strategy,
        layer_pattern=args.kd_layer_pattern,
        proj_hidden=args.kd_proj_hidden,
        ema_momentum=args.kd_ema_momentum,
        ema_final_momentum=args.kd_ema_final_momentum,
    )


def _resolve_labels(blob, labels):
  """Labels from the keyword argument, else from ``blob.meta['labels']``."""
  return labels if labels is not None else blob.meta.get("labels")


def _step_budget(args, pipeline):
  """Total optimizer steps for the EMA ramp, or None when it cannot be sized.

  ``global_step`` counts optimizer steps (the trainer increments it once per
  ``grad_accum_steps`` flushes), so the budget divides the micro-batch total.
  """
  loader = getattr(pipeline, "train_loader", None)
  epochs = args.epochs
  if loader is None or not epochs:
    return None
  try:
    total = len(loader) * epochs
  except TypeError:  # iterable-only dataset (no len()): leave the ramp a no-op
    return None
  return total // max(args.grad_accum_steps or 1, 1)


def _flatten(feature):
  """Flatten every non-batch axis: ``(B, ...) -> (B, prod(...))``."""
  return feature.flatten(1)


def _first_feature(feats):
  """First non-None activation in *feats*, or None when all are None."""
  for feat in feats:
    if feat is not None:
      return feat
  return None


def _projector_fn(proj):
  """Adapt a projector module to the ``kd_feature_loss`` callable contract.

  ``nn.Identity`` stands for "no projection needed" (equal widths), which
  ``kd_feature_loss`` expresses as ``projector=None``.
  """
  if isinstance(proj, nn.Identity):
    return None
  return proj


def _activation_dims(args, model, modules, device):
  """Infer each module's flattened feature width with one dummy forward pass.

  Returns a list parallel to *modules*: the flattened width of each captured
  activation, falling back to the backbone-dim probe when a module did not fire
  for the dummy input.
  """
  if not modules:
    return []
  size = args.image_size or 224
  # Dummy batch (1, 3, S, S): shapes are batch-independent for these modules.
  dummy = torch.zeros(1, 3, size, size, device=device)
  # One forward pass captures every module's activation (list of (1, ...) or None).
  _, feats = _forward_with_capture(model, dummy, modules)
  dims = []
  for i, feat in enumerate(feats):
    if feat is None:
      # Module did not fire for the dummy: fall back to the attribute probe.
      dims.append(detect_backbone_dim(modules[i]))
    else:
      # Flatten (1, ...) -> (1, prod(...)) and take the feature width.
      dims.append(int(_flatten(feat).shape[1]))
  return dims
