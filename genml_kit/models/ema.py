"""Shared target-network and EMA utilities.

Several methods keep a second copy of a network that is *not* trained by
gradients: the target/teacher network blended from the source/student
network.

- Self-supervised pre-training: ``BYOL``, ``DINO`` (models) and ``I-JEPA``
  (method) maintain an EMA teacher.
- Off-policy RL: ``QNetwork`` (DQN) and ``SACCritic`` (SAC) maintain a
  frozen target used for bootstrapping, updated by a hard copy or a
  Polyak (soft) blend.

All of these reduce to the same two operations -- build a frozen copy and
then move it toward the source -- so they are extracted here once instead
of being re-implemented at every call site.

``EMAModel`` is deliberately a **plain object, not an ``nn.Module``**.
It only *references* the target and source modules; it does not own them.
That is what lets each parent keep the target as a direct attribute (e.g.
``BYOL.target_encoder``, ``QNetwork.target``) so the parent's
``state_dict`` keys are byte-for-byte identical to before this refactor --
existing checkpoints keep loading.  Attaching an ``nn.Module`` holder
instead would rename those keys (``target_encoder.*`` ->
``ema.target.*``) and silently break resume.

Two EMA idioms coexist and are kept deliberately separate:

1. **Loss-coupled** target/teacher networks (the ``EMAModel`` users above,
   plus distillation's ``--teacher_kind ema``): the EMA copy is *part of
   the objective* -- the loss reads it -- so it is owned, updated and
   checkpointed by the method itself.
2. **Eval-time** weights EMA (:class:`WeightsEMA`): the shadow is *not*
   part of the loss.  It exists so validation and the best checkpoint can
   be computed on a low-pass-filtered copy of the optimisation trajectory
   (the timm / MMDetection weights-EMA).  The trainer drives its cadence
   (one update per optimizer step, swap-in around validation); the
   *method* decides whether it applies, because only the method knows its
   topology (DPO owns a frozen reference tower, distillation owns teacher
   towers -- neither is EMA material).
"""

import copy

import torch

from genml_kit.datasets.iterable import safe_len
from genml_kit.utils.distributed import unwrap


class EMAModel:
  """Frozen target module moved toward a source module.

  The constructor does **not** copy: the caller passes an already-built
  target (typically ``copy.deepcopy(source)``) so that the target remains
  registered under the parent's own attribute name.  This class freezes
  the target's parameters and provides the update primitives.

  Args:
      target: The frozen module to update in place.
      source: The trainable module whose parameters drive the update.
  """

  def __init__(self, target, source):
    self.target = target
    self.source = source
    for p in self.target.parameters():
      p.requires_grad = False

  @torch.no_grad()
  def update(self, decay):
    """EMA blend: ``target <- decay * target + (1 - decay) * source``.

    Args:
        decay: EMA coefficient.  For self-supervised teachers this is the
            momentum ``m`` (close to 1 keeps the target nearly frozen);
            for a Polyak update with coefficient ``tau`` pass ``1 - tau``.
    """
    for sp, tp in zip(self.source.parameters(), self.target.parameters(), strict=True):
      tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)

  @torch.no_grad()
  def copy_hard(self):
    """Hard copy: ``target <- source`` (parameter-wise)."""
    self.target.load_state_dict(self.source.state_dict())


class WeightsEMA:
  """Shadow copy of the trained weights, driven by the trainer.

  Unlike :class:`EMAModel` (loss-coupled, method-owned), a ``WeightsEMA``
  is a pure eval-time artifact: it never enters a loss.  The trainer
  updates it once per optimizer step and may swap it in around validation
  so the best checkpoint carries the EMA weights (see
  :meth:`Method.wire_ema`).

  Args:
      model: The training module; a DDP facade is accepted -- the shadow
          is built from the *semantic* module underneath it, so the
          state-dict keys match the portable checkpoint contract.
      decay: EMA coefficient; the update is
          ``shadow <- decay * shadow + (1 - decay) * model``.
      final_decay: Optional ramp target: the effective decay moves linearly
          from *decay* to *final_decay* over *total_steps* (the DINO-style
          momentum schedule, via :func:`linear_momentum`).  Ignored when
          ``None`` or when *total_steps* is not computable (iterable-only
          datasets keep the schedule at *decay*).
      warmup_steps: Without *final_decay*: linear warmup of the effective
          decay from 0 up to *decay* over this many optimizer steps
          (0 disables the warmup).
      total_steps: Total optimizer steps of the run, when known (see
          :func:`compute_step_budget`).
      eval_enabled: Whether the trainer may evaluate and best-save the EMA
          weights.  Decided by the method (via its ``WEIGHTS_EMA`` flag and
          :meth:`Method.wire_ema`), *not* read from args by the trainer --
          other trainers share the loop and must never read a
          supervised-only flag.
      device: Optional device to hold the shadow on (e.g. ``"cpu"`` to
          save accelerator memory, at the cost of a per-step copy).
  """

  def __init__(self,
               model,
               decay,
               final_decay=None,
               warmup_steps=0,
               total_steps=None,
               eval_enabled=True,
               device=None):
    self.decay = decay
    self.final_decay = final_decay
    self.warmup_steps = warmup_steps
    self.total_steps = total_steps
    self.eval_enabled = eval_enabled
    # Deepcopy the semantic module, never the DDP facade: the shadow must
    # carry the portable state-dict keys, and facades may not deepcopy.
    self.target = copy.deepcopy(unwrap(model))
    if device is not None:
      self.target.to(device)
    for p in self.target.parameters():
      p.requires_grad = False
    self.step = 0

  @torch.no_grad()
  def update(self, model, step):
    """One blend toward the live weights; call once per optimizer step.

    Args:
        model: The training module (facade-safe: parameters are matched
            positionally, and facades expose the same order).
        step: Current optimizer step, driving the decay schedule.
    """
    effective = self._effective_decay(step)
    t_params = list(self.target.parameters())
    m_params = list(model.parameters())
    if len(t_params) == len(m_params):
      # The length guard above is deliberately lenient: on structure drift we
      # skip the blend rather than raise, so ``strict=False`` keeps that
      # contract explicit.
      for tp, mp in zip(t_params, m_params, strict=False):
        tp.data.mul_(effective).add_(mp.data, alpha=1.0 - effective)
    self._update_buffers(model, effective)
    self.step = step

  @torch.no_grad()
  def _update_buffers(self, model, effective):
    """Blend float buffers, copy integer buffers (BN stats vs. counters)."""
    t_buffers = list(self.target.buffers())
    m_buffers = list(model.buffers())
    if len(t_buffers) != len(m_buffers):
      return  # Structure drift: skip rather than corrupt the shadow.
    # Lenient by design, matching the skip above: ``strict=False`` is the
    # contract, not an oversight.
    for tb, mb in zip(t_buffers, m_buffers, strict=False):
      if tb.is_floating_point() and mb.is_floating_point():
        tb.data.mul_(effective).add_(mb.data, alpha=1.0 - effective)
      elif tb.shape == mb.shape:
        tb.copy_(mb)  # Integer counters (e.g. BN num_batches_tracked).

  def _effective_decay(self, step):
    """Decay actually applied at *step* (ramp, warmup, or flat)."""
    if self.final_decay is not None:
      return linear_momentum(step, self.total_steps, self.decay, self.final_decay)
    if self.warmup_steps and step < self.warmup_steps:
      return self.decay * (step / self.warmup_steps)
    return self.decay

  @torch.no_grad()
  def state_dict(self):
    """Checkpoint payload: shadow tensors plus the schedule position."""
    return {
        "state": {
            k: v.detach().clone() for k, v in self.target.state_dict().items()
        },
        "step": self.step,
    }

  @torch.no_grad()
  def load_state_dict(self, state):
    """Restore shadow tensors and schedule position (tolerant filtering).

    Keys absent from the current model (architecture drift) are skipped, so
    a shadow saved for an older topology never hard-fails a resume; the
    skipped keys keep their re-seeded values.
    """
    target_state = self.target.state_dict()
    filtered = {k: v for k, v in state["state"].items() if k in target_state}
    self.target.load_state_dict(filtered, strict=False)
    self.step = int(state.get("step", 0))

  @torch.no_grad()
  def copy_to(self, model):
    """Overwrite the live weights with the shadow (filtered, tolerant)."""
    model_state = model.state_dict()
    filtered = {k: v for k, v in self.target.state_dict().items() if k in model_state}
    model.load_state_dict(filtered, strict=False)

  @torch.no_grad()
  def swap(self, model):
    """Exchange the live weights with the shadow.

    Paired with :meth:`restore` around the validation window: the model
    evaluates (and best-saves) the EMA weights, then gets the raw
    trajectory back bit-for-bit.
    """
    self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    self.copy_to(model)
    self._swapped = True

  @torch.no_grad()
  def restore(self, model):
    """Undo :meth:`swap` (the inverse exchange)."""
    if not getattr(self, "_swapped", False):
      raise RuntimeError("restore() called without a matching swap()")
    model.load_state_dict(self._backup)
    del self._backup
    self._swapped = False


def linear_momentum(step, total, start, end):
  """Linear momentum ramp from *start* to *end* over *total* steps.

  Returns *start* when *total* is falsy (e.g. an iterable dataset with no
  ``len()``), so callers never divide by zero and the schedule is a no-op
  until the budget is known.

  Args:
      step: Current optimizer step.
      total: Total optimizer steps for the run.
      start: Momentum at ``step == 0``.
      end: Momentum at ``step >= total``.
  """
  if not total:
    return start
  ratio = min(step / total, 1.0)
  return start + (end - start) * ratio


def compute_step_budget(args, pipeline):
  """Total optimizer steps for an EMA ramp, or ``None`` when not computable.

  ``global_step`` counts optimizer steps (the trainer increments it once per
  ``grad_accum_steps`` flush), so the budget divides the micro-batch total
  ``len(train_loader) * epochs`` by ``grad_accum_steps``.  Returns ``None``
  when the loader is absent or has no ``len()`` (an iterable-only stream), so
  callers leave the ramp a no-op until the budget is known.

  Args:
      args: Parsed CLI args; uses ``epochs`` and ``grad_accum_steps``.
      pipeline: Data pipeline exposing ``train_loader``.

  Returns:
      int | None: Total optimizer steps, or ``None`` when not computable.
  """
  loader = getattr(pipeline, "train_loader", None)
  epochs = args.epochs
  if loader is None or not epochs:
    return None
  batches = safe_len(loader)
  if batches is None:  # iterable-only dataset (no len()): leave the ramp a no-op
    return None
  return (batches * epochs) // max(args.grad_accum_steps or 1, 1)
