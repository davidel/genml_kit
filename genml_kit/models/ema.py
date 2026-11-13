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
"""

import torch


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
    for sp, tp in zip(self.source.parameters(), self.target.parameters()):
      tp.data.mul_(decay).add_(sp.data, alpha=1.0 - decay)

  @torch.no_grad()
  def copy_hard(self):
    """Hard copy: ``target <- source`` (parameter-wise)."""
    self.target.load_state_dict(self.source.state_dict())


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
  try:
    total = len(loader) * epochs
  except TypeError:  # iterable-only dataset (no len()): leave the ramp a no-op
    return None
  return total // max(args.grad_accum_steps or 1, 1)
