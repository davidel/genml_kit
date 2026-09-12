"""Base trainer class owning the shared training-loop mechanics.

Extracted from ``training/train.py::run_training`` (first via the
``run_training_loop`` skeleton, now as a base class) so every trainer --
supervised classification, self-supervised pre-training and the VO
front-end -- extends one loop instead of a structural copy.  ``run()``
owns, verbatim from the original:

- ``parse_state_flags`` / ``create_grad_monitor`` / model report;
- the single ``CheckpointSaver`` and its save-on-exit ``finally``;
- signal handling: interrupts land a consistent checkpoint before exit;
- the best-checkpoint cycle, driven by ``has_metric_improved`` and the
  named metric ``BEST_METRIC`` inside ``validate()``'s return.

Subclasses override the hooks, never the loop:

- ``train_epoch(epoch, saver, step, monitor)``: one training epoch,
  returning ``(loss, new_step)`` -- abstract;
- ``validate()``: validation returning a *named tuple* of scalar
  metrics (or ``None`` to disable validation); the best-checkpoint
  field is picked by name via ``BEST_METRIC`` -- no positional indices;
- ``has_metric_improved(old, new)``: best-checkpoint direction;
  the default maximizes, a minimizing metric (e.g. pixel error)
  overrides it -- no sign negation at the boundary;
- ``epoch_end()``: per-epoch hook (e.g. method momentum ramping);
- ``saver_extra()`` / ``ckpt_extra(best, step)``: extra checkpoint
  state (method state, LoRA blob).

The classification trainer keeps the ``best_macro_f1`` checkpoint key
via ``BEST_METRIC_KEY``; the pre-trainer passes ``best_metric=0.0`` and
defines no ``validate()``.
"""

import logging
from collections import namedtuple

from genml_kit.io.checkpointing import (
    CheckpointSaver,
    create_model_report,
    parse_state_flags,
)
from genml_kit.training.grad_monitor import create_grad_monitor
from genml_kit.utils.signal import InterruptedException, sigexcept

TrainingResult = namedtuple("TrainingResult", [
    "completed_epoch",
    "best_metric",
    "global_step",
    "interrupt_signals",
])


class BaseTrainer:
  """Task-agnostic training loop; subclasses supply the epoch body."""

  #: Checkpoint dict key carrying the best metric value.
  BEST_METRIC_KEY = "best_metric"
  #: Name of the best-checkpoint metric inside ``validate()``'s return.
  BEST_METRIC = None
  #: Whether frozen-parameter state is included in checkpoints.
  SAVE_FROZEN = False

  def __init__(self, args, model, optimization, device, writer, start_epoch,
               best_metric, global_step):
    """Bind the run's collaborators and restored training state.

    Args:
        args: Parsed CLI args (epochs, batch/accum, checkpoint, monitor
            and save settings).
        model: The prepared model; mutated in place by training.
        optimization: Object with ``optimizer``, optional ``scheduler``
            and optional ``scaler``.
        device: The run's ``torch.device``.
        writer: TensorBoard ``SummaryWriter`` (closed here on exit).
        start_epoch: First epoch to run (0 on a fresh run).
        best_metric: Best validation metric so far, in this trainer's
            own direction (e.g. ``float("inf")`` before the first
            validation of a minimizing metric).
        global_step: Optimizer step counter restored from the checkpoint.
    """
    self.args = args
    self.model = model
    self.optimization = optimization
    self.device = device
    self.writer = writer
    self.start_epoch = start_epoch
    self.best_metric = best_metric
    self.global_step = global_step
    # Last fully completed epoch (-1 = none yet); set by ``train_epoch``
    # and read by ``validate`` so logging lands on the right epoch.
    self.epoch = start_epoch - 1

  # --- Override surface ---------------------------------------------------

  def train_epoch(self, epoch, saver, step, monitor):
    """Run one training epoch and return ``(loss, new_step)``.

    Args:
        epoch: Zero-based index of the epoch being run.
        saver: The loop's ``CheckpointSaver`` (in-epoch saves).
        step: Optimizer step counter entering the epoch.
        monitor: Gradient monitor (or null object).

    Returns:
        Tuple ``(average_loss, new_global_step)``.
    """
    raise NotImplementedError

  def validate(self):
    """Validate and return a named tuple of scalar metrics, or ``None``."""
    return None

  def has_metric_improved(self, old, new):
    """Return True when *new* beats the best *old* (default: maximize)."""
    return new > old

  def epoch_end(self):
    """Per-epoch hook (e.g. momentum ramping); no-op by default."""
    return None

  def saver_extra(self):
    """Extra state attached to every checkpoint write."""
    return {}

  def ckpt_extra(self, best, step):
    """Extra state attached to best-checkpoint and exit saves."""
    return {}

  # --- Loop mechanics -----------------------------------------------------

  def run(self):
    """Run the training loop and return a ``TrainingResult``."""
    args = self.args
    states_to_save = parse_state_flags(args.state_save)
    completed_epoch = self.start_epoch - 1  # last fully completed (-1 = none)
    grad_monitor = create_grad_monitor(args, self.model)
    if grad_monitor is None:

      class _NoMonitor:
        """Null-object monitor: step() is a no-op."""

        def step(self, *_args, **_kwargs):
          return None

      grad_monitor = _NoMonitor()

    # Report the final model state after checkpoint restoration and all
    # training initialization, immediately before training begins.
    logging.info(create_model_report(self.model))

    # All checkpoint writes go through one saver: state sources bound once,
    # per-save data (epoch, global_step, metrics) passed per call.
    saver = CheckpointSaver(
        self.model,
        self.optimization.optimizer,
        self.optimization.scheduler,
        root=args.checkpoint,
        states_to_save=states_to_save,
        scaler=self.optimization.scaler,
        save_frozen=self.SAVE_FROZEN,
        remote_uri=args.remote_checkpoint,
        save_every=args.save_every,
        extra_fn=self.saver_extra,
    )

    # Signals arriving inside the loop become InterruptedException, so the
    # finally-block below still runs and a consistent checkpoint lands on
    # disk before a clean exit.  __exit__ restores the handlers only after
    # that save completed (the with-block encloses the try/finally).
    with sigexcept() as interrupts:
      try:
        for epoch in range(self.start_epoch, args.epochs):
          logging.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
          _, self.global_step = self.train_epoch(epoch, saver, self.global_step,
                                                 grad_monitor)
          self.epoch = epoch

          metrics = self.validate()
          if metrics is not None:
            value = getattr(metrics, self.BEST_METRIC)
            if self.has_metric_improved(self.best_metric, value):
              prev_best = self.best_metric
              self.best_metric = value
              saver.save_best(epoch,
                              **{self.BEST_METRIC_KEY: self.best_metric},
                              global_step=self.global_step,
                              **self.ckpt_extra(self.best_metric, self.global_step))
              logging.info(f"New best {self.BEST_METRIC_KEY}: "
                           f"{prev_best:.2f} -> {self.best_metric:.2f}")

          self.epoch_end()
          completed_epoch = epoch  # epoch fully done: train, validate, hooks
      except InterruptedException:
        logging.warning(f"Interrupted by {interrupts.received}; saving checkpoint.")
      finally:
        saver.save_latest(
            completed_epoch,
            **{self.BEST_METRIC_KEY: self.best_metric},
            global_step=self.global_step,
            **self.ckpt_extra(self.best_metric, self.global_step),
        )
        logging.info("Checkpoint saved on exit.")
        if self.writer is not None:
          self.writer.close()

    return TrainingResult(completed_epoch, self.best_metric, self.global_step,
                          interrupts.received)
