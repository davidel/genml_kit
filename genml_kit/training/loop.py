"""Task-agnostic training-loop skeleton shared by all trainers.

Extracted verbatim from ``training/train.py::run_training`` (plan §12.6,
item 3) so every trainer -- supervised classification, self-supervised
pre-training and the VO front-end -- is a *consumer* of the same loop
instead of a structural copy.  The loop owns, verbatim from the original
``run_training``:

- ``parse_state_flags`` / ``create_grad_monitor`` / model report;
- the single ``CheckpointSaver`` all checkpoint writes go through;
- the ``sigexcept()`` block turning signals into ``InterruptedException``
  so the ``finally``-block still lands a consistent checkpoint on disk;
- the per-epoch cycle with best-checkpoint selection and the
  save-on-exit in the ``finally`` block.

The task-specific parts are injected:

- ``train_epoch_fn``: one epoch (closures keep their own writer logging);
- ``validate_fn``: optional validation returning a metric sequence;
- ``best_index``: which element of that sequence is "higher is better";
- ``epoch_end_fn``: per-epoch hook (e.g. method momentum ramping);
- ``ckpt_extra_fn``: per-save checkpoint extras (lora blob, method state).

The classification trainer keeps the ``best_macro_f1`` checkpoint key
via ``best_metric_key``; the pre-trainer passes ``best_metric=0.0`` and
no ``validate_fn``.
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

TrainingResult = namedtuple(
    "TrainingResult",
    "completed_epoch, best_metric, global_step, interrupt_signals")


def run_training_loop(args,
                      model,
                      optimization,
                      device,
                      *,
                      train_epoch_fn,
                      validate_fn=None,
                      best_index=3,
                      best_metric_key="best_metric",
                      epoch_end_fn=None,
                      saver_extra_fn=None,
                      ckpt_extra_fn=None,
                      save_frozen=False,
                      writer=None,
                      start_epoch=0,
                      best_metric=0.0,
                      global_step=0):
  """Run the generic per-epoch training loop.

  Args:
      args: Parsed CLI args (``epochs``, ``state_save``, ``checkpoint``,
          ``remote_checkpoint``, ``save_every``, grad-monitor settings).
      model: The prepared ``torch.nn.Module``; mutated in place.
      optimization: Object with ``optimizer`` and optional ``scheduler``
          and ``scaler`` attributes (as built by ``optim_factory``).
      device: The run's ``torch.device`` (passed to ``validate_fn``).
      train_epoch_fn: Callable ``(epoch, saver, global_step) ->
          (loss, global_step)``; logs its own per-epoch writer scalars.
      validate_fn: Optional zero-arg callable returning a metric
          sequence; ``None`` disables validation and best-checkpoint
          selection (pre-training and VO-supervised-only runs).
      best_index: Index of the "higher is better" metric inside
          ``validate_fn``'s return value.
     best_metric_key: Checkpoint key under which the best metric is
         stored ("best_macro_f1" for the classification trainer).
     epoch_end_fn: Optional callable ``(model, epoch, writer)`` run at
          the end of every epoch (e.g. BYOL teacher momentum).
      saver_extra_fn: Optional zero-arg callable returning a dict merged
          into every checkpoint (pre-training ``method_state``).
      ckpt_extra_fn: Optional callable ``(best_metric, global_step)``
          returning per-save kwargs (classification: the LoRA blob).
      save_frozen: Forwarded to the ``CheckpointSaver``.
      writer: TensorBoard ``SummaryWriter``; closed on exit when given.
      start_epoch: First epoch to run (0 on a fresh run).
      best_metric: Best metric value restored from a checkpoint.
      global_step: Optimizer step counter restored from a checkpoint.

  Returns:
      ``TrainingResult`` with the completed epoch, the best metric, the
      global step and any interrupt signals received.
  """
  del device  # validate_fn closes over the device it needs
  states_to_save = parse_state_flags(args.state_save)
  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  grad_monitor = create_grad_monitor(args, model)
  if grad_monitor is None:

    class _NoMonitor:
      """Null-object monitor: step() is a no-op."""

      def step(self, *_args, **_kwargs):
        return None

    grad_monitor = _NoMonitor()

  # Report the final model state after checkpoint restoration and all
  # training initialization, immediately before training begins.
  logging.info(create_model_report(model))

  # All checkpoint writes go through one saver: state sources bound once,
  # per-save data (epoch, global_step, metrics) passed per call.
  saver = CheckpointSaver(
      model,
      optimization.optimizer,
      optimization.scheduler,
      root=args.checkpoint,
      states_to_save=states_to_save,
      scaler=optimization.scaler,
      save_frozen=save_frozen,
      remote_uri=args.remote_checkpoint,
      save_every=args.save_every,
      extra_fn=saver_extra_fn,
  )

  # Signals arriving inside the loop become InterruptedException, so the
  # finally-block below still runs and a consistent checkpoint lands on
  # disk before a clean exit.  __exit__ restores the handlers only after
  # that save completed (the with-block encloses the try/finally).
  with sigexcept() as interrupts:
    try:
      for epoch in range(start_epoch, args.epochs):
        logging.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        _, global_step = train_epoch_fn(epoch, saver, global_step, grad_monitor)

        if validate_fn is not None:
          metrics = validate_fn()
          if metrics[best_index] > best_metric:
            prev_best = best_metric
            best_metric = metrics[best_index]
            saver.save_best(
                epoch,
                **{best_metric_key: best_metric},
                global_step=global_step,
                **(ckpt_extra_fn(best_metric, global_step) if ckpt_extra_fn else {}))
            logging.info(f"New best {best_metric_key}: "
                         f"{prev_best:.2f}% -> {best_metric:.2f}%")

        if epoch_end_fn is not None:
          epoch_end_fn(model, epoch, writer)
        completed_epoch = epoch
    except InterruptedException:
      logging.warning(f"Interrupted by {interrupts.received}; saving checkpoint.")
    finally:
      saver.save_latest(
          completed_epoch,
          **{best_metric_key: best_metric},
          global_step=global_step,
          **(ckpt_extra_fn(best_metric, global_step) if ckpt_extra_fn else {}),
      )
      logging.info("Checkpoint saved on exit.")
      if writer is not None:
        writer.close()

  return TrainingResult(completed_epoch, best_metric, global_step, interrupts.received)
