"""Generic branchless trainer over the pipeline + method contract.

v4.2 (plans/GENERIC_PIPELINE.md s 4): one loop that is data- and
objective-agnostic.  ``BaseTrainer.run()`` owns, verbatim from the
original (train.py / pretrain/cli.py / train_vo.py):

- ``parse_state_flags`` / ``create_grad_monitor`` / model report;
- the single ``CheckpointSaver`` and its save-on-exit ``finally``;
- signal handling: interrupts land a consistent checkpoint before exit;
- the best-checkpoint cycle driven by ``method.has_metric_improved`` and
  ``method.metric_key`` (the checkpoint value is always stored under
  ``best_<metric_key>`` -- the loop never negates).

The epoch body (s 4) is the union of the two legacy loops: AMP autocast +
GradScaler, grad-accumulation with an end-of-epoch flush, the gradient
monitor placed AFTER ``unscale_`` and BEFORE clip, and per-epoch scheduler
step + ``method.on_epoch_end`` in ``run()``.
"""

import logging
from collections import namedtuple

import torch

from genml_kit.io.checkpointing import (
    CheckpointSaver,
    create_model_report,
    parse_state_flags,
)
from genml_kit.training.grad_monitor import create_grad_monitor
from genml_kit.training.model_utils import set_train_mode
from genml_kit.utils.signal import InterruptedException, sigexcept

TrainingResult = namedtuple("TrainingResult", [
    "completed_epoch",
    "best_metric",
    "global_step",
    "interrupt_signals",
])


class BaseTrainer:
  """Task-agnostic training loop over ``method`` + ``pipeline``.

  Subclasses override ``validate`` only when they need loop-level
  validation; the base one calls ``method.evaluate(...)``.
  """

  #: Whether frozen-parameter state is included in checkpoints.
  SAVE_FROZEN = False

  def __init__(self, args, model, method, pipeline, optimization, device, writer,
               start_epoch, best_metric, global_step):
    """Bind the run's collaborators and restored training state.

    Args:
        args: Parsed CLI args (epochs, batch/accum, checkpoint, monitor
            and save settings).
        model: The prepared model; mutated in place by training.
        method: The objective (model builder + loss + metric) instance.
        pipeline: The data pipeline (owns ``train_loader``/``val_loader``
            and ``to_device``).
        optimization: Object with ``optimizer``, optional ``scheduler``
            and optional ``scaler``.
        device: The run's ``torch.device``.
        writer: TensorBoard ``SummaryWriter`` (closed here on exit).
        start_epoch: First epoch to run (0 on a fresh run).
        best_metric: Best validation metric so far, in this method's own
            direction (e.g. ``float("inf")`` before the first validation
            of a minimizing metric).
        global_step: Optimizer step counter restored from the checkpoint.
    """
    self.args = args
    self.model = model
    self.method = method
    self.pipeline = pipeline
    self.optimization = optimization
    self.device = device
    self.writer = writer
    self.start_epoch = start_epoch
    self.best_metric = best_metric
    self.global_step = global_step
    # Last fully completed epoch (-1 = none yet); set by the loop and
    # read by ``validate`` so logging lands on the right epoch.
    self.epoch = start_epoch - 1

  # --- Epoch body (s 4) ---------------------------------------------------

  def train_epoch(self, epoch, saver, step, monitor):
    """Run one training epoch; return ``(avg_loss, new_step)``."""
    set_train_mode(self.model, "train")  # loop owns train/eval mode
    total, batches = 0.0, 0
    scaler = self.optimization.scaler  # None unless fp16-on-CUDA
    amp_dtype = getattr(self.args, "amp_dtype", None)
    total_batches = len(self.pipeline.train_loader)

    for step_in_epoch, blob in enumerate(self.pipeline.train_loader):
      blob = self.pipeline.to_device(blob, self.device)  # data AND meta
      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and self.device.type == "cuda"),
      ):
        loss_out = self.method.train_step(self.model, blob, step)
        loss = loss_out.loss  # raw, UNSCALED mean batch objective
      grad = loss / self.args.grad_accum_steps  # scale ONLY for grad
      if scaler is not None:
        scaler.scale(grad).backward()
      else:
        grad.backward()

      if ((step_in_epoch + 1) % self.args.grad_accum_steps == 0 or
          (step_in_epoch + 1) == total_batches):  # flush partial tail
        if scaler is not None:
          scaler.unscale_(self.optimization.optimizer)
        if monitor is not None:  # true grads: post-unscale, pre-clip
          monitor.step(step)
        if self.args.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.args.grad_clip)
        if scaler is not None:
          scaler.step(self.optimization.optimizer)
          scaler.update()
        else:
          self.optimization.optimizer.step()
        self.optimization.optimizer.zero_grad(set_to_none=True)
        step += 1

      total += loss.item()  # report RAW loss (see s 4 notes)
      batches += 1

    if self.writer is not None:
      self.writer.add_scalar("train/loss", total / max(batches, 1), epoch)
    return total / max(batches, 1), step

  def validate(self):
    """Evaluate and return ``metrics[method.METRIC_KEY]`` (or ``None``).

    ``evaluate`` owns its mode: the default implementation wraps in
    ``torch.no_grad()``; overrides that need eval-mode (SimMIM
    reconstructions, VO mce) use ``model_mode(model, "eval")`` internally.
    """
    if self.pipeline.val_loader is None:
      return None
    metrics = self.method.evaluate(self.model, self.pipeline.val_loader, self.device,
                                   self.pipeline.to_device)
    key = self.method.METRIC_KEY
    val = metrics[key]
    if self.writer is not None:
      self.writer.add_scalar(f"val/{key}", val, self.epoch)
    return val

  def has_metric_improved(self, old, new):
    """Route direction to the method (default: maximize)."""
    return self.method.has_metric_improved(new, old)

  def epoch_end(self):
    """Per-epoch hook routed to the method (e.g. momentum ramping)."""
    self.method.on_epoch_end(self.model, self.epoch, self.writer)

  def saver_extra(self):
    """Extra state attached to every checkpoint write (method state)."""
    return {"method_state": self.method.get_checkpoint_state(self.model, self.args)}

  def ckpt_extra(self, best, step):
    """Extra state attached to best-checkpoint and exit saves."""
    return {}

  @property
  def best_metric_key(self):
    """Checkpoint dict key carrying the best metric value."""
    return f"best_{self.method.METRIC_KEY}"

  def _init_grad_monitor(self):
    grad_monitor = create_grad_monitor(self.args, self.model)
    if grad_monitor is None:

      class _NoMonitor:
        """Null-object monitor: step() is a no-op."""

        def step(self, *_args, **_kwargs):
          return None

      grad_monitor = _NoMonitor()
    return grad_monitor

  # --- Loop ----------------------------------------------------------------

  def run(self):
    """Run the training loop and return a ``TrainingResult``."""
    args = self.args
    states_to_save = parse_state_flags(args.state_save)
    completed_epoch = self.start_epoch - 1  # last fully completed (-1 = none)
    grad_monitor = self._init_grad_monitor()

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

          if self.optimization.scheduler is not None:
            self.optimization.scheduler.step()

          metrics = self.validate()
          if (metrics is not None and
              self.has_metric_improved(self.best_metric, metrics)):
            best_metric = self.best_metric
            self.best_metric = metrics
            saver.save_best(epoch,
                            **{self.best_metric_key: self.best_metric},
                            global_step=self.global_step,
                            **self.ckpt_extra(self.best_metric, self.global_step))
            logging.info(f"New best {self.best_metric_key}: "
                         f"{best_metric:.2f} -> {self.best_metric:.2f}")

          self.epoch_end()
          completed_epoch = epoch  # epoch fully done: train, validate, hooks
      except InterruptedException:
        logging.warning(f"Interrupted by {interrupts.received}; saving "
                        "checkpoint.")
      finally:
        saver.save_latest(
            completed_epoch,
            **{self.best_metric_key: self.best_metric},
            global_step=self.global_step,
            **self.ckpt_extra(self.best_metric, self.global_step),
        )
        logging.info("Checkpoint saved on exit.")
        if self.writer is not None:
          self.writer.close()

    return TrainingResult(completed_epoch, self.best_metric, self.global_step,
                          interrupts.received)
