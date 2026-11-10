"""Generic branchless trainer over the pipeline + method contract.

v4.2 (plans/GENERIC_PIPELINE.md s 4): one loop that is data- and
objective-agnostic.  ``BaseTrainer.run()`` owns, verbatim from the
original (train.py / pretrain/cli.py / train_vo.py):

- ``parse_state_flags`` / ``create_grad_monitor`` / model report;
- the single ``CheckpointSaver`` and its save-on-exit ``finally``;
- signal handling: interrupts land a consistent checkpoint before exit;
- the best-checkpoint cycle driven by ``method.has_metric_improved`` and
  ``method.METRIC_KEY`` (the checkpoint value is always stored under
  ``best_<metric_key>`` -- the loop never negates).

The epoch body (s 4) is the union of the two legacy loops: AMP autocast +
GradScaler, grad-accumulation with an end-of-epoch flush, the gradient
monitor placed AFTER ``unscale_`` and BEFORE clip, and per-epoch scheduler
step + ``method.on_epoch_end`` in ``run()``.
"""

import contextlib
import logging
from collections import namedtuple
from itertools import islice

import torch

from genml_kit.io.checkpointing import (
    CheckpointSaver,
    create_model_report,
    parse_state_flags,
)
from genml_kit.training.grad_monitor import create_grad_monitor
from genml_kit.training.metrics_sinks import ensure_logger
from genml_kit.training.model_utils import set_train_mode
from genml_kit.training.train_reporting import ImageTrainReporting, Metric, metric
from genml_kit.utils.args import amp_dtype_from_args
from genml_kit.utils.attr import MISSING, maybe_call
from genml_kit.utils.distributed import (
    all_gather_object,
    barrier,
    is_distributed,
    is_main_process,
    sampler_epoch,
    unwrap,
)
from genml_kit.utils.signal import InterruptedException, sigexcept

TrainingResult = namedtuple("TrainingResult", [
    "completed_epoch",
    "best_metric",
    "global_step",
    "interrupt_signals",
])


class _NoMonitor:
  """Null-object grad monitor: step() is a no-op."""

  def step(self, *_args, **_kwargs):
    pass


class BaseTrainer:
  """Task-agnostic training loop over ``method`` + ``pipeline``.

  Subclasses override ``validate`` only when they need loop-level
  validation; the base one calls ``method.evaluate(...)``.
  """

  #: Whether frozen-parameter state is included in checkpoints.
  SAVE_FROZEN = False

  def __init__(self,
               args,
               model,
               method,
               pipeline,
               optimization,
               device,
               logger,
               start_epoch,
               best_metric,
               global_step,
               resume_progress=None):
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
        logger: ``MetricsLogger`` fan-out (closed here on exit).
        start_epoch: First epoch to run (0 on a fresh run).
        best_metric: Best validation metric so far, in this method's own
            direction (e.g. ``float("inf")`` before the first validation
            of a minimizing metric).
        global_step: Optimizer step counter restored from the checkpoint.
        resume_progress: The checkpoint's ``in_progress`` marker (or
            ``None``): ``{"epoch", "micro_step", ...}`` describing an
            interrupted epoch.  Only honoured on the first epoch of the run.
    """
    self.args = args
    self.model = model
    self.method = method
    self.pipeline = pipeline
    self.optimization = optimization
    self.device = device
    # Always present: a NullSink-backed logger when none is supplied.
    self.logger = ensure_logger(logger)
    self.start_epoch = start_epoch
    self.best_metric = best_metric
    self.global_step = global_step
    # Mid-epoch resume marker restored from the checkpoint (or ``None``).
    # ``resume_step_in_epoch`` is the micro-batch position already consumed in
    # ``start_epoch``; it is fast-forwarded (grad-accum aligned) on the first
    # epoch and then reset so later epochs start at the top.
    self.resume_progress = dict(resume_progress) if resume_progress else None
    self.resume_step_in_epoch = 0
    if self.resume_progress and self.resume_progress.get("epoch") == start_epoch:
      self.resume_step_in_epoch = max(0,
                                      int(self.resume_progress.get("micro_step") or 0))
    # Last fully completed epoch (-1 = none yet); set by the loop and
    # read by ``validate`` so logging lands on the right epoch.
    self.epoch = start_epoch - 1
    # Progress marker for the epoch currently running: ``{"epoch", "micro_step"}``
    # at the last flush boundary (grad-accum aligned), or ``None`` between
    # epochs.  ``run``'s exit ``finally`` reads it to record where an
    # interrupted epoch should resume.
    self._epoch_progress = None

  def _resume_skip(self, epoch):
    """Return the grad-accum-aligned fast-forward for *epoch*.

    Only the first epoch of the run (``start_epoch``) is skipped: it is the
    one a mid-epoch checkpoint pointed into.  Every later epoch starts at 0.
    Rounding down to ``grad_accum_steps`` keeps the first accumulation window
    after the skip whole, so the DDP all-reduce never fires on a partial one.
    """
    if epoch != self.start_epoch or self.resume_step_in_epoch <= 0:
      return 0
    accum = max(1, self.args.grad_accum_steps)
    return (self.resume_step_in_epoch // accum) * accum

  def train_epoch(self, epoch, saver, step, monitor, skip=0):
    """Run one training epoch; return ``(avg_loss, new_step)``.

    ``skip`` fast-forwards the loader by that many micro-batches (used to
    resume inside an interrupted epoch).  It is ``grad_accum_steps``-aligned
    by the caller, so the first flush window after the skip is whole.
    """
    # Loop owns train/eval mode.
    set_train_mode(self.model, "train")
    # None unless fp16-on-CUDA.
    scaler = self.optimization.scaler
    amp_dtype = amp_dtype_from_args(self.args)
    total_batches = len(self.pipeline.train_loader)
    accum_steps = self.args.grad_accum_steps

    reporter = ImageTrainReporting(
        total_batches=total_batches,
        log_every=self.args.log_every,
        logger=self.logger,
        device=self.device,
        optimizer=self.optimization.optimizer,
        throughput_unit=getattr(self.pipeline, "THROUGHPUT_UNIT", "sample"),
        epoch=epoch,
    )

    # ``step_in_epoch`` restarts at ``skip`` so the accumulation boundary,
    # the tail flush and the reporter all see the true position.
    batch_iter = islice(self.pipeline.train_loader, skip, None)
    for step_in_epoch, blob in enumerate(batch_iter, start=skip):
      # Data AND meta.
      blob = self.pipeline.to_device(blob, self.device)
      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and self.device.type == "cuda"),
      ):
        loss_out = self.method.train_step(self.model, blob, step)
        # Raw, UNSCALED mean batch objective.
        loss = loss_out.loss
      # Accumulation boundary: shared by the no_sync gate below and the flush.
      is_flush = ((step_in_epoch + 1) % accum_steps == 0 or
                  (step_in_epoch + 1) == total_batches)
      # Scale ONLY for grad.
      grad = loss / accum_steps
      # Defer the DDP all-reduce on non-final micro-steps: gradients still
      # accumulate locally, and the flush step's backward performs the single
      # synchronized reduction over the whole window.  ``no_sync`` is absent on
      # an unwrapped model (world_size == 1), so fall back to a no-op context.
      no_sync = getattr(self.model, "no_sync", None)
      ctx = (contextlib.nullcontext() if (is_flush or no_sync is None) else no_sync())
      with ctx:
        if scaler is not None:
          scaler.scale(grad).backward()
        else:
          grad.backward()

      # Flush partial tail.
      if is_flush:
        if scaler is not None:
          scaler.unscale_(self.optimization.optimizer)
        # True grads: post-unscale, pre-clip.
        if monitor is not None:
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
        # Flush boundary reached: record progress so an interrupt after this
        # point resumes from here (grad-accum-aligned) instead of replaying.
        self._epoch_progress = {"epoch": epoch, "micro_step": step_in_epoch + 1}
        # Periodic mid-epoch checkpoint (--save_every).  Records the *current*
        # epoch plus the micro-step reached, so a resume continues here rather
        # than replaying the epoch.  ``barrier`` keeps peer ranks from running
        # ahead into the next collective while rank 0 writes (a slow save would
        # otherwise risk an NCCL watchdog timeout).
        if saver is not None and saver.should_save(step):
          barrier()
          # ``epoch - 1`` is the last *fully completed* epoch (the checkpoint's
          # ``epoch`` field keeps that meaning); the ``in_progress`` overlay
          # carries the true position inside ``epoch``.
          saver.save_latest(
              epoch - 1,
              **{self.best_metric_key: self.best_metric},
              global_step=step,
              in_progress=dict(self._epoch_progress),
          )
          barrier()

      # Sample count and targets.  Collators publish ``size`` on the blob, so
      # the batch dimension is never re-derived by sniffing the shape of
      # ``data`` (which may be a tensor, a tuple of views, or a dict).
      if isinstance(blob, dict):
        batch_size = blob["pixel_values"].shape[0]
        targets = blob.get("labels", None)
      else:
        batch_size = blob.size
        targets = blob.meta.get("labels", None) if isinstance(blob.meta, dict) else None

      # Surface the method's per-component loss metrics (mce, dlog_s,
      # dtheta, conf, ...).  These are already returned by every method's
      # train_step; without this they were dropped on the floor for the
      # image/pretraining/VO path and never reached the log or
      # TensorBoard.  Only the RL trainer was building extra_metrics.
      #
      # "loss" is skipped: it is already rendered from loss_value above,
      # and several methods re-emit it inside metrics, which would print
      # the loss twice.  A value that is already a ``Metric`` (e.g. a
      # DERIVED metric such as causal-LM perplexity) is passed through
      # untouched; a bare tensor is wrapped as an AVERAGE metric.
      extra_metrics = None
      if getattr(loss_out, "metrics", None):
        extra_metrics = {
            name: value if isinstance(value, Metric) else metric(name, value)
            for name, value in loss_out.metrics.items()
            if name != "loss"
        }
        extra_metrics = extra_metrics or None

      reporter.step(
          batch_idx=step_in_epoch,
          batch_size=batch_size,
          loss_value=loss.item(),
          logits=getattr(loss_out, "logits", None),
          targets=targets,
          global_step=step,
          report_now=(step_in_epoch + 1 == total_batches),
          extra_metrics=extra_metrics,
      )

    reporter.summary()
    return reporter.epoch_avg_loss()[0], step

  def validate(self):
    """Evaluate and return ``metrics[method.METRIC_KEY]`` (or ``None``).

    ``evaluate`` owns its mode: the default implementation wraps in
    ``torch.no_grad()``; overrides that need eval-mode (SimMIM
    reconstructions, VO mce) use ``model_mode(model, "eval")`` internally.

    Distributed policy: the validation set is sharded across ranks (see
    ``shard_val_loader``), so every rank evaluates its disjoint slice, the
    per-rank ``ValidationResult``s are gathered and merged by
    ``Method.merge_validation``, and only rank 0 logs the whole-set metrics.
    A single rank alone would see just its 1/N slice, hence the mandatory merge.
    """
    if self.pipeline.val_loader is None:
      return None
    result = self.method.evaluate(self.model, self.pipeline.val_loader, self.device,
                                  self.pipeline.to_device)
    if is_distributed():
      metrics = self.method.merge_validation(all_gather_object(result))
    else:
      metrics = result.metrics
    key = self.method.METRIC_KEY
    val = metrics[key]
    # Log every scalar metric the method reports (loss, mdice, pixel_acc,
    # per-class aggregates, ...) so nothing computed by ``evaluate`` is
    # dropped on the floor.  Non-scalar entries (e.g. ``per_class_iou``
    # lists) are skipped -- they carry no scalar meaning.  Validation is
    # epoch-scope, so these are emitted as epoch records.  Only rank 0 logs;
    # the merged metrics describe the whole set, so the others stay silent.
    if is_main_process():
      for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
          self.logger.log_epoch(f"val/{name}", value, epoch=self.epoch)
    return val

  def _should_validate(self):
    """True when this rank owns validation.

    Rank 0 only.  Used by env-based overrides (``RLTrainer.validate``) whose
    "validation" is a policy rollout that is not sharded -- unlike the
    DataLoader path, which shards the val set and merges across ranks.
    """
    return is_main_process()

  def has_metric_improved(self, old, new):
    """Route direction to the method (default: maximize)."""
    return self.method.has_metric_improved(new, old)

  def epoch_end(self):
    """Per-epoch hook routed to the method (e.g. momentum ramping)."""
    self.method.on_epoch_end(self.model, self.epoch, self.logger)

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
      grad_monitor = _NoMonitor()
    return grad_monitor

  def run(self):
    """Run the training loop and return a ``TrainingResult``."""
    args = self.args
    states_to_save = parse_state_flags(args.state_save)
    # Last fully completed (-1 = none).
    completed_epoch = self.start_epoch - 1
    grad_monitor = self._init_grad_monitor()

    # Report the final model state after checkpoint restoration and all
    # training initialization, immediately before training begins.
    logging.info(create_model_report(self.model))

    # All checkpoint writes go through one saver: state sources bound once,
    # per-save data (epoch, global_step, metrics) passed per call.
    # Pass the full optimization object (not just .optimizer) so that
    # methods with custom optimization wrappers (e.g. SACOptimization)
    # save/restore the complete state via their own state_dict().
    #
    # ``unwrap`` peels infrastructure wrappers only (DDP facade).  Saving must
    # see the semantic model: its state-dict keys are the portable checkpoint
    # contract, and ``serialize_lora_state`` tests ``isinstance(model,
    # PeftModel)``, which a facade would fail.
    saver = CheckpointSaver(
        unwrap(self.model),
        self.optimization,
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
          # Cooperative guard: if an inner ``except Exception`` swallowed an
          # InterruptedException, the latch would keep logging later signals
          # as "latched" without ever stopping.  Re-raise at this safe
          # epoch boundary so the interrupt is honoured regardless.
          interrupts.check_interrupt()
          # Advance every per-epoch sampler on the loader so each epoch
          # reshuffles: the pipeline's own batch sampler, the
          # ``DistributedSampler`` one level down, and the rank-strided shard
          # offset.  ``sampler_epoch`` walks the wrapper chain by capability, so
          # a plain ``BatchSampler`` cannot mask the ``DistributedSampler`` the
          # way a first-truthy lookup did.  No-op single-process.
          #
          # Preferred path is the pipeline, which owns the loader.  The direct
          # call covers a pipeline that does not define ``set_train_epoch`` (the
          # trainer only requires a ``train_loader`` attribute), so the epoch is
          # still delivered instead of being silently skipped.
          if maybe_call(self.pipeline, "set_train_epoch", epoch) is MISSING:
            sampler_epoch(self.pipeline.train_loader, epoch)
          logging.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
          # Fast-forward past what an interrupted checkpoint already consumed,
          # but only for the first epoch of the run: later epochs start clean.
          skip = self._resume_skip(epoch)
          self.resume_step_in_epoch = 0
          self._epoch_progress = {"epoch": epoch, "micro_step": skip}
          _, self.global_step = self.train_epoch(epoch,
                                                 saver,
                                                 self.global_step,
                                                 grad_monitor,
                                                 skip=skip)
          self.epoch = epoch
          # Epoch fully ran: it is no longer "in progress".
          self._epoch_progress = None

          if self.optimization.scheduler is not None:
            self.optimization.scheduler.step()

          metrics = self.validate()
          # Ranks that skipped validation (rank 0 owns it by default) wait here,
          # so a slow validation cannot let a peer run ahead and hit the next
          # epoch's all-reduce first (which can time out the NCCL watchdog).
          barrier()
          if (metrics is not None and self.args.vis_every > 0 and
              self.epoch % self.args.vis_every == 0):
            try:
              self.method.log_validation(
                  self.model,
                  self.pipeline.val_loader,
                  self.pipeline.to_device,
                  self.logger,
                  self.global_step,
                  self.device,
              )
            except InterruptedException:
              raise
            except Exception as e:
              logging.warning("Failed to log validation images: %s", e)
          if (metrics is not None and
              self.has_metric_improved(self.best_metric, metrics)):
            best_metric = self.best_metric
            self.best_metric = metrics
            saver.save_best(epoch,
                            **{self.best_metric_key: self.best_metric},
                            global_step=self.global_step,
                            **self.ckpt_extra(self.best_metric, self.global_step))
            logging.info(f"New best {self.best_metric_key}: "
                         f"{best_metric:.4f} -> {self.best_metric:.4f}")

          self.epoch_end()
          # Epoch fully done: train, validate, hooks.
          completed_epoch = epoch
      except InterruptedException:
        logging.warning(f"Interrupted by {interrupts.received}; saving "
                        "checkpoint.")
      finally:
        # The ``epoch`` field always means the last *fully completed* epoch;
        # when the interrupt landed inside one, the ``in_progress`` overlay
        # records that epoch plus the micro-step reached so resume continues
        # there instead of replaying it.
        progress = self._epoch_progress
        if progress is not None and progress["epoch"] > completed_epoch:
          in_progress = dict(progress)
        else:
          in_progress = None
        saver.save_latest(
            completed_epoch,
            **{self.best_metric_key: self.best_metric},
            global_step=self.global_step,
            **self.ckpt_extra(self.best_metric, self.global_step),
            **({
                "in_progress": in_progress
            } if in_progress else {}),
        )
        logging.info("Checkpoint saved on exit.")
        self.logger.close()

    return TrainingResult(completed_epoch, self.best_metric, self.global_step,
                          interrupts.received)
