"""Centralised metrics tracking and periodic logging for training."""

import logging
import time

import torch
from sklearn.metrics import f1_score

from genml_kit.training.optim_factory import report_lr
from genml_kit.utils.gpu import gpu_stats_str


class TrainReporting:
  """Accumulate per-batch training statistics and report periodically.

  This class replaces the inline metric bookkeeping that was previously
  scattered throughout ``train_one_epoch``.  It owns:

  * cumulative epoch-level counters (loss, accuracy, sample count),
  * a sliding *window* of recent results used for the periodic log lines,
  * timing helpers (epoch start, last-log timestamps), and
  * TensorBoard scalar writes.

  Parameters
  ----------
  total_batches : int
      Number of batches in the current epoch (used for progress display).
  log_every : int
      Emit a log line every *log_every* batches (and at the last batch).
  writer : SummaryWriter or None
      Optional TensorBoard writer.
  device : torch.device
      Device the model lives on (for GPU-stat reporting).
  optimizer : torch.optim.Optimizer
      Optimizer whose learning-rate(s) should be logged.
  """

  def __init__(
      self,
      total_batches,
      log_every,
      writer=None,
      device=None,
      optimizer=None,
  ):
    self._total_batches = total_batches
    self._log_every = log_every
    self._writer = writer
    self._device = device
    self._optimizer = optimizer

    # Cumulative epoch-level counters.
    self._total_loss = 0.0
    self._correct_top1 = 0
    self._total_samples = 0

    # Timing (internal).
    self._start_time = time.time()
    self._last_log_time = time.time()

    # Window buffers (reset every *log_every* steps).
    self._window_samples = 0
    self._window_correct = 0
    self._window_loss = 0.0
    self._window_preds = []
    self._window_labels = []

  def step(
      self,
      batch_idx,
      batch_size,
      loss_value,
      logits,
      targets,
      global_step,
      report_now=False,
  ):
    """Update cumulative and window stats, and optionally log.

    Parameters
    ----------
    batch_idx : int
        Zero-based batch index inside the current epoch.
    batch_size : int
        Number of samples in this micro-batch.
    loss_value : float
        **Unscaled** loss for this micro-batch
        (i.e. ``loss.item() * batch_size * grad_accum_steps``).
    logits : Tensor
        Model output logits ``[batch, num_classes]``.
    targets : Tensor
        Ground-truth label tensor for this micro-batch.
    global_step : int
        Running optimizer-step counter (for TensorBoard x-axis).
    report_now : bool
        If *True*, force a log report after updating stats (used for
        the very last batch of the epoch even when it doesn't fall on
        a ``log_every`` boundary).
    """
    preds = logits.argmax(dim=1)

    # Cumulative epoch-level counters.
    self._total_loss += loss_value
    self._correct_top1 += (preds == targets).sum().item()
    self._total_samples += batch_size

    # Window buffers.
    self._window_samples += batch_size
    self._window_loss += loss_value
    self._window_correct += (preds == targets).sum().item()
    self._window_preds.extend(preds.cpu().tolist())
    self._window_labels.extend(targets.cpu().tolist())

    # Decide whether to emit a report.
    if report_now or (batch_idx + 1) % self._log_every == 0:
      self._log_step(batch_idx, global_step)

  def summary(self):
    """Return final epoch-level metrics and log a summary line.

    Returns
    -------
    tuple[float, float]
        ``(avg_loss, top1)`` — the epoch-level average cross-entropy loss
        and top-1 accuracy (percentage).
    """
    avg_loss = self._total_loss / self._total_samples if self._total_samples else 0.0
    top1 = (self._correct_top1 / self._total_samples *
            100.0 if self._total_samples else 0.0)
    elapsed = time.time() - self._start_time
    logging.info(f"  Train stats -> loss: {avg_loss:.4f}"
                 f" | top1: {top1:.2f}%"
                 f" | time: {elapsed:.1f}s")
    return avg_loss, top1

  def _log_step(self, batch_idx, global_step):
    """Compute windowed & cumulative metrics and emit a log line."""
    elapsed = time.time() - self._last_log_time
    w_samples = self._window_samples
    throughput = w_samples / elapsed if elapsed > 0 else 0.0
    w_loss = self._window_loss / w_samples if w_samples > 0 else 0.0
    w_top1 = (self._window_correct / w_samples * 100.0 if w_samples > 0 else 0.0)

    # Window macro F1.
    w_macro_f1 = 0.0
    if self._window_preds:
      w_macro_f1 = (f1_score(
          self._window_labels, self._window_preds, average="macro", zero_division=0) *
                    100.0)

    # Cumulative metrics.
    avg_loss = self._total_loss / self._total_samples
    top1 = (self._correct_top1 / self._total_samples) * 100.0

    # Hardware / optimizer info.
    gpu = gpu_stats_str(self._device)
    lr_str = report_lr(self._optimizer, writer=self._writer, step=global_step)

    # Console log.
    msg = (f"  [Step {batch_idx + 1}/{self._total_batches}]"
           f" loss={w_loss:.4f} ({avg_loss:.4f})"
           f" top1={w_top1:.2f}% ({top1:.2f}%)"
           f" macro_f1={w_macro_f1:.2f}%"
           f" {lr_str}"
           f" img/s={throughput:.0f}")
    logging.info(msg)
    if gpu:
      logging.info(f"  [Step {batch_idx + 1}/{self._total_batches}] {gpu}")

    # TensorBoard scalars.
    if self._writer is not None:
      self._writer.add_scalar("Train/loss", w_loss, global_step)
      self._writer.add_scalar("Train/top1", w_top1, global_step)
      self._writer.add_scalar("Train/macro_f1", w_macro_f1, global_step)
      self._writer.add_scalar("Train/loss_avg", avg_loss, global_step)
      self._writer.add_scalar("Train/top1_avg", top1, global_step)
      self._writer.add_scalar("Train/throughput", throughput, global_step)
      if self._device is not None and self._device.type == "cuda":
        self._writer.add_scalar(
            "GPU/memory_MB",
            torch.cuda.memory_allocated(self._device) / 1024**2,
            global_step,
        )
        if hasattr(torch.cuda, "utilization"):
          self._writer.add_scalar(
              "GPU/utilization_pct",
              torch.cuda.utilization(self._device),
              global_step,
          )

    # Reset window buffers and update timestamp.
    self._last_log_time = time.time()
    self._window_samples = 0
    self._window_correct = 0
    self._window_loss = 0.0
    self._window_preds = []
    self._window_labels = []
