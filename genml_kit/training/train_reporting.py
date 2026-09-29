"""Centralised metrics tracking and periodic logging for training.

``TrainReporting`` is the generic base used by both image and RL trainers.
``ImageTrainReporting`` adds accuracy/F1 tracking from logits+targets.

Extra metrics passed to :meth:`TrainReporting.step` are described by a
:class:`Metric`, which carries the accumulation semantics (average vs last
value) and the print format.  This lets the epoch summary render every
metric as space-separated ``name=value`` pairs without the reporter having
to guess whether a value is a loss (averaged) or a monotonic counter such
as ``env_steps`` (last value only).
"""

import collections
import enum
import logging
import time

import torch

from genml_kit.training.optim_factory import report_lr
from genml_kit.utils.attr import maybe_call, MISSING
from genml_kit.utils.gpu import gpu_stats_str

# How a Metric is folded across an epoch.
#   AVERAGE -- mean of the per-step values (losses, gauges).
#   LAST    -- final value seen (monotonic counters, e.g. env_steps).
MetricKind = enum.Enum("MetricKind", ["AVERAGE", "LAST"])

# A single scalar training metric.
#   name  -- log key (e.g. "critic_loss").
#   value -- current scalar value.
#   kind  -- accumulation semantics for the epoch summary.
#   fmt   -- printf-style format spec used to render the value.
Metric = collections.namedtuple("Metric", ["name", "value", "kind", "fmt"],
                                defaults=[MetricKind.AVERAGE, ".4f"])


def metric(name, value, kind=MetricKind.AVERAGE, fmt=".4f"):
  """Build a :class:`Metric` from a raw value.

  Args:
      name: The log key.
      value: The scalar value (a torch tensor is detached/``item()``-ed).
      kind: Accumulation semantics (:class:`MetricKind`).
      fmt: printf-style format spec used to render the value.

  Returns:
      A :class:`Metric` instance.
  """
  if hasattr(value, "item"):
    value = value.item()
  return Metric(name, value, kind, fmt)


class TrainReporting:
  """Accumulate per-batch training statistics and report periodically.

  This class owns:

  * cumulative epoch-level counters (loss, sample count),
  * a sliding *window* of recent results used for the periodic log lines,
  * timing helpers (epoch start, last-log timestamps), and
  * TensorBoard scalar writes (including GPU metrics).

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
  throughput_unit : str
      Label for throughput in the log line.  ``"img"`` → ``img/s``,
      ``"step"`` → ``step/s``.
  """

  def __init__(
      self,
      total_batches,
      log_every,
      writer=None,
      device=None,
      optimizer=None,
      throughput_unit="img",
  ):
    self._total_batches = total_batches
    self._log_every = log_every
    self._writer = writer
    self._device = device
    self._optimizer = optimizer
    self._throughput_unit = throughput_unit

    # Cumulative epoch-level counters.
    self._total_loss = 0.0
    self._total_samples = 0

    # Timing (internal).
    self._start_time = time.time()
    self._last_log_time = time.time()

    # Window buffers (reset every *log_every* steps).
    self._window_samples = 0
    self._window_loss = 0.0

    # Latest extra metric seen per name, as ``{name: Metric}``.  Persists
    # across log emissions so the epoch summary can render each metric's
    # format and, for ``LAST`` metrics, its final value.
    self._extra = {}

    # Epoch-level accumulation of ``AVERAGE`` extra metrics (e.g. SAC's
    # critic_loss / actor_loss / alpha_loss).  ``LAST`` metrics need no
    # extra state: their final value lives in ``_extra``.
    self._extra_totals = {}
    self._extra_counts = {}

  def step(
      self,
      batch_idx,
      batch_size,
      loss_value,
      global_step,
      *,
      report_now=False,
      extra_metrics=None,
  ):
    """Update cumulative and window stats, and optionally log.

    Parameters
    ----------
    batch_idx : int
        Zero-based batch index inside the current epoch.
    batch_size : int
        Number of samples in this micro-batch.
    loss_value : float
        Loss for this micro-batch as a **per-sample mean** -- i.e. the
        mean over the batch dimension, which is what every method's loss
        already is (``F.cross_entropy`` with default reduction, the
        ``.mean()`` terms of ``vo_losses``, the RL objectives).  The
        reporter applies the sample weighting itself using the
        ``batch_size`` passed alongside, so ``epoch_avg_loss`` is the
        true per-sample mean of the epoch.

        It previously documented an *unscaled* loss
        (``loss.item() * batch_size * grad_accum_steps``) while the
        arithmetic here expected a sum over samples, and the trainer
        passed a mean.  Sum-of-means divided by sample count made every
        reported loss too small by exactly ``batch_size`` -- invisible
        for classification, but for VO it put ``loss`` below its own
        ``mce`` component, which cannot happen since all loss terms are
        non-negative.
    global_step : int
        Running optimizer-step counter (for TensorBoard x-axis).
    report_now : bool
        If *True*, force a log report after updating stats (used for
        the very last batch of the epoch even when it doesn't fall on
        a ``log_every`` boundary).
    extra_metrics : dict[str, Metric] or None
        Additional metrics to log, keyed by name (e.g. ``{"epsilon": metric(
        "epsilon", 0.5)}``).  Values must be :class:`Metric` instances, which
        carry the accumulation semantics and print format.
    """
    # Cumulative epoch-level counters.  ``loss_value`` is a per-sample
    # mean, so weight it by the samples it covers; otherwise the epoch
    # mean would be a sum of means divided by a sample count.
    self._total_loss += loss_value * batch_size
    self._total_samples += batch_size

    # Window buffers (same weighting, so the step line's loss= and the
    # summary's loss= are both per-sample means).
    self._window_samples += batch_size
    self._window_loss += loss_value * batch_size

    # Record the latest value per extra metric (drives both the per-step
    # line and the epoch summary), and fold ``AVERAGE`` metrics into the
    # epoch mean.  ``LAST`` metrics need no accumulation.
    if extra_metrics:
      for name, m in extra_metrics.items():
        self._extra[name] = m
        if m.kind is not MetricKind.LAST:
          self._extra_totals[name] = self._extra_totals.get(name, 0.0) + m.value
          self._extra_counts[name] = self._extra_counts.get(name, 0) + 1

    # Decide whether to emit a report.
    if report_now or (batch_idx + 1) % self._log_every == 0:
      self._log_step(batch_idx, global_step)

  def epoch_avg_loss(self):
    """Return the epoch-level average loss (unscaled)."""
    if self._total_samples == 0:
      return 0.0
    return self._total_loss / self._total_samples

  def summary(self):
    """Return final epoch-level metrics and log a summary line.

    The headline ``loss`` is whatever the method reports as its scalar
    loss and is not comparable across methods.  For SAC it is the *sum*
    of three heterogeneous objectives (critic + actor + alpha), so it is
    dominated by ``actor_loss`` and conveys little on its own; the per-
    component values appended below are the informative numbers.

    The line is rendered as space-separated ``name=value`` pairs, matching
    the per-step log line.  ``AVERAGE`` metrics report their epoch mean;
    ``LAST`` metrics (monotonic counters such as ``env_steps``) report
    their final value.

    Returns
    -------
    float
        The epoch-level average loss.
    """
    avg_loss = self.epoch_avg_loss()
    elapsed = time.time() - self._start_time
    gpu = gpu_stats_str(self._device)
    parts = [f"loss={avg_loss:.4f}"]
    # Decompose the aggregate ``loss`` above with the per-component
    # metrics.  Sorted for a stable, diff-friendly log line.
    for k in sorted(self._extra):
      m = self._extra[k]
      if k in self._extra_totals:
        count = self._extra_counts.get(k, 0)
        if count:
          parts.append(f"{k}={self._extra_totals[k] / count:{m.fmt}}")
      else:
        parts.append(f"{k}={m.value:{m.fmt}}")
    parts.append(f"time={elapsed:.1f}s")
    if gpu:
      parts.append(gpu)
    logging.info("  Train Summary: " + " ".join(parts))
    return avg_loss

  def _log_step(self, batch_idx, global_step):
    """Compute windowed & cumulative metrics and emit a log line."""
    elapsed = time.time() - self._last_log_time
    w_samples = self._window_samples
    throughput = w_samples / elapsed if elapsed > 0 else 0.0
    w_loss = self._window_loss / w_samples if w_samples > 0 else 0.0

    # Cumulative metrics.
    avg_loss = self.epoch_avg_loss()

    # Hardware / optimizer info.
    gpu = gpu_stats_str(self._device)
    lr_str = report_lr(self._optimizer, writer=self._writer, step=global_step)

    # Throughput label.
    unit = f"{self._throughput_unit}/s"

    # Console log.
    msg = (f"  train [{batch_idx + 1}/{self._total_batches}]"
           f" loss={w_loss:.4f}"
           f" {unit}={throughput:.0f}")
    if gpu:
      msg += f" {gpu}"
    msg += f" {lr_str}"
    # Append extra metrics, each rendered with its declared format.
    for name, m in self._extra.items():
      msg += f" {name}={m.value:{m.fmt}}"
    logging.info(msg)

    # TensorBoard scalars.
    if self._writer is not None:
      self._writer.add_scalar("Train/loss", w_loss, global_step)
      self._writer.add_scalar("Train/loss_avg", avg_loss, global_step)
      self._writer.add_scalar("Train/throughput", throughput, global_step)
      for name, m in self._extra.items():
        self._writer.add_scalar(f"Train/{name}", m.value, global_step)
      if self._device is not None and self._device.type == "cuda":
        self._writer.add_scalar(
            "GPU/memory_MB",
            torch.cuda.memory_allocated(self._device) / 1024**2,
            global_step,
        )
        utilization = maybe_call(torch.cuda, "utilization", self._device)
        if utilization is not MISSING:
          self._writer.add_scalar(
              "GPU/utilization_pct",
              utilization,
              global_step,
          )

    # Reset window buffers and update timestamp.  ``self._extra`` is
    # intentionally NOT cleared: it carries the latest metric per name so
    # the epoch summary can render each one's format and last value.
    self._last_log_time = time.time()
    self._window_samples = 0
    self._window_loss = 0.0


class ImageTrainReporting(TrainReporting):
  """TrainReporting with image-classification accuracy/F1 tracking.

  Extends the base class to accumulate per-batch accuracy and macro-F1
  from model logits and ground-truth targets.  When logits/targets are
  not provided, falls back to tracking top1 from ``extra_metrics``.
  """

  def __init__(
      self,
      total_batches,
      log_every,
      writer=None,
      device=None,
      optimizer=None,
      throughput_unit="img",
  ):
    super().__init__(
        total_batches=total_batches,
        log_every=log_every,
        writer=writer,
        device=device,
        optimizer=optimizer,
        throughput_unit=throughput_unit,
    )

    # Image-specific cumulative counters.  ``_has_accuracy`` stays False
    # until a step actually provides logits/targets (or a top1 metric),
    # so methods without a classification head do not print accuracy.
    self._has_accuracy = False
    self._correct_top1 = 0

    # Image-specific window buffers.
    self._window_correct = 0
    self._window_preds = []
    self._window_labels = []

  def step(
      self,
      batch_idx,
      batch_size,
      loss_value,
      global_step,
      *,
      logits=None,
      targets=None,
      report_now=False,
      extra_metrics=None,
  ):
    """Update stats including accuracy/F1 from logits+targets.

    Parameters
    ----------
    logits : Tensor or None
        Model output logits ``[batch, num_classes]``.
    targets : Tensor or None
        Ground-truth label tensor for this micro-batch.
    extra_metrics : dict[str, Metric] or None
        Additional metrics.  If ``"top1"`` is present and logits/targets
        are not provided, its value is used for accuracy tracking.
    """
    # Track accuracy from logits/targets.
    if logits is not None and targets is not None:
      preds = logits.argmax(dim=1)
      self._correct_top1 += (preds == targets).sum().item()
      self._window_correct += (preds == targets).sum().item()
      self._window_preds.extend(preds.cpu().tolist())
      self._window_labels.extend(targets.cpu().tolist())
      self._has_accuracy = True
    elif extra_metrics and "top1" in extra_metrics:
      # Fallback: use top1 from method metrics (already a percentage).
      top1_val = extra_metrics["top1"].value
      self._correct_top1 += int(top1_val * batch_size / 100.0)
      self._window_correct += int(top1_val * batch_size / 100.0)
      self._has_accuracy = True

    super().step(
        batch_idx=batch_idx,
        batch_size=batch_size,
        loss_value=loss_value,
        global_step=global_step,
        report_now=report_now,
        extra_metrics=extra_metrics,
    )

  def _window_accuracy(self):
    """Top-1 accuracy over the current window (percent)."""
    if self._window_samples <= 0:
      return 0.0
    return self._window_correct / self._window_samples * 100.0

  def _window_macro_f1(self):
    """Macro-F1 over the current window's predictions (percent)."""
    if not self._window_preds:
      return 0.0
    from sklearn.metrics import f1_score
    return f1_score(
        self._window_labels, self._window_preds, average="macro",
        zero_division=0) * 100.0

  def epoch_avg_loss(self):
    """Return ``(avg_loss, top1_accuracy)`` tuple."""
    avg_loss = super().epoch_avg_loss()
    top1 = (self._correct_top1 / self._total_samples *
            100.0 if self._total_samples else 0.0)
    return avg_loss, top1

  def _extra_str(self):
    """Render every extra metric as name=value pairs, honouring its kind.

    Shared by ``summary`` and ``_log_step`` so the epoch line and the
    per-step line list the same metrics.  ``top1`` is skipped because it
    is already rendered from the accuracy counters above -- mirroring the
    guard the TensorBoard block uses.

    ``AVERAGE`` metrics report their epoch mean from
    ``_extra_totals``/``_extra_counts``; ``LAST`` metrics (monotonic
    counters such as ``env_steps``) report their final value from
    ``_extra``.  Printing ``_extra`` for every metric regardless of kind
    made the epoch summary report the *last batch* of a 625-step epoch,
    which for a pixel-domain metric like VO's ``mce`` is far too noisy to
    read as an epoch statistic -- and sat next to a true epoch mean
    (``loss=``) with silently different semantics.  The base
    ``TrainReporting.summary`` already aggregated this way; this override
    was the outlier.
    """
    parts = []
    for name, m in self._extra.items():
      if name == "top1":
        continue
      if name in self._extra_totals and self._extra_counts.get(name, 0):
        value = self._extra_totals[name] / self._extra_counts[name]
      else:
        value = m.value
      parts.append(f" {name}={value:{m.fmt}}")
    return "".join(parts)

  def summary(self):
    """Log image-specific summary and return ``(avg_loss, top1)``.

    ``top1`` is the epoch accuracy.  It is only present when the run
    actually tracked accuracy -- a method with no classification head
    would otherwise print a permanent, misleading 0.00%.
    """
    avg_loss, top1 = self.epoch_avg_loss()
    elapsed = time.time() - self._start_time
    gpu = gpu_stats_str(self._device)
    parts = [
        f"loss={avg_loss:.4f}",
    ]
    if self._has_accuracy:
      parts.append(f"top1={top1:.2f}%")
    parts.append(f"time={elapsed:.1f}s")
    parts.append(self._extra_str())
    if gpu:
      parts.append(gpu)
    logging.info("  Train Summary: " + " ".join(parts))
    return avg_loss, top1

  def _log_step(self, batch_idx, global_step):
    """Emit the windowed log line, plus accuracy when the method has it."""
    elapsed = time.time() - self._last_log_time
    w_samples = self._window_samples
    throughput = w_samples / elapsed if elapsed > 0 else 0.0
    w_loss = self._window_loss / w_samples if w_samples > 0 else 0.0

    # Cumulative metrics.
    avg_loss, top1 = self.epoch_avg_loss()

    # Hardware / optimizer info.
    gpu = gpu_stats_str(self._device)
    lr_str = report_lr(self._optimizer, writer=self._writer, step=global_step)

    # Console log.  Accuracy is appended only when this run actually
    # tracked it: a method with no classification head would otherwise
    # print a permanent, meaningless top1=0.00% / macro_f1=0.00%.
    msg = (f"  train [{batch_idx + 1}/{self._total_batches}]"
           f" loss={w_loss:.4f} ({avg_loss:.4f})")
    if self._has_accuracy:
      msg += f" top1={self._window_accuracy():.2f}% ({top1:.2f}%)"
      msg += f" macro_f1={self._window_macro_f1():.2f}%"
    msg += f" img/s={throughput:.0f}"
    msg += self._extra_str()
    if gpu:
      msg += f" {gpu}"
    msg += f" {lr_str}"
    logging.info(msg)

    # TensorBoard scalars.
    if self._writer is not None:
      self._writer.add_scalar("Train/loss", w_loss, global_step)
      if self._has_accuracy:
        self._writer.add_scalar("Train/top1", self._window_accuracy(), global_step)
        self._writer.add_scalar("Train/macro_f1", self._window_macro_f1(), global_step)
        self._writer.add_scalar("Train/top1_avg", top1, global_step)
      self._writer.add_scalar("Train/loss_avg", avg_loss, global_step)
      self._writer.add_scalar("Train/throughput", throughput, global_step)
      for name, m in self._extra.items():
        if name == "top1":
          continue  # Already tracked separately.
        self._writer.add_scalar(f"Train/{name}", m.value, global_step)
      if self._device is not None and self._device.type == "cuda":
        self._writer.add_scalar(
            "GPU/memory_MB",
            torch.cuda.memory_allocated(self._device) / 1024**2,
            global_step,
        )
        utilization = maybe_call(torch.cuda, "utilization", self._device)
        if utilization is not MISSING:
          self._writer.add_scalar(
              "GPU/utilization_pct",
              utilization,
              global_step,
          )

    # Reset window buffers and update timestamp.
    self._last_log_time = time.time()
    self._window_samples = 0
    self._window_correct = 0
    self._window_loss = 0.0
    self._window_preds = []
    self._window_labels = []
