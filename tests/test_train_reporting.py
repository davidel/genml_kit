"""Tests for genml_kit.train_reporting.TrainReporting.

These are white-box tests: they deliberately assert on the private
(underscore-prefixed) attributes of ``TrainReporting`` to verify its
internal state transitions.
"""

import logging
from unittest.mock import MagicMock

import torch


def _make_reporter(**kwargs):
  """Create a TrainReporting with sensible defaults for testing."""
  from genml_kit.training.train_reporting import TrainReporting
  opt = MagicMock()
  opt.param_groups = [{"lr": 1e-4}]
  defaults = {
      "total_batches": 10,
      "log_every": 5,
      "writer": None,
      "device": torch.device("cpu"),
      "optimizer": opt,
  }
  defaults.update(kwargs)
  return TrainReporting(**defaults)


def _dummy_batch(batch_size=4, num_classes=3, num_correct=4):
  """Return (logits, targets) with *num_correct* correct predictions.

  The first *num_correct* samples are made correct by assigning a high logit
  to the target class.  All remaining samples are made *incorrect* by
  assigning the high logit to the *wrong* class.
  """
  targets = torch.arange(batch_size) % num_classes
  logits = torch.zeros(batch_size, num_classes)
  for i in range(batch_size):
    if i < num_correct:
      logits[i, targets[i]] = 10.0
    else:
      # Pick a wrong class: shift by +1 (mod num_classes).
      wrong = (targets[i].item() + 1) % num_classes
      logits[i, wrong] = 10.0
  return logits, targets


def test_init_counters_are_zero():
  r = _make_reporter()
  assert r._total_loss == 0.0
  assert r._correct_top1 == 0
  assert r._total_samples == 0
  assert r._window_samples == 0
  assert r._window_loss == 0.0
  assert r._window_correct == 0
  assert r._window_preds == []
  assert r._window_labels == []


def test_init_stores_references():
  device = torch.device("cpu")
  writer = MagicMock()
  opt = MagicMock()
  r = _make_reporter(writer=writer, device=device, optimizer=opt)
  assert r._writer is writer
  assert r._device is device
  assert r._optimizer is opt
  assert r._total_batches == 10
  assert r._log_every == 5


def test_step_accumulates_loss():
  r = _make_reporter(log_every=100)  # large log_every to suppress logging
  logits, targets = _dummy_batch(batch_size=4)
  r.step(0, 4, 2.5, logits, targets, 0)
  assert r._total_loss == 2.5
  assert r._total_samples == 4
  r.step(1, 4, 1.0, logits, targets, 1)
  assert r._total_loss == 3.5
  assert r._total_samples == 8


def test_step_accumulates_top1():
  r = _make_reporter(log_every=100)
  logits, targets = _dummy_batch(batch_size=4, num_classes=3, num_correct=3)
  r.step(0, 4, 1.0, logits, targets, 0)
  assert r._correct_top1 == 3
  assert r._total_samples == 4


def test_step_accumulates_window():
  r = _make_reporter(log_every=100)
  logits, targets = _dummy_batch(batch_size=4, num_correct=2)
  r.step(0, 4, 2.0, logits, targets, 0)
  assert r._window_samples == 4
  assert r._window_loss == 2.0
  assert r._window_correct == 2
  assert len(r._window_preds) == 4
  assert len(r._window_labels) == 4
  assert r._window_labels == targets.tolist()


def test_log_triggered_on_log_every_boundary(caplog):
  r = _make_reporter(log_every=3)
  logits, targets = _dummy_batch()
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 1.0, logits, targets, 0)  # batch_idx=0, (0+1)%3 != 0
    r.step(1, 4, 1.0, logits, targets, 1)  # batch_idx=1, (1+1)%3 != 0
    r.step(2, 4, 1.0, logits, targets, 2)  # batch_idx=2, (2+1)%3 == 0 -> log
  assert "[Step 3/10]" in caplog.text
  assert "loss=" in caplog.text


def test_no_log_when_not_on_boundary(caplog):
  r = _make_reporter(log_every=10)
  logits, targets = _dummy_batch()
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 1.0, logits, targets, 0)
    r.step(1, 4, 1.0, logits, targets, 1)
  assert "[Step" not in caplog.text


def test_report_now_forces_log(caplog):
  r = _make_reporter(log_every=100)  # would NOT normally log
  logits, targets = _dummy_batch()
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 1.0, logits, targets, 0, report_now=True)
  assert "[Step 1/10]" in caplog.text
  assert "loss=" in caplog.text


def test_window_resets_after_log():
  r = _make_reporter(log_every=1)
  logits, targets = _dummy_batch(batch_size=4, num_correct=2)
  r.step(0, 4, 3.0, logits, targets, 0)
  # After the log, window buffers should be reset.
  assert r._window_samples == 0
  assert r._window_loss == 0.0
  assert r._window_correct == 0
  assert r._window_preds == []
  assert r._window_labels == []
  # But cumulative counters should still hold.
  assert r._total_samples == 4
  assert r._total_loss == 3.0
  assert r._correct_top1 == 2


def test_window_accumulates_between_logs():
  r = _make_reporter(log_every=3)
  logits, targets = _dummy_batch(batch_size=4, num_correct=4)
  r.step(0, 4, 1.0, logits, targets, 0)
  r.step(1, 4, 1.0, logits, targets, 1)
  # No log yet — window should have accumulated 8 samples.
  assert r._window_samples == 8
  assert r._window_loss == 2.0
  r.step(2, 4, 1.0, logits, targets, 2)  # triggers log
  assert r._window_samples == 0  # reset


def test_log_contains_key_fields(caplog):
  r = _make_reporter(log_every=1, total_batches=5)
  logits, targets = _dummy_batch(batch_size=4, num_classes=3, num_correct=3)
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 2.0, logits, targets, 0)
  assert "[Step 1/5]" in caplog.text
  assert "loss=" in caplog.text
  assert "top1=" in caplog.text
  assert "macro_f1=" in caplog.text
  assert "img/s=" in caplog.text


def test_log_metrics_are_correct(caplog):
  """Verify the windowed and cumulative numbers in the log line."""
  r = _make_reporter(log_every=2, total_batches=4)
  logits, targets = _dummy_batch(batch_size=4, num_classes=3, num_correct=4)
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 4.0, logits, targets, 0)
    r.step(1, 4, 2.0, logits, targets, 1)

  # Window: loss=6.0/8=0.75, top1=100%, samples=8
  lines = [ln for ln in caplog.text.splitlines() if "loss=0.7500" in ln]
  assert len(lines) >= 1
  # Cumulative also 0.75 so (0.7500) should appear.
  assert "0.7500" in caplog.text
  assert "100.00%" in caplog.text


def test_summary_returns_correct_values():
  r = _make_reporter(log_every=100)
  logits, targets = _dummy_batch(batch_size=4, num_classes=3, num_correct=3)
  r.step(0, 4, 4.0, logits, targets, 0)
  r.step(1, 4, 2.0, logits, targets, 1)
  avg_loss, top1 = r.summary()
  # avg_loss = 6.0 / 8 = 0.75
  assert abs(avg_loss - 0.75) < 1e-6
  # top1 = 6/8 * 100 = 75.0
  assert abs(top1 - 75.0) < 1e-6


def test_summary_logs_final_line(caplog):
  r = _make_reporter(log_every=100)
  logits, targets = _dummy_batch(batch_size=4, num_correct=2)
  r.step(0, 4, 3.0, logits, targets, 0)
  with caplog.at_level(logging.INFO):
    r.summary()
  assert "Train stats ->" in caplog.text
  assert "loss:" in caplog.text
  assert "top1:" in caplog.text
  assert "time:" in caplog.text


def test_summary_zero_samples():
  r = _make_reporter()
  avg_loss, top1 = r.summary()
  assert avg_loss == 0.0
  assert top1 == 0.0


def test_macro_f1_in_log(caplog):
  """All-correct single-class window should produce macro F1 = 100%."""
  r = _make_reporter(log_every=1)
  batch_size = 4
  targets = torch.zeros(batch_size, dtype=torch.long)
  logits = torch.zeros(batch_size, 2)
  logits[:, 0] = 10.0  # all predict class 0
  with caplog.at_level(logging.INFO):
    r.step(0, batch_size, 1.0, logits, targets, 0)
  assert "macro_f1=100.00%" in caplog.text


def test_macro_f1_zero_when_no_correct(caplog):
  """Single class always predicted wrong -> macro F1 = 0."""
  r = _make_reporter(log_every=1)
  targets = torch.tensor([0, 0, 0, 0])
  logits = torch.zeros(4, 2)
  logits[:, 1] = 10.0  # all predict class 1
  with caplog.at_level(logging.INFO):
    r.step(0, 4, 1.0, logits, targets, 0)
  assert "macro_f1=0.00%" in caplog.text


def test_tensorboard_scalars_written():
  writer = MagicMock()
  r = _make_reporter(log_every=1, writer=writer)
  logits, targets = _dummy_batch(batch_size=4, num_correct=3)
  r.step(0, 4, 2.0, logits, targets, 42)

  expected_calls = {
      ("Train/loss",),
      ("Train/top1",),
      ("Train/macro_f1",),
      ("Train/loss_avg",),
      ("Train/top1_avg",),
      ("Train/throughput",),
  }
  written = {(c[0][0],) for c in writer.add_scalar.call_args_list}
  assert expected_calls <= written
  # Verify step=42 was used.
  for call in writer.add_scalar.call_args_list:
    assert call[0][2] == 42


def test_no_tensorboard_calls_when_writer_none():
  r = _make_reporter(log_every=1, writer=None)
  logits, targets = _dummy_batch()
  # Should not raise.
  r.step(0, 4, 1.0, logits, targets, 0)


def test_single_batch_epoch():
  """Epoch with exactly 1 batch and log_every=1."""
  r = _make_reporter(total_batches=1, log_every=1)
  logits, targets = _dummy_batch(batch_size=2, num_correct=2)
  r.step(0, 2, 1.5, logits, targets, 0, report_now=True)
  avg_loss, top1 = r.summary()
  assert abs(avg_loss - 0.75) < 1e-6
  assert abs(top1 - 100.0) < 1e-6


def test_report_now_false_on_boundary_still_logs():
  """When report_now=False but log_every boundary is hit, should still log."""
  r = _make_reporter(log_every=2)
  logits, targets = _dummy_batch()
  # (batch_idx=1, (1+1)%2==0) triggers log even without report_now.
  r.step(0, 4, 1.0, logits, targets, 0)
  assert r._window_samples == 4  # no log yet
  r.step(1, 4, 1.0, logits, targets, 1)
  assert r._window_samples == 0  # log happened, window reset
