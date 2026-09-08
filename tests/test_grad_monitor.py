"""Tests for genml_kit.grad_monitor.GradMonitor."""

import logging
import warnings

import torch
import torch.nn as nn

from genml_kit.training.grad_monitor import GradMonitor


class TinyModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(10, 20)
    self.fc2 = nn.Linear(20, 5)

  def forward(self, x):
    return self.fc2(torch.relu(self.fc1(x)))


def _train_step(model, x, target, optimizer):
  optimizer.zero_grad()
  loss = nn.functional.mse_loss(model(x), target)
  loss.backward()
  optimizer.step()
  return loss


def test_step_does_not_log_off_step(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=50)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(1)

  assert "Gradient Report" not in caplog.text


def test_step_logs_on_log_step(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=10)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(10)

  assert "[Step 10]" in caplog.text
  assert "Gradient Report" in caplog.text


def test_report_contains_param_count(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "4 params" in caplog.text


def test_grad_norms_positive_no_anomaly(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "STL" not in caplog.text
  assert "OVF" not in caplog.text


def test_nan_detection(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=100, detect_nan=True)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.fill_(float("nan"))
      break

  with caplog.at_level(logging.CRITICAL):
    mon.step(1)

  assert "NaN/Inf" in caplog.text


def test_no_stall_on_first_step(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, stall_window=5)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "STL" not in caplog.text


def test_stall_detection_after_window(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, stall_window=3)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)
  mon.step(0)

  with caplog.at_level(logging.INFO):
    for step in range(1, 10):
      opt.zero_grad()
      loss = nn.functional.mse_loss(model(x), target)
      loss.backward()
      model.fc1.weight.grad.zero_()
      opt.step()
      mon.step(step)

  assert "STL" in caplog.text
  assert "fc1.weight" in caplog.text


def test_explosion_detection(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_ceiling=1.0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  for p in model.parameters():
    if p.grad is not None:
      p.grad.data.fill_(100.0)
      break

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "OVF" in caplog.text


def test_report_without_grad(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "grad_rms: mean=0.00e+00" in caplog.text


def test_imbalance_detection(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  for p in model.parameters():
    if p.grad is not None and p is not model.fc2.weight:
      p.grad.data.mul_(0.0001)
  model.fc2.weight.grad.data.fill_(10.0)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "IMB" in caplog.text
  assert "fc2.weight" in caplog.text


def test_zero_grad_handling(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1)

  opt = torch.optim.SGD(model.parameters(), lr=0.01)
  opt.zero_grad()

  for p in model.parameters():
    p.grad = torch.zeros_like(p.data)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "nan" not in caplog.text.lower()


def test_log_every_zero_disables_logging(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=0)

  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  with caplog.at_level(logging.INFO):
    for i in range(100):
      _train_step(model, x, target, opt)
      mon.step(i)

  assert "Gradient Report" not in caplog.text


def test_log_every_skips_steps(caplog):
  """With log_every=3, only step 0, 3, 6, ... produce logs."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=3)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(0)
  assert "Gradient Report" in caplog.text

  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(1)
  assert "Gradient Report" not in caplog.text

  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(2)
  assert "Gradient Report" not in caplog.text

  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(3)
  r_first = caplog.text
  assert "Gradient Report" in r_first

  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(4)
  assert "Gradient Report" not in caplog.text

  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(5)
  assert "Gradient Report" not in caplog.text

  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(6)
  r_second = caplog.text
  assert "Gradient Report" in r_second
  assert r_first != r_second


def test_gpr_detection(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, gpr_ceiling=0.5)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  # Make the gradient large relative to the parameter norm.
  for p in model.parameters():
    if p.grad is not None and p is not model.fc2.weight:
      p.grad.data.fill_(0.0)
  model.fc2.weight.grad.data.fill_(100.0)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "GPR" in caplog.text
  assert "fc2.weight" in caplog.text


def test_gpr_no_flag_when_below_ceiling(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, gpr_ceiling=1000.0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  _train_step(model, x, target, opt)

  with caplog.at_level(logging.INFO):
    mon.step(0)

  assert "GPR" not in caplog.text


# norm_history tests


def test_no_trend_when_norm_history_disabled(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_history=0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(5):
    _train_step(model, x, target, opt)
    mon.step(i)

  assert "Norm Trends" not in caplog.text


def test_trend_appears_after_two_snapshots(caplog):
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_history=10)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  # First snapshot — only one entry, no trend yet.
  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(0)
  assert "Norm Trends" not in caplog.text

  # Second snapshot — now we have two entries.
  _train_step(model, x, target, opt)
  caplog.clear()
  with caplog.at_level(logging.INFO):
    mon.step(1)
  assert "Norm Trends" in caplog.text
  assert "g_dir" in caplog.text
  assert "p_dir" in caplog.text


def test_norm_history_rolling_window(caplog):
  """Deque maxlen limits history to norm_history entries."""
  model = TinyModel()
  history_size = 5
  mon = GradMonitor(model, log_every=1, norm_history=history_size)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(20):
    _train_step(model, x, target, opt)
    mon.step(i)

  # Check that no deque exceeds the maxlen.
  for buf in mon._norm_buf.values():
    assert len(buf) <= history_size


def test_trend_direction_growing(caplog):
  """When param norm grows, trend reports UP."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_history=10)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  # Train normally for a few steps to build history.
  for i in range(5):
    _train_step(model, x, target, opt)
    mon.step(i)

  # Artificially inflate gradients to push param norms up.
  with caplog.at_level(logging.INFO):
    for i in range(5, 15):
      _train_step(model, x, target, opt)
      # Scale up weights directly to make norm growth obvious.
      with torch.no_grad():
        for p in model.parameters():
          p.data.mul_(1.5)
      mon.step(i)

  # At least one param should show UP in the trend.
  assert "UP" in caplog.text


def test_trend_top_n_limits_output(caplog):
  """trend_top_n caps the number of rows in the trend table."""
  model = TinyModel()
  # 4 params total; limit to 2.
  mon = GradMonitor(model, log_every=1, norm_history=10, trend_top_n=2)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(10):
    _train_step(model, x, target, opt)
    mon.step(i)

  with caplog.at_level(logging.INFO):
    mon.step(10)

  trend_headers = [r.message for r in caplog.records if "params by change" in r.message]
  assert len(trend_headers) == 1
  assert "top 2/4" in trend_headers[0]


def test_trend_top_n_zero_shows_all(caplog):
  """trend_top_n=0 shows all params."""
  model = TinyModel()
  mon = GradMonitor(model, log_every=1, norm_history=10, trend_top_n=0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(10):
    _train_step(model, x, target, opt)
    mon.step(i)

  with caplog.at_level(logging.INFO):
    mon.step(10)

  trend_headers = [r.message for r in caplog.records if "params by change" in r.message]
  assert len(trend_headers) == 1
  assert "top 4/4" in trend_headers[0]


def test_trend_strips_common_prefix(caplog):
  """Trend table shows 'Prefix (stripped):' when params share a prefix."""

  class NestedModel(nn.Module):

    def __init__(self):
      super().__init__()
      self.block = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

    def forward(self, x):
      return self.block(x)

  model = NestedModel()
  mon = GradMonitor(model, log_every=1, norm_history=10, trend_top_n=0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(10):
    _train_step(model, x, target, opt)
    mon.step(i)

  with caplog.at_level(logging.INFO):
    mon.step(10)

  prefix_lines = [r.message for r in caplog.records if "Prefix (stripped)" in r.message]
  # All params share "block." prefix; trend table should strip it.
  assert len(prefix_lines) == 1
  assert "block." in prefix_lines[0]


def test_trend_disambiguates_duplicate_suffixes(caplog):
  """When stripping a prefix produces identical names, the full name is used."""

  class MultiHead(nn.Module):

    def __init__(self):
      super().__init__()
      self.head_a = nn.Linear(10, 5)
      self.head_b = nn.Linear(10, 5)

    def forward(self, x):
      return self.head_a(x) + self.head_b(x)

  model = MultiHead()
  mon = GradMonitor(model, log_every=1, norm_history=10, trend_top_n=0)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)
  opt = torch.optim.SGD(model.parameters(), lr=0.01)

  for i in range(10):
    _train_step(model, x, target, opt)
    mon.step(i)

  with caplog.at_level(logging.INFO):
    mon.step(10)

  trend_lines = [r.message for r in caplog.records if "Norm Trends" in r.message]
  assert len(trend_lines) == 1

  # Both .weight and .bias share the suffix "head_a.weight" / "head_b.weight"
  # after stripping "multi_head." — they must not collide.
  param_lines = [
      r.message
      for r in caplog.records
      if "head_a" in r.message or "head_b" in r.message
  ]
  # Should see both head_a and head_b in the trend table.
  assert any("head_a" in msg for msg in param_lines)
  assert any("head_b" in msg for msg in param_lines)


def test_zero_element_parameter_reports_zero_stats():
  """Regression: zero-element params (e.g. a headless model) must not
  crash RMS computation; they report zero norm statistics."""
  model = nn.Sequential()
  model.fc = nn.Linear(10, 5)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # nn.Linear warns on out=0
    model.unused_head = nn.Linear(5, 0)  # zero-element weight and bias

  mon = GradMonitor(model, log_every=1)
  x = torch.randn(4, 10)
  target = torch.randn(4, 5)

  # Step through the real sub-network only (unused_head outputs nothing).
  loss = nn.functional.mse_loss(model.fc(x), target)
  loss.backward()  # would raise ZeroDivisionError inside monitor below

  with caplog_disabled():
    mon.step(1)  # must not raise


class caplog_disabled:
  """Context manager suppressing log noise for one-off assertions."""

  def __enter__(self):
    logging.disable(logging.CRITICAL)
    return self

  def __exit__(self, *exc):
    logging.disable(logging.NOTSET)
