"""Lightweight gradient inspector for PyTorch models.

Usage::

    monitor = GradMonitor(model, log_every=50)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        monitor.step(global_step)
        optimizer.step()
        optimizer.zero_grad()

The monitor walks ``model.named_parameters()`` on each call to ``step()``
and computes per-parameter gradient statistics (norm, mean, max, sparsity)
and parameter norms.  Reports are logged via ``logging.info()`` every
``log_every`` steps.

When ``detect_nan=True`` the monitor checks every step (not just log
steps) for NaN / Inf in gradient tensors and logs a critical warning
immediately if any are found.

When ``norm_history`` > 0, per-layer gradient and parameter norms are
accumulated in a rolling window and a trend summary is appended to each
report.

All norms reported are **RMS** (root mean square): L2 norm divided by
``sqrt(numel)``.  This makes them independent of tensor shape and
directly comparable across parameters of different sizes.
"""

import collections
import logging

from scdiag.table_utils import format_table


def _find_common_root(strings):
  """Return the longest dotted prefix shared by all *strings*.

  Splits each string on ``'.'`` and compares component-by-component,
  so the result is always a clean dotted path (never cuts mid-component).

  Returns ``""`` when no common root exists.
  """
  if not strings:
    return ""
  split = [s.split(".") for s in strings]
  common = []
  for components in zip(*split):
    if len(set(components)) == 1:
      common.append(components[0])
    else:
      break
  return ".".join(common)


def _strip_common_prefix(rows):
  """Strip the common dotted prefix from parameter names.

  Returns ``(prefix, stripped_rows)``.  Each row is a
  ``(name, stats_dict, status)`` tuple.  When no meaningful prefix
  exists the original rows are returned unchanged.
  """
  prefix = _find_common_root([r[0] for r in rows])
  if not prefix:
    return "", rows
  pfx = prefix + "."
  return pfx, [(name[len(pfx):], s, st) for name, s, st in rows]


def create_grad_monitor(args, model):
  """Build a :class:`GradMonitor` from the shared ``--grad_monitor`` CLI args.

  Both training scripts used identical inline blocks for this; centralised
  here so they cannot drift apart.

  Args:
      args: Parsed CLI args (``grad_monitor``, ``norm_history``,
          ``trend_top_n``).
      model: The model whose gradients will be monitored.

  Returns:
      A configured ``GradMonitor``, or ``None`` when gradient monitoring
      is disabled (``--grad_monitor < 0``).
  """
  monitor = None
  if args.grad_monitor >= 0:
    monitor = GradMonitor(
        model,
        log_every=args.grad_monitor,
        norm_history=args.norm_history,
        trend_top_n=args.trend_top_n,
    )
    logging.info(f"Gradient monitoring enabled (every {args.grad_monitor} steps).")
    if args.norm_history > 0:
      logging.info(f"  Norm trend history: last {args.norm_history} snapshots")
  return monitor


class GradMonitor:
  """Architecture-agnostic gradient inspector.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters will be inspected.
    log_every : int
        Compute and log gradient stats every N training steps.
    detect_nan : bool
        If *True*, every call to ``step()`` checks all gradient tensors
        for NaN / Inf (slight overhead).
    stall_window : int
        Number of consecutive log-steps a layer's gradient norm must
        stay below ``norm_floor`` before it is flagged as stalled.
    norm_floor : float
        Gradient RMS threshold below which a parameter is considered
        inactive.  Default ``1e-7``.
    norm_ceiling : float
        Gradient RMS threshold above which a parameter is considered
        exploding.  Default ``1.0``.
    top_k : int
        Number of healthy (non-anomalous) params to show in the report,
        sorted by gradient norm descending.
    imbalance_factor : float
        A parameter whose gradient norm exceeds this factor times the
        median gradient norm is flagged as IMBALANCED.  Defaults to 100.
    gpr_ceiling : float
        Gradient-to-parameter ratio threshold above which a parameter
        is flagged as GPR (G/P Ratio).  A value > 1.0 means the
        gradient update is larger than the weight itself.  Defaults to
        1.0.
    norm_history : int
        Maximum number of recent (step, grad_norm, param_norm) snapshots
        to keep per parameter for trend analysis.  0 disables (default).
    trend_top_n : int
        Number of top-changing parameters to show in the trend table,
        sorted by max absolute change percentage.  0 shows all.
    """

  def __init__(
      self,
      model,
      log_every=50,
      detect_nan=False,
      stall_window=50,
      norm_floor=1e-7,
      norm_ceiling=1.0,
      top_k=5,
      imbalance_factor=100.0,
      gpr_ceiling=1.0,
      norm_history=0,
      trend_top_n=10,
  ):
    self._model = model
    self._log_every = log_every
    self._detect_nan = detect_nan
    self._stall_window = stall_window
    self._norm_floor = norm_floor
    self._norm_ceiling = norm_ceiling
    self._top_k = top_k
    self._imbalance_factor = imbalance_factor
    self._gpr_ceiling = gpr_ceiling
    self._norm_history = norm_history
    self._trend_top_n = trend_top_n

    self._consecutive_low = {}
    self._norm_buf = collections.defaultdict(
        lambda: collections.deque(maxlen=norm_history or None))

  def step(self, global_step):
    """Advance the monitor by one training step.

        If ``detect_nan`` is enabled, checks for NaN/Inf every call.
        Computes stats and logs a report every ``log_every`` steps.
        """
    if self._detect_nan:
      self._check_nan()

    if self._log_every > 0 and (global_step % self._log_every == 0):
      self._snapshot(global_step)

  def _check_nan(self):
    for name, p in self._model.named_parameters():
      if p.grad is None:
        continue
      if p.grad.isnan().any() or p.grad.isinf().any():
        logging.critical("GradMonitor: NaN/Inf detected in gradient of '%s'", name)

  def _snapshot(self, global_step):
    stats = {}
    anomalies = []

    for name, p in self._model.named_parameters():
      if p.requires_grad:
        entry = {}
        numel = p.numel()
        # 0-element tensors (norm 0.0) divide by the neutral 1.0.
        rms_scale = numel**0.5 if numel else 1.0

        param_rms = float(p.data.norm()) / rms_scale
        entry["param_norm"] = param_rms

        if p.grad is None:
          entry["grad_norm"] = 0.0
          entry["grad_mean"] = 0.0
          entry["grad_max"] = 0.0
          entry["grad_sparsity"] = 1.0
          entry["grad_param_ratio"] = 0.0
        else:
          g = p.grad.data
          grad_rms = float(g.norm()) / rms_scale
          entry["grad_norm"] = grad_rms
          entry["grad_mean"] = float(g.float().mean())
          entry["grad_max"] = float(g.abs().max())
          entry["grad_sparsity"] = float((g.abs() < 1e-7).float().mean())
          entry["grad_param_ratio"] = grad_rms / (param_rms + 1e-12)

        is_low = entry["grad_norm"] < self._norm_floor
        prev = self._consecutive_low.get(name, 0)
        self._consecutive_low[name] = prev + 1 if is_low else 0
        if self._consecutive_low[name] >= self._stall_window:
          anomalies.append(f"STALLED({name}, {self._consecutive_low[name]} steps)")

        if entry["grad_norm"] > self._norm_ceiling:
          anomalies.append(f"EXPLODING({name}, norm={entry['grad_norm']:.2e})")

        if entry["grad_param_ratio"] > self._gpr_ceiling:
          anomalies.append(f"GPR({name}, ratio={entry['grad_param_ratio']:.2e})")

        stats[name] = entry

    norms = [v["grad_norm"] for v in stats.values() if v["grad_norm"] > 0]
    if norms:
      norms_sorted = sorted(norms)
      median = norms_sorted[len(norms_sorted) // 2]
      for name, v in stats.items():
        if (v["grad_norm"] > 0 and v["grad_norm"] > median * self._imbalance_factor):
          anomalies.append(f"IMBALANCED({name})")

    # Accumulate history
    if self._norm_history > 0:
      for name, s in stats.items():
        self._norm_buf[name].append((global_step, s["grad_norm"], s["param_norm"]))

    self._log(global_step, stats, anomalies)

  def _log(self, global_step, stats, anomalies):
    norms = [v["grad_norm"] for v in stats.values()]
    ratios = [
        v["grad_param_ratio"] for v in stats.values() if v["grad_param_ratio"] > 0
    ]

    lines = [(f"[Step {global_step}] Gradient Report:"
              f" {len(stats)} params"
              f" | grad_rms:"
              f" mean={sum(norms)/len(norms) if norms else 0:.2e}"
              f" max={max(norms) if norms else 0:.2e}"
              f" min={min(norms) if norms else 0:.2e}"
              f" | grad/param:"
              f" mean={sum(ratios)/len(ratios) if ratios else 0:.2e}")]

    name_status = {}
    for a in anomalies:
      for tag, short in [
          ("STALLED(", "STL"),
          ("EXPLODING(", "OVF"),
          ("IMBALANCED(", "IMB"),
          ("GPR(", "GPR"),
      ]:
        if tag in a:
          start = a.index(tag) + len(tag)
          end = a.index(")", start)
          pname = a[start:end].split(",")[0]
          prev = name_status.get(pname)
          name_status[pname] = f"{prev}|{short}" if prev else short

    SEVERITY = {"STL": 0, "OVF": 1, "IMB": 2, "GPR": 3}

    def sort_key(kv):
      name, s = kv
      sev = SEVERITY.get(name_status.get(name, ""), 99)
      return (sev, -s["grad_norm"])

    ranked = sorted(stats.items(), key=sort_key)
    rows = []
    for name, s in ranked:
      if len(rows) >= self._top_k + len(name_status):
        break
      if s["grad_norm"] > 0 or name in name_status:
        rows.append((name, s, name_status.get(name, "OK")))

    if rows:
      prefix, rows = _strip_common_prefix(rows)
      if prefix:
        lines.append(f"  Prefix (stripped): {prefix}")

      headers = ["Name", "g_rms", "p_rms", "g/p", "g_max", "sparse", "status"]
      aligns = ["left", "right", "right", "right", "right", "right", "left"]
      table_rows = []
      for name, s, status in rows:
        table_rows.append([
            name,
            f"{s['grad_norm']:>10.2e}",
            f"{s['param_norm']:>10.2e}",
            f"{s['grad_param_ratio']:>10.2e}",
            f"{s['grad_max']:>10.2e}",
            f"{s['grad_sparsity']:>6.1%}",
            f"{status:<12}",
        ])
      lines.extend(format_table(headers, table_rows, align=aligns, prefix="  "))

    # Norm trend summary
    if self._norm_history > 0 and self._norm_buf:
      lines.extend(self._log_trends(global_step))

    logging.info("\n".join(lines))

  def _log_trends(self, global_step):
    """Append a norm-trend table to the current report.

    For each parameter with at least two history entries, reports
    the direction (growing/shrinking/stable), percentage change
    over the window, and min/max values.
    """
    rows = []
    for name, buf in sorted(self._norm_buf.items()):
      if len(buf) < 2:
        continue
      first_step, first_gn, first_pn = buf[0]
      last_step, last_gn, last_pn = buf[-1]
      span = last_step - first_step
      if span <= 0:
        continue

      # Grad norm trend
      if first_gn > 0:
        gn_change = (last_gn - first_gn) / first_gn * 100
        gn_vals = [e[1] for e in buf]
        gn_direction = ("UP"
                        if gn_change > 10 else "DOWN" if gn_change < -10 else "---")
      else:
        gn_change = 0.0
        gn_vals = [e[1] for e in buf]
        gn_direction = "---"

      # Param norm trend
      if first_pn > 0:
        pn_change = (last_pn - first_pn) / first_pn * 100
        pn_vals = [e[2] for e in buf]
        pn_direction = ("UP"
                        if pn_change > 10 else "DOWN" if pn_change < -10 else "---")
      else:
        pn_change = 0.0
        pn_vals = [e[2] for e in buf]
        pn_direction = "---"

      rows.append((
          name,
          gn_direction,
          gn_change,
          min(gn_vals),
          max(gn_vals),
          pn_direction,
          pn_change,
          min(pn_vals),
          max(pn_vals),
      ))

    if not rows:
      return []

    # Sort by max absolute change percentage descending, take top_n.
    rows.sort(key=lambda r: max(abs(r[2]), abs(r[6])), reverse=True)
    total = len(rows)
    if self._trend_top_n > 0 and len(rows) > self._trend_top_n:
      rows = rows[:self._trend_top_n]

    # Strip common prefix for readability.
    names = [r[0] for r in rows]
    prefix = _find_common_root(names)
    if prefix:
      prefix += "."
      stripped = [(name.removeprefix(prefix), name) for name in names]
      # Detect duplicates — fall back to full name for any collision.
      counts = collections.Counter(s for s, _ in stripped)
      rows = [(full if counts[short] > 1 else short,) + r[1:]
              for (short, full), r in zip(stripped, rows)]

    lines = []
    lines.append(
        f"  Norm Trends (last {len(self._norm_buf[next(iter(self._norm_buf))])}"
        f" snapshots, top {len(rows)}/{total} params by change):")
    if prefix:
      lines.append(f"  Prefix (stripped): {prefix}")

    headers = [
        "Param",
        "g_dir",
        "g_chg%",
        "g_min",
        "g_max",
        "p_dir",
        "p_chg%",
        "p_min",
        "p_max",
    ]
    aligns = [
        "left",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
    ]
    table_rows = []
    for r in rows:
      (name, gn_dir, gn_chg, gn_lo, gn_hi, pn_dir, pn_chg, pn_lo, pn_hi) = r
      table_rows.append([
          name,
          f"{gn_dir:>5}",
          f"{gn_chg:>+8.1f}%",
          f"{gn_lo:>9.2e}",
          f"{gn_hi:>9.2e}",
          f"{pn_dir:>5}",
          f"{pn_chg:>+8.1f}%",
          f"{pn_lo:>9.2e}",
          f"{pn_hi:>9.2e}",
      ])
    lines.extend(format_table(headers, table_rows, align=aligns, prefix="  "))
    return lines
