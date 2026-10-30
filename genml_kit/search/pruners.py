"""Pluggable early-stopping pruners (``PRUNERS`` registry).

Pruning lets a multi-fidelity sweep kill hopeless trials before they consume
their full epoch budget.  A pruner is *stateful*: it observes every trial's
running objective series and returns a stop decision per epoch.  State is
guarded by a lock so the same pruner instance is safe under the multi-process
pool (each worker thread shares it).

Three pruners ship in-tree:

``none``
    Never stop; every trial runs to its budget.
``median``
    Stop a trial at a rung when its value falls below the running median of
    all values seen at that rung.
``asha``
    Asynchronous Successive Halving / Hyperband: keep the top ``1/rf`` of
    trials at each geometric rung, prune the rest.

Interface
---------
Constructed with ``(objective_is_min, **kwargs)`` and exposing::

    should_stop(trial_id, series) -> bool

where *series* is the trial's chronological ``[(epoch, value)]`` objective
history.  ``series`` values are in the study's native direction; the pruner
converts to a maximise-internal view using ``objective_is_min``.
"""

import threading

import numpy as np

from genml_kit.utils.registry import Registry

PRUNERS = Registry("pruner")


class Pruner:
  """Base class: stateful per-rung early-stopping policy."""

  NAME = None

  def __init__(self,
               objective_is_min=False,
               min_resource=1,
               reduction_factor=3,
               **kwargs):
    self.objective_is_min = bool(objective_is_min)
    self.sign = -1.0 if self.objective_is_min else 1.0
    self.min_resource = max(1, int(min_resource))
    self.reduction_factor = max(2, int(reduction_factor))
    self._lock = threading.Lock()
    self._rung_values = {}  # rung -> list of maximise-internal values
    self._recorded = {}  # trial_id -> set of rungs already recorded

  def _rungs_up_to(self, n_epochs):
    """Geometric rung resources ``min_resource * rf**k <= n_epochs``."""
    rungs = []
    resource = self.min_resource
    while resource <= n_epochs:
      rungs.append(resource)
      resource *= self.reduction_factor
    return rungs

  def _rung_value(self, series, rung):
    """Value to compare at *rung*: best over the first *rung* epochs."""
    values = [self.sign * value for epoch, value in series if epoch < rung]
    return max(values) if values else None

  def _value_at_new_rung(self, trial_id, series):
    """Record and return the trial's value at the newest rung it reached."""
    rungs = self._rungs_up_to(len(series))
    if not rungs:
      return None, None
    rung = rungs[-1]
    seen = self._recorded.setdefault(trial_id, set())
    if rung in seen:
      return rung, None
    value = self._rung_value(series, rung)
    if value is None:
      return rung, None
    seen.add(rung)
    self._rung_values.setdefault(rung, []).append(value)
    return rung, value

  def should_stop(self, trial_id, series):
    """Return whether the trial identified by *trial_id* should be pruned."""
    if len(series) < self.min_resource:
      return False
    with self._lock:
      rung, value = self._value_at_new_rung(trial_id, series)
      if rung is None or value is None:
        return False
      return self._decide(rung, value)

  def _decide(self, rung, value):
    """Policy hook: whether *value* at *rung* should be pruned."""
    raise NotImplementedError


@PRUNERS.register
class NoPruner(Pruner):
  """Never prune; the safe default (full budget for every trial)."""

  NAME = "none"

  def _decide(self, rung, value):
    return False


@PRUNERS.register
class MedianPruner(Pruner):
  """Prune when the trial's rung value is below the running median."""

  NAME = "median"

  def __init__(self,
               objective_is_min=False,
               min_resource=1,
               reduction_factor=3,
               n_warmup=1,
               **kwargs):
    super().__init__(objective_is_min, min_resource, reduction_factor, **kwargs)
    self.n_warmup = int(n_warmup)

  def _decide(self, rung, value):
    values = self._rung_values[rung]
    if len(values) <= self.n_warmup:
      return False
    return value < float(np.median(values))


@PRUNERS.register
class AshaPruner(Pruner):
  """Asynchronous Successive Halving (Hyperband) promotion policy."""

  NAME = "asha"

  def __init__(self,
               objective_is_min=False,
               min_resource=1,
               reduction_factor=3,
               min_trials_per_rung=None,
               **kwargs):
    super().__init__(objective_is_min, min_resource, reduction_factor, **kwargs)
    self.min_trials_per_rung = (min_trials_per_rung or self.reduction_factor)

  def _decide(self, rung, value):
    values = self._rung_values[rung]
    if len(values) < self.min_trials_per_rung:
      return False
    keep = max(1, len(values) // self.reduction_factor)
    cutoff = sorted(values, reverse=True)[keep - 1]
    return value < cutoff


def build_pruner(name, objective_is_min=False, **kwargs):
  """Instantiate a pruner by name (registry lookup + construction)."""
  cls = PRUNERS.get(name)
  return cls(objective_is_min=objective_is_min, **kwargs)
