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
``hyperband``
    Hyperband bracketing: multiple ASHA brackets with progressively smaller
    rung grids, so aggressive early halving and wide late coverage are
    both available.  Each trial is assigned to a bracket round-robin by
    trial id hash; the bracket with the most rungs (``s_max``) uses rungs
    ``min_resource * rf**k``, while shallower brackets start at
    ``s_max - s`` and keep the same geometric spacing.

Interface
---------
Constructed with ``(objective_is_min, **kwargs)`` and exposing::

    should_stop(trial_id, series) -> bool

where *series* is the trial's chronological ``[(epoch, value)]`` objective
history.  ``series`` values are in the study's native direction; the pruner
converts to a maximise-internal view using ``objective_is_min``.
"""

import threading
from types import SimpleNamespace

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
    """Record and return the trial's value at the newest rung it reached.

    A trivial grouping of the rung index and the value at it (``None``
    when the trial has not advanced to a fresh rung yet).
    """
    rungs = self._rungs_up_to(len(series))
    if not rungs:
      return SimpleNamespace(rung=None, value=None)
    rung = rungs[-1]
    seen = self._recorded.setdefault(trial_id, set())
    if rung in seen:
      return SimpleNamespace(rung=rung, value=None)
    value = self._rung_value(series, rung)
    if value is None:
      return SimpleNamespace(rung=rung, value=None)
    seen.add(rung)
    self._rung_values.setdefault(rung, []).append(value)
    return SimpleNamespace(rung=rung, value=value)

  def should_stop(self, trial_id, series):
    """Return whether the trial identified by *trial_id* should be pruned."""
    if len(series) < self.min_resource:
      return False
    with self._lock:
      newest = self._value_at_new_rung(trial_id, series)
      if newest.rung is None or newest.value is None:
        return False
      return self._decide(newest.rung, newest.value)

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


@PRUNERS.register
class HyperbandPruner(Pruner):
  """Hyperband: several ASHA brackets over nested rung grids.

  A single ASHA pruner fixes one trade-off: aggressive rungs (many halvings,
  ``s = s_max``) prune fast but may kill late bloomers, while shallow
  brackets (few halvings) keep everything but prune rarely.  Hyperband runs
  ``s_max + 1`` brackets with ``s = 0..s_max`` and lets trials spread across
  them, so the sweep gets both behaviours.

  Trials are assigned to brackets deterministically (``trial_id % n_brackets``)
  so the same trial always lands in the same bracket across pool workers.
  Within a bracket, pruning is exactly ASHA's rule over the bracket's rung
  grid; bracket ``s`` starts at rung ``min_resource * rf**s`` (bracket 0 is
  the deepest: first rung ``min_resource``, the most halvings available).
  """

  NAME = "hyperband"

  def __init__(self,
               objective_is_min=False,
               min_resource=1,
               reduction_factor=3,
               min_trials_per_rung=None,
               s_max=None,
               **kwargs):
    super().__init__(objective_is_min, min_resource, reduction_factor, **kwargs)
    self.min_trials_per_rung = (min_trials_per_rung or self.reduction_factor)
    # Number of halvings in the deepest bracket.  ``None`` means "derive
    # from the observed epoch budget" (set on first observation).
    self._s_max = int(s_max) if s_max is not None else None
    self._asha = {}  # bracket index -> AshaPruner over that bracket's grid

  def _n_brackets(self):
    """``s_max + 1`` brackets (the classic Hyperband set)."""
    return (self._s_max or 1) + 1

  def _bracket_of(self, trial_id):
    """Deterministic bracket assignment: stable across workers."""
    return int(trial_id) % self._n_brackets()

  def _ensure_bracket(self, n_epochs):
    """Lazily size ``s_max`` and build one ASHA pruner per bracket."""
    if self._s_max is None:
      rungs = self._rungs_up_to(n_epochs)
      self._s_max = max(1, len(rungs) - 1)
    for s in range(self._n_brackets()):
      if s in self._asha:
        continue
      # Bracket s starts at rung min_resource * rf**s.  Bracket 0 therefore
      # has the smallest first rung and the most halvings available before
      # the epoch budget; higher s are shallower and prune later.
      first_rung = self.min_resource * (self.reduction_factor**s)
      self._asha[s] = AshaPruner(objective_is_min=self.objective_is_min,
                                 min_resource=first_rung,
                                 reduction_factor=self.reduction_factor,
                                 min_trials_per_rung=self.min_trials_per_rung)

  def should_stop(self, trial_id, series):
    if len(series) < self.min_resource:
      return False
    with self._lock:
      self._ensure_bracket(len(series))
      bracket = self._bracket_of(trial_id)
      return self._asha[bracket].should_stop(trial_id, series)


def build_pruner(name, objective_is_min=False, **kwargs):
  """Instantiate a pruner by name (registry lookup + construction)."""
  cls = PRUNERS.get(name)
  return cls(objective_is_min=objective_is_min, **kwargs)
