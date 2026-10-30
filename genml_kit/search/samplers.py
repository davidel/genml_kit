"""Pluggable hyperparameter samplers (``SEARCHERS`` registry).

A sampler proposes fresh configurations given the trials observed so far.  It
is the analogue, for the search subsystem, of what a ``Method`` is for
training: a small object selected by name from a :class:`Registry`.

Four samplers ship in-tree:

``grid``
    Deterministic Cartesian product; hands out unseen grid points in order.
``random``
    Uniform sampling over the space (numpy only).
``tpe``
    Tree-structured Parzen Estimator -- Bayesian optimization with only
    numpy (the effective default for expensive training trials).
``gpei``
    Gaussian-Process Expected Improvement -- a scipy-backed Bayesian
    sampler, registered only when scipy is importable.

Interface
---------
Every sampler is constructed with ``(space, direction, seed, **kwargs)`` and
exposes::

    suggest(observed, n=1) -> list[dict]

where ``observed`` is a list of ``(config, score)`` pairs (scores in the
study's *native* direction) and each returned dict maps dimension name to a
value.  Samplers always *maximise* internally.
"""

import abc
import math

import numpy as np

from genml_kit.search.errors import SearchSpaceExhausted
from genml_kit.utils.logging import fatal
from genml_kit.utils.registry import Registry

SEARCHERS = Registry("sampler")

_DIRECTION = {"max": 1.0, "min": -1.0}


def _as_max(observed, direction):
  """Return ``[(config, y_maximise)]`` for a list of native-direction scores."""
  sign = _DIRECTION[direction]
  return [(config, sign * score) for config, score in observed]


class Sampler(abc.ABC):
  """Base class: a named proposal strategy over a :class:`SearchSpace`."""

  NAME = None

  def __init__(self, space, direction="max", seed=0, **kwargs):
    self.space = space
    self.direction = direction
    if direction not in _DIRECTION:
      fatal(f"Unknown direction {direction!r}; expected 'max' or 'min'.", ValueError)
    self.seed = int(seed)
    self.rng = np.random.default_rng(self.seed)
    self.options = kwargs

  @abc.abstractmethod
  def suggest(self, observed, n=1):
    """Return ``n`` fresh config dicts given observed ``(config, score)``."""
    raise NotImplementedError

  def _unit(self, config):
    """Encode a config as a unit-hypercube vector."""
    return np.array(
        [dim.to_unit(config[name]) for name, dim in self.space.dimensions.items()])

  def _decode(self, vector):
    """Decode a unit-hypercube vector back into a config dict."""
    return {
        name: dim.from_unit(u)
        for (name, dim), u in zip(self.space.dimensions.items(), vector)
    }


@SEARCHERS.register
class GridSampler(Sampler):
  """Deterministic Cartesian-product sampler (resume-safe, stateless)."""

  NAME = "grid"

  def __init__(self, space, direction="max", seed=0, max_points=None, **kwargs):
    super().__init__(space, direction, seed, **kwargs)
    self.max_points = max_points
    self._grid = space.grid(max_points=max_points)

  def suggest(self, observed, n=1):
    seen = {tuple(sorted(config.items())) for config, _ in observed}
    fresh = [c for c in self._grid if tuple(sorted(c.items())) not in seen]
    if not fresh:
      raise SearchSpaceExhausted("Grid space exhausted; no fresh configs left.")
    return fresh[:n]


@SEARCHERS.register
class RandomSampler(Sampler):
  """Independent uniform draws over every dimension."""

  NAME = "random"

  def suggest(self, observed, n=1):
    return [self.space.sample(self.rng) for _ in range(n)]


def _gaussian_logpdf(x, mean, bandwidth):
  """Per-dimension univariate Gaussian log-density, vectorised over *x*."""
  bandwidth = max(bandwidth, 1e-9)
  z = (x - mean) / bandwidth
  return -0.5 * z * z - math.log(bandwidth) - 0.5 * math.log(2 * math.pi)


@SEARCHERS.register
class TpeSampler(Sampler):
  """Tree-structured Parzen Estimator (numpy-only Bayesian optimizer).

  Observed scores are split at the ``gamma`` quantile into a "good" and a
  "bad" set; each set is modelled with a per-dimension Parzen (Gaussian KDE)
  estimator.  Candidates drawn from the good density are ranked by the
  likelihood ratio ``l(x)/g(x)`` and the best one is proposed -- the standard
  TPE acquisition.  Falls back to random until ``n_startup_trials`` exist.
  """

  NAME = "tpe"

  def __init__(self,
               space,
               direction="max",
               seed=0,
               gamma=0.25,
               n_startup_trials=5,
               n_candidates=None,
               **kwargs):
    super().__init__(space, direction, seed, **kwargs)
    self.gamma = min(max(float(gamma), 0.05), 0.9)
    self.n_startup_trials = int(n_startup_trials)
    self.n_candidates = n_candidates

  def suggest(self, observed, n=1):
    scored = _as_max(observed, self.direction)
    if len(scored) < self.n_startup_trials:
      return [self.space.sample(self.rng) for _ in range(n)]

    scored.sort(key=lambda pair: pair[1], reverse=True)
    n_good = max(1, int(round(self.gamma * len(scored))))
    good = [self._unit(config) for config, _ in scored[:n_good]]
    bad = [self._unit(config) for config, _ in scored[n_good:]]
    good = np.vstack(good)
    bad = np.vstack(bad) if bad else good

    n_candidates = self.n_candidates or (max(24, 64 * len(self.space)))
    proposals = []
    for _ in range(n):
      candidates = self._draw_good_candidates(good, n_candidates)
      ratio = self._log_density(candidates, good) - self._log_density(candidates, bad)
      proposals.append(self._decode(candidates[int(np.argmax(ratio))]))
    return proposals

  def _draw_good_candidates(self, good, n_candidates):
    """Sample from the good KDE (pick a kernel centre, add noise)."""
    idx = self.rng.integers(0, len(good), size=n_candidates)
    centres = good[idx]
    bandwidth = self._bandwidth(good)
    noise = self.rng.normal(0.0, bandwidth, size=centres.shape)
    return np.clip(centres + noise, 0.0, 1.0)

  def _bandwidth(self, points):
    """Scott's-rule bandwidth, clamped to a small positive floor."""
    if len(points) < 2:
      return np.full(points.shape[1], 0.2)
    spread = points.std(axis=0)
    return np.clip(spread * len(points)**(-1.0 / 5.0), 1e-3, None)

  def _log_density(self, candidates, points):
    """Log product-density of *candidates* under a per-dim Gaussian KDE."""
    bandwidth = self._bandwidth(points)
    total = np.zeros(len(candidates))
    for d in range(candidates.shape[1]):
      pdfs = np.exp(
          _gaussian_logpdf(candidates[:, d, None], points[None, :, d], bandwidth[d]))
      total += np.log(np.clip(pdfs.mean(axis=1), 1e-12, None))
    return total


class _GpEiSampler(Sampler):
  """Gaussian-Process Expected Improvement (requires scipy).

  A compact GP with an RBF kernel fitted by Cholesky-solved kernel ridge
  regression on the unit-hypercube-encoded trials; EI is maximised over a
  random candidate pool.  Chosen for scipy availability, not speed: it is the
  textbook BayesOpt baseline that ``tpe`` complements.
  """

  NAME = "gpei"

  def __init__(self,
               space,
               direction="max",
               seed=0,
               n_startup_trials=5,
               length_scale=None,
               noise=1e-6,
               n_candidates=None,
               **kwargs):
    super().__init__(space, direction, seed, **kwargs)
    self.n_startup_trials = int(n_startup_trials)
    self.length_scale = length_scale
    self.noise = float(noise)
    self.n_candidates = n_candidates

  def suggest(self, observed, n=1):
    scored = _as_max(observed, self.direction)
    if len(scored) < self.n_startup_trials:
      return [self.space.sample(self.rng) for _ in range(n)]

    x = np.vstack([self._unit(config) for config, _ in scored])
    y = np.array([score for _, score in scored])
    y = (y - y.mean()) / (y.std() + 1e-12)
    length_scale = self.length_scale or 0.3 * math.sqrt(len(self.space))
    kernel = self._kernel(x, x, length_scale)
    chol = self._cholesky(kernel + self.noise * np.eye(len(x)))
    alpha = self._solve(chol, y)
    best = y.max()

    n_candidates = self.n_candidates or (max(64, 256 * len(self.space)))
    proposals = []
    for _ in range(n):
      candidates = self.rng.random((n_candidates, len(self.space)))
      mean, var = self._posterior(candidates, x, kernel, chol, alpha, length_scale)
      ei = self._expected_improvement(mean, var, best)
      proposals.append(self._decode(candidates[int(np.argmax(ei))]))
    return proposals

  @staticmethod
  def _kernel(a, b, length_scale):
    sq = ((a[:, None, :] - b[None, :, :])**2).sum(axis=2)
    return np.exp(-0.5 * sq / length_scale**2)

  @staticmethod
  def _cholesky(matrix):
    from scipy import linalg
    return linalg.cho_factor(matrix)

  @staticmethod
  def _solve(chol, rhs):
    from scipy import linalg
    return linalg.cho_solve(chol, rhs)

  def _posterior(self, candidates, x, kernel, chol, alpha, length_scale):
    from scipy import linalg
    k_star = self._kernel(candidates, x, length_scale)
    mean = k_star @ alpha
    v = linalg.cho_solve(chol, k_star.T)
    var = np.clip(1.0 - (k_star * v.T).sum(axis=1), 1e-12, None)
    return mean, var

  @staticmethod
  def _expected_improvement(mean, var, best):
    from scipy.stats import norm
    sigma = np.sqrt(var)
    z = (mean - best) / sigma
    return (mean - best) * norm.cdf(z) + sigma * norm.pdf(z)


def _register_optional():
  """Register scipy-backed samplers only when scipy is importable."""
  try:
    import scipy  # noqa: F401
  except ImportError:  # pragma: no cover - exercised only without scipy
    return
  SEARCHERS.register(_GpEiSampler)


_register_optional()


def build_sampler(name, space, direction="max", seed=0, **kwargs):
  """Instantiate a sampler by name (registry lookup + construction)."""
  cls = SEARCHERS.get(name)
  return cls(space, direction=direction, seed=seed, **kwargs)
