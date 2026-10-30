"""Hyperparameter search: declarative studies over the unified trainer.

This subsystem is deliberately decoupled from the training code.  A *study*
(YAML/JSON or CLI flags) names a :class:`~genml_kit.search.space.SearchSpace`,
a sampler, a pruner, an objective and a budget.  Each trial runs as an
isolated ``genml-kit-train`` subprocess; the search reads the objective from
the trial's existing ``metrics.jsonl`` sink and prunes hopeless trials via a
clean signal-driven shutdown.

The console entry point (:mod:`genml_kit.cli.tune`, ``genml-kit-tune``) lives
in :mod:`genml_kit.cli` alongside the other binaries; this package holds only
the reusable library.
"""

from genml_kit.search.pruners import PRUNERS, build_pruner
from genml_kit.search.runner import StudyRunner, run_study
from genml_kit.search.samplers import SEARCHERS, build_sampler
from genml_kit.search.space import Dimension, SearchSpace
from genml_kit.search.store import StudyStore
from genml_kit.search.study import StudySpec
from genml_kit.search.trial import Trial

__all__ = [
    "Dimension",
    "PRUNERS",
    "SEARCHERS",
    "SearchSpace",
    "StudyRunner",
    "StudySpec",
    "StudyStore",
    "Trial",
    "build_pruner",
    "build_sampler",
    "run_study",
]
