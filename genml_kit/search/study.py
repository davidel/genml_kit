"""Study specification: the declarative "what to search" document.

A study is *data, not code*: a YAML (or JSON, or dict) document that names the
space, the sampler/pruner, the objective, the budget and the resources.  The
same document produces the same search regardless of who runs it, which is
what makes a sweep reproducible and comparable across machines.

Minimal example::

    objective: epoch/macro_f1
    direction: max
    sampler: tpe
    pruner: asha
    max_trials: 50
    max_epochs_per_trial: 20
    min_resource: 2
    reduction_factor: 3
    space:
      opt.lr: log(1e-4:1e-1)
      batch_size: [32, 64, 128]
      dropout: 0.0:0.5
    base_args: ["--pipeline", "images", "--method", "classification"]
"""

import json
import os
from collections import OrderedDict
from types import SimpleNamespace

from genml_kit.search.space import SearchSpace
from genml_kit.utils.logging import fatal


def load_document(path):
  """Load a study document from a ``.yaml``/``.yml``/``.json`` file."""
  if not os.path.isfile(path):
    fatal(f"Study file not found: {path!r}", FileNotFoundError)
  with open(path, encoding="utf-8") as handle:
    text = handle.read()
  if path.endswith(".json"):
    return json.loads(text)
  try:
    import yaml
  except ImportError:
    fatal(
        "Reading a YAML study requires PyYAML; use a .json study or "
        "install PyYAML.", ImportError)
  return yaml.safe_load(text)


class StudySpec:
  """Parsed, validated study document."""

  def __init__(self, data, override=None):
    if not isinstance(data, dict):
      fatal("Study document must be a mapping at the top level.", ValueError)
    self.raw = data
    override = override or {}

    self.space = self._build_space(data)
    self.objective = override.get("objective") or data.get("objective")
    if not self.objective:
      fatal("Study must define an 'objective' (a metric tag).", ValueError)
    self.direction = (override.get("direction") or data.get("direction") or
                      "max").lower()
    if self.direction not in ("max", "min"):
      fatal(f"Study 'direction' must be 'max' or 'min', got "
            f"{self.direction!r}.", ValueError)

    self.sampler = override.get("sampler") or data.get("sampler") or "tpe"
    self.pruner = override.get("pruner") or data.get("pruner") or "none"
    self.sampler_args = OrderedDict(data.get("sampler_args") or {})
    self.pruner_args = OrderedDict(data.get("pruner_args") or {})

    self.max_trials = int(override.get("max_trials") or data.get("max_trials") or 0)
    self.max_epochs_per_trial = int(
        data.get("max_epochs_per_trial") or override.get("epochs") or 0)
    if self.max_trials <= 0 and self.max_epochs_per_trial <= 0:
      self.max_trials = len(self.space.grid()) or 1

    self.seed = int(data.get("seed", 42))
    self.retries = int(data.get("retries", 0))
    self.base_args = list(data.get("base_args") or [])
    self.resources = dict(data.get("resources") or {})
    self.study_dir = override.get("study_dir") or data.get("study_dir")

  @classmethod
  def from_file(cls, path, override=None):
    """Load and validate a study from a YAML/JSON file."""
    return cls(load_document(path), override=override)

  @classmethod
  def from_space_pairs(cls, pairs, **fields):
    """Build an ad-hoc study from ``--space`` tokens plus explicit fields."""
    data = dict(fields)
    data["space"] = OrderedDict()
    for token in pairs:
      if "=" not in token:
        fatal(f"Expected NAME=SPEC search-space token, got: {token!r}.", ValueError)
      name, spec = token.split("=", 1)
      data["space"][name.strip()] = spec.strip()
    return cls(data)

  # -- helpers -------------------------------------------------------------

  @staticmethod
  def _build_space(data):
    space = data.get("space")
    if not space:
      fatal("Study must define a non-empty 'space' mapping.", ValueError)
    pairs = []
    items = space.items() if isinstance(space, dict) else []
    if not items:
      fatal("Study 'space' must be a mapping of name -> spec.", ValueError)
    for name, spec in items:
      pairs.append(f"{name}={spec}")
    return SearchSpace.from_pairs(pairs)

  def to_dict(self):
    """Serialise the effective spec (for the study ledger header)."""
    return {
        "objective": self.objective,
        "direction": self.direction,
        "sampler": self.sampler,
        "pruner": self.pruner,
        "sampler_args": dict(self.sampler_args),
        "pruner_args": dict(self.pruner_args),
        "max_trials": self.max_trials,
        "max_epochs_per_trial": self.max_epochs_per_trial,
        "seed": self.seed,
        "retries": self.retries,
        "base_args": list(self.base_args),
        "resources": dict(self.resources),
        "space": {
            d.name: d.to_json() for d in self.space
        },
    }

  def as_namespace(self):
    """Expose the spec as an attribute namespace (mirrors parsed CLI args)."""
    return SimpleNamespace(**self.to_dict())
