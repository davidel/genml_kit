"""Search-space model: typed dimensions, sampling and argv encoding.

A :class:`SearchSpace` is an ordered set of named :class:`Dimension` objects.
Each dimension knows how to (a) sample a value given an RNG and (b) enumerate
a finite grid of candidate values.  The space also knows how to translate a
resolved ``{name: value}`` config into the ``argv`` tail of a real
``genml-kit-train`` invocation.

The grammar is intentionally close to the existing ``--KEY=VALUE`` override
convention (:mod:`genml_kit.utils.cli`), so a name like ``opt.lr`` becomes
``--opt_arg lr=...`` and ``model.embed_dim`` becomes ``--model_arg
embed_dim=...``; a bare name (``batch_size``) becomes ``--batch_size ...``.

Example specs (one per ``--space`` token)::

    lr=1e-4,1e-3,1e-2          # categorical choice
    batch_size=32,64,128       # categorical choice (ints)
    dropout=0.0:0.5            # uniform float range
    embed_dim=64:1024          # integer range
    opt.lr=log(1e-4:1e-1)      # log-uniform float range
    depth=2:12:step=2          # integer range with stride
"""

import math
from collections import OrderedDict

import numpy as np

from genml_kit.utils.logging import fatal

# Prefix -> train-CLI override flag.  ``None`` means "bare flag" (``--name``).
_PREFIX_FLAGS = OrderedDict([
    ("opt.", "--opt_arg"),
    ("sched.", "--sched_arg"),
    ("model.", "--model_arg"),
    ("proc.", "--proc_arg"),
])

_GRID_INTS = 12
_GRID_FLOATS = 8


def _is_int_str(token):
  """Whether *token* looks like an integer literal (``12``, ``-3``)."""
  try:
    int(token)
  except (TypeError, ValueError):
    return False
  return True


def _is_number_str(token):
  """Whether *token* parses as an int or float (``1e-3`` included)."""
  try:
    float(token)
  except (TypeError, ValueError):
    return False
  return True


def _as_number(token):
  """Return an ``int`` when *token* is integral, else a ``float``."""
  value = float(token)
  return int(value) if value.is_integer() and _is_int_str(token) else value


def format_value(value):
  """Render *value* the way :func:`genml_kit.utils.cli.parse_value` reads it.

  The output round-trips through the existing ``KEY=VALUE`` parser: bools are
  lower-case, floats keep full precision, lists are bracketed.
  """
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, float):
    return repr(value)
  if isinstance(value, (list, tuple)):
    return "[" + ",".join(format_value(v) for v in value) + "]"
  return str(value)


class Dimension:
  """A single tunable axis: a choice set or a numeric (int/float) range."""

  def __init__(self,
               name,
               kind,
               values=None,
               low=None,
               high=None,
               log=False,
               step=None):
    self.name = name
    self.kind = kind  # 'choice' | 'int' | 'float'
    self.values = list(values) if values is not None else None
    self.low = low
    self.high = high
    self.log = log
    self.step = step

  # -- construction --------------------------------------------------------

  @classmethod
  def from_spec(cls, name, spec):
    """Parse one ``name=<spec>`` body into a :class:`Dimension`."""
    spec = spec.strip()
    if not spec:
      fatal(f"Empty search-space spec for {name!r}.", ValueError)

    log = False
    step = None
    # Optional ``log(...)`` wrapper.
    if spec.startswith("log(") and spec.endswith(")"):
      log = True
      spec = spec[len("log("):-1].strip()
    # Optional type hint wrapper ``int(...)`` / ``float(...)``.
    hint = None
    for prefix in ("int(", "float("):
      if spec.startswith(prefix) and spec.endswith(")"):
        hint = prefix[:-1]
        spec = spec[len(prefix):-1].strip()

    # ``low:high[:step=N]`` numeric range takes precedence over commas.
    if ":" in spec and "=" not in spec:
      return cls._range(name, spec, hint, log)
    if ":" in spec and "=" in spec:
      # e.g. ``0.0:1.0:step=0.1`` -- split the trailing ``step=`` clause.
      parts = spec.split(":")
      step_clause = parts[-1]
      if step_clause.startswith("step="):
        step = _as_number(step_clause.split("=", 1)[1])
        spec = ":".join(parts[:-1])
      return cls._range(name, spec, hint, log, step)

    # Otherwise a categorical choice list.
    items = _split_choices(spec)
    values = [
        _as_number(item) if _is_number_str(item.strip()) else item.strip()
        for item in items
    ]
    return cls(name, "choice", values=values)

  @classmethod
  def _range(cls, name, spec, hint, log, step=None):
    low_str, high_str = spec.split(":", 1)
    low, high = _as_number(low_str), _as_number(high_str)
    if isinstance(low, int) and isinstance(high, int) and hint != "float":
      return cls(name, "int", low=low, high=high, log=log, step=step)
    return cls(name, "float", low=float(low), high=float(high), log=log, step=step)

  # -- sampling ------------------------------------------------------------

  def sample(self, rng):
    """Draw one value for this dimension from *rng* (a ``numpy`` Generator)."""
    if self.kind == "choice":
      return self.values[int(rng.integers(len(self.values)))]
    if self.log:
      lo, hi = math.log(self.low), math.log(self.high)
      raw = math.exp(lo + float(rng.random()) * (hi - lo))
    else:
      raw = self.low + float(rng.random()) * (self.high - self.low)
    return self._quantize(raw)

  def grid(self, max_points=None):
    """Enumerate this dimension's candidate values for grid search."""
    if self.kind == "choice":
      return list(self.values)
    n = max_points or (_GRID_INTS if self.kind == "int" else _GRID_FLOATS)
    if self.step:
      values = []
      current = self.low
      while current <= self.high + 1e-12:
        values.append(self._quantize(current))
        current += self.step
      return values
    if self.log:
      lo, hi = math.log(self.low), math.log(self.high)
      points = np.linspace(lo, hi, n)
      return [self._quantize(math.exp(p)) for p in points]
    points = np.linspace(self.low, self.high, n)
    return [self._quantize(float(p)) for p in points]

  def _quantize(self, value):
    """Snap *value* to the dimension's type / step."""
    if self.step:
      value = round((value - self.low) / self.step) * self.step + self.low
    value = min(max(value, self.low), self.high)
    if self.kind == "int":
      return int(round(value))
    return float(value)

  # -- unit-hypercube encoding (for model-based samplers) ------------------

  def to_unit(self, value):
    """Map *value* onto ``[0, 1]`` for the model-based samplers."""
    if self.kind == "choice":
      n = len(self.values)
      return 0.0 if n <= 1 else self.values.index(value) / (n - 1)
    if self.log:
      lo, hi = math.log(self.low), math.log(self.high)
      return (math.log(value) - lo) / (hi - lo)
    return (value - self.low) / (self.high - self.low)

  def from_unit(self, u):
    """Inverse of :meth:`to_unit`: decode ``u in [0, 1]`` back to a value."""
    u = min(max(float(u), 0.0), 1.0)
    if self.kind == "choice":
      n = len(self.values)
      return self.values[0 if n <= 1 else int(round(u * (n - 1)))]
    if self.log:
      lo, hi = math.log(self.low), math.log(self.high)
      return self._quantize(math.exp(lo + u * (hi - lo)))
    return self._quantize(self.low + u * (self.high - self.low))

  def to_json(self):
    """Serialise to a plain dict for the study ledger."""
    return {
        "name": self.name,
        "kind": self.kind,
        "values": self.values,
        "low": self.low,
        "high": self.high,
        "log": self.log,
        "step": self.step,
    }


def _split_choices(spec):
  """Split a choice body on commas, respecting ``[...]`` brackets."""
  if spec.startswith("[") and spec.endswith("]"):
    spec = spec[1:-1]
  items, depth, current = [], 0, []
  for ch in spec:
    if ch in "[(":
      depth += 1
      current.append(ch)
    elif ch in "])":
      depth -= 1
      current.append(ch)
    elif ch == "," and depth == 0:
      items.append("".join(current))
      current = []
    else:
      current.append(ch)
  items.append("".join(current))
  return items


class SearchSpace:
  """Ordered mapping of dimension name to :class:`Dimension`."""

  def __init__(self, dimensions):
    self.dimensions = OrderedDict((d.name, d) for d in dimensions)

  @classmethod
  def from_pairs(cls, pairs):
    """Build a space from ``name=spec`` strings (one per ``--space`` token)."""
    dims = []
    for token in pairs:
      if "=" not in token:
        fatal(f"Expected NAME=SPEC search-space token, got: {token!r}.", ValueError)
      name, spec = token.split("=", 1)
      dims.append(Dimension.from_spec(name.strip(), spec))
    return cls(dims)

  def __len__(self):
    return len(self.dimensions)

  def __iter__(self):
    return iter(self.dimensions.values())

  @property
  def names(self):
    return list(self.dimensions.keys())

  def sample(self, rng):
    """Return a config dict by sampling every dimension."""
    return {name: dim.sample(rng) for name, dim in self.dimensions.items()}

  def grid(self, max_points=None):
    """Return the cartesian product of all dimension grids."""
    per_dim = [dim.grid(max_points) for dim in self.dimensions.values()]
    combos = [{}]
    for name, values in zip(self.dimensions.keys(), per_dim):
      combos = [dict(base, **{name: v}) for base in combos for v in values]
    return combos

  def decode_to_argv(self, config):
    """Encode a resolved *config* into ``genml-kit-train`` argv tokens.

    Namespaced keys (``opt.``, ``model.``, ``sched.``, ``proc.``) are grouped
    under their override flag; other keys become bare ``--name value`` pairs.
    """
    overrides = OrderedDict((flag, []) for flag in _PREFIX_FLAGS.values())
    argv = []
    for name, value in config.items():
      flag = None
      key = name
      for prefix, override_flag in _PREFIX_FLAGS.items():
        if name.startswith(prefix):
          flag, key = override_flag, name[len(prefix):]
          break
      if flag is None:
        argv.extend([f"--{name}", format_value(value)])
      else:
        overrides[flag].append(f"{key}={format_value(value)}")
    for flag, tokens in overrides.items():
      if tokens:
        argv.append(flag)
        argv.extend(tokens)
    return argv
