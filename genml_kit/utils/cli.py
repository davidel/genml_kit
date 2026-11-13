"""CLI utilities for parsing repeatable ``--KEY=VALUE`` arguments.

Used by ``train.py`` and ``infer.py`` to allow users to
override model, processor, optimizer, and scheduler parameters from the
command line without changing the argument parser for every possible knob.

Example::

    genml_kit-train --model convvit --model_arg depth=6 num_heads=8
    genml_kit-train --model convvit --opt_arg betas=[0.9,0.999]
    genml_kit-train --model convvit --model_arg depths=[3,6,12,6]
"""

import argparse
import collections

from genml_kit.utils.logging import fatal

# What ``parse_inline_spec`` returns: the name and the parsed kwargs.
InlineSpec = collections.namedtuple("InlineSpec", ["name", "kwargs"])


def parse_value(s):
  """Auto-convert a CLI string to a Python value.

    Conversion rules (applied in order):

    1. ``"true"`` / ``"false"`` (case-insensitive) → ``bool``
    2. ``"none"`` (case-insensitive) → ``None``
    3. Bare integers (e.g. ``"3"``, ``"-7"``) → ``int``
    4. Bare floats (e.g. ``"0.1"``, ``"1e-3"``) → ``float``
    5. List literal ``"[1, 2, 3]"`` or ``"(1, 2, 3)"`` → ``list`` with each
       element recursively converted via ``parse_value``
    6. Everything else is returned as a ``str``.

    Quoting overrides the rules above: a token wrapped in *matching* single
    or double quotes is returned verbatim (quotes stripped) as a ``str``,
    with no coercion -- so ``'"10"'`` stays ``"10"`` and ``'"true"'`` stays
    ``"true"``.  This is the escape hatch for numeric-looking strings.
    """
  if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
    return s[1:-1]

  if s.lower() in ("true", "false"):
    return s.lower() == "true"

  if s.lower() == "none":
    return None

  try:
    return int(s)
  except ValueError:
    pass

  try:
    return float(s)
  except ValueError:
    pass

  stripped = s.strip()
  if (stripped.startswith("[") and stripped.endswith("]")) or \
     (stripped.startswith("(") and stripped.endswith(")")):
    inner = stripped[1:-1].strip()
    if not inner:
      return []
    items = [parse_value(item.strip()) for item in _split_list_items(inner)]
    return items

  return s


def _split_list_items(s):
  """Split a list-body string by commas, respecting nested brackets."""
  items = []
  depth = 0
  current = []
  for ch in s:
    if ch in ("[", "("):
      depth += 1
      current.append(ch)
    elif ch in ("]", ")"):
      depth -= 1
      current.append(ch)
    elif ch == "," and depth == 0:
      items.append("".join(current).strip())
      current = []
    else:
      current.append(ch)
  if current:
    items.append("".join(current).strip())
  return items


def parse_inline_spec(spec, kind="spec"):
  """Split an inline ``"name:key=value,..."`` string into an :class:`InlineSpec`.

  Shared by the classifier and decoder registries.  The part before the
  first ``:`` is the name (a registered name or a ``.py`` path); the
  remainder is a comma-separated ``key=value`` list converted with
  :func:`parse_value` (so ints, floats, booleans, and bracketed lists
  work).  Bracket-aware splitting keeps values such as ``(0, 1)`` intact.

  A spec without a colon is just a name with no kwargs.

  Parameters
  ----------
  spec : str
      The inline specification.
  kind : str
      Human-readable kind, used only in the error message.

  Returns
  -------
  (str, dict)
      The name and its constructor kwargs.
  """
  spec = str(spec).strip()
  name, sep, body = spec.partition(":")
  if not sep or not body.strip():
    return InlineSpec(name=name, kwargs={})
  kwargs = {}
  for item in _split_list_items(body.strip()):
    if "=" not in item:
      fatal(
          f"Expected key=value in {kind} spec {spec!r}, got: {item!r}",
          ValueError,
      )
    key, val = item.split("=", 1)
    kwargs[key.strip()] = parse_value(val.strip())
  return InlineSpec(name=name, kwargs=kwargs)


class KVPairAction(argparse.Action):
  """Argparse action that collects ``--KEY=VALUE`` pairs into a dict.

    Multiple occurrences are accumulated into a single dict::

        --model_arg depth=6 --model_arg num_heads=8
        → {"depth": 6, "num_heads": 8}

    A bare token (no ``=``) is shorthand for a boolean true, so these are
    equivalent::

        --model_arg use_cls
        --model_arg use_cls=true

    Values are auto-converted via :func:`parse_value` (which already maps
    ``true``/``True``/``false``/``False`` to ``bool``).  If a key is given
    more than once, the last value wins.
    """

  def __init__(self, *args, default=None, **kwargs):
    # Normalize the default so consumers can always ``**unpack`` the
    # attribute as a dict (e.g. ``**args.model_arg``).  Parser call sites
    # pass ``default=None``; treat any falsy default as empty.
    super().__init__(*args, default=dict(default) if default else {}, **kwargs)

  def __call__(self, parser, namespace, values, option_string=None):
    # argparse sets the per-action default (normalized in __init__) as the
    # SAME object on every namespace that omits the flag, so never mutate
    # it in place: copy first, then rebind the namespace to the fresh dict
    # (also keeps a reused namespace from accumulating stale keys).
    d = dict(getattr(namespace, self.dest))
    for token in values:
      if "=" not in token:
        # Bare token is shorthand for ``key=True`` (flag-style usage).
        d[token] = True
        continue
      key, val = token.split("=", 1)
      d[key] = parse_value(val)
    setattr(namespace, self.dest, d)


def apply_arg_defaults(parser, owners):
  """Merge owner-declared defaults into *parser*; explicit CLI still wins.

  Owners (pipelines and methods) may declare per-flag default values via
  ``arg_defaults()``.  This applies them *after* every ``add_argument``
  and *before* ``parse_args``, which is the only window where the selected
  owner is known and argparse still treats the values as defaults -- so an
  explicit ``--flag`` on the command line always takes precedence.

  Scalar dests are set outright.  Dict dests (``KVPairAction`` flags such
  as ``dist_arg``) are merged per key with the owner value as the base, so
  a user ``--dist_arg static_graph`` keeps a method-declared
  ``find_unused_parameters``.  This works because ``KVPairAction.__call__``
  copies the namespace value (seeded from the action default rewritten
  here) before applying its own tokens.

  A declared dest that names no flag is a hard error (a typo would
  otherwise silently do nothing forever).  Validation happens against the
  parser's own dests *before* anything is written: checking the parsed
  namespace afterwards cannot work, because ``set_defaults`` materializes
  the very key that has to be reported as unknown.

  Args:
      parser: The ``argparse.ArgumentParser`` under construction.
      owners: Iterable of owner classes exposing ``arg_defaults()``.

  Returns:
      set: The dest names applied.
  """
  declared = {}
  for owner in owners:
    for dest, value in (owner.arg_defaults() or {}).items():
      # Later owners win on a scalar; dicts merge with the later owner as
      # the base, matching the CLI-over-default precedence below.
      prior = declared.get(dest)
      if isinstance(prior, dict) and isinstance(value, dict):
        value = {**value, **prior}
      declared[dest] = value

  known = {action.dest for action in parser._actions}
  unknown = sorted(set(declared) - known)
  if unknown:
    fatal(f"arg_defaults() names unknown CLI flag(s): {', '.join(unknown)}")

  for dest, value in declared.items():
    current = parser.get_default(dest)
    if isinstance(current, dict) and isinstance(value, dict):
      value = {**value, **current}
    parser.set_defaults(**{dest: value})
  return set(declared)
