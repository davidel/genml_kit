"""CLI utilities for parsing repeatable ``--KEY=VALUE`` arguments.

Used by ``train.py`` and ``infer.py`` to allow users to
override model, processor, optimizer, and scheduler parameters from the
command line without changing the argument parser for every possible knob.

Example::

    genml_kit-train --model convvit --model_arg depth=6 num_heads=8
    genml_kit-train --model convvit --opt_arg betas=0.9,0.999
    genml_kit-train --model convvit --model_arg depths=[3,6,12,6]
"""

import argparse

from genml_kit.utils.logging import fatal


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
    """
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
  """Split an inline ``"name:key=value,..."`` string into ``(name, kwargs)``.

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
    return name, {}
  kwargs = {}
  for item in _split_list_items(body.strip()):
    if "=" not in item:
      fatal(
          f"Expected key=value in {kind} spec {spec!r}, got: {item!r}",
          ValueError,
      )
    key, val = item.split("=", 1)
    kwargs[key.strip()] = parse_value(val.strip())
  return name, kwargs


class KVPairAction(argparse.Action):
  """Argparse action that collects ``--KEY=VALUE`` pairs into a dict.

    Multiple occurrences are accumulated into a single dict::

        --model_arg depth=6 --model_arg num_heads=8
        → {"depth": 6, "num_heads": 8}

    Values are auto-converted via :func:`parse_value`.  If a key is given
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
        parser.error(f"Expected KEY=VALUE pair, got: {token!r} "
                     f"(for {option_string})")
      key, val = token.split("=", 1)
      d[key] = parse_value(val)
    setattr(namespace, self.dest, d)
