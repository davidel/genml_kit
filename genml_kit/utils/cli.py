"""CLI utilities for parsing repeatable ``--KEY=VALUE`` arguments.

Used by ``train.py``, ``pretrain.py``, and ``infer.py`` to allow users to
override model, processor, optimizer, and scheduler parameters from the
command line without changing the argument parser for every possible knob.

Example::

    genml_kit-train --model convvit --model_arg depth=6 num_heads=8
    genml_kit-train --model convvit --opt_arg betas=0.9,0.999
    genml_kit-train --model convvit --model_arg depths=[3,6,12,6]
"""

import argparse


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
  # Bool check
  if s.lower() in ("true", "false"):
    return s.lower() == "true"

  # None check
  if s.lower() == "none":
    return None

  # Int check
  try:
    return int(s)
  except ValueError:
    pass

  # Float check
  try:
    return float(s)
  except ValueError:
    pass

  # List check — explicit brackets or parens
  stripped = s.strip()
  if (stripped.startswith("[") and stripped.endswith("]")) or \
     (stripped.startswith("(") and stripped.endswith(")")):
    inner = stripped[1:-1].strip()
    if not inner:
      return []
    items = [parse_value(item.strip()) for item in _split_list_items(inner)]
    return items

  # Fallback: raw string
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


class KVPairAction(argparse.Action):
  """Argparse action that collects ``--KEY=VALUE`` pairs into a dict.

    Multiple occurrences are accumulated into a single dict::

        --model_arg depth=6 --model_arg num_heads=8
        → {"depth": 6, "num_heads": 8}

    Values are auto-converted via :func:`parse_value`.  If a key is given
    more than once, the last value wins.
    """

  def __call__(self, parser, namespace, values, option_string=None):
    d = getattr(namespace, self.dest) or {}
    if not isinstance(d, dict):
      d = {}
    for token in values:
      if "=" not in token:
        parser.error(f"Expected KEY=VALUE pair, got: {token!r} "
                     f"(for {option_string})")
      key, val = token.split("=", 1)
      d[key] = parse_value(val)
    setattr(namespace, self.dest, d)
