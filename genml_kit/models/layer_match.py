"""Layer-matching helpers for feature-based knowledge distillation.

FitNets and attention transfer distil *intermediate* activations rather than
final logits: they capture a hidden activation from the teacher (the "hint")
and one from the student (the "guided" layer) and compare the two.  Which
teacher layer faces which student layer is a *policy* — it must not be
hard-coded inside a training loop, or the method stops reading as
orchestration and the pairing logic cannot be unit-tested in isolation.

Three strategies are provided:

- ``"by_index"``  -- positional pairing; requires equal depth.
- ``"uniform"``   -- evenly spaced subsampling when the two depths differ.
- ``"by_name"``   -- pairing on a regex match of the module names.

A pair is a :data:`LayerPair` namedtuple holding the two ``nn.Module``
objects, so a caller can register forward hooks on them directly (the same
pattern already used by ``genml_kit.training.model_utils``).  The matched and
skipped layer *names* are emitted through ``logging`` rather than carried in
the returned struct.
"""

import collections
import logging
import re

LayerPair = collections.namedtuple("LayerPair", ["teacher", "student"])


def iter_named_layers(model, pattern=None):
  """Enumerate candidate layers of *model* as ``(name, module)`` pairs.

  Walks ``model.named_modules()`` in registration order (numeric for
  ``nn.ModuleList`` / ``nn.Sequential``, so ``"blocks.10"`` does not sort
  before ``"blocks.2"``).  The root container (empty name) is always skipped:
  it is the whole model, never a distillation target.

  Args:
      model: The ``nn.Module`` to inspect.
      pattern: Optional Python regex.  When given, only modules whose dotted
          name matches it (via ``re.search``, the same semantics as
          ``genml_kit.io.checkpointing.rename_keys``) are returned.

  Returns:
      A list of ``(name, module)`` tuples in registration order.  Modules
      reachable through more than one name (tied weights) are kept once, under
      their first name.

  Raises:
      ValueError: If the pattern is given but matches no module, or if the
          selection is empty — an empty candidate list would silently pair
          nothing and train a distillation loss on zero layers.
  """
  regex = re.compile(pattern) if pattern is not None else None
  seen = set()
  layers = []
  for name, module in model.named_modules():
    if not name:
      continue
    if regex is not None and not regex.search(name):
      continue
    if id(module) in seen:
      continue
    seen.add(id(module))
    layers.append((name, module))
  if not layers:
    if regex is not None:
      raise ValueError(f"No module name matched pattern {pattern!r}.")
    raise ValueError("No layers found in the given model.")
  return layers


def _even_indices(total, count):
  """Return *count* evenly spaced indices spanning ``[0, total - 1]``.

  The first and last index are always included, so the deepest (most
  semantic) layer is never dropped, and the sequence is strictly increasing.
  A single index selects the middle layer.

  Args:
      total: Number of available positions (``>= 1``).
      count: Number of indices to select (``1 <= count <= total``).

  Returns:
      A list of *count* integers.
  """
  if count == 1:
    return [total // 2]
  return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _name_key_map(layers, regex):
  """Map each layer's match key to its ``(name, module)`` pair.

  The key is the substring the regex matched (``re.search(...).group(0)``) so
  that differing path prefixes still pair — e.g. ``model.blocks.3`` on the
  teacher and ``blocks.3`` on the student share the key ``blocks.3``.  Without
  a regex the whole dotted name is the key.  The first layer wins when two
  share a key.
  """
  key_map = {}
  for name, module in layers:
    if regex is None:
      key = name
    else:
      match = regex.search(name)
      if match is None:
        continue
      key = match.group(0)
    key_map.setdefault(key, (name, module))
  return key_map


def match_layers(teacher_layers, student_layers, strategy="uniform", pattern=None):
  """Pair teacher and student layers according to *strategy*.

  Args:
      teacher_layers: List of ``(name, module)`` tuples, as returned by
          :func:`iter_named_layers`, in teacher order.
      student_layers: Same, for the student.
      strategy: One of ``"by_index"``, ``"uniform"``, ``"by_name"``.
      pattern: Regex used to extract the pairing key for ``"by_name"`` (see
          :func:`_name_key_map`).  Ignored by the other strategies.

  Returns:
      A list of :data:`LayerPair` (``.teacher`` / ``.student`` hold the two
      ``nn.Module`` objects).  Matched and skipped layers are also reported
      through ``logging`` (INFO for matches, WARNING for skips).

  Raises:
      ValueError: On an unknown *strategy*; on unequal depth for
          ``"by_index"``; or when the pairing yields zero pairs (a mis-typed
          ``"by_name"`` pattern, typically).
  """
  if not teacher_layers or not student_layers:
    raise ValueError("Both teacher_layers and student_layers must be non-empty.")

  if strategy == "by_index":
    if len(teacher_layers) != len(student_layers):
      raise ValueError(
          f"strategy='by_index' needs equal depth; got {len(teacher_layers)} "
          f"teacher and {len(student_layers)} student layers.  Use 'uniform' "
          "for different depths.")
    pairs = [
        LayerPair(t_mod, s_mod)
        for (_, t_mod), (_, s_mod) in zip(teacher_layers, student_layers)
    ]
    _log_matches([(t[0], s[0]) for t, s in zip(teacher_layers, student_layers)])
    return pairs

  if strategy == "uniform":
    count = min(len(teacher_layers), len(student_layers))
    t_idx = _even_indices(len(teacher_layers), count)
    s_idx = _even_indices(len(student_layers), count)
    pairs = [
        LayerPair(teacher_layers[i][1], student_layers[j][1])
        for i, j in zip(t_idx, s_idx)
    ]
    _log_matches([
        (teacher_layers[i][0], student_layers[j][0]) for i, j in zip(t_idx, s_idx)
    ])
    return pairs

  if strategy == "by_name":
    regex = re.compile(pattern) if pattern is not None else None
    teacher_map = _name_key_map(teacher_layers, regex)
    student_map = _name_key_map(student_layers, regex)
    pairs = []
    matched = []
    skipped = []
    for key, (t_name, t_mod) in teacher_map.items():
      if key in student_map:
        s_name, s_mod = student_map[key]
        pairs.append(LayerPair(t_mod, s_mod))
        matched.append((t_name, s_name))
      else:
        skipped.append(t_name)
    for key, (s_name, _) in student_map.items():
      if key not in teacher_map:
        skipped.append(s_name)
    _log_matches(matched)
    if skipped:
      logging.warning("  Skipped %d unmatched layer(s): %s", len(skipped),
                      ", ".join(skipped))
    if not pairs:
      raise ValueError(f"strategy='by_name' matched no teacher/student layers"
                       f"{f' for pattern {pattern!r}' if pattern is not None else ''}.")
    return pairs

  raise ValueError(f"Unknown strategy {strategy!r}; expected one of "
                   "'by_index', 'uniform', 'by_name'.")


def _log_matches(matched):
  """Log the matched ``(teacher_name, student_name)`` pairs at INFO level."""
  if not matched:
    return
  lines = "\n".join(f"    {t} <-> {s}" for t, s in matched)
  logging.info("  Matched %d layer(s):\n%s", len(matched), lines)
