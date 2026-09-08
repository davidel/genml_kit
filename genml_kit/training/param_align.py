"""Automatic state-dict key alignment between two models.

Given an *old* state-dict (e.g. pretrained checkpoint) and a *new* state-dict
(e.g. target model to initialise), produce a ``{new_key: old_key}`` mapping.

Matching is driven by:
  1. **Shape** — old and new parameter shapes must be identical.
  2. **Name tokens** — keys are split on ``"."`` into tokens, reversed (leaf
     first), and compared using a weighted edit distance that penalises
     leaf-side mismatches more heavily than root-side mismatches.

The best same-shape candidate (lowest distance) is selected for each new key.
"""

from collections import defaultdict, namedtuple

AlignConfig = namedtuple("AlignConfig", ["max_distance"], defaults=[0.25])


def _tokenize(key):
  """Split *key* on ``"."`` and reverse the tokens (leaf first)."""
  return list(reversed(key.split(".")))


def weighted_token_distance(tokens1, tokens2):
  """Weighted edit distance between two pre-reversed token lists.

  Uses geometric weights ``1/2^k`` where *k* is the 0-indexed position.
  A mismatch at position 0 (leaf) costs 1.0, position 1 costs 0.5,
  position 2 costs 0.25, etc.  The sum of all weights beyond position
  *k* is strictly less than the weight at *k*, so leaf-side mismatches
  always dominate.
  """
  if len(tokens1) < len(tokens2):
    tokens1, tokens2 = tokens2, tokens1

  len1, len2 = len(tokens1), len(tokens2)
  prev_row = [0.0] * (len2 + 1)
  curr_row = [0.0] * (len2 + 1)

  for i in range(1, len1 + 1):
    curr_row[0] = prev_row[0] + 1.0 / 2**(i - 1)
    t1 = tokens1[i - 1]

    for j in range(1, len2 + 1):
      if t1 == tokens2[j - 1]:
        curr_row[j] = prev_row[j - 1]
      else:
        weight = 1.0 / 2**(max(i, j) - 1)
        curr_row[j] = min(
            prev_row[j] + weight,
            curr_row[j - 1] + weight,
            prev_row[j - 1] + weight,
        )
    prev_row = curr_row[:]

  return prev_row[len2]


def _build_shape_dict(sd):
  """Build a shape → ``[(reversed_tokens, key)]`` mapping from *sd*."""
  shape_dict = defaultdict(list)
  for key, tensor in sd.items():
    shape = tuple(tensor.shape)
    tokens = _tokenize(key)
    shape_dict[shape].append((tokens, key))
  return dict(shape_dict)


def _find_best_match(shape_dict, shape, new_tokens, max_distance=float('inf')):
  """Find the best matching old key for *new_tokens* within the same shape.

  Returns ``(old_key, distance)`` or ``(None, float('inf'))``.
  """
  candidates = shape_dict.get(shape, [])
  if not candidates:
    return None, float('inf')

  best_distance = float('inf')
  best_key = None

  for old_tokens, old_key in candidates:
    dist = weighted_token_distance(new_tokens, old_tokens)
    if dist < best_distance:
      best_distance = dist
      best_key = old_key

  if best_distance <= max_distance:
    return best_key, best_distance
  return None, float('inf')


AlignReport = namedtuple("AlignReport",
                         ["mapping", "unmatched_new", "unused_old", "divergent", "ok"])


def report_to_str(report):
  lines = []

  if report.ok:
    n = len(report.mapping)
    lines.append(f"param_align: all {n} new keys matched, {n} old keys used.")
  else:
    if report.unmatched_new:
      lines.append(f"param_align: {len(report.unmatched_new)} new keys UNMATCHED"
                   " (no same-shape old key):")
      for k in report.unmatched_new:
        lines.append(f"  {k} → ?")

    if report.unused_old:
      lines.append(f"param_align: {len(report.unused_old)} old keys"
                   " UNUSED (will not be copied):")
      for k in report.unused_old:
        lines.append(f"  UNUSED ← {k}")

    if report.divergent:
      lines.append(f"param_align: {len(report.divergent)} old keys claimed"
                   " by multiple new keys:")
      for old_key, new_keys in report.divergent:
        lines.append(f"  1→M ← {old_key} → {new_keys}")

  return "\n".join(lines)


def align_state_dicts(old_sd, new_sd, config=None):
  """Align *new_sd* parameters to *old_sd* parameters.

  Each new key is matched to the closest old key (lowest
  :func:`weighted_token_distance`) that shares the exact same tensor
  shape.  The shape match is mandatory.
  """
  config = config or AlignConfig()
  shape_dict = _build_shape_dict(old_sd)
  all_old_keys = set(old_sd.keys())
  used_old = set()

  mapping = {}
  unmatched_new = []
  old_to_new = defaultdict(list)

  for key, tensor in new_sd.items():
    shape = tuple(tensor.shape)
    new_tokens = _tokenize(key)

    old_key, _dist = _find_best_match(shape_dict,
                                      shape,
                                      new_tokens,
                                      max_distance=config.max_distance)

    if old_key is not None:
      mapping[key] = old_key
      used_old.add(old_key)
      old_to_new[old_key].append(key)
    else:
      unmatched_new.append(key)

  unused_old = sorted(all_old_keys - used_old)

  divergent = []
  for old_key, new_keys in old_to_new.items():
    if len(new_keys) > 1:
      divergent.append((old_key, new_keys))

  ok = not unmatched_new and not unused_old and not divergent

  return AlignReport(mapping, unmatched_new, unused_old, divergent, ok)
