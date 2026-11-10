"""Regression guard: no ad-hoc ``args`` mutation (plans/FIX_FRAP.md).

The parsed ``argparse.Namespace`` must be a pure, read-only view of the
CLI.  Every attribute written via ``args.<attr> = ...`` in ``genml_kit/``
must be a *declared* CLI flag -- otherwise the code is smuggling runtime
values through the shared namespace (e.g. the old
``args.rollout_len = args.ppo_rollout_len`` alias, the classification
``args.train_transforms = ...`` object smuggling, or the simmim
``args.mask_ratio`` round-trip).

This is intentionally a *static* guard (no runtime cost): it scans the
source tree for the mutation pattern and cross-checks against the set of
dests the unified parser declares.
"""

import ast
import pathlib

from genml_kit.cli.train import build_parser


def _declared_cli_dests():
  """Collect every dest the unified training parser declares."""
  parser = build_parser()
  return {action.dest for action in parser._actions if action.dest is not None}


def _args_mutations(paths):
  """Yield ``(file, lineno, attr)`` for every ``args.<attr> = ...`` write."""
  for path in paths:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
      if isinstance(node, ast.Assign):
        for target in node.targets:
          if (isinstance(target, ast.Attribute) and
              isinstance(target.value, ast.Name) and target.value.id == "args"):
            yield str(path), node.lineno, target.attr


def test_no_undeclared_args_mutation():
  root = pathlib.Path(__file__).resolve().parents[1]
  pkg = root / "genml_kit"
  declared = _declared_cli_dests()

  offenders = []
  for fpath, lineno, attr in _args_mutations(pkg.rglob("*.py")):
    if attr not in declared:
      offenders.append(f"{fpath}:{lineno}: args.{attr} = ... (not a CLI flag)")

  assert not offenders, ("Undeclared `args` mutations found (plans/FIX_FRAP.md):\n" +
                         "\n".join(offenders))
