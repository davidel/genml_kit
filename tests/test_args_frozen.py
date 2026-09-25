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

from genml_kit.training.train import build_parser, register_all_owners
from genml_kit.utils.args import (add_checkpoint_args, add_logging_args,
                                  add_optimization_args, add_source_checkpoint_args,
                                  add_training_state_args)


def _declared_cli_dests():
  """Collect every dest the unified training parser declares."""
  parser = build_parser()
  register_all_owners(parser)
  add_checkpoint_args(parser, checkpoint_default="genml_kit", resume_default=True)
  add_optimization_args(parser)
  add_training_state_args(parser,
                          state_save="opt,sched,amp",
                          state_load="opt,sched,amp")
  add_logging_args(parser)
  add_source_checkpoint_args(parser)
  # mirror parse_args()'s inline optimizer group
  parser.add_argument("--lr", type=float, default=3e-5)
  parser.add_argument("--weight_decay", type=float, default=0.0)
  parser.add_argument("--llrd_decay", type=float, default=None)
  parser.add_argument("--lr_group", action="store_true")
  parser.add_argument("--optimizer", type=str, default="AdamW")
  parser.add_argument("--opt_arg", action="store_true")
  parser.add_argument("--scheduler", type=str, default=None)
  parser.add_argument("--sched_arg", action="store_true")
  parser.add_argument("--save_every", type=int, default=500)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument("--log_dir", type=str, default=None)
  parser.add_argument("--hf_token", type=str, default=None)
  parser.add_argument("--in_ch", type=int, default=1)
  parser.add_argument("--vis_every", type=int, default=0)
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
