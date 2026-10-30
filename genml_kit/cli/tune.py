"""``genml-kit-tune``: hyperparameter search over the unified trainer.

Two subcommands::

    genml-kit-tune run --study study.yaml -- "--pipeline images --method classification"
    genml-kit-tune run --method classification \
        --space opt.lr=log(1e-4:1e-1) --space batch_size=32,64,128 \
        --objective val/macro_f1 --sampler tpe --pruner asha --workers 2 \
        -- --pipeline images
    genml-kit-tune report runs/study-20250101-120000 --top 10

Fixed ``genml-kit-train`` arguments are supplied after a ``--`` separator (or
via ``--train``); they are applied verbatim to every trial subprocess.

The argparse surface and driver live here; the reusable search library
(:class:`~genml_kit.search.study.StudySpec`, the samplers/pruners, the
subprocess runner and the ledger) stays in :mod:`genml_kit.search`.
"""

import argparse
import datetime
import logging
import os
import shlex

from genml_kit.search import (PRUNERS, SEARCHERS, StudySpec, StudyStore, run_study)
from genml_kit.search.runner import DEFAULT_TRAIN_MODULE
from genml_kit.search.study import load_document
from genml_kit.utils.cli import KVPairAction
from genml_kit.utils.logging import fatal, setup_logging

DEFAULT_STUDY_ROOT = "runs"


def build_parser():
  """Build the ``genml-kit-tune`` parser (``run``/``report`` subcommands)."""
  parser = argparse.ArgumentParser(
      description="Hyperparameter search over genml-kit training.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  sub = parser.add_subparsers(dest="command", required=True)

  run = sub.add_parser(
      "run",
      help="Run a study (subprocess-isolated trials).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  run.add_argument("--study",
                   type=str,
                   default=None,
                   help="Path to a YAML/JSON study document.")
  run.add_argument(
      "--space",
      type=str,
      action="append",
      default=None,
      metavar="NAME=SPEC",
      help="Search dimension (repeatable). SPEC is a choice list "
      "(lr=1e-4,1e-3), a range (dropout=0.0:0.5), a log range "
      "(opt.lr=log(1e-4:1e-1)), or an int range (depth=2:12).",
  )
  run.add_argument("--objective",
                   type=str,
                   default=None,
                   help="Metric tag to optimize (e.g. val/macro_f1).")
  run.add_argument("--direction",
                   type=str,
                   default=None,
                   choices=["max", "min"],
                   help="Whether to maximize or minimize the objective.")
  run.add_argument("--sampler",
                   type=str,
                   default=None,
                   choices=SEARCHERS.list_names(),
                   help="Sampler (proposal strategy).")
  run.add_argument("--pruner",
                   type=str,
                   default=None,
                   choices=PRUNERS.list_names(),
                   help="Early-stopping pruner.")
  run.add_argument("--sampler_arg",
                   nargs="+",
                   action=KVPairAction,
                   default=None,
                   metavar="KEY=VALUE",
                   help="Extra sampler kwargs (repeatable).")
  run.add_argument("--pruner_arg",
                   nargs="+",
                   action=KVPairAction,
                   default=None,
                   metavar="KEY=VALUE",
                   help="Extra pruner kwargs (repeatable).")
  run.add_argument("--max_trials",
                   type=int,
                   default=None,
                   help="Maximum number of trials to run.")
  run.add_argument("--epochs",
                   type=int,
                   default=None,
                   help="Max epochs per trial (caps each run's budget).")
  run.add_argument("--min_resource",
                   type=int,
                   default=None,
                   help="Epochs before the pruner may stop a trial.")
  run.add_argument("--reduction_factor",
                   type=int,
                   default=None,
                   help="Successive-halving promotion factor.")
  run.add_argument("--seed",
                   type=int,
                   default=None,
                   help="Base seed for sampling and per-trial seeds.")
  run.add_argument("--retries",
                   type=int,
                   default=None,
                   help="Retries per trial on non-zero exit.")
  run.add_argument("--study_dir",
                   type=str,
                   default=None,
                   help="Study directory (resume target / output location).")
  run.add_argument("--resume",
                   action="store_true",
                   help="Resume an existing study directory.")
  run.add_argument("--workers",
                   type=int,
                   default=1,
                   help="Concurrent trials (1 = single process).")
  run.add_argument("--gpus",
                   type=str,
                   default=None,
                   help="Comma-separated GPU ids to pin trials to.")
  run.add_argument(
      "--train",
      type=str,
      default=None,
      help="Fixed genml-kit-train arguments (shell-quoted), applied to every "
      "trial. Example: \"--pipeline images --method classification\". "
      "Everything after a '--' separator is also forwarded.")
  run.add_argument("--python",
                   type=str,
                   default=None,
                   help="Python interpreter for trial subprocesses "
                   "(defaults to the current interpreter).")
  run.add_argument("--train-module",
                   type=str,
                   default=None,
                   help="Module launched for each trial "
                   "(defaults to genml_kit.cli.train).")
  run.add_argument("--log_level",
                   type=str,
                   default="INFO",
                   help="Tuner logging level (DEBUG, INFO, WARNING, ...).")
  run.add_argument("--log_targets",
                   type=str,
                   default="STDERR",
                   help="Tuner log destinations (STDERR or file paths).")
  run.add_argument(
      "train_extra",
      nargs="*",
      default=[],
      metavar="-- TRAIN_ARG",
      help="Fixed genml-kit-train arguments forwarded verbatim to every trial "
      "when placed after a '--' separator.")

  report = sub.add_parser(
      "report",
      help="Print the leaderboard of a finished study.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  report.add_argument("study_dir",
                      type=str,
                      help="Study directory containing study.jsonl.")
  report.add_argument("--top",
                      type=int,
                      default=20,
                      help="Show the top N trials (0 = all).")
  return parser


def parse_args(argv=None):
  """Parse ``genml-kit-tune`` arguments.

  Fixed ``genml-kit-train`` arguments placed after a literal ``--`` are
  captured by the ``train_extra`` positional (the highest-fidelity way to pass
  through flags that would otherwise collide with the tuner's own options).
  """
  return build_parser().parse_args(argv)


def _default_study_dir():
  stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  return os.path.join(DEFAULT_STUDY_ROOT, f"study-{stamp}")


def _train_argv(args):
  """Compose the fixed ``genml-kit-train`` argv from ``--train`` and ``--``."""
  argv = list(getattr(args, "train_extra", None) or [])
  if getattr(args, "train", None):
    argv = shlex.split(args.train) + argv
  return argv


def _parse_gpus(spec):
  if not spec:
    return None
  return [int(token) for token in spec.split(",") if token.strip()]


def _build_spec(args):
  """Assemble a :class:`StudySpec` from the study file plus CLI overrides."""
  data = load_document(args.study) if args.study else {}
  if args.space:
    space = data.setdefault("space", {})
    for token in args.space:
      if "=" not in token:
        fatal(f"Expected NAME=SPEC for --space, got: {token!r}.", ValueError)
      name, spec = token.split("=", 1)
      space[name.strip()] = spec.strip()
  if args.sampler_arg:
    data.setdefault("sampler_args", {}).update(args.sampler_arg)
  pruner_args = data.setdefault("pruner_args", {})
  if args.pruner_arg:
    pruner_args.update(args.pruner_arg)
  if args.min_resource is not None:
    pruner_args["min_resource"] = args.min_resource
  if args.reduction_factor is not None:
    pruner_args["reduction_factor"] = args.reduction_factor
  override = {
      "objective": args.objective,
      "direction": args.direction,
      "sampler": args.sampler,
      "pruner": args.pruner,
      "max_trials": args.max_trials,
      "epochs": args.epochs,
      "seed": args.seed,
      "study_dir": args.study_dir,
  }
  override = {key: value for key, value in override.items() if value is not None}
  if args.retries is not None:
    data["retries"] = args.retries
  if args.seed is not None:
    data["seed"] = args.seed
  return StudySpec(data, override=override)


def run_command(args):
  """Execute the ``run`` subcommand."""
  spec = _build_spec(args)
  study_dir = args.study_dir or _default_study_dir()
  base_argv = _train_argv(args)
  if args.resume:
    store = StudyStore.open(study_dir)
    logging.info("Resuming study at %s (%s trials).", study_dir, len(store.trials))
  else:
    store = StudyStore.create(study_dir, spec.to_dict())
    logging.info("Created study at %s.", study_dir)
  run_study(spec,
            store,
            base_argv,
            workers=args.workers,
            gpus=_parse_gpus(args.gpus),
            python=args.python,
            train_module=args.train_module or DEFAULT_TRAIN_MODULE)
  _print_leaderboard(store, spec.direction, top=None)
  logging.info("Study complete: %s trial(s) in %s.", len(store.trials), study_dir)


def _print_leaderboard(store, direction, top=None):
  """Log the ranked trials (best objective first)."""
  board = store.leaderboard(top=top, direction=direction)
  if not board:
    logging.info("No scored trials to report.")
    return
  logging.info("Leaderboard (%s):", direction)
  for rank, trial in enumerate(board, start=1):
    logging.info(
        "  #%d  trial %s  %s=%.6g  %s", rank, trial.trial_id,
        store.header.get("objective", "objective") if store.header else "objective",
        trial.best, trial.config)


def report_command(args):
  """Execute the ``report`` subcommand."""
  store = StudyStore.open(args.study_dir)
  direction = "max"
  if store.header:
    direction = store.header.get("direction", "max")
  top = args.top or None
  _print_leaderboard(store, direction, top=top)


def main(argv=None):
  """Dispatch to the selected subcommand."""
  args = parse_args(argv)
  setup_logging(getattr(args, "log_level", "INFO"),
                getattr(args, "log_targets", "STDERR"))
  if args.command == "run":
    run_command(args)
  elif args.command == "report":
    report_command(args)
  else:  # pragma: no cover - argparse enforces the choice
    fatal(f"Unknown command: {args.command!r}", ValueError)
  logging.shutdown()


if __name__ == "__main__":
  main()
