"""Build real CLI args namespaces from the entry-point parsers.

Tests used to hand-build ``argparse.Namespace`` fixtures that mirrored the
CLI surface.  Those fixtures silently drifted whenever a flag was added or
a ``getattr`` fallback was removed (e.g. the RL convergence tests broke
when ``args.batch_size`` started being read directly).  Deriving the
namespace from the *real* parser keeps tests in lock-step with the CLI.

Two helpers are provided:

- :func:`train_args` parses with :func:`genml_kit.cli.train.build_parser`,
  i.e. the plain argparse defaults (no owner-declared ``arg_defaults``).
- :func:`train_args_full` parses with
  :func:`genml_kit.cli.train.parse_args`, which *also* applies the owner
  declared ``arg_defaults`` for the selected ``--pipeline`` / ``--method``.

Both apply ``overrides`` by assigning onto the returned namespace, which is
purely a test-side convenience (``tests/test_args_frozen.py`` only guards
against undeclared mutation inside ``genml_kit/``).
"""

from genml_kit.cli.llm import build_parser as build_llm_parser
from genml_kit.cli.train import build_parser, parse_args


def _apply_overrides(args, overrides):
  """Assign ``overrides`` onto *args* and return it."""
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def train_args(argv=None, **overrides):
  """Return a train namespace from the plain parser defaults.

  Args:
      argv: Optional argument list parsed by ``build_parser`` (owner-declared
          ``arg_defaults`` are *not* applied).
      **overrides: Attribute overrides applied after parsing.

  Returns:
      argparse.Namespace: The parsed (and overridden) arguments.
  """
  args = build_parser().parse_args(list(argv or []))
  return _apply_overrides(args, overrides)


def train_args_full(argv=None, **overrides):
  """Return a train namespace with owner-declared defaults applied.

  Same as :func:`train_args` but routes through ``parse_args`` so the
  ``--pipeline`` / ``--method`` ``arg_defaults`` are honoured.

  Args:
      argv: Optional argument list parsed by ``parse_args``.
      **overrides: Attribute overrides applied after parsing.

  Returns:
      argparse.Namespace: The parsed (and overridden) arguments.
  """
  args = parse_args(list(argv) if argv is not None else None)
  return _apply_overrides(args, overrides)


def llm_args(command, argv=None, **overrides):
  """Return a ``genml-kit-llm`` namespace for *command*.

  Args:
      command: Subcommand name (``"generate"`` or ``"eval"``); placed first
          in the parsed argv.
      argv: Optional additional arguments appended after *command*.
      **overrides: Attribute overrides applied after parsing.

  Returns:
      argparse.Namespace: The parsed (and overridden) arguments.
  """
  # ``--model`` is a required flag; the caller supplies it via *overrides*
  # (that is the historical ``_common(model_dir, ...)`` contract), so seed a
  # placeholder here and let the override assign the real value.
  argv = list(argv or [])
  if "--model" not in argv:
    argv = ["--model", "<placeholder>"] + argv
  args = build_llm_parser().parse_args([command] + argv)
  return _apply_overrides(args, overrides)
