"""``genml-kit-llm``: generation and evaluation for causal language models.

A single entry point with two subcommands::

    genml-kit-llm generate --model sshleifer/tiny-gpt2 --prompts prompts.txt
    genml-kit-llm eval --model sshleifer/tiny-gpt2 --dataset ./corpus

Both subcommands load their model through the shared registry
(:mod:`genml_kit.training.llm_model`), so HuggingFace repos and registered
custom models are handled identically.
"""

import argparse
import logging

from genml_kit.training.generate import add_generate_args, run_generate
from genml_kit.training.llm_eval import add_eval_args, run_eval
from genml_kit.utils.logging import fatal


def build_parser():
  """Build the top-level ``genml-kit-llm`` parser with its subcommands."""
  parser = argparse.ArgumentParser(
      prog="genml-kit-llm",
      description="Generate completions and evaluate causal language models.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  generate = subparsers.add_parser("generate",
                                   help="Generate completions for prompts (JSONL out).")
  add_generate_args(generate)

  evaluate = subparsers.add_parser(
      "eval", help="Perplexity / exact-match / token-F1 evaluation.")
  add_eval_args(evaluate)

  return parser


def main(argv=None):
  """Dispatch to the selected subcommand."""
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.command == "generate":
    run_generate(args)
  elif args.command == "eval":
    run_eval(args)
  else:  # pragma: no cover - argparse enforces the choice
    fatal(f"Unknown command: {args.command!r}", ValueError)
  logging.shutdown()


if __name__ == "__main__":
  main()
