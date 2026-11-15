"""``genml-kit-tokenizer``: build a BPE tokenizer artifact offline.

Kept deliberately separate from ``genml-kit-train``: the tokenizer is trained
once, from a corpus subset that must be reproducible and reviewable, and the
resulting directory is then handed to every training run via ``--tokenizer``.
Mixing the two would make the vocabulary a side effect of a training run.

The heavyweight work lives in :mod:`genml_kit.models.tokenizer_train`; this
module is only the argument surface.
"""

import argparse
import logging

from genml_kit.models.tokenizer_train import DEFAULT_VOCAB_SIZE, train_tokenizer
from genml_kit.utils.script import load_extern
from genml_kit.utils.shared_args import (ARG_CACHE_DIR, ARG_DATASET, ARG_DATASET_ARG,
                                         ARG_SEED, ARG_SPLIT, add_shared_args)


def _fraction(value):
  """argparse type for ``--fraction``: a float in ``(0.0, 1.0]``."""
  try:
    fraction = float(value)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"not a float: {value!r}") from e
  if not 0.0 < fraction <= 1.0:
    raise argparse.ArgumentTypeError(f"must be in (0.0, 1.0], got {fraction!r}")
  return fraction


def _positive_int(value):
  """argparse type for integer caps that must be > 0."""
  try:
    number = int(value)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"not an int: {value!r}") from e
  if number <= 0:
    raise argparse.ArgumentTypeError(f"must be positive, got {number}")
  return number


def _sampler_spec(value):
  """Load a ``MODULE:FUNC`` sampler hook, resolved once at parse time."""
  module, _, func_name = value.partition(":")
  if not module or not func_name:
    raise argparse.ArgumentTypeError(f"expected MODULE:FUNC, got {value!r}")
  return load_extern(module, func_name)


def build_parser():
  """Build the ``genml-kit-tokenizer`` argument parser."""
  parser = argparse.ArgumentParser(
      prog="genml-kit-tokenizer",
      description="Train a byte-level BPE tokenizer and write a "
      "HuggingFace-format directory consumable with --tokenizer <dir>.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  add_shared_args(parser, ARG_DATASET, ARG_SPLIT, ARG_DATASET_ARG, ARG_SEED,
                  ARG_CACHE_DIR)
  parser.set_defaults(seed=0)

  parser.add_argument(
      "--text_column",
      required=True,
      help="Column holding the raw training text.",
  )
  parser.add_argument(
      "--out",
      required=True,
      help="Destination directory for tokenizer.json, tokenizer_config.json "
      "and tokenizer_meta.json.",
  )
  parser.add_argument(
      "--vocab_size",
      type=int,
      default=DEFAULT_VOCAB_SIZE,
      help="Target vocabulary size.",
  )
  parser.add_argument(
      "--fraction",
      type=_fraction,
      default=None,
      help="Fraction of rows to train on, in (0.0, 1.0]. Omit to use every "
      "row. Exactly round(n*fraction) rows for random-access datasets; an "
      "expected fraction for streaming ones.",
  )
  parser.add_argument(
      "--streaming",
      action="store_true",
      help="Load the corpus as a stream instead of materialising it. "
      "Required for corpora too large to fit in the cache.",
  )
  parser.add_argument(
      "--max_rows",
      type=_positive_int,
      default=None,
      help="Consider at most this many rows, before --fraction applies. "
      "Bounds how much of a stream is read.",
  )
  parser.add_argument(
      "--sampler",
      type=_sampler_spec,
      default=None,
      metavar="MODULE:FUNC",
      help="Custom row selector fn(dataset, fraction=, seed=) -> iterable of "
      "rows, overriding the built-in strategy.",
  )
  parser.add_argument(
      "--force",
      action="store_true",
      help="Rebuild even when the output directory already holds a tokenizer.",
  )
  return parser


def run_train(args):
  """Execute a tokenizer build from parsed arguments."""
  train_tokenizer(
      args.dataset,
      text_column=args.text_column,
      out_dir=args.out,
      vocab_size=args.vocab_size,
      fraction=args.fraction,
      seed=args.seed,
      split=args.split,
      dataset_name=(args.dataset_arg or {}).get("name"),
      dataset_arg=args.dataset_arg,
      cache_dir=args.cache_dir,
      streaming=args.streaming,
      max_rows=args.max_rows,
      sampler=args.sampler,
      force=args.force,
  )


def main(argv=None):
  """Entry point for ``genml-kit-tokenizer``."""
  parser = build_parser()
  args = parser.parse_args(argv)
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
  run_train(args)


if __name__ == "__main__":
  main()
