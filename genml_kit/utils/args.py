"""Shared argparse blocks for the training CLIs.

``train.py`` and ``pretrain.py`` previously carried near-verbatim copies
of the checkpoint/optimization argparse blocks (including help text),
which had already drifted apart (e.g. a mangled ``--param_rename``
example in one of the two).  These helpers add the shared flags so the
two CLIs cannot drift again.
"""

import argparse

import torch


def normalize_args(args):
  """Post-process parsed CLI args shared by the training entry points.

  Performs the ``--amp_dtype`` string -> ``torch.dtype`` conversion so the
  rest of the code can rely on a dtype object (or ``None``).

  Args:
      args: The ``argparse.Namespace`` returned by a training script's
          ``parse_args()``.  Modified in place.

  Returns:
      The same ``args`` namespace, for call-site readability.
  """
  # Convert string amp_dtype to torch.dtype.
  args.amp_dtype = getattr(torch, args.amp_dtype, None) if args.amp_dtype else None
  return args


def add_checkpoint_args(parser, checkpoint_default, resume_default=None):
  """Add the shared checkpoint-location flags.

      Args:
          parser: An ``argparse.ArgumentParser``.
          checkpoint_default: Default for ``--checkpoint`` (path prefix;
              ``_latest.pt`` / ``_best.pt`` suffixes are appended by the
              training scripts).
          resume_default: Default for ``--resume``.  ``None`` omits the
              flag entirely (``train.py`` always attempts a resume, so it
              exposes no flag).
    """
  parser.add_argument(
      "--checkpoint",
      type=str,
      default=checkpoint_default,
      help="Checkpoint path prefix (without extension). "
      "_latest.pt and _best.pt are appended automatically.",
  )
  parser.add_argument(
      "--remote_checkpoint",
      type=str,
      help="Remote URI to sync checkpoints to "
      "(format: gs://BUCKET/PREFIX, r2://BUCKET/PREFIX, or "
      "s3://BUCKET/PREFIX).",
  )
  if resume_default is not None:
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=resume_default,
        help="Resume training from latest checkpoint if one exists.",
    )


def add_training_state_args(parser, state_save, state_load):
  """Add the ``--state_save`` / ``--state_load`` flag pair.

      Args:
          parser: An ``argparse.ArgumentParser``.
          state_save: Default for ``--state_save``.
          state_load: Default for ``--state_load``.
    """
  parser.add_argument(
      "--state_save",
      type=str,
      default=state_save,
      help="Comma-separated list of states to save in "
      "checkpoints. One or more of: opt, sched, amp, none.",
  )
  parser.add_argument(
      "--state_load",
      type=str,
      default=state_load,
      help="Comma-separated list of states to restore from checkpoint "
      "on resume. One or more of: opt, sched, amp, none.",
  )


def add_optimization_args(parser,
                          grad_clip_default=1.0,
                          with_amp=True,
                          with_grad_accum=True):
  """Add the shared optimization flags (AMP, clipping, accumulation).

      Args:
          parser: An ``argparse.ArgumentParser``.
          grad_clip_default: Default maximum gradient norm.
          with_amp: Whether to add ``--amp_dtype``.
          with_grad_accum: Whether to add ``--grad_accum_steps``.
    """
  if with_amp:
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["float16", "bfloat16"],
        help="AMP dtype for mixed precision. Omit to disable AMP. "
        "float16 requires GradScaler; bfloat16 is recommended for "
        "Ampere+ GPUs.",
    )
  parser.add_argument(
      "--grad_clip",
      type=float,
      default=grad_clip_default,
      help="Maximum gradient norm for clipping. 0 disables clipping.",
  )
  if with_grad_accum:
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch "
        "size = batch_size * grad_accum_steps.",
    )


def add_source_checkpoint_args(parser):
  """Add the shared ``--source_checkpoint`` / ``--param_rename`` flags.

      ``--source_checkpoint`` absorbs pretrained parameters (typically
      from ``genml_kit-pretrain``); ``--param_rename`` optionally rewrites
      keys before shape-based alignment.
    """
  parser.add_argument(
      "--source_checkpoint",
      type=str,
      help="Path to a source checkpoint to absorb parameters from. "
      "Keys are aligned by shape and name before loading. "
      "Typically produced by genml_kit-pretrain.",
  )
  parser.add_argument(
      "--param_rename",
      nargs="+",
      help="Regex-based key rename patterns for --source_checkpoint. "
      "Each pattern is 'SEARCH;REPLACE' where SEARCH is a Python regex "
      "and REPLACE may use $1, $2, \u2026 for capture groups. "
      "Applied before shape-based alignment. "
      "Example: 'encoder\\\\\\\\.(.*);model\\\\\\\\.$1'.",
  )


def add_logging_args(parser):
  """Add the shared ``--log_level`` / ``--log_targets`` flags.

  ``--log_targets`` is a comma-separated list of log destinations.  The
  special token ``STDERR`` selects standard error (the default); every
  other entry is a file path appended to.  Multiple destinations receive
  the same log records simultaneously, e.g. ``STDERR,/tmp/train.log``.
  """
  parser.add_argument(
      "--log_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      help="Minimum logging level.",
  )
  parser.add_argument(
      "--log_targets",
      type=str,
      default="STDERR",
      help="Comma-separated list of log destinations. 'STDERR' logs to "
      "standard error (default); any other entry is a log file path "
      "appended to. Example: 'STDERR,/tmp/train.log' logs to both.",
  )
