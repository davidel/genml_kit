"""Shared argparse blocks for the training CLIs.

``train.py`` and ``pretrain.py`` previously carried near-verbatim copies
of the checkpoint/optimization argparse blocks (including help text),
which had already drifted apart (e.g. a mangled ``--param_rename``
example in one of the two).  These helpers add the shared flags so the
two CLIs cannot drift again.

The individually shared flags (``--loss``, ``--grad_clip``, ...) are
declared once in :mod:`genml_kit.utils.shared_args`; the helpers below are
thin wrappers that register them, preserving the historical call sites.
"""

import argparse

import torch

from genml_kit.utils.shared_args import (ARG_GRAD_ACCUM_STEPS, ARG_GRAD_CLIP, ARG_LOSS,
                                         ARG_LOSS_ARGS, ARG_SOURCE_CHECKPOINT,
                                         add_shared_arg)
from genml_kit.utils.shared_args import json_dict_type

_AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}

__all__ = [
    "json_dict_type",
    "amp_dtype_from_args",
    "add_checkpoint_args",
    "add_training_state_args",
    "add_optimization_args",
    "add_source_checkpoint_args",
    "add_loss_args",
    "add_logging_args",
]


def amp_dtype_from_args(args):
  """Resolve the ``--amp_dtype`` CLI string to a ``torch.dtype`` (or None).

  Kept at the use site so the parsed ``args`` namespace stays a pure,
  read-only view of the command line: ``args.amp_dtype`` is always the
  raw string the user passed (or ``None``), and consumers convert on
  demand via this helper.
  """
  name = getattr(args, "amp_dtype", None)
  return _AMP_DTYPES[name] if name else None


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
  add_shared_arg(parser,
                 ARG_GRAD_CLIP,
                 default=grad_clip_default,
                 help="Maximum gradient norm for clipping. 0 disables "
                 "clipping.")
  if with_grad_accum:
    add_shared_arg(
        parser,
        ARG_GRAD_ACCUM_STEPS,
        help="Gradient accumulation steps. Effective batch "
        "size = batch_size * grad_accum_steps.",
    )
  parser.add_argument(
      "--log_every",
      type=int,
      default=50,
      help="Log training metrics every N batches.",
  )


def add_source_checkpoint_args(parser):
  """Add the shared ``--source_checkpoint`` / ``--param_rename`` flags.

      ``--source_checkpoint`` absorbs pretrained parameters (typically
      from ``genml_kit-pretrain``); ``--param_rename`` optionally rewrites
      keys before shape-based alignment.
    """
  add_shared_arg(
      parser,
      ARG_SOURCE_CHECKPOINT,
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


def add_loss_args(parser):
  """Add the shared ``--loss`` / ``--loss_args`` flags.

  ``--loss`` is a generic/global concept: any method (classification,
  PPO, SAC, DQN, ...) resolves it through the ``LOSSES`` registry.  The
  default is ``None`` and each method selects its own built-in bundle
  when unset (e.g. ``'focal'`` for classification, ``'ppo'`` for PPO).
  ``--loss_args`` is a JSON dict forwarded to the loss builder.
  """
  add_shared_arg(parser, ARG_LOSS)
  add_shared_arg(parser, ARG_LOSS_ARGS)


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
      "standard error; any other entry is a log file path "
      "appended to. Example: 'STDERR,/tmp/train.log' logs to both.",
  )
  parser.add_argument(
      "--metrics_sinks",
      type=str,
      default="tb,jsonl",
      help="Comma-separated list of metric sinks. 'tb' writes TensorBoard "
      "scalars; 'jsonl' appends structured records to <log_dir>/metrics.jsonl; "
      "'none' disables metrics output. Example: 'tb,jsonl'.",
  )
