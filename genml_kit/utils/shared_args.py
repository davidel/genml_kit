"""Globally declared CLI arguments shared across components.

Some ``argparse`` flags have a genuinely global meaning that several
components rely on (e.g. ``--cache_dir``, ``--model``, ``--loss``).  When
each component declared its own copy -- or, worse, silently read them with
``getattr(args, "flag", None)`` -- the flag could vanish from the CLI the
moment the *declaring* component was removed or renamed, leaving the
consumer permanently stuck on the ``None``/default fallback.

This module centralizes those declarations so that:

* every ``ARG_*`` constant is a ``(dest, add_argument-kwargs)`` tuple
  declared **once**, globally;
* :func:`add_shared_arg` registers one such constant on a parser (or an
  argument group), with optional per-CLI overrides (``required`` /
  ``default`` / ``help``) so an entry point can tweak a default without
  duplicating the whole spec;
* the :data:`SHARED_ARGS` registry makes it easy to audit later whether a
  shared argument ended up being used in a single place only, in which case
  it can be downgraded back to a plain ``add_argument`` call.

Convention: only flags that are *truly global* live here.  Flags whose
meaning/constraints differ between components that can be active at the
same time must be prefixed by their owner (e.g. ``--ppo_gamma``,
``--dqn_gamma``, ``--sac_gamma``) and stay in that component's ``add_args``.
"""

import argparse
import json

from genml_kit.utils.cli import KVPairAction


def json_dict_type(value):
  """Argparse type converter: a JSON string that must parse to a dict.

  Used for the ``--loss_args`` flag.  ``None``/empty maps to ``{}``; a
  JSON non-dict (e.g. ``"[]"``) raises ``ArgumentTypeError`` so the CLI
  fails fast instead of silently building a wrong loss.
  """
  if not value:
    return {}
  try:
    data = json.loads(value)
  except json.JSONDecodeError as e:
    raise argparse.ArgumentTypeError(f"Invalid JSON: {e}") from e
  if not isinstance(data, dict):
    raise argparse.ArgumentTypeError(
        f"Expected a JSON object (dict), got {type(data).__name__}: {value!r}")
  return data


# --- Model / downloads -------------------------------------------------------

ARG_CACHE_DIR = ("cache_dir", {
    "type": str,
    "help": "HuggingFace cache directory.",
})

ARG_MODEL = ("model", {
    "type": str,
    "help": "HF model name/path or timm model (e.g. 'timm:eva02_base_patch14'). "
            "RL methods accept registered factories (e.g. 'rl/qnet_dueling').",
})

ARG_MODEL_ARG = ("model_arg", {
    "nargs": "+",
    "action": KVPairAction,
    "metavar": "KEY=VALUE",
    "help": "Override model configuration (repeatable). "
            "Example: --model_arg depth=6 num_heads=8.",
})

ARG_TOKENIZER = ("tokenizer", {
    "type": str,
    "help": "Tokenizer name/path; defaults to --model.",
})

ARG_TRUST_REMOTE_CODE = ("trust_remote_code", {
    "action": "store_true",
    "help": "Allow custom code from the model hub.",
})

ARG_ATTN_IMPLEMENTATION = ("attn_implementation", {
    "type": str,
    "help": "Attention backend (e.g. 'sdpa').",
})

ARG_HF_TOKEN = ("hf_token", {
    "type": str,
    "help": "HuggingFace token.",
})

ARG_IN_CH = ("in_ch", {
    "type": int,
    "default": 1,
    "help": "Input channels (VO pipelines use single-channel imagery).",
})

# --- Data --------------------------------------------------------------------

ARG_DATASET = ("dataset", {
    "type": str,
    "help": "Dataset: HF name/path, 'imagefolder/<dir>', or a save_to_disk dir.",
})

ARG_SPLIT = ("split", {
    "type": str,
    "default": "train",
    "help": "HF dataset split to load.",
})

ARG_VAL_SPLIT = ("val_split", {
    "type": float,
    "default": 0.2,
    "help": "Fraction held out as validation (classification, --dataset).",
})

ARG_IMAGE_COLUMN = ("image_column", {
    "type": str,
    "default": "image",
    "help": "Dataset column holding the image.",
})

ARG_MAX_ITEMS = ("max_items", {
    "type": int,
    "help": "Cap the number of items (smoke tests).",
})

# --- Training loop -----------------------------------------------------------

ARG_IMAGE_SIZE = ("image_size", {
    "type": int,
    "help": "Model input resolution.",
})

ARG_BATCH_SIZE = ("batch_size", {
    "type": int,
    "default": 32,
    "help": "Batch size per step.",
})

ARG_NUM_WORKERS = ("num_workers", {
    "type": int,
    "default": 4,
    "help": "DataLoader worker processes.",
})

ARG_SEED = ("seed", {
    "type": int,
    "help": "RNG seed (deterministic kernels when set).",
})

ARG_EPOCHS = ("epochs", {
    "type": int,
    "default": 5,
    "help": "Number of training epochs.",
})

ARG_METHOD = ("method", {
    "type": str,
    "help": "Objective/method name.",
})

ARG_GRAD_CLIP = ("grad_clip", {
    "type": float,
    "default": 1.0,
    "help": "Max gradient norm for clipping (0 disables).",
})

ARG_GRAD_ACCUM_STEPS = ("grad_accum_steps", {
    "type": int,
    "default": 1,
    "help": "Number of micro-batches to accumulate before an optimizer step.",
})

ARG_GRAD_CHECKPOINT = ("grad_checkpoint", {
    "action": "store_true",
    "help": "Enable gradient checkpointing to reduce VRAM usage.",
})

ARG_LORA = ("lora", {
    "action": "store_true",
    "help": "Enable LoRA fine-tuning.",
})

ARG_PARAM_INIT = ("param_init", {
    "type": str,
    "choices": [None, "ortho"],
    "help": "Parameter init strategy (default None = PyTorch default init).",
})

ARG_SOURCE_CHECKPOINT = ("source_checkpoint", {
    "type": str,
    "help": "Checkpoint whose weights are loaded before training (transfer).",
})

# --- Loss --------------------------------------------------------------------

ARG_LOSS = ("loss", {
    "type": str,
    "help": "Loss spec: registered name or a path/URL to a .py script "
            "defining build_loss(**kwargs).  None (default) selects the "
            "method's built-in loss (e.g. 'focal' for classification, 'ppo' "
            "for PPO).",
})

ARG_LOSS_ARGS = ("loss_args", {
    "type": json_dict_type,
    "help": "JSON kwargs forwarded to the loss builder, e.g. "
            "'{\"gamma\": 1.0, \"label_smoothing\": 0.1}'.",
})

# --- Text / preference -------------------------------------------------------

ARG_MAX_SEQ_LEN = ("max_seq_len", {
    "type": int,
    "default": 512,
    "help": "Token sequence length (truncation / packing window).",
})

ARG_MASK_PROMPT = ("mask_prompt", {
    "action": "store_true",
    "help": "Mask prompt tokens out of the loss (instruction mode).",
})

ARG_PAD_TO_MULTIPLE_OF = ("pad_to_multiple_of", {
    "type": int,
    "help": "Round the padded length up to this multiple "
            "(e.g. 8 for tensor-core friendliness).",
})

#: Registry of every globally shared argument, keyed by its ``dest``.  Use
#: this to audit whether a shared flag is only consumed in one place (and can
#: therefore be downgraded back to a local ``add_argument`` call).
SHARED_ARGS = {
    arg[0]: arg for arg in (
        ARG_CACHE_DIR,
        ARG_MODEL,
        ARG_MODEL_ARG,
        ARG_TOKENIZER,
        ARG_TRUST_REMOTE_CODE,
        ARG_ATTN_IMPLEMENTATION,
        ARG_HF_TOKEN,
        ARG_IN_CH,
        ARG_DATASET,
        ARG_SPLIT,
        ARG_VAL_SPLIT,
        ARG_IMAGE_COLUMN,
        ARG_MAX_ITEMS,
        ARG_IMAGE_SIZE,
        ARG_BATCH_SIZE,
        ARG_NUM_WORKERS,
        ARG_SEED,
        ARG_EPOCHS,
        ARG_METHOD,
        ARG_GRAD_CLIP,
        ARG_GRAD_ACCUM_STEPS,
        ARG_GRAD_CHECKPOINT,
        ARG_LORA,
        ARG_PARAM_INIT,
        ARG_SOURCE_CHECKPOINT,
        ARG_LOSS,
        ARG_LOSS_ARGS,
        ARG_MAX_SEQ_LEN,
        ARG_MASK_PROMPT,
        ARG_PAD_TO_MULTIPLE_OF,
    )
}


def add_shared_arg(parser, shared_arg, **overrides):
  """Register a globally declared shared argument on *parser*.

  Args:
      parser: An ``argparse.ArgumentParser`` or ``argparse._ArgumentGroup``
          to register the flag on.  Passing a group keeps ``--help`` nicely
          sectioned; the ``add_argument`` call is identical either way.
      shared_arg: One of the module-level ``ARG_*`` tuples, i.e. a
          ``(dest, add_argument_kwargs)`` pair.
      **overrides: Per-CLI overrides merged over the canonical kwargs (e.g.
          ``required=True`` or ``default=None``) so an entry point can adapt a
          shared flag without duplicating its whole spec.

  Returns:
      The ``argparse.Action`` created by ``add_argument``.
  """
  dest, kwargs = shared_arg
  merged = {**kwargs, **overrides}
  return parser.add_argument(f"--{dest}", **merged)


def add_shared_args(parser, *shared_args):
  """Register several shared arguments at once (see :func:`add_shared_arg`)."""
  for shared_arg in shared_args:
    add_shared_arg(parser, shared_arg)
