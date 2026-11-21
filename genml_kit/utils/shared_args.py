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


def _non_negative_int(value):
  """``argparse`` type for an integer cap that may be zero but not negative."""
  try:
    number = int(value)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"not an int: {value!r}") from e
  if number < 0:
    raise argparse.ArgumentTypeError(f"must be non-negative, got {number}")
  return number


def _positive_float(value):
  """``argparse`` type for a duration that must be strictly positive."""
  try:
    number = float(value)
  except ValueError as e:
    raise argparse.ArgumentTypeError(f"not a number: {value!r}") from e
  if number <= 0:
    raise argparse.ArgumentTypeError(f"must be positive, got {number}")
  return number


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


ARG_CACHE_DIR = ("cache_dir", {
    "type": str,
    "help": "HuggingFace cache directory.",
})

# Default of ``--model``: the generic vision backbone.  RL factories must NOT
# treat this default as an explicit user choice, so ``RLMethod`` compares
# against this single constant to decide whether to fall back to its own
# registered factory.  Defined here so the CLI default and the RL fallback can
# never drift apart.
DEFAULT_VISION_MODEL = "google/vit-base-patch16-224"

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

ARG_DIST_ARG = ("dist_arg", {
    "nargs": "+",
    "action": KVPairAction,
    "metavar": "KEY=VALUE",
    "help": "Distributed-training overrides (repeatable). Keys: backend "
            "(nccl|gloo), shard (bool, default on when >1 rank), "
            "find_unused_parameters (bool), static_graph (bool). "
            "Bare keys mean true. "
            "Example: --dist_arg backend=nccl find_unused_parameters.",
})

ARG_TOKENIZER = ("tokenizer", {
    "type": str,
    "help": "Tokenizer name/path; defaults to --model.",
})

ARG_TOKENIZER_ARG = ("tokenizer_arg", {
    "nargs": "+",
    "action": KVPairAction,
    "metavar": "KEY=VALUE",
    "default": None,
    "help": "Extra kwargs forwarded to the tokenizer factory "
            "(repeatable; external tokenizer scripts receive them). "
            "Example: --tokenizer_arg vocab_size=16384.",
})

ARG_TRUST_REMOTE_CODE = ("trust_remote_code", {
    "action": "store_true",
    "help": "Allow custom code from the model hub.",
})

ARG_ATTN_IMPLEMENTATION = ("attn_implementation", {
    "type": str,
    "help": "Attention backend (e.g. 'sdpa').",
})

ARG_IN_CH = ("in_ch", {
    "type": int,
    "default": 1,
    "help": "Input channels (VO pipelines use single-channel imagery).",
})

ARG_DATASET = ("dataset", {
    "type": str,
    "help": "Dataset: HF name/path, 'imagefolder/<dir>', or a save_to_disk dir.",
})

ARG_SPLIT = ("split", {
    "type": str,
    "default": "train",
    "help": "HF dataset split to load.",
})

ARG_DATASET_ARG = ("dataset_arg", {
    "nargs": "+",
    "action": KVPairAction,
    "metavar": "KEY=VALUE",
    "default": None,
    "help": "Extra kwargs forwarded verbatim to datasets.load_dataset "
            "(repeatable). Example: --dataset_arg name=sample-10BT "
            "revision=main.",
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
    "help": "Cap the number of items per dataset/split (smoke tests). "
            "Applied to each split after the train/val split.",
})

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

ARG_SHUFFLE_BUFFER_K = ("shuffle_buffer_k", {
    "type": float,
    "default": 10.0,
    "help": "Round-robin shuffle round for streaming sources, in batches per "
            "rank: the buffer holds world_size*K batches. 1 is the minimal "
            "round-robin; 0 round-robins without permuting. Ignored for "
            "random-access datasets, which shuffle by sampler.",
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

ARG_MAX_BATCHES = ("max_batches", {
    "type": _non_negative_int,
    "help": "Cap the micro-batches per epoch (applied to every epoch). "
            "The next epoch continues until --epochs. Unset means drain the "
            "loader each epoch.",
})

ARG_MAX_TRAIN_TIME = ("max_train_time", {
    "type": _positive_float,
    "help": "Wall-clock training budget in seconds for the whole run. Stops "
            "once exceeded (DDP-synchronized). Unset means no time limit.",
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

ARG_EMA_DECAY = ("ema_decay", {
    "type": float,
    "default": 0.0,
    "help": "Weights-EMA decay for validation and the best checkpoint. "
            "0 disables the shadow entirely (default); only methods that "
            "opt in via their WEIGHTS_EMA flag wire it.",
})

ARG_EMA_FINAL_DECAY = ("ema_final_decay", {
    "type": float,
    "help": "Optional ramp target for --ema_decay: the effective decay moves "
            "linearly to this value over the run (DINO-style momentum "
            "schedule). Omit for a constant decay.",
})

ARG_EMA_WARMUP_STEPS = ("ema_warmup_steps", {
    "type": int,
    "default": 0,
    "help": "Linear warmup of the effective decay from 0 up to --ema_decay "
            "over this many optimizer steps. Ignored when "
            "--ema_final_decay is set.",
})

ARG_EMA_EVAL = ("ema_eval", {
    "action": "store_true",
    "help": "Evaluate and best-save the EMA weights instead of the raw "
            "trajectory (best.pt then carries the EMA weights; latest.pt "
            "always carries the raw trajectory plus the shadow).",
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

ARG_MAX_SEQ_LEN = ("max_seq_len", {
    "type": int,
    "default": 512,
    "help": "Token sequence length (truncation / packing window).",
})

ARG_SEQUENCE_MODE = ("sequence_mode", {
    "type": str,
    "default": "document",
    "help": "Text sequence recipe compiled by genml_kit.training.sequence_mode: "
            "'document' (no packing, document-exact boundary loss, prompt "
            "framed as a document prefix) or 'open' (packed concatenation, "
            "boundary loss kept, prompt framed as a whole document -- the "
            "GPT-2/GPT-3 pre-training recipe).",
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

ARG_LOG_LEVEL = ("log_level", {
    "type": str,
    "default": "INFO",
    "choices": ["DEBUG", "INFO", "WARNING", "ERROR"],
    "help": "Minimum logging level.",
})

ARG_LOG_TARGETS = ("log_targets", {
    "type": str,
    "default": "STDERR",
    "help": "Comma-separated log destinations. 'STDERR' logs to "
            "standard error; any other entry is a log file path "
            "appended to. Example: 'STDERR,/tmp/train.log' logs to both.",
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
        ARG_TOKENIZER_ARG,
        ARG_TRUST_REMOTE_CODE,
        ARG_ATTN_IMPLEMENTATION,
        ARG_IN_CH,
        ARG_DATASET,
        ARG_DATASET_ARG,
        ARG_SPLIT,
        ARG_VAL_SPLIT,
        ARG_IMAGE_COLUMN,
        ARG_MAX_ITEMS,
        ARG_IMAGE_SIZE,
        ARG_BATCH_SIZE,
        ARG_NUM_WORKERS,
        ARG_SHUFFLE_BUFFER_K,
        ARG_SEED,
        ARG_EPOCHS,
        ARG_MAX_BATCHES,
        ARG_MAX_TRAIN_TIME,
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
        ARG_SEQUENCE_MODE,
        ARG_MASK_PROMPT,
        ARG_PAD_TO_MULTIPLE_OF,
        ARG_LOG_LEVEL,
        ARG_LOG_TARGETS,
    )
}


def add_shared_arg(parser, shared_arg, **overrides):
  """Register a globally declared shared argument on *parser*.

  Idempotent per container: a shared argument that is already present on
  *parser* (or on the root parser of the group it belongs to) is skipped,
  because the whole point of a shared arg is that it is declared once --
  composable callers (e.g. several methods chaining a common mixin's
  ``add_args``) can re-request it without argparse raising a duplicate
  option conflict.  The skip is only valid for an *identical* spec: the
  re-request must match the registered flag on name, default, type,
  description (``help``), ``required`` and ``nargs``; a differing spec
  raises ``ValueError`` instead of silently keeping whichever variant was
  registered first (that would make ``--help`` and parsed defaults depend
  on registration order).

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
      The ``argparse.Action`` created by ``add_argument``, or ``None`` when
      the shared argument was already registered with the same spec.
  """
  dest, kwargs = shared_arg
  merged = {**kwargs, **overrides}
  existing = _find_option(parser, dest)
  if existing is not None:
    _assert_same_spec(dest, existing, merged)
    return None
  return parser.add_argument(f"--{dest}", **merged)


def add_shared_args(parser, *shared_args):
  """Register several shared arguments at once (see :func:`add_shared_arg`)."""
  for shared_arg in shared_args:
    add_shared_arg(parser, shared_arg)


def _has_option(container, dest):
  """Whether ``--{dest}`` is already registered on *container*.

  For an ``_ArgumentGroup`` the check walks up to the owning root parser,
  so a flag registered anywhere in the same parser counts as present --
  argparse treats option strings parser-wide, and the duplicate conflict
  fires at that level regardless of which group holds the flag.
  """
  root = container
  while getattr(root, "parent", None) is not None:
    root = root.parent
  return any(f"--{dest}" in action.option_strings for action in root._actions)


def _find_option(container, dest):
  """The registered action for ``--{dest}`` on *container*, or ``None``.

 For an ``_ArgumentGroup`` the search walks up to the owning root parser:
 argparse treats option strings parser-wide, so a flag registered via one
 group is present for every other group of the same parser too.
 """
  root = container
  while getattr(root, "parent", None) is not None:
    root = root.parent
  for action in root._actions:
    if f"--{dest}" in action.option_strings:
      return action
  return None


_ABSENT = object()

# argparse's signature-level defaults, used when a re-request omits a field
# that the registered action filled with an action-specific value (e.g.
# ``store_true`` implies ``default=False`` even when the spec does not say
# so -- comparing against ``None`` would falsely report a mismatch).
_ARGPARSE_DEFAULTS = {
    "default": None,
    "type": None,
    "help": None,
    "required": False,
    "nargs": None,
    "const": None,
}

# Fields an action class fills in at signature level when the spec omits
# them (e.g. ``store_true`` implies ``default=False``, ``nargs=0``,
# ``const=True`` -- comparing against the generic fallbacks would falsely
# report a mismatch on a bare, consistent re-request).
_ACTION_SIGNATURE_DEFAULTS = {
    "default": {
        argparse._StoreTrueAction: False,
        argparse._StoreFalseAction: False
    },
    "nargs": {
        argparse._StoreTrueAction: 0,
        argparse._StoreFalseAction: 0
    },
    "const": {
        argparse._StoreTrueAction: True,
        argparse._StoreFalseAction: False
    },
}


def _assert_same_spec(dest, existing, merged):
  """A skipped re-registration must request the identical spec.

 Compares the fields that define the flag's observable behaviour: default
 value, type converter, help text, ``required`` and ``nargs``.  Raises
 ``ValueError`` on the first difference, naming both sides.
 """
  mismatched = []
  for name, fallback in _ARGPARSE_DEFAULTS.items():
    wanted = merged.get(name, _ABSENT)
    if wanted is _ABSENT:
      wanted = _ACTION_SIGNATURE_DEFAULTS.get(name, {}).get(type(existing), fallback)
    have = getattr(existing, name, fallback)
    if not _spec_equal(have, wanted):
      mismatched.append(f"{name}: registered={have!r} incoming={wanted!r}")
  if mismatched:
    raise ValueError(f"--{dest} is already registered with a different spec: " +
                     "; ".join(mismatched))


def _spec_equal(have, wanted):
  """Strict equality for specs: type-strict for values, identity for callables."""
  if callable(wanted) or callable(have):
    return have is wanted
  if type(have) is not type(wanted):
    return False
  return have == wanted
