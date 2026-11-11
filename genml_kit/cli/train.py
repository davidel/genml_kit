"""``genml-kit-train``: unified training entry point -- one binary.

v4.2 (plans/GENERIC_PIPELINE.md s 5): "train" vs "pretrain" are
configurations, not programs.  ``--pipeline`` selects the data side
(images / vo_pair), ``--method`` the objective (classification, vo_pair,
simmim, supcon, dino, byol, ijepa).  Defaults keep the legacy CLI UX:
``--pipeline images --method classification``.

The argparse surface and driver live here; the reusable data-loading
helper (:func:`genml_kit.training.train.load_and_split_dataset`) stays in
:mod:`genml_kit.training.train`.
"""

import argparse
import logging
import sys

from genml_kit.io.checkpointing import (
    open_resume_context,
    parse_state_flags,
    warn_world_size_mismatch,
)
from genml_kit.methods import METHODS, build_method
from genml_kit.pipelines import PIPELINES, build_pipeline
from genml_kit.training.optim_factory import build_optimization
from genml_kit.utils.args import (
    add_checkpoint_args,
    add_logging_args,
    add_loss_args,
    add_optimization_args,
    add_source_checkpoint_args,
    add_training_state_args,
)
from genml_kit.utils.attr import maybe_call
from genml_kit.utils.cli import KVPairAction, apply_arg_defaults
from genml_kit.utils.distributed import (
    broadcast_model_state,
    destroy_distributed,
    init_distributed,
    is_main_process,
    rank,
    world_size,
)
from genml_kit.utils.gpu import resolve_device
from genml_kit.training.metrics_sinks import build_metrics_logger
from genml_kit.utils.logging import setup_logging
from genml_kit.utils.seed import resolve_seed, seed_everything
from genml_kit.utils.shared_args import (
    ARG_BATCH_SIZE, ARG_CACHE_DIR, ARG_DATASET, ARG_DATASET_ARG, ARG_DIST_ARG,
    ARG_EPOCHS, ARG_GRAD_CHECKPOINT, ARG_IMAGE_COLUMN, ARG_IMAGE_SIZE, ARG_IN_CH,
    ARG_LORA, ARG_MAX_ITEMS, ARG_MAX_SEQ_LEN, ARG_MASK_PROMPT, ARG_METHOD, ARG_MODEL,
    ARG_MODEL_ARG,
    ARG_NUM_WORKERS,
    ARG_PAD_TO_MULTIPLE_OF,
    ARG_PARAM_INIT,
    ARG_SEED,
    ARG_SHUFFLE_BUFFER_K,
    ARG_SEQUENCE_MODE, ARG_SPLIT, ARG_VAL_SPLIT, add_shared_arg)


def build_parser():
  """Build the complete ``genml-kit-train`` parser.

  Returns a parser declaring *every* flag a run can read: the generic
  surface (``--pipeline`` / ``--method`` / model / data), all pipeline and
  method owner args, and the shared checkpoint / optimization / training
  state / logging / loss / source-checkpoint / distributed groups.

  Owner-declared ``arg_defaults`` are *not* applied here (they depend on
  the selected owners); :func:`parse_args` applies them before parsing.
  """
  parser = argparse.ArgumentParser(
      description="Train a model: classification, VO or self-supervised "
      "pre-training (v4.2 unified harness).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--pipeline",
      type=str,
      default="images",
      choices=PIPELINES.list_names(),
      help="Data pipeline (available: {}).".format(", ".join(PIPELINES.list_names())),
  )
  add_shared_arg(
      parser,
      ARG_METHOD,
      default="classification",
      choices=METHODS.list_names(),
      help="Objective (available: {}).".format(", ".join(METHODS.list_names())),
  )
  add_shared_arg(parser, ARG_MODEL, default="google/vit-base-patch16-224")
  add_shared_arg(
      parser,
      ARG_MODEL_ARG,
      help="Override model configuration (repeatable). "
      "Example: --model_arg depth=6 num_heads=8. For "
      "--model cls_model_wrapper:<hf_name> the classifier head travels "
      "here too: --model_arg classifier=mlp:hidden=512,dropout=0.3",
  )
  add_shared_arg(parser,
                 ARG_GRAD_CHECKPOINT,
                 action=argparse.BooleanOptionalAction,
                 default=False)
  init_group = parser.add_argument_group("param init")
  add_shared_arg(
      init_group,
      ARG_PARAM_INIT,
      help="Parameter init strategy (default None = PyTorch default "
      "init). 'ortho' applies SB3-style orthogonal init (gain sqrt(2) for "
      "policy/hidden layers, 1.0 for value/head layers, zero biases) -- the "
      "init SB3's PPO/SAC are tuned around, so it helps MLP-based RL "
      "policies/critics on continuous control and small-to-mid MLPs. "
      "Generally neutral-to-mild (or harmful) for transformer/ViT and conv "
      "backbones whose own init is already tuned -- keep None there.",
  )
  lora_group = parser.add_argument_group("lora")
  add_shared_arg(lora_group, ARG_LORA, help="Apply LoRA adapters to the model.")
  lora_group.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
  lora_group.add_argument("--lora_alpha",
                          type=float,
                          default=16.0,
                          help="LoRA alpha scaling.")
  lora_group.add_argument("--lora_dropout",
                          type=float,
                          default=0.0,
                          help="LoRA dropout.")
  lora_group.add_argument(
      "--lora_target_modules",
      type=str,
      default="",
      help="Comma-separated LoRA target module names.",
  )
  add_shared_arg(parser, ARG_IMAGE_SIZE, default=448)
  add_shared_arg(parser, ARG_BATCH_SIZE, help="Training batch size per step.")
  add_shared_arg(parser, ARG_NUM_WORKERS)
  add_shared_arg(parser, ARG_SHUFFLE_BUFFER_K)
  add_shared_arg(parser, ARG_SEED)
  add_shared_arg(parser, ARG_EPOCHS)

  # Data-side shared args: declared here (once, globally) rather than by a
  # selectable pipeline, so consuming components can rely on them existing.
  data_group = parser.add_argument_group("data")
  add_shared_arg(data_group,
                 ARG_DATASET,
                 help="Single dataset (classification): HF name/path, or "
                 "'imagefolder/<dir>'.")
  add_shared_arg(data_group, ARG_DATASET_ARG)
  add_shared_arg(data_group,
                 ARG_IMAGE_COLUMN,
                 default="image",
                 help="Dataset column holding the image.")
  add_shared_arg(data_group, ARG_SPLIT)
  add_shared_arg(data_group,
                 ARG_VAL_SPLIT,
                 help="Fraction held out as validation "
                 "(classification, --dataset).")
  add_shared_arg(data_group, ARG_CACHE_DIR)
  add_shared_arg(data_group, ARG_MAX_ITEMS)
  add_shared_arg(data_group, ARG_MAX_SEQ_LEN)
  add_shared_arg(data_group, ARG_SEQUENCE_MODE)
  add_shared_arg(data_group,
                 ARG_MASK_PROMPT,
                 help="Mask prompt tokens out of the loss (instruction mode).")
  add_shared_arg(data_group,
                 ARG_PAD_TO_MULTIPLE_OF,
                 help="Round the padded length up to this multiple "
                 "(e.g. 8 for tensor-core friendliness).")
  parser.add_argument(
      "--grad_monitor",
      type=int,
      default=-1,
      help="Log gradient stats every N optimizer steps. -1 disables.",
  )
  parser.add_argument(
      "--norm_history",
      type=int,
      default=0,
      help="Keep last N norm snapshots per param for trend analysis.",
  )
  parser.add_argument(
      "--trend_top_n",
      type=int,
      default=10,
      help="Show top N params in trend table by change percent. 0 = all.",
  )

  # Owner + shared groups: added here so ``build_parser`` alone returns the
  # complete CLI surface (every flag a run can read).  Tests and tooling can
  # therefore obtain a *real* default namespace without hand-building one.
  register_all_owners(parser)

  add_checkpoint_args(parser, checkpoint_default="genml_kit", resume_default=True)
  add_optimization_args(parser)
  add_training_state_args(parser,
                          state_save="opt,sched,amp",
                          state_load="opt,sched,amp")
  add_logging_args(parser)
  add_loss_args(parser)
  add_source_checkpoint_args(parser)

  opt = parser.add_argument_group("optimizer")
  opt.add_argument("--lr", type=float, default=3e-5, help="Base learning rate")
  opt.add_argument("--weight_decay", type=float, default=0.0)
  opt.add_argument("--llrd_decay",
                   type=float,
                   help="Layer-wise LR decay factor. Default None = no "
                   "layer-wise decay (flat LR); set e.g. 0.85 to enable.")
  opt.add_argument(
      "--lr_group",
      nargs="+",
      metavar="REGEX=LR",
      help="Per-parameter-group learning rates. "
      "E.g. --lr_group 'backbone.*=1e-5' 'classifier.*=1e-3'. "
      "Regexes matched against named_parameters(); first match wins. "
      "Unmatched trainable params use --lr.",
  )
  opt.add_argument(
      "--optimizer",
      type=str,
      default="AdamW",
      help="Optimizer class name (AdamW, Adam, SGD).",
  )
  opt.add_argument(
      "--opt_arg",
      nargs="+",
      action=KVPairAction,
      metavar="KEY=VALUE",
      help="Extra optimizer kwargs (repeatable). "
      "Example: --opt_arg betas=[0.9,0.999] momentum=0.9",
  )
  opt.add_argument(
      "--scheduler",
      type=str,
      help="torch.optim.lr_scheduler class name (e.g. "
      "CosineAnnealingLR, StepLR) or a path/URL.",
  )
  opt.add_argument(
      "--sched_arg",
      nargs="+",
      action=KVPairAction,
      metavar="KEY=VALUE",
      help="Extra scheduler kwargs (repeatable). "
      "Example: --sched_arg T_max=50 eta_min=1e-6",
  )
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N optimizer steps. 0 disables.",
  )

  parser.add_argument(
      "--device",
      type=str,
      help="Device: cpu, cuda, or cuda:INDEX.",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      help="TensorBoard log directory.",
  )
  add_shared_arg(parser, ARG_IN_CH)
  parser.add_argument(
      "--vis_every",
      type=int,
      default=0,
      help="Log validation images every N epochs (0 disables).",
  )

  dist_group = parser.add_argument_group("distributed")
  add_shared_arg(dist_group, ARG_DIST_ARG)
  return parser


def register_all_owners(parser):
  """Register the CLI args of every pipeline and method.

    Called by :func:`build_parser` so every owner's flags exist on the one
    parser (``--help`` renders them all).  If two owners ever register the
    same option string, ``add_args`` raises ``ArgumentError`` here -- an
    explicit signal to rename one of the flags.
    """
  for name in PIPELINES.list_names():
    PIPELINES.get(name).add_args(parser)
  for name in METHODS.list_names():
    METHODS.get(name).add_args(parser)


def selected_owners(parser, argv):
  """Resolve the selected pipeline and method before the real parse.

  ``arg_defaults`` must be applied while argparse still treats the values
  as defaults (so an explicit flag wins), but at that point the namespace
  does not exist yet.  A throwaway parser that knows only the two selector
  flags recovers the choice from the raw argv; the defaults are seeded
  from *parser* so they can never drift from the real flag defaults.

  Args:
      parser: The fully built parser (used only for its selector defaults).
      argv: The raw argument list being parsed.

  Returns:
      list: The owning classes, pipeline first then method.
  """
  pre = argparse.ArgumentParser(add_help=False)
  pre.add_argument("--pipeline", default=parser.get_default("pipeline"))
  pre.add_argument("--method", default=parser.get_default("method"))
  known, _ = pre.parse_known_args(argv)
  return [PIPELINES.get(known.pipeline), METHODS.get(known.method)]


def parse_args(argv=None):
  """Build the full parser, apply owner defaults, then parse ``argv``."""
  parser = build_parser()

  # Owner-declared defaults go in last, once every flag exists: the method
  # and pipeline selected on this command line may override the plain
  # argparse defaults, but never an explicit flag.
  argv = sys.argv[1:] if argv is None else list(argv)
  apply_arg_defaults(parser, selected_owners(parser, argv))
  return parser.parse_args(argv)


def main(argv=None):
  """Run the unified training harness.

    The driver is branchless: it calls the ``Method`` lifecycle hooks
    (see the Method class docstring in ``genml_kit/methods/base.py``) in
    one fixed order and never inspects which method/pipeline it is
    running.

        parse -> seed -> device -> pipeline/method objects
          -> method.prepare_transforms   (processor normalization, if any)
          -> pipeline.build_loader x2    (data side)
          -> method.wire_data            (label space, criterion, ...)
          -> method.build_model          (model side, sized from wire_data)
          -> resume -> optimization -> BaseTrainer.run()
          -> [interrupt guard] -> method.post_train

    Invariant: transforms are resolved BEFORE the loaders are built (the
    classification data path consumes them during loader construction), and
    wire_data runs BEFORE build_model (the HF head is sized from the label
    space).  Both are enforced by the hook order, not by convention.
    """
  args = parse_args(argv)
  # Join the process group before anything rank-dependent: seeding, device
  # resolution and log-file ownership all key off the rank.  A no-op outside
  # ``torchrun``, so the single-device path is unchanged.
  init_distributed(backend=args.dist_arg.get("backend"))
  try:
    return _run(args)
  finally:
    destroy_distributed()


def _run(args):
  """Driver body, after the process group is up (see :func:`main`)."""
  # Rank-aware logging: only rank 0 owns the file targets, so concurrent ranks
  # do not interleave writes to the same file.  Every rank keeps stderr.
  log_targets = args.log_targets if is_main_process() else "STDERR"
  setup_logging(args.log_level, log_targets)
  # Rank-aware seeding: offset by global rank so ranks draw different
  # augmentation/dropout streams (identical streams would waste the ensemble
  # and can mask a sharding bug).
  seed_everything(resolve_seed(args.seed) + rank(),
                  deterministic=(args.seed is not None))

  device = resolve_device(args.device)
  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  logging.info(
      "Resolved run: pipeline=%s method=%s (metric_key=%s)",
      pipeline.NAME,
      method.NAME,
      method.METRIC_KEY,
  )

  # Method lifecycle, in the one order the driver guarantees:
  #   1. prepare_transforms -- model-processor normalization before loaders
  #   2. loaders            -- the pipeline owns the data
  #   3. wire_data          -- the method consumes pipeline data attributes
  #   4. build_model        -- the method constructs the model (sized from
  #                            the label space wire_data provided)
  method.prepare_transforms(args, device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)
  # Shard the training set across ranks (no-op single-process, or when
  # --dist_arg shard=false).  Must run before the epoch-budget math below, which
  # reads len(train_loader).
  maybe_call(pipeline, "shard_train_loader", args)
  # Shard the validation set too (no-op single-process, or when
  # --dist_arg shard=false).  Its per-rank results are merged across ranks in
  # BaseTrainer.validate, so a single rank's slice is never reported alone.
  maybe_call(pipeline, "shard_val_loader", args)
  # RL pipelines need environment initialized before wire_data reads n_actions/obs_dim
  # Default no-op in DataPipeline; RL pipelines override.
  pipeline.init_env(args)
  method.wire_data(args, pipeline)
  model = method.build_model(args, device)
  method.load_checkpoint_state(model, {}, args)

  states_to_load = parse_state_flags(args.state_load)
  model, start_epoch, best_metric, ckpt_extra = open_resume_context(
      args,
      model,
      device,
      metric_key=f"best_{method.METRIC_KEY}",
      default_metric=method.initial_best_metric,
  )
  method.load_checkpoint_state(model, ckpt_extra.get("method_state", {}), args)
  # Restore pipeline state (e.g., observation-normalization RMS for RL) saved
  # by RLTrainer.saver_extra() into the checkpoint.  Only pipelines that opt
  # in (RLPipeline) define load_checkpoint_state; no-op for image/vo pipelines
  # (mirrors genml_kit.utils.attr.maybe_call semantics).
  maybe_call(pipeline, "load_checkpoint_state", ckpt_extra)

  optimization = build_optimization(args,
                                    model,
                                    device,
                                    ckpt_extra,
                                    states_to_load,
                                    method=method)
  train_loader = getattr(pipeline, "train_loader", None)
  global_step = ckpt_extra.get(
      "global_step",
      (start_epoch *
       (len(train_loader) // args.grad_accum_steps) if train_loader is not None else 0),
  )
  # Mid-epoch resume: a checkpoint taken on an interrupt carries ``in_progress``
  # pointing at the epoch that was cut short.  ``start_epoch`` already equals
  # that epoch (``resume_checkpoint`` resumes *into* it), so surface the
  # micro-step for the loop to fast-forward to.
  resume_progress = ckpt_extra.get("in_progress")
  if not (resume_progress and resume_progress.get("epoch") == start_epoch):
    resume_progress = None
  # A changed rank count re-slices every rank's shard (shard boundaries are a
  # function of ``world_size``); warn (never abort) -- see the helper for why
  # the residual is only padding/truncation.
  warn_world_size_mismatch(ckpt_extra.get("world_size"), world_size() or 1)
  # Drop the reference to the full checkpoint extras dict so the potentially
  # large optimizer/scheduler/scaler state can be GC'd before the trainer is
  # constructed. This is intentional memory hygiene, not vestigial code.
  del ckpt_extra

  # Wrap LAST: the checkpoint restore and the optimizer were built against the
  # semantic model, so both see unwrapped parameter names and keys.  From here
  # on the trainer holds whatever the method wants to train with.  The default
  # is a transparent facade whose forward runs through DDP (gradient
  # all-reduce); RL methods override it to stay unwrapped and reduce gradients
  # explicitly -- see ``RLMethod.wrap_model``.  Identity when world_size == 1.
  model = method.wrap_model(model, device, args)
  # Every rank seeded differently (``seed + rank()``) to diversify its data
  # stream, which also makes ``build_model`` draw different initial weights.
  # Replicas must start from one model: broadcast rank 0's.  The supervised
  # path gets this implicitly from ``DistributedDataParallel.__init__``; doing
  # it explicitly covers the unwrapped (RL) path too.
  broadcast_model_state(model)

  logger = build_metrics_logger(args)
  # Select trainer class via Method.get_trainer_class() (C13 fix).
  trainer_cls = method.get_trainer_class()
  trainer = trainer_cls(
      args=args,
      model=model,
      method=method,
      pipeline=pipeline,
      optimization=optimization,
      device=device,
      logger=logger,
      start_epoch=start_epoch,
      best_metric=best_metric,
      global_step=global_step,
      resume_progress=resume_progress,
  )
  result = trainer.run()
  _post_train(args, pipeline, method, device, result)


def _post_train(args, pipeline, method, device, result):
  """Run the method's post-training stages, unless the run was interrupted.

    The interrupt guard is driver orchestration (whether post stages run at
    all); what runs is the method's business (post_train hook).
    """
  if result.interrupt_signals and result.interrupt_signals != ["SIGINT"]:
    logging.info("Interrupted by %s; checkpoint saved, exiting.",
                 result.interrupt_signals)
    return
  method.post_train(args, pipeline, device, result)


if __name__ == "__main__":
  main()
