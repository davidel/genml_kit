"""Unified training entry point -- one ``genml-kit-train`` binary.

v4.2 (plans/GENERIC_PIPELINE.md s 5): "train" vs "pretrain" are
configurations, not programs.  ``--pipeline`` selects the data side
(images / vo_pair), ``--method`` the objective (classification, vo_pair,
simmim, supcon, dino, byol, ijepa).  Defaults keep the legacy CLI UX:
``--pipeline images --method classification``.
"""

import argparse
import logging
import os

import datasets as _datasets

from genml_kit.datasets.hf_proxy import HFDatasetProxy  # noqa: F401
from genml_kit.io.checkpointing import open_resume_context, parse_state_flags
from genml_kit.methods import build_method, get_method, list_methods
from genml_kit.models import load_model, load_processor  # noqa: F401
from genml_kit.pipelines import build_pipeline, get_pipeline, list_pipelines
from genml_kit.pipelines.images import (  # noqa: F401
    build_pretrain_dataset, build_pretrain_transform, compute_class_weights,
    log_validation_images,
)
from genml_kit.training.optim_factory import build_optimization
from genml_kit.training.train_compat import (
    CombinedFocalLoss,  # noqa: F401
    build_transforms,  # noqa: F401
    evaluate_performance,  # noqa: F401
    load_augmentation_script,  # noqa: F401
    mixup_data,  # noqa: F401
    parse_class_multipliers,  # noqa: F401
)
from genml_kit.training.trainer import (  # noqa: F401 (compat export)
    BaseTrainer, TrainingResult,
)
from genml_kit.utils.args import (
    add_checkpoint_args,
    add_logging_args,
    add_optimization_args,
    add_source_checkpoint_args,
    add_training_state_args,
    normalize_args,
)
from genml_kit.utils.cli import KVPairAction
from genml_kit.utils.gpu import resolve_device
from genml_kit.utils.logging import fatal, open_writer, setup_logging
from genml_kit.utils.seed import resolve_seed, seed_everything

load_dataset = _datasets.load_dataset  # patchable via 'training.train.load_dataset'


def load_and_split_dataset(
    dataset_name,
    cache_dir=None,
    test_size=0.2,
    seed=None,
    train_transform=None,
    val_transform=None,
    image_column=None,
    label_column=None,
):
  """Load a HuggingFace dataset, return ``(train_proxy, val_proxy)``."""
  if dataset_name.startswith("imagefolder/"):
    data_dir = dataset_name.split("/", 1)[1]
    raw = load_dataset("imagefolder", data_dir=data_dir, cache_dir=cache_dir)
  else:
    raw = load_dataset(dataset_name, cache_dir=cache_dir)

  # Single split: validate, split and wrap.
  if isinstance(raw, _datasets.Dataset):
    detected_image_column = image_column or HFDatasetProxy.detect_image_column(raw)
    if detected_image_column is None:
      fatal(
          f"No image column detected in {dataset_name}. "
          f"Columns: {list(raw.features.keys())}",
          ValueError,
      )
    split = raw.train_test_split(test_size=test_size, seed=seed)
    return (
        HFDatasetProxy(
            split["train"],
            transform=train_transform,
            image_column=image_column,
            label_column=label_column,
        ),
        HFDatasetProxy(
            split["test"],
            transform=val_transform,
            image_column=image_column,
            label_column=label_column,
        ),
    )

  # DatasetDict: validate and ensure train/test splits exist.
  for split_name in raw:
    detected_image_column = image_column or HFDatasetProxy.detect_image_column(
        raw[split_name])
    if detected_image_column is None:
      fatal(
          f"No image column in split '{split_name}' of {dataset_name}. "
          f"Columns: {list(raw[split_name].features.keys())}",
          ValueError,
      )

  splits = set(raw.keys())
  if "train" not in splits or "test" not in splits:
    if "train" in splits:
      split = raw["train"].train_test_split(test_size=test_size, seed=seed)
      raw = _datasets.DatasetDict(split)
    elif len(splits) == 1:
      only = next(iter(splits))
      split = raw[only].train_test_split(test_size=test_size, seed=seed)
      raw = _datasets.DatasetDict(split)
    else:
      names = list(raw.keys())
      raw = _datasets.DatasetDict({"train": raw[names[0]], "test": raw[names[1]]})

  logging.info(
      "Using image column: %s%s",
      image_column or "auto-detected",
      " (explicit)" if image_column else "",
  )
  logging.info(
      "Using label column: %s%s",
      label_column or "auto-detected",
      " (explicit)" if label_column else "",
  )
  return (
      HFDatasetProxy(
          raw["train"],
          transform=train_transform,
          image_column=image_column,
          label_column=label_column,
      ),
      HFDatasetProxy(
          raw["test"],
          transform=val_transform,
          image_column=image_column,
          label_column=label_column,
      ),
  )


def build_parser():
  """Build the generic parser (--pipeline, --method, model, shared args)."""
  parser = argparse.ArgumentParser(
      description="Train a model: classification, VO or self-supervised "
      "pre-training (v4.2 unified harness).",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--pipeline",
      type=str,
      default="images",
      choices=list_pipelines(),
      help="Data pipeline (available: {}).".format(", ".join(list_pipelines())),
  )
  parser.add_argument(
      "--method",
      type=str,
      default="classification",
      choices=list_methods(),
      help="Objective (available: {}).".format(", ".join(list_methods())),
  )
  parser.add_argument(
      "--model",
      type=str,
      default="google/vit-base-patch16-224",
      help="HF model name/path or timm model (e.g. 'timm:eva02_base_patch14').",
  )
  parser.add_argument(
      "--model_arg",
      action=KVPairAction,
      default={},
      help="Model constructor kwargs: --model_arg key=value --model_arg k2=v2.",
  )
  parser.add_argument(
      "--grad_checkpoint",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable gradient checkpointing to reduce VRAM usage.",
  )
  lora_group = parser.add_argument_group("lora")
  lora_group.add_argument("--lora",
                          action="store_true",
                          default=False,
                          help="Apply LoRA adapters to the model.")
  lora_group.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
  lora_group.add_argument("--lora_alpha",
                          type=float,
                          default=16.0,
                          help="LoRA alpha scaling.")
  lora_group.add_argument("--lora_dropout",
                          type=float,
                          default=0.0,
                          help="LoRA dropout.")
  lora_group.add_argument("--lora_target_modules",
                          type=str,
                          default="",
                          help="Comma-separated LoRA target module names.")
  parser.add_argument("--image_size",
                      type=int,
                      default=448,
                      help="Model input resolution.")
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Training batch size per step.")
  parser.add_argument("--log_every",
                      type=int,
                      default=20,
                      help="Log training stats every N steps.")
  parser.add_argument("--num_workers",
                      type=int,
                      default=4,
                      help="DataLoader worker processes.")
  parser.add_argument("--seed",
                      type=int,
                      default=None,
                      help="RNG seed (deterministic kernels when set).")
  parser.add_argument("--epochs",
                      type=int,
                      default=5,
                      help="Number of training epochs.")
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
  return parser


def parse_args(argv=None):
  """Two-pass parse: generic flags first, then pipeline + method flags."""
  parser = build_parser()
  known, _ = parser.parse_known_args(argv)
  pipeline_cls = get_pipeline(known.pipeline)
  method_cls = get_method(known.method)
  # ORDER MATTERS (s 5.1): pipeline added first, then method, so
  # method-level flags override pipeline-level ones with the same name.
  pipeline_cls().add_args(parser)
  method_cls().add_args(parser)

  add_checkpoint_args(parser, checkpoint_default="genml_kit", resume_default=True)
  add_optimization_args(parser)
  add_training_state_args(parser,
                          state_save="opt,sched,amp",
                          state_load="opt,sched,amp")
  add_logging_args(parser)
  add_source_checkpoint_args(parser)

  opt = parser.add_argument_group("optimizer")
  opt.add_argument("--lr", type=float, default=3e-5, help="Base learning rate")
  opt.add_argument("--weight_decay", type=float, default=0.0)
  opt.add_argument("--llrd_decay",
                   type=float,
                   default=0.0,
                   help="Layer-wise LR decay factor.")
  opt.add_argument("--lr_group",
                   action=KVPairAction,
                   default={},
                   help="Per-group LRs: --lr_group layer_name=lr ...")
  opt.add_argument("--optimizer",
                   type=str,
                   default="AdamW",
                   help="Optimizer class name (AdamW, Adam, SGD).")
  opt.add_argument("--opt_arg",
                   action=KVPairAction,
                   default={},
                   help="Optimizer constructor kwargs.")
  opt.add_argument("--scheduler",
                   type=str,
                   default=None,
                   help="torch.optim.lr_scheduler class name (e.g. "
                   "CosineAnnealingLR, StepLR) or a path/URL; default: none.")
  opt.add_argument("--sched_arg",
                   action=KVPairAction,
                   default={},
                   help="Scheduler constructor kwargs.")
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N optimizer steps. 0 disables.",
  )

  parser.add_argument(
      "--device",
      type=str,
      help="Device: cpu, cuda, or cuda:INDEX (default: auto-detect).",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=None,
      help="TensorBoard log directory (default: <checkpoint_dir>/logs).",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      default=None,
      help="HuggingFace token (default: $HF_TOKEN).",
  )
  parser.add_argument(
      "--in_ch",
      type=int,
      default=1,
      help="Input channels (VO pipelines use single-channel imagery).",
  )
  parser.add_argument(
      "--vis_every",
      type=int,
      default=0,
      help="Log validation images every N epochs (0 disables).",
  )
  args = parser.parse_args(argv)
  _post_process(parser, args)
  return args


def _post_process(parser, args):
  """Shared default-filling done after parsing (kept importable)."""
  if args.log_dir is None:
    args.log_dir = os.path.dirname(args.checkpoint) or "."
    args.log_dir = os.path.join(args.log_dir, "logs")
  if args.hf_token is None:
    args.hf_token = os.environ.get("HF_TOKEN")
  # Classification path needs these present even when the pipeline does
  # not define them (two-pass parse, defaults section 6.1).
  for name, default in (("class_multipliers", None), ("sampler_weights", "frequency"),
                        ("sampler", "none"), ("val_split", 0.2),
                        ("train_transforms", None), ("val_transforms", None),
                        ("tta_transform", None), ("strict_datasets", False),
                        ("needs_labels", False), ("grad_accum_steps", 1), ("grad_clip",
                                                                           0.0)):
    if not hasattr(args, name):
      setattr(args, name, default)
  return args


def main(argv=None):
  """Run the unified training harness.

  The driver is branchless: it calls the ``Method`` lifecycle hooks
  (plans/B3_PLAN.md s 2.1) in one fixed order and never inspects which
  method/pipeline it is running.

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
  args = normalize_args(parse_args(argv))
  setup_logging(args.log_level, args.log_targets)
  seed_everything(resolve_seed(args.seed), deterministic=(args.seed is not None))

  device = resolve_device(args.device)
  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  logging.info("Resolved run: pipeline=%s method=%s (metric_key=%s)", pipeline.NAME,
               method.NAME, method.METRIC_KEY)

  # Method lifecycle (plans/B3_PLAN.md s 2.1), in the one order the driver
  # guarantees:
  #   1. prepare_transforms -- model-processor normalization before loaders
  #   2. loaders            -- the pipeline owns the data
  #   3. wire_data          -- the method consumes pipeline data attributes
  #   4. build_model        -- the method constructs the model (sized from
  #                            the label space wire_data provided)
  method.prepare_transforms(args, device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)
  method.wire_data(args, pipeline)
  model = method.build_model(args, device)
  method.load_checkpoint_state(model, {}, args)

  states_to_load = parse_state_flags(args.state_load)
  model, start_epoch, best_metric, ckpt_extra = open_resume_context(
      args,
      model,
      device,
      metric_key=f"best_{method.METRIC_KEY}",
      default_metric=_default_metric(method),
  )
  method.load_checkpoint_state(model, ckpt_extra.get("method_state", {}), args)

  optimization = build_optimization(args, model, device, ckpt_extra, states_to_load)
  global_step = ckpt_extra.get(
      "global_step",
      start_epoch * (len(pipeline.train_loader) // args.grad_accum_steps),
  )
  # Drop the reference to the full checkpoint extras dict so the potentially
  # large optimizer/scheduler/scaler state can be GC'd before the trainer is
  # constructed. This is intentional memory hygiene, not vestigial code.
  del ckpt_extra

  writer = open_writer(log_dir=args.log_dir)
  trainer = BaseTrainer(
      args=args,
      model=model,
      method=method,
      pipeline=pipeline,
      optimization=optimization,
      device=device,
      writer=writer,
      start_epoch=start_epoch,
      best_metric=best_metric,
      global_step=global_step,
  )
  result = trainer.run()
  _post_train(args, pipeline, method, device, result)


def _default_metric(method):
  """Best-metric sentinel for the first validation of a run.

  The sentinel is the worst possible value in the method's metric direction,
  so the first real metric always wins.  Probe the direction with two
  constants rather than duplicating maximize/minimize knowledge.
  """
  return float("inf") if method.has_metric_improved(0.0, 1.0) else float("-inf")


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
