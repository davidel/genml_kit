"""Self-supervised pre-training for ConvViT.

Supports multiple pre-training algorithms via the ``--method`` flag:
``simmim`` (masked-image modeling) and ``ijepa`` (joint-embedding
predictive architecture).

Usage::

    genml_kit-pretrain \
        --method simmim \
        --datasets "my-images" "other-images" \
        --image_size 448 \
        --batch_size 32 \
        --epochs 200 \
        --output_dir ./checkpoints/pretrain

After training, the encoder weights can be loaded into a classification
model via ``--source_checkpoint`` in ``genml_kit-train``.
"""

import argparse
import collections
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from genml_kit.datasets.balanced_sampler import BalancedBatchSampler
from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.datasets.field_dataset import FieldSectorDataset
from genml_kit.io.checkpointing import (
    CheckpointSaver,
    create_model_report,
    open_resume_context,
    parse_state_flags,
)
from genml_kit.models.registry import load_model
from genml_kit.pretrain.methods import get_method, list_methods
from genml_kit.training.grad_monitor import create_grad_monitor
from genml_kit.training.model_utils import (
    enable_grad_checkpointing,
    model_mode,
    set_train_mode,
)
from genml_kit.training.optim_factory import build_optimization, report_lr
from genml_kit.utils.args import (
    add_checkpoint_args,
    add_logging_args,
    add_optimization_args,
    add_source_checkpoint_args,
    add_training_state_args,
    normalize_args,
)
from genml_kit.utils.cli import KVPairAction
from genml_kit.utils.gpu import gpu_stats_str, resolve_device
from genml_kit.utils.logging import fatal, open_writer, setup_logging
from genml_kit.utils.seed import resolve_seed, seed_everything, seed_worker
from genml_kit.utils.signal import InterruptedException, sigexcept


def build_pretrain_transform(image_size=448):
  """Default augmentations for pre-training.

    Resize to *image_size* (shortest side), center crop, then horizontal
    and vertical flips.  Used by any method without a ``build_transform``
    override (e.g. I-JEPA).
    """
  return v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])


class _TransformWrapper:
  """Apply a transform to the image field of a dict-returning dataset."""

  def __init__(self, dataset, transform, image_field="image"):
    self._dataset = dataset
    self._transform = transform
    self._image_field = image_field

  def __len__(self):
    return len(self._dataset)

  def __getitem__(self, idx):
    item = self._dataset[idx]
    return {**item, self._image_field: self._transform(item[self._image_field])}


def build_pretrain_dataset(args, needs_labels=False, transform=None):
  """Build the DatasetEnsemble + transform pipeline.

      If *transform* is ``None``, the default pretrain transform is used.
  """
  configs = []
  label_column = getattr(args, "label_column", None)
  for name in args.datasets:
    name = name.strip()
    if not name:
      continue
    if name.startswith("imagefolder/"):
      data_dir = name.split("/", 1)[1]
      configs.append({"name": data_dir, "source": "imagefolder"})
    elif os.path.isdir(name):
      configs.append({"name": name, "source": "imagefolder"})
    else:
      cfg = {
          "name": name,
          "source": "hf",
          "split": "train",
          "image_column": args.image_column,
      }
      if label_column:
        cfg["label_column"] = label_column
      configs.append(cfg)

  if not configs:
    fatal("No datasets specified. Use --datasets <name1> <name2> ...", ValueError)

  ensemble = DatasetEnsemble(
      configs,
      cache_dir=args.cache_dir,
      hf_token=args.hf_token,
      strict=args.strict_datasets,
  )
  if needs_labels:
    ensemble.ensure_label_space()
    if ensemble.unlabeled_datasets:
      fatal(
          "These datasets provide no labels, but this pre-training method "
          f"requires them: {', '.join(ensemble.unlabeled_datasets)}.  Pass "
          f"only labeled datasets (or drop the unlabeled ones), or switch "
          f"to a label-free method (e.g. --method simmim).", ValueError)
  if transform is None:
    transform = build_pretrain_transform(args.image_size)
  dataset = _TransformWrapper(ensemble, transform, image_field=ensemble.image_column)
  if not needs_labels:
    # Methods that ignore labels get image-only items: a mixed ensemble
    # (some sources labeled, some not) would otherwise produce batches
    # with inconsistent keys that default_collate cannot handle.
    dataset = FieldSectorDataset(dataset,
                                 fields={ensemble.image_column: ensemble.image_column})
  logging.info(ensemble.summary())
  return dataset, ensemble


def log_validation_images(method,
                          model,
                          loader,
                          writer,
                          global_step,
                          device,
                          image_column,
                          num_samples=8):
  """Log method-specific validation images to TensorBoard.

  Pulls a single batch from *loader*, extracts the images stored under
  ``batch[image_column]`` (the loader yields dicts), slices the first
  *num_samples* and hands them to ``method.validate()``, which returns
  optional reconstruction images.  If the method returns ``None``
  (methods without pixel-space visualisations, e.g. I-JEPA), nothing
  is logged.

  The model is restored to train mode even when nothing is logged.
  """
  with model_mode(model, "eval"):
    batch = next(iter(loader))
    raw = batch[image_column]
    if isinstance(raw, (tuple, list)):
      images = type(raw)(v[:num_samples].to(device) for v in raw)
    else:
      images = raw[:num_samples].to(device)
    with torch.no_grad():
      recon = method.validate(model, images, num_samples)
    if recon is None:
      return
    # recon is (N, C, H, W) — log first sample.
    writer.add_image("recon/original", images[0], global_step)
    writer.add_image("recon/reconstructed", recon[0].clamp(0, 1), global_step)


def train_one_epoch(
    method,
    model,
    loader,
    ensemble,
    optimizer,
    device,
    amp_dtype,
    epoch,
    global_step,
    writer,
    log_every=50,
    vis_every=0,
    monitor=None,
    grad_accum_steps=1,
    scaler=None,
    *,
    saver,
):
  """Run one epoch of self-supervised pre-training.

    If *grad_accum_steps* > 1, gradients are accumulated over that many
    micro-batches before the optimizer steps and gradients are zeroed.

    When *scaler* is provided (float16 AMP), gradient scaling is applied
    to avoid underflow of small gradients.

    When *saver* has a positive ``save_every``, a periodic checkpoint is
    written to ``<root>_latest.pt`` every N optimizer steps (recording
    ``epoch - 1`` as the last fully completed epoch).

    Returns ``(avg_loss, global_step)``.
  """
  set_train_mode(model, 'train')
  total_loss = 0.0
  total_samples = 0
  start_time = time.time()
  last_log_time = start_time
  window_samples = 0
  window_loss = 0.0

  total_batches = len(loader)

  needs_labels = method.needs_labels

  for step, batch in enumerate(loader):
    raw = batch[ensemble.image_column]
    if isinstance(raw, (tuple, list)):
      images = type(raw)(v.to(device, non_blocking=True) for v in raw)
    else:
      images = raw.to(device, non_blocking=True)
    labels = None
    if needs_labels:
      labels = batch[ensemble.label_column].to(device, non_blocking=True)

    with torch.amp.autocast(
        "cuda",
        dtype=amp_dtype,
        enabled=(amp_dtype is not None and device.type == "cuda"),
    ):
      loss, _info = method.train_step(
          model,
          images,
          global_step,
          labels=labels,
      )

    loss = loss / grad_accum_steps
    if amp_dtype == torch.float16 and scaler is not None:
      scaler.scale(loss).backward()
    else:
      loss.backward()

    # Step optimizer only every grad_accum_steps batches (or at end of epoch).
    if (step + 1) % grad_accum_steps == 0 or (step + 1) == total_batches:
      if amp_dtype == torch.float16 and scaler is not None:
        scaler.unscale_(optimizer)
        # Gradient monitor must read AFTER unscale_() so it sees the true
        # gradient magnitudes, not the scaled values.
        if monitor is not None:
          monitor.step(global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
      else:
        if monitor is not None:
          monitor.step(global_step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
      optimizer.zero_grad(set_to_none=True)

      global_step += 1

      # Mid-epoch periodic checkpoint. Records *epoch - 1* (the last fully
      # completed epoch): on resume the interrupted epoch restarts instead
      # of being skipped. The running loss average is unavailable mid-epoch,
      # and the resume path does not consume it.
      if saver.should_save(global_step):
        saver.save_latest(epoch - 1, global_step=global_step)

    bs = images.shape[0]
    total_loss += loss.item() * bs * grad_accum_steps
    total_samples += bs
    window_samples += bs
    window_loss += loss.item() * bs * grad_accum_steps

    elapsed = time.time() - last_log_time
    if (step +
        1) % grad_accum_steps == 0 and global_step > 0 and global_step % log_every == 0:
      throughput = window_samples / elapsed if elapsed > 0 else 0
      w_loss = window_loss / window_samples if window_samples > 0 else 0.0
      avg_loss_so_far = total_loss / total_samples
      gpu = gpu_stats_str(device)
      lr_str = report_lr(optimizer, writer=writer, step=global_step)
      msg = (f"  [Step {step + 1}/{total_batches}]"
             f" loss={w_loss:.4f} ({avg_loss_so_far:.4f})"
             f" {lr_str}"
             f" img/s={throughput:.0f}")
      logging.info(msg)
      if gpu:
        logging.info(f"  [Step {step + 1}/{total_batches}] {gpu}")
      if writer is not None:
        writer.add_scalar("Train/loss_step", w_loss, global_step)
        writer.add_scalar("Train/loss_avg", avg_loss_so_far, global_step)
        writer.add_scalar("Train/throughput", throughput, global_step)
        if device.type == "cuda":
          writer.add_scalar(
              "GPU/memory_MB",
              torch.cuda.memory_allocated(device) / 1024**2,
              global_step,
          )
          if hasattr(torch.cuda, "utilization"):
            writer.add_scalar(
                "GPU/utilization_pct",
                torch.cuda.utilization(device),
                global_step,
            )
      last_log_time = time.time()
      window_samples = 0
      window_loss = 0.0

    if (vis_every > 0 and global_step % vis_every == 0 and writer is not None):
      log_validation_images(method,
                            model,
                            loader,
                            writer,
                            global_step,
                            device,
                            image_column=ensemble.image_column)

  avg_loss = total_loss / max(total_samples, 1)
  elapsed = time.time() - start_time
  logging.info(f"  Epoch {epoch + 1} Train stats -> loss: {avg_loss:.4f}"
               f" | time: {elapsed:.1f}s")
  return avg_loss, global_step


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
      description="Self-supervised pre-training for ConvViT",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  available = ", ".join(list_methods()) or "(none)"
  parser.add_argument(
      "--method",
      type=str,
      default="simmim",
      choices=list_methods(),
      help=f"Pre-training method (available: {available}).",
  )
  parser.add_argument(
      "--model",
      type=str,
      default="convvit",
      help="Model name registered in the genml_kit registry "
      "(e.g. 'convvit', a HuggingFace model ID, or "
      "'timm:<model_name>' for timm models).",
  )
  parser.add_argument(
      "--datasets",
      nargs="+",
      required=True,
      help="Dataset names or local paths to include in ensemble. "
      "Use HuggingFace IDs (e.g. 'HAM10000') or directories.",
  )
  parser.add_argument(
      "--image_column",
      type=str,
      help="Image column name for HF datasets; auto-detected when omitted.",
  )
  parser.add_argument(
      "--label_column",
      type=str,
      help="Label column name for HF datasets; auto-detected when omitted. "
      "Required when using a label-aware pre-training method.",
  )
  parser.add_argument(
      "--strict_datasets",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Abort instead of skipping a dataset that fails to load.",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      help="HuggingFace datasets cache directory",
  )
  parser.add_argument(
      "--hf_token",
      type=str,
      help="HuggingFace token for gated datasets (or set HF_TOKEN "
      "env var)",
  )
  parser.add_argument("--image_size",
                      type=int,
                      default=448,
                      help="Input image size (square)")
  parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

  parser.add_argument(
      "--batch_size",
      type=int,
      default=32,
      help="Per-GPU batch size. Reduce if OOM on "
      "consumer GPUs at 448px.",
  )
  parser.add_argument(
      "--samples_per_class",
      type=int,
      default=4,
      help="Samples per class in each batch.  Only used by "
      "label-aware methods (e.g. supcon).  batch_size must "
      "be divisible by this value.",
  )
  parser.add_argument("--epochs",
                      type=int,
                      default=200,
                      help="Total pre-training epochs.")
  parser.add_argument(
      "--lr",
      type=float,
      default=1e-4,
      help="Peak learning rate for AdamW. Linear warmup "
      "from 1%% of this value.",
  )
  parser.add_argument("--weight_decay",
                      type=float,
                      default=0.05,
                      help="AdamW weight decay.")
  parser.add_argument(
      "--lr_group",
      nargs="+",
      metavar="REGEX=LR",
      help="Per-parameter-group learning rates. "
      "E.g. --lr_group 'backbone.*=1e-5' 'classifier.*=1e-3'. "
      "Regexes matched against named_parameters(); first match wins. "
      "Unmatched trainable params use --lr.",
  )
  parser.add_argument(
      "--llrd_decay",
      type=float,
      metavar="FACTOR",
      help="Layer-wise learning rate decay factor. "
      "When set, learning rates decay by this factor per depth level "
      "(shallow layers get lower LR). E.g. --llrd_decay 0.85.",
  )
  add_optimization_args(parser)
  add_checkpoint_args(parser,
                      checkpoint_default="./checkpoints/convvit_simmim",
                      resume_default=True)
  add_training_state_args(parser, state_save="opt,sched", state_load="opt,sched")

  add_logging_args(parser)
  parser.add_argument(
      "--log_dir",
      type=str,
      help="TensorBoard log directory. Defaults to "
      "<checkpoint_dir>/logs if not specified.",
  )
  parser.add_argument(
      "--log_every",
      type=int,
      default=50,
      help="Log training metrics every N optimization steps.",
  )
  parser.add_argument(
      "--grad_monitor",
      type=int,
      default=-1,
      help="Log gradient stats every N steps. "
      "-1 (default) = disabled.",
  )
  parser.add_argument(
      "--norm_history",
      type=int,
      default=0,
      help="Keep last N norm snapshots per param for trend analysis. "
      "0 (default) = disabled. Requires --grad_monitor.",
  )
  parser.add_argument(
      "--trend_top_n",
      type=int,
      default=10,
      help="Show top N params in trend table by absolute change percent. "
      "0 = show all. Requires --norm_history.",
  )
  parser.add_argument(
      "--vis_every",
      type=int,
      default=0,
      help="Log reconstruction visualisation to TensorBoard "
      "every N steps. 0 (default) = disabled.",
  )
  parser.add_argument(
      "--save_every",
      type=int,
      default=500,
      help="Save checkpoint every N optimizer steps. 0 disables.",
  )

  parser.add_argument(
      "--model_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override model configuration (repeatable). "
      "Example: --model_arg depth=6 num_heads=8",
  )
  parser.add_argument(
      "--proc_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Override processor configuration (repeatable).",
  )
  parser.add_argument(
      "--optimizer",
      type=str,
      default="AdamW",
      help="torch.optim optimizer class name (case-sensitive), "
      "or a path to a custom .py script. "
      "Examples: AdamW (default), Adam, SGD.",
  )
  parser.add_argument(
      "--opt_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra optimizer kwargs (repeatable). "
      "Example: --opt_arg betas=0.9,0.999 momentum=0.9",
  )
  parser.add_argument(
      "--scheduler",
      type=str,
      help="torch.optim.lr_scheduler class name (case-sensitive), "
      "or a path to a custom .py script. "
      "Examples: CosineAnnealingLR, StepLR. "
      "Default: None (no scheduler).",
  )
  parser.add_argument(
      "--sched_arg",
      nargs="+",
      action=KVPairAction,
      default={},
      metavar="KEY=VALUE",
      help="Extra scheduler kwargs (repeatable). "
      "Example: --sched_arg T_max=50 eta_min=1e-6",
  )
  add_source_checkpoint_args(parser)

  parser.add_argument(
      "--grad_checkpoint",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable gradient checkpointing to reduce VRAM usage "
      "(~40-50%% less activation memory, ~25-35%% more compute per step). "
      "Enables larger batch sizes. Net throughput gain depends on "
      "whether the GPU is memory-bound or compute-saturated.",
  )

  # Two-pass parse: first to get --method, then add its args.
  known, _ = parser.parse_known_args(argv)
  method_cls = get_method(known.method)
  method_cls().add_args(parser)

  parser.add_argument(
      "--seed",
      type=int,
      default=None,
      help="Explicit RNG seed and full determinism (cuDNN deterministic "
      "kernels, benchmark off). Omit to keep runs fast: RNG streams are "
      "still seeded internally from $GENML_KIT_SEED (default 42), just "
      "not bit-exact.",
  )
  parser.add_argument(
      "--device",
      type=str,
      help="Device: cpu, cuda, or cuda:INDEX (default: auto-detect).",
  )
  args = parser.parse_args(argv)

  if args.log_dir is None:
    args.log_dir = os.path.dirname(args.checkpoint) or "."
    args.log_dir = os.path.join(args.log_dir, "logs")

  if args.hf_token is None:
    args.hf_token = os.environ.get("HF_TOKEN")

  return args


# Bookkeeping handed back after the pre-training loop finishes.
# ``best_macro_f1`` is not tracked during pre-training and is always 0.0.
# ``interrupt_signals`` mirrors train.py's TrainingResult for shape
# symmetry; there is no post-loop stage, so it is not consumed here.
TrainingResult = collections.namedtuple(
    "TrainingResult", "completed_epoch, best_macro_f1, global_step, interrupt_signals")


def build_pretrain_loaders(args, dataset, ensemble, device, needs_labels):
  """Build the training DataLoader for a pre-training run.

  Args:
      args: Parsed CLI args (batch_size, num_workers, samples_per_class).
      dataset: The wrapped dataset to iterate.
      ensemble: The ``DatasetEnsemble`` feeding label info (used by the
          balanced batch sampler).
      device: The run's ``torch.device`` (selects ``pin_memory``).
      needs_labels: Whether the method needs labels; selects the balanced
          batch sampler vs plain shuffled loader.

  Returns:
      Tuple ``(loader, data_generator)``.
  """
  # Generator for DataLoader shuffling.  With an explicit --seed it is
  # seeded so shuffling is reproducible; otherwise it draws fresh entropy
  # per run (global RNG seeding still applies to workers via seed_worker).
  data_generator = torch.Generator()
  if args.seed is not None:
    data_generator.manual_seed(args.seed)

  if needs_labels:
    sampler = BalancedBatchSampler(
        labels=ensemble.labels_array,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=(device.type == "cuda"),
    )
  else:
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=data_generator,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
  return loader, data_generator


def build_pretrain_model(args, device, method):
  """Load the backbone, build the method's model and apply source weights.

  Args:
      args: Parsed CLI args (model spec, grad_checkpoint,
          source_checkpoint, LoRA settings).
      device: Device to place the model on.
      method: The pre-training method instance (builds the model and
          owns method-specific state).

  Returns:
      The prepared ``torch.nn.Module``.
  """
  logging.info("Loading model '%s' via registry ...", args.model)
  base_model = load_model(
      args.model,
      num_labels=0,
      id2label={},
      label2id={},
      image_size=args.image_size,
      cache_dir=args.cache_dir,
      device=device,
      **args.model_arg,
  )

  if args.grad_checkpoint:
    enable_grad_checkpointing(base_model)

  model = method.build(args, base_model, device)
  method.load_checkpoint_state(model, {}, args)  # initialize method state

  if args.source_checkpoint:
    from genml_kit.io.checkpointing import load_checkpoint_weights

    logging.info(f"Loading source checkpoint: {args.source_checkpoint}")
    if args.param_rename:
      logging.info(f"  Key renames: {args.param_rename}")
    load_checkpoint_weights(
        args.source_checkpoint,
        model,
        device=device,
        param_rename=args.param_rename,
    )
  return model


def run_pretraining(args, model, loader, ensemble, method, optimization, device, writer,
                    start_epoch, global_step):
  """Run the self-supervised pre-training loop.

  Owns the model report, the gradient monitor, the ``CheckpointSaver``
  (with the method-state ``extra_fn`` and the save-on-exit in the
  ``finally`` block) and the per-epoch training cycle.

  Args:
      args: Parsed CLI args (epochs, logging, monitor and save settings).
      model: The prepared model; mutated in place by training.
      loader: The training DataLoader from :func:`build_pretrain_loaders`.
      ensemble: The ``DatasetEnsemble`` consumed by the method hooks.
      method: The pre-training method instance.
      optimization: The ``Optimization`` namedtuple from
          :func:`build_optimization`.
      device: The run's ``torch.device``.
      writer: TensorBoard ``SummaryWriter`` (closed here on exit).
      start_epoch: First epoch to run (0 on a fresh run).
      global_step: Optimizer step counter restored from the checkpoint.

  Returns:
      A ``TrainingResult`` namedtuple.
  """
  states_to_save = parse_state_flags(args.state_save)

  completed_epoch = start_epoch - 1  # last fully completed (-1 = none yet)
  # Loss of the last *completed* epoch (0.0 before the first epoch
  # finishes).  The finally-block below checkpoints it as-is, so on an
  # interrupt mid-epoch this value is intentionally the previous epoch's.
  avg_loss = 0.0
  grad_monitor = create_grad_monitor(args, model)

  # Report the final model state after checkpoint restoration and all training
  # initialization, immediately before pre-training begins.
  logging.info(create_model_report(model))

  # All checkpoint writes go through one saver: state sources bound once,
  # per-save data (epoch, global_step, method state) passed per call.
  # extra_fn computes the method-specific state at save time.
  saver = CheckpointSaver(
      model,
      optimization.optimizer,
      optimization.scheduler,
      root=args.checkpoint,
      states_to_save=states_to_save,
      scaler=optimization.scaler,
      remote_uri=args.remote_checkpoint,
      save_every=args.save_every,
      extra_fn=lambda: {"method_state": method.get_checkpoint_state(model, args)},
  )

  # Signals arriving inside the loop become InterruptedException, so the
  # finally-block below still runs and a consistent checkpoint lands on
  # disk before a clean exit.  __exit__ restores the handlers only after
  # that save completed (the with-block encloses the try/finally).
  with sigexcept() as interrupts:
    try:
      for epoch in range(start_epoch, args.epochs):
        logging.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        avg_loss, global_step = train_one_epoch(
            method,
            model,
            loader,
            ensemble,
            optimization.optimizer,
            device,
            args.amp_dtype,
            epoch,
            global_step,
            writer,
            log_every=args.log_every,
            vis_every=args.vis_every,
            monitor=grad_monitor,
            grad_accum_steps=args.grad_accum_steps,
            scaler=optimization.scaler,
            saver=saver,
        )
        writer.add_scalar("Train/loss_epoch", avg_loss, epoch)
        if optimization.scheduler is not None:
          optimization.scheduler.step()
        completed_epoch = epoch
        method.on_epoch_end(model, epoch, writer)

        saver.save_latest(completed_epoch, global_step=global_step, loss=avg_loss)

    except InterruptedException:
      logging.warning(f"Interrupted by {interrupts.received}; saving checkpoint.")
    finally:
      saver.save_latest(completed_epoch, global_step=global_step, loss=avg_loss)
      logging.info("Checkpoint saved on exit.")
      writer.close()

  return TrainingResult(completed_epoch,
                        best_macro_f1=0.0,
                        global_step=global_step,
                        interrupt_signals=interrupts.received)


def main(argv=None):
  args = normalize_args(parse_args(argv))
  setup_logging(args.log_level, args.log_targets)
  # An explicit --seed asks for deterministic kernels; the internally
  # resolved default seed only seeds the RNG streams (benchmark stays on).
  seed_everything(resolve_seed(args.seed),
                  deterministic=(args.seed is not None))

  method_cls = get_method(args.method)
  method = method_cls()

  logging.info("=" * 60)
  logging.info(f"Pre-training method: {args.method}")
  logging.info("=" * 60)
  logging.info(f"Args: {vars(args)}")

  device = resolve_device(args.device)
  if device.type == "cuda":
    logging.info(gpu_stats_str(device))

  logging.info("Building dataset ...")
  transform = method.build_transform(args.image_size)
  dataset, ensemble = build_pretrain_dataset(
      args,
      needs_labels=method.needs_labels,
      transform=transform,
  )
  logging.info(f"Total images: {len(dataset):,}")
  if len(dataset) == 0:
    fatal("No images loaded from any dataset. "
          "Check --datasets, --hf_token, and --cache_dir.")

  loader, data_generator = build_pretrain_loaders(args,
                                                  dataset,
                                                  ensemble,
                                                  device,
                                                  needs_labels=method.needs_labels)

  model = build_pretrain_model(args, device, method)

  logging.info(f"Effective batch size: {args.batch_size} x {args.grad_accum_steps}"
               f" = {args.batch_size * args.grad_accum_steps}")

  start_epoch = 0
  ckpt_extra = {}
  if args.resume:
    model, start_epoch, _, ckpt_extra = open_resume_context(args, model, device)
    # Restore method-specific state from checkpoint.
    method_state = ckpt_extra.get("method_state", {})
    method.load_checkpoint_state(model, method_state, args)

  states_to_load = parse_state_flags(args.state_load)
  optimization = build_optimization(args, model, device, ckpt_extra, states_to_load)

  writer = open_writer(log_dir=args.log_dir)

  global_step = ckpt_extra.get(
      "global_step",
      start_epoch * (len(loader) // args.grad_accum_steps),
  )
  del ckpt_extra

  run_pretraining(
      args,
      model,
      loader,
      ensemble,
      method,
      optimization,
      device,
      writer,
      start_epoch=start_epoch,
      global_step=global_step,
  )


if __name__ == "__main__":
  main()
