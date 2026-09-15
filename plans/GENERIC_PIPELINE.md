# Generic Pipeline + Method Registry — Detailed Plan (v4.2)

Status: **DRAFT for review** (no code written yet)
Date: 2026-XX-XX (resumable tomorrow)
Scope: Introduce a **pipeline** abstraction (data only) and **reshape `PretrainMethod`
  into a general "method"** (objective: loss + model + metric), so the training loop
  is data- and objective-agnostic, and VO trains through the **same** harness as
  classification and pre-training.

Revision note (v4.1 → v4.2): incorporates the verified findings of
`plans/REVIEW.md` (#1–#22, reviewed 2026-09-15) after line-by-line
re-verification against the source, plus the joint review of
`plans/COUNTER_REVIEW.md` (2026-09-15).  The loop in §4 now carries the real
AMP/scaler/monitor/accumulation mechanics verbatim from `train.py` and
`pretrain/cli.py`; `Method` restores the `train_step(model, blob, global_step)`
signature (instead of the loss-only `loss_fn`) plus the checkpoint/epoch-end
lifecycle hooks; `DataPipeline` owns both loaders and
`to_device(blob, device) -> DataBlob` (meta included, pipeline-specific
traversal, no type annotations); `--vo_stage` uses descriptive CLI names
(supervised|photometric) with an int internal contract and default `supervised`
(v4.2 joint review); the doc/test inventories in §11/§12 are corrected to the
verified blast radius.  Remaining joint-review items are tracked in
`plans/COUNTER_REVIEW.md`.

**Design north-star: mirror the repo's existing registries verbatim, and separate
  responsibilities.**

Two orthogonal registries:

| Registry | Owns | Registered | CLI |
|---|---|---|---|
| **pipeline** | DATA: loader, blob contract, `to_device` | `images`, `vo_pair` | `--pipeline` |
| **method** | OBJECTIVE: model + loss + metric (incl. `PretrainMethod` reshaped) | `classification`, `vo_pair`, `simmim`, `supcon`, `dino`, `byol`, `ijepa` | `--method` |

- **pipeline = "what data do I consume?"** (loader + blob shape + device transfer)
- **method = "what objective am I optimizing?"** (model + loss + metric direction)

The generic loop is **branchless**:

```python
for blob in pipeline.train_loader:
  blob = pipeline.to_device(blob, device)     # moves BOTH data and meta
  loss_out = method.train_step(model, blob, global_step)  # method owns loss
  ...
```

`--method` semantics from pretraining are **preserved** — each method keeps its own
loss (SimMIM MIM loss, SupCon contrastive, DINO distillation, VO staged
loss/photometric residual). The pipeline never steals the loss.

---

## 1. Existing Patterns To Copy (source of truth)

### 1.1 Classifier registry — `genml_kit/training/classifiers/__init__.py`
```python
_CLASSIFIERS = {}
def register_classifier(name):
  def wrapper(cls):
    if name in _CLASSIFIERS:
      fatal(f"Classifier {name!r} already registered", ValueError)
    _CLASSIFIERS[name] = cls
    return cls
  return wrapper
def _register_builtins():
  from genml_kit.training.classifiers import cls_attention, mlp  # noqa: F401
_register_builtins()
```

### 1.2 Method registry — `genml_kit/pretrain/methods/registry.py`
```python
_METHODS = {}
def register_method(cls):
  name = cls.NAME
  if name in _METHODS:
    fatal(f"Duplicate pre-training method name '{name}'", RuntimeError)
  _METHODS[name] = cls
  return cls
def get_method(name): ...
def list_methods(): return sorted(_METHODS)
```

### 1.3 Two-pass CLI parse — `genml_kit/pretrain/cli.py:540-543`
```python
known, _ = parser.parse_known_args(argv)
method_cls = get_method(known.method)
method_cls().add_args(parser)   # instance method; see §5.1 for the
                                # pipeline-before-method ordering
```

### 1.4 Pair-aware transform — `genml_kit/datasets/transforms.py`
`DictFieldTransform(ds, t, fields=("image",))` or `fields=("A","B")`.

---

## 2. Pipeline Registry (DATA ONLY) — `genml_kit/pipelines/`

### 2.1 `genml_kit/pipelines/__init__.py`
```python
"""Pipeline registry — data-agnostic DataLoader + blob contract.

A pipeline owns ONLY the data side of training:
  - add_args(parser)             -> registers data CLI flags
  - build_loader(args)           -> populates self.train_loader / self.val_loader
  - to_device(blob, device)      -> DataBlob with BOTH data and meta on device

It does NOT own the model, the loss, or the validation metric.  Those
belong to the method registry (PretrainMethod becomes a general "method").
"""

from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import (
    build_pipeline, get_pipeline, list_pipelines, register_pipeline,
)

def _register_builtins():
  from genml_kit.pipelines import images, vo_pair  # noqa: F401

_register_builtins()

__all__ = [
    "DataBlob", "DataPipeline",
    "build_pipeline", "get_pipeline", "list_pipelines", "register_pipeline",
]
```

### 2.2 `genml_kit/pipelines/contracts.py`
```python
"""Batch/contract types shared by all pipelines."""

import collections

# One DataLoader step.  data is opaque to the loop; meta carries anything
# the method's train_step needs (labels, GT similarity, ...).
DataBlob = collections.namedtuple("DataBlob", ["data", "meta"])

# NOTE: do NOT define a generic "ModelOutput" here -- the name is taken by
# genml_kit.models.registry.ModelOutput (holds .logits, used by every custom
# model loader).  The loop passes the raw `model(...)` return straight to
# method.train_step, which decodes its own shape (dict for VO, tensor for
# classification, tuple for SimMIM, ...).

# What method.train_step returns.  loss is the (already averaged-across-the-batch),
# UNSCALED objective; metrics is dict[str, torch.Tensor] for logging.
LossOutput = collections.namedtuple("LossOutput", ["loss", "metrics"])
```

### 2.3 `genml_kit/pipelines/base.py`
```python
"""Base class for all data pipelines."""

import abc
from torch.utils.data import DataLoader
from genml_kit.pipelines.contracts import DataBlob


class DataPipeline(abc.ABC):
  """Data side of training: loader + blob contract + device transfer.

  Registered via @register_pipeline, constructed via build_pipeline(name).
  Does NOT build models or compute losses (that is Method's job).  The
  concrete pipeline caches its loaders on first build so the loop and the
  trainer can reference ``self.pipeline.train_loader`` / ``val_loader``
  without extra plumbing.
  """

  NAME = ""                 # registry key (matches Method.NAME convention)

  def __init__(self):
    self.train_loader = None
    self.val_loader = None

  def add_args(self, parser):
    """Register this pipeline's data CLI flags (mirrors PretrainMethod)."""

  # NOTE: add_args is an INSTANCE method here (the plan instantiates
  # `pipeline = pipeline_cls()` before calling `pipeline.add_args(parser)`,
  # see §5.1).  §1.1/#22 nits settle on instance methods for pipelines and
  # methods alike; the old @classmethod on PretrainMethod.add_args is not
  # carried over.

  @abc.abstractmethod
  def build_loader(self, args, mode="train"):
    """Build the loader for *mode* and store it on self (train/val_loader).

    Idempotent: returning the cached loader on repeat calls with the same
    mode avoids rebuilding the dataset (and keeps the loop's
    ``self.pipeline.train_loader`` reference valid).
    """

  @abc.abstractmethod
  def to_device(self, blob, device):
    """Move BOTH blob.data and blob.meta onto device.

    PIPELINE-SPECIFIC: each pipeline implements the move for its own blob
    shape and meta nesting (a plain tensor for images; a tuple + nested
    meta dict for VO pairs -- note ``meta.gt`` is itself a dict of tensors
    {log_s, theta, t} that must be moved recursively; a list of crops for
    DINO).  No generic recursive utility is imposed; the PIPELINE owns the
    exact traversal.

    Return a new DataBlob (e.g. ``blob._replace(...)``) so the device
    transfer is visible to the method's train_step.
    """

  # NOTE: this interface follows the project's no-typing-annotation
  # convention (see .style.yapf / Ruff config): no PEP-484 annotations
  # anywhere in the codebase, including these signatures.

  def build_transform(self, args, method):
    """Optional generic preprocessing for a method (default: identity).

    Default no-op; a pipeline MAY apply its generic preprocessing here
    (resize/normalize/crop-flip-jitter).  Objective-defined augmentation
    (DualView / MultiCrop) is composed by the METHOD on top -- see "two-phase
    contract" in §6.1.  Returns a callable applied to each raw item before it
    becomes a DataBlob.
    """
    return lambda x: x
```

### 2.4 `genml_kit/pipelines/registry.py`
```python
"""Pipeline registry — mirrors genml_kit/training/classifiers/__init__.py."""

import logging
from genml_kit.utils.logging import fatal

_PIPELINES = {}

def register_pipeline(name):
  def wrapper(cls):
    if name in _PIPELINES:
      fatal(f"Pipeline {name!r} already registered", ValueError)
    _PIPELINES[name] = cls
    return cls
  return wrapper

def get_pipeline(name):
  if name not in _PIPELINES:
    available = ", ".join(sorted(_PIPELINES)) or "(none)"
    fatal(f"Unknown pipeline '{name}'. Available: {available}", ValueError)
  return _PIPELINES[name]

def build_pipeline(name, **kwargs):
  cls = get_pipeline(name)
  logging.info("Pipeline kwargs: %s", kwargs)
  return cls(**kwargs)

def list_pipelines():
  return sorted(_PIPELINES)
```

### 2.5 `genml_kit/pipelines/images.py` — the image pipeline
```python
"""Single-image pipeline: HF datasets / ImageFolder / ensemble."""

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import register_pipeline


@register_pipeline("images")
class ImagesPipeline(DataPipeline):
  NAME = "images"

  def add_args(self, parser):
    # Data flags moved from train.py: --dataset, --datasets (pretrain),
    # --label_column, --image_column, --image_size, --split, --cache_dir,
    # --sampler, ...
    ...

  def __init__(self, ...): ...

  def build_loader(self, args, mode="train"):
    # Moves build_data()/load_and_split_dataset()/build_pretrain_dataset()
    # from train.py / pretrain/cli.py verbatim.
    # Returns a loader whose items -> DataBlob(data=images, meta={"labels": ...});
    # stores self.train_loader / self.val_loader per mode (§2.3).
    ...
```

### 2.6 `genml_kit/pipelines/vo_pair.py` — the VO pair pipeline
```python
"""VO pair pipeline: synthetic pairs from VOPairDataset."""

from torch.utils.data import DataLoader
from genml_kit.datasets.vo_pairs import VOPairDataset
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import register_pipeline
from genml_kit.utils.seed import seed_worker


@register_pipeline("vo_pair")
class VOPairPipeline(DataPipeline):
  NAME = "vo_pair"

  def add_args(self, parser):
    g = parser.add_argument_group("vo_pair pipeline")
    g.add_argument("--vo_length", type=int, default=20000)
    g.add_argument("--vo_val_length", type=int, default=2000)
    g.add_argument("--pitch_deg", type=float, default=45.0)

  def build_loader(self, args, mode="train"):
    # No constructor args: the pipeline derives everything from args
    # (mirrors today's VO CLI, which builds VOPairDataset from --vo_length /
    # --vo_val_length / --image_size / --pitch_deg / --seed).  This keeps
    # build_pipeline(name) == cls() uniform for both pipelines and makes
    # `pipeline_cls()` in §5.1 valid without kwargs.
    length = args.vo_length if mode == "train" else args.vo_val_length
    ds = VOPairDataset(length=length, size=(args.image_size, args.image_size),
                       seed=resolve_seed(args.seed)
                       + (0 if mode == "train" else 1000),
                       pitch_deg=args.pitch_deg)
    shuffle = (mode == "train")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                        num_workers=args.num_workers, worker_init_fn=seed_worker)
    if mode == "train":
      self.train_loader = loader
    else:
      self.val_loader = loader
    return loader

  def to_device(self, blob, device):
    # data is (image_a, image_b); meta carries gt / residual / terrain (CPU).
    # PIPELINE-SPECIFIC traversal: meta.gt is a NESTED dict of tensors
    # {log_s, theta, t} and must be moved recursively, not just top-level.
    a, b = blob.data

    def _move(v):
      if hasattr(v, "to"):
        return v.to(device, non_blocking=True)
      if isinstance(v, dict):
        return {k: _move(x) for k, x in v.items()}
      return v  # ints (range_bin), strings (terrain) stay on CPU

    moved_meta = {k: _move(v) for k, v in blob.meta.items()}
    return DataBlob((a.to(device, non_blocking=True),
                     b.to(device, non_blocking=True)), moved_meta)
```

---

## 3. Method Registry — reshape `PretrainMethod` into a general "Method"

### 3.1 New base: `genml_kit/methods/base.py` (or reshape in place)
```python
"""Base class for all training methods (supervised + self-supervised)."""

import abc
import collections

import torch

class Method(abc.ABC):
  """Objective side of training: model + loss + metric.

  ``PretrainMethod`` reshapes into this: a method builds the model,
  computes the loss from (model, blob, global_step), and declares the
  metric contract — for supervised (classification, VO) and self-supervised
  (SimMIM, SupCon, DINO, BYOL, IJEPA, VO photometric) alike.
  """

  NAME = ""                 # registry key (matches PretrainMethod.NAME today)

  metric_key = "loss"       # best-checkpoint metric field
  has_metric_improved = staticmethod(lambda old, new: new < old)

  def add_args(self, parser):
    """Register this method's objective/model CLI flags.

    Instance method: the CLI instantiates the method first and calls
    ``method.add_args(parser)``; this is the majority pattern in today's
    methods (simmim/supcon/ijepa/vo_pair) and keeps a single convention
    with pipelines (§2.3).  BYOL/DINO currently declare this as
    ``@classmethod``; they must switch to instance methods.

    Descriptive-CLI convention (v4.2): option VALUES the user types on the
    CLI are descriptive names, not raw ints, e.g. the vo_pair method:
      parser.add_argument("--vo_stage",
                          choices=("supervised", "photometric"),
                          default="supervised")
    and converts to the internal int at parse time (e.g. via a tiny
    ``type=`` mapping to STAGES) so internal code (vo_losses/TrainStep)
    keeps its int contract.  See §5.2 for the full vo_pair resolution.
    """

  @abc.abstractmethod
  def build_model(self, args, device):
    """Return the model.  Model's forward(data) takes the blob.data shape."""

  @abc.abstractmethod
  def train_step(self, model, blob, global_step, *, labels=None):
    """Compute the loss for one (already device-moved) blob.

    Returns ``LossOutput(loss, metrics)``.  The method receives BOTH the
    model and global_step because existing methods perform per-step
    side effects inside their old ``train_step``:

      BYOL   set_train_mode(model, "train"); model.update_momentum(...)
      DINO   set_train_mode(model, "train"); model.update_momentum(...)
      IJEPA  set_train_mode(model, "train")
      SimMIM builds its mask from blob.data inside train_step

    ``labels`` is passed for backward-compat with today's keyword-call
    convention; the images-pipeline usually delivers labels via blob.meta.
    The loop owns AMP / grad-accum / clipping around this call.
    """

  def get_checkpoint_state(self, model, args):
    """Optional: dict of method-owned state persisted to every checkpoint.

    Carried over verbatim from PretrainMethod (BYOL momentum bounds, DINO
    center/EMA bounds, SimMIM mask/decoder config, IJEPA momentum).
    Called by BaseTrainer while saving; merged into the checkpoint dict.
    """
    return {}

  def load_checkpoint_state(self, model, state, args):
    """Optional: restore method-owned state from a checkpoint dict."""

  def evaluate(self, model, loader, device, to_device):
    """Return dict[str, float] of validation metrics (default: loss mean).

    Default implementation iterates *loader*, moves each blob with the
    pipeline's ``to_device`` callback, calls ``train_step`` under
    ``torch.no_grad()`` and returns the mean of each metric key.  Override
    for objective-specific metrics: VO (mce), classification (macro-F1 /
    confusion matrix).
    """
    metrics_acc = collections.defaultdict(float)
    n = 0
    with torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        out = self.train_step(model, blob, 0)
        for k, v in out.metrics.items():
          metrics_acc[k] += float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in metrics_acc.items()}
```

### 3.2 Registry — extend `genml_kit/pretrain/methods/registry.py`
```python
# Rename _METHODS' decorator semantics but keep API:
# register_method(cls) already does: name = cls.NAME; _METHODS[name] = cls.
# Add a builder mirroring classifiers.  Methods take NO constructor args
# (config flows through add_args/args at build time, not __init__ kwargs);
# the bare builder keeps the registry consistent with pipelines.
def build_method(name):
  cls = get_method(name)
  return cls()
```
(No structural change to `register_method` — only the docstring + maybe rename
file to `genml_kit/methods/` if we want it task-agnostic; keep backward-compat
aliases.  Note: because registry keys stay ``cls.NAME`` (the existing
convention), the base class attribute here is ``NAME = ""``, matching
``PretrainMethod.NAME`` today, NOT a lowercase ``name``.)

### 3.3 Ported methods

| Method | name | model | loss | metric |
|---|---|---|---|---|
| classification | `classification` | registry `load_model` | `CombinedFocalLoss`/CE | `macro_f1` (max) |
| VO supervised | `vo_pair` | `VOSimilarityNet` | `vo_losses` (staged) | `mce` (min) |
| VO self-supervised | `vo_pair` (stage=photometric) | `VOSimilarityNet` | `photometric_residual` | `mce` (min) |
| SimMIM | `simmim` | existing | MIM loss | `loss` (min) |
| SupCon | `supcon` | existing | contrastive | `loss` (min) |
| DINO | `dino` | existing | distillation | `loss` (min) |
| BYOL | `byol` | existing | MAE/feature | `loss` (min) |
| IJEPA | `ijepa` | existing | latent-pred | `loss` (min) |

Porting a `PretrainMethod` to `Method` is a rename + reshape:
```python
# before (VOPairMethod.train_step)
def train_step(self, model, batch, global_step, *, labels=None):
  image_a = batch["image_a"]; image_b = batch["image_b"]
  out = model(image_a, image_b)
  residual = photometric_residual(image_a, image_b, out["params"])
  return residual.mean(), {"vo_photo": ...}

# after (VOPairMethod.train_step -> Method.train_step)
def train_step(self, model, blob, global_step, *, labels=None):
  a, b = blob.data            # from VOPairPipeline (data-only, already moved)
  out = model(a, b)           # raw model return; no ModelOutput wrapper
  residual = photometric_residual(a, b, out["params"])
  return LossOutput(loss=residual.mean(),
                    metrics={"vo_photo": residual.mean().detach()})
```

### 3.4 The `--method` flag now covers BOTH supervised and self-supervised
`genml-kit-pretrain` is gone in v4 (§5.2); every run is `genml-kit-train`:
- `genml-kit-train --method classification --pipeline images ...`  (supervised)
- `genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage supervised ...`
  (supervised VO; maps to STAGES["supervised"] == 0)
- `genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage photometric ...`
  (self-supervised VO; maps to STAGES["photometric"] == 1)
- `genml-kit-train --method simmim     --pipeline images ...`     (existing UX)

---

## 4. Generic BaseTrainer — single branchless loop

The loop below is the union of what `train.py:778-810` and `pretrain/cli.py:200-245`
already do: AMP autocast + GradScaler, grad-accumulation with an end-of-epoch
flush, the gradient monitor placed AFTER unscale_ and BEFORE clip, and a
per-epoch scheduler step.  `BaseTrainer.run()` (today `trainer.py:131-207`)
owns the epoch loop; it calls `train_epoch` then `validate`, saves best/latest
checkpoints, and calls `epoch_end()` — which the generic base routes to the
method (see below).

```python
def train_epoch(self, epoch, saver, step, monitor):
  set_train_mode(self.model, "train")          # loop owns train/eval mode
  total, batches = 0.0, 0
  scaler = self.optimization.scaler            # None unless fp16-on-CUDA
  amp_dtype = self.args.amp_dtype
  total_batches = len(self.pipeline.train_loader)

  for step_in_epoch, blob in enumerate(self.pipeline.train_loader):
    blob = self.pipeline.to_device(blob, self.device)   # data AND meta
    with torch.amp.autocast("cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None
                                     and self.device.type == "cuda")):
      loss_out = self.method.train_step(self.model, blob, step)  # method owns loss
      loss = loss_out.loss            # raw, UNSCALED mean batch objective
    grad = loss / self.args.grad_accum_steps   # scale ONLY for grad correctness
    if scaler is not None:
      scaler.scale(grad).backward()
    else:
      grad.backward()

    if ((step_in_epoch + 1) % self.args.grad_accum_steps == 0
        or (step_in_epoch + 1) == total_batches):       # flush partial tail
      if scaler is not None:
        scaler.unscale_(self.optimization.optimizer)
      if monitor is not None:                 # true grads: post-unscale,
        monitor.step(step)                    # pre-clip
      if self.args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self.args.grad_clip)
      if scaler is not None:
        scaler.step(self.optimization.optimizer)
        scaler.update()
      else:
        self.optimization.optimizer.step()
      self.optimization.optimizer.zero_grad(set_to_none=True)
      step += 1

    total += loss.item()          # report RAW loss (see §4 notes)
    batches += 1

  self.writer.add_scalar("train/loss", total / max(batches, 1), epoch)
  return total / max(batches, 1), step

def validate(self):
  # evaluate() owns its mode: the default impl wraps in torch.no_grad();
  # overrides that need eval-mode (simmim reconstructions, VO mce) use
  # model_mode(model, "eval") internally as they do today.
  metrics = self.method.evaluate(self.model, self.pipeline.val_loader,
                                 self.device, self.pipeline.to_device)
  key = self.method.metric_key
  val = metrics[key]
  self.writer.add_scalar(f"val/{key}", val, self.epoch)
  return val
```

**§4 notes (resolved decisions):**

- **Reported epoch loss uses the RAW (undivided) per-micro-batch loss.**
  Both existing loops compensate for accumulation when reporting
  (`train.py:841` multiplies by `grad_accum_steps`, `cli.py:257` does the
  same); the reported number is the mean per-sample loss, not the
  per-optimizer-step scaled gradient.  Dividing the accumulator would
  under-report by a factor of `grad_accum_steps` — do NOT do that.
- **AMP** is exactly the `train.py:778-810` / `cli.py:200-245` structure:
  `torch.amp.autocast("cuda", dtype=..., enabled=...)` around the forward +
  loss, `scaler.scale(loss).backward()`, and inside the accumulation guard
  `scaler.unscale_ → monitor.step → clip → scaler.step/update → zero_grad`.
  `scaler` is `None` unless `amp_dtype == float16` on CUDA
  (`optim_factory.py:74-75`).
- **End-of-epoch flush.** The step condition is the current codebase's
  `(i+1) % accum == 0 or (i+1) == total_batches`; without the second clause a
  loader whose length is not a multiple of `grad_accum_steps` would strand the
  tail gradients (and their `zero_grad`) into the next epoch.
- **Grad clip** uses `args.grad_clip` when `> 0` (`utils/args.py:114-117`,
  default `1.0`).  Today pretrain hardcodes `max_norm=1.0` (cli.py:237/243)
  and IGNORES a user-supplied `--grad_clip`; the unified loop honors it.
  Since the shared default is `1.0`, default behaviour is unchanged — only the
  "user passes `--grad_clip 0`" case now actually disables clipping for
  pretrain-style runs (see §12.Q9).
- **Scheduler step** belongs in `BaseTrainer.run()`, once per epoch, right
  after `train_epoch` returns — the shared position matching today's two
  trainers (`train.py:1213-1214`, `cli.py:713-714`) — so no subclass can
  forget it and the plan's loop above does NOT include it (it is
  `run()`-level, like `epoch_end`).

`BaseTrainer.__init__` becomes (+ epoch_end routing + checkpoint state):
```python
def __init__(self, args, model, method, pipeline, optimization, device, writer, ...):
  ...

def epoch_end(self):                 # BaseTrainer, generic (trainer.py:117-119)
  self.method.on_epoch_end(self.model, self.epoch, self.writer)

def saver_extra(self):               # BaseTrainer, generic
  return self.method.get_checkpoint_state(self.model, self.args)
```
Note the method lifecycle hooks (`on_epoch_end`, `get/load_checkpoint_state`)
are restored from `PretrainMethod` — WITHOUT them BYOL/DINO momentum ramps and
SimMIM/IJEPA resume round-trips silently no-op (see §12.Q2).

---

## 5. Unified CLIs

### 5.1 `genml-kit-train` (supervised + VO) — two-pass parse on BOTH registries
```python
def main(argv=None):
  parser = build_parser()          # generic: --pipeline, --method, --model, --
                                  # image_size, batch_size, grad_accum,
                                  # optimization/checkpoint/logging args...
  known, _ = parser.parse_known_args(argv)

  # ORDER MATTERS: pipeline FIRST, then method.  The pipeline registers the
  # data flags the method's add_args may need to inspect; registering the
  # method before the pipeline could shadow/duplicate shared options
  # (e.g. --vo_stage is defined by the vo_pair METHOD only; no pipeline
  # re-adds it, so the merged binary must guarantee single registration).
  pipeline_cls = get_pipeline(known.pipeline)
  method_cls = get_method(known.method)
  pipeline = pipeline_cls()
  method = method_cls()
  pipeline.add_args(parser)        # data flags (--dataset or --vo_length...)
  method.add_args(parser)          # objective flags (--vo_stage, --loss weights)
  args = parser.parse_args(argv)

  logging.info("Resolved pipeline=%s method=%s", known.pipeline, known.method)

  device = resolve_device(args.device)
  seed_everything(resolve_seed(args.seed), deterministic=True)

  model = method.build_model(args, device)
  pipeline.build_loader(args, mode="train")   # populates pipeline.train_loader
  pipeline.build_loader(args, mode="val")     # populates pipeline.val_loader

  optimization = build_optimization(args, model, device)
  writer = open_writer(args, ...)
  BaseTrainer(args, model, method, pipeline, optimization, device, writer).run()
```

**§5.1 notes:**
- The loader is owned by the **pipeline instance** (`self.train_loader` /
  `self.val_loader`, populated by `build_loader` per §2.3), and the loop reads
  `self.pipeline.train_loader`.  `BaseTrainer` does NOT accept loaders as
  constructor args — it reaches them through the pipeline.
- `--pipeline images --method classification` are the defaults for
  `genml-kit-train` (these replace the old `genml-kit-train` and
  `genml-kit-pretrain` entry points).

### 5.2 `genml-kit-pretrain` is **removed entirely** (no alias, no backward compat)
With v4, "train" vs "pretrain" are **configurations, not programs**:

| axis | train (classification) | train (supervised VO) | pretrain (simmim) | pretrain (vo_pair) |
|---|---|---|---|---|
| pipeline | `images` | `vo_pair` | `images` | `vo_pair` |
| method | `classification` | `vo_pair` | `simmim` | `vo_pair` |
| `--vo_stage` | (n/a) | `supervised` | (n/a) | `photometric` |
| model | headful | `VOSimilarityNet` | headless | `VOSimilarityNet` |
| loss | CE/focal | `vo_losses` (staged) | MIM | photometric residual |
| val | macro-F1 | mce | none | mce |

Every difference is now a `--pipeline` + `--method` choice, so ONE binary
`genml-kit-train` covers all four quadrants; **`genml-kit-pretrain` is deleted**
(console script removed from `pyproject.toml`, `genml_kit/pretrain/cli.py` deleted,
`PretrainTrainer` deleted):

```bash
# supervised classification (formerly genml-kit-train)
genml-kit-train --method classification --pipeline images ...
# self-supervised SimMIM (formerly genml-kit-pretrain --method simmim)
genml-kit-train --method simmim --pipeline images --datasets ... --image_size 224 ...
# supervised VO (default stage supervised; --vo_stage may be omitted)
genml-kit-train --method vo_pair --pipeline vo_pair ...
# supervised VO, explicit
genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage supervised ...
# self-supervised VO (formerly genml-kit-pretrain --method vo_pair --vo_stage 1)
genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage photometric ...
```

> **`--vo_stage` — RESOLVED (v4.2 per joint review): descriptive CLI names,
> int internally, default `supervised`.**
> - CLI surface: `--vo_stage supervised|photometric` (argparse `type=` /
>   `choices`), **default `supervised`**.  Self-documenting and removes the
>   v4.1 default foot-gun (a plain `--method vo_pair` trains the supervised
>   objective — the natural reading of a binary named `train`).
> - Internal contract UNCHANGED: the parse-time conversion yields the int
>   (`STAGES["supervised"] == 0`, `STAGES["photometric"] == 1`);
>   `vo_losses(stage)` and the numeric `stage >= STAGES["photometric"]`
>   comparison (`train_vo.py:51`) are untouched.
> - `tests/test_trainer.py:23` passes the internal int (`vo_stage = 0`)
>   directly, so it keeps working.
> - BREAKING (documented): old numeric invocations `--vo_stage 1` (historic
>   pretrain form) must become `--vo_stage photometric`; old `--vo_stage 0`
>   becomes `--vo_stage supervised`.  Both are within v4 scope since
>   `genml-kit-pretrain` is deleted anyway.

---

## 6. DataBlob Shapes

| Pipeline | `blob.data` | `blob.meta` |
|---|---|---|
| images | `images` `(B,C,H,W)` | `{"labels": (B,), ...}` |
| vo_pair | `(image_a, image_b)` tuple | `{"gt": {...}, "residual": ..., "terrain": ..., "range_bin": ...}` |

### 6.1 `ImagesPipeline` — merging train (`--dataset`) and pretrain (`--datasets`) data paths

**This is NOT a pure copy-paste of `build_data`/`build_pretrain_dataset`** — the two
old CLIs expose *different* data flags and *different* label semantics, and one
`ImagesPipeline` must support both supervised (classification) and self-supervised
(SimMIM/SupCon/...) methods.  Explicit reconciliation:

- **Flag namespace**: keep BOTH spellings.  `--dataset` = a single HF dataset name or
  `imagefolder/PATH` (classification's `load_and_split_dataset`, `train.py:134-271`);
  `--datasets` = a list for the ensemble (`build_pretrain_dataset`, `pretrain/cli.py:77-118`).
  Whichever is present decides the path; if both → `fatal`.
- **Labels**: `ImagesPipeline` does NOT decide whether batches carry labels — the
  **method** does, via a `needs_labels` attribute (today `PretrainMethod.needs_labels`);
  the classification method sets `True`, self-supervised methods `False`.  The
  pipeline reads `method.needs_labels` when building the loader: when `False` the
  existing `FieldSectorDataset` label-stripping is applied; when `True` labels flow
  into `DataBlob.meta["labels"]`.  This replaces the old
  `process=build_pretrain_dataset(needs_labels=method.needs_labels)` toggle.
- **Transform — two-phase contract**: the pipeline owns GENERIC preprocessing
  only (`build_transforms`/`DictFieldTransform` with `fields=("image",)`: resize,
  normalize, basic crop/flip/jitter), NOT objective-defined augmentation.  The
  objective-specific transforms are **method-owned** and stay with the method:
  `DualViewTransform` (BYOL) and `MultiCropTransform` (DINO) live in
  `pretrain/augmentations/` and are referenced from each method's
  `build_transform` (byol.py:44, dino.py:93).  SimMIM does **not** implement
  `build_transform` at all — it builds its mask inside `train_step`
  (`make_mask`, simmim.py:97) — so the old claim "`Method.build_transform`
  (used by SimMIM etc.) moves into the pipeline" was wrong on both counts.
  A method MAY declare an optional `build_transform(blob)` composed on top by
  the pipeline when present; the default is identity.
- **Hard-coded key**: the ensemble `image_column` defaults to `"image"` and is
  consumed at `pretrain/cli.py:202`; in the pipeline this becomes
  `DataBlob(data=images, meta={"labels": ...})` — the loop no longer knows the key.
- **DataLoader**: `num_workers`, `batch_size`, `sampler` (balanced/weighted for
  classification) all live in the pipeline.

Resulting `ImagesPipeline.add_args` exposes: `--dataset`, `--datasets`,
`--label_column`, `--image_column`, `--split`, `--cache_dir`, `--sampler`,
`--image_size`, `--batch_size`, `--num_workers`.

---

## 7. Checkpoint Schema (critical — avoids resume breakage)

TODAY the codebase already uses per-trainer keys: `best_macro_f1`
(`train.py:1160`, `pretrain/cli.py:682`) and `best_mce` (`train_vo.py:160`,
stored POSITIVE — the old negated `best_mce_negated` variant is already gone,
asserted by `tests/test_trainer.py:262`). A generic `BaseTrainer` must **not**
hardcode a metric key.

**New rule:** the checkpoint's best-metric key comes from the **method**:
`method.metric_key` (e.g. `macro_f1`, `mce`, `loss`), and the stored value is
always keyed `best_` + `metric_key` so resume is unambiguous and key-name
independent of direction:

- `best_mce` (minimize), `best_macro_f1` (maximize), `best_loss` (minimize).
- Direction is expressed ONLY by the method's `has_metric_improved`; the loop
  never negates, so `best_mce` holds the positive mean corner error (as today's
  `train_vo.py:160` already does).

`BaseTrainer` reads/writes `best_{method.metric_key}` (and the method's
`has_metric_improved`) on load/save.  **Explicit decision:** old checkpoint files
whose schema differs (`best_macro_f1` under the old key, VO checkpoints that used
`best_mce_negated`) are **not migrated** — the clean break means older checkpoints
need re-saving under the new convention.  (No backward compatibility, per project
decision.)

Negation stays OUT of the loop: `has_metric_improved` (per method) is the only
place direction is expressed, and the stored value is always positive-as-measured
(`mce` stored positive, matching today's `train_vo.py:160`).

---

## 8. CLI Examples

### Supervised VO
```bash
genml-kit-train --pipeline vo_pair --method vo_pair \
  --vo_length 20000 --vo_val_length 2000 --pitch_deg 45.0 \
  --image_size 128 --batch_size 32 --grad_accum_steps 4 \
  --lr 1e-3 --epochs 50 --checkpoint /tmp/vo.pt --log_dir /tmp/vo_logs
```

### Self-supervised VO pre-train (same binary as supervised)
```bash
genml-kit-train --pipeline vo_pair --method vo_pair --vo_stage photometric ...
```

### Classification (unchanged UX)
```bash
genml-kit-train --method classification --pipeline images \
  --model google/vit-base-patch16-224 --dataset ... --label_column ...
```

### SimMIM pre-train (formerly `genml-kit-pretrain --method simmim`)
```bash
genml-kit-train --method simmim --pipeline images --datasets ... --image_size 224
```

---

## 9. Exact Files & Impact

| File | Change |
|---|---|
| `genml_kit/pipelines/{__init__,contracts,base,registry,images,vo_pair}.py` | new (data-only) |
| `genml_kit/methods/__init__.py` | new; `register_method`/`build_method`/`get_method`/`list_methods` exported |
| `genml_kit/methods/base.py` | new; `Method` ABC (= reshaped `PretrainMethod`, renamed) |
| `genml_kit/methods/registry.py` | new; `register_method` decorator + builder (copied from `pretrain/methods/registry.py`) |
| `genml_kit/methods/classification.py` | new; `ClassificationMethod` (loss+metric: `CombinedFocalLoss`/macro-F1) |
| `genml_kit/methods/vo_pair.py` | new; `VOPairMethod` (supervised staged loss / photometric) |
| `genml_kit/methods/simmim.py`, `supcon.py`, `dino.py`, `byol.py`, `ijepa.py` | moved + reshaped from `pretrain/methods/` |
| `genml_kit/pretrain/methods/` | **deleted** (moved to `genml_kit/methods/`) |
| `genml_kit/pretrain/cli.py` | **deleted** (console script removed) |
| `genml_kit/training/trainer.py` | `BaseTrainer(args, model, method, pipeline, optimization, device, writer, ...)`; one loop (§4) |
| `genml_kit/training/train.py` | two-pass `--pipeline`+`--method`; move classification data → `images` pipeline, classification loss → `classification` method |
| `genml_kit/training/vo/train_vo.py` | keep `vo_losses`/`evaluate_vo`; delete `VOTrainer` (loss moves to `vo_pair` method) |
| `genml_kit/models/vo/vo_similar.py` | `forward(data)` tuple reshape |
| `genml_kit/training/infer.py` | **unchanged** (stays separate; consumes checkpoints) |
| `pyproject.toml` | remove `genml-kit-pretrain`; keep `genml-kit-train` + `genml-kit-infer` |
| `genml_kit/pretrain/losses/` | **kept in place** — imported by production code that stays (`train.py:26` imports `genml_kit.pretrain.losses.focal`) |
| `genml_kit/pretrain/augmentations/` | **kept in place** — `DualViewTransform` / `MultiCropTransform` are method-owned (§6.1) and stay importable by the moved methods |
| tests | construct `BaseTrainer(args, model, method, pipeline, ...)` (see §11) |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Circular imports | `pipelines` imports `train_vo` lazily inside the `vo_pair` **method** (`train_step`); `trainer` imports only `pipelines.base`/`methods.base` |
| Classification UX regression | `images` pipeline + `classification` method move code verbatim; defaults keep `--pipeline images --method classification` |
| `--method` backward compat | `register_method`/`get_method`/`list_methods` unchanged; only hooks reshaped |
| Namedtuple picklability | fine (already used) |
| Existing tests construct trainers | update to `BaseTrainer(args, model, method, pipeline, optimization, device, writer, ...)` (§4) |
| Metric direction | `Method.metric_key` + `has_metric_improved` |
| `pretrain/losses` + `pretrain/augmentations` import churn | **kept in place** (§9); only `pretrain/cli.py` + `pretrain/methods/` are deleted |
| `--vo_stage` drift | **RESOLVED**: descriptive CLI names (supervised\|photometric), int internal, default supervised (§5.2) |

---

## 11. Test Plan & Test-Change Inventory

### 11.1 New tests
- **Unit — pipeline registry**: `register_pipeline` decorator; duplicate-name fatal;
  `get_pipeline` unknown fatal; `list_pipelines` sorted; `build_pipeline` instantiates.
- **Unit — method registry**: `register_method`/`build_method`/`get_method`/`list_methods`.
- **Unit — contracts**: `DataBlob` and `LossOutput` namedtuple fields/defaults;
  NO `ModelOutput` here (name is owned by `genml_kit.models.registry`, §2.2).
- **Unit — `ImagesPipeline`**: `build_loader` yields `DataBlob` with `meta["labels"]`
  (supervised) and no labels (self-supervised toggle).
- **Unit — `VOPairPipeline`**: `build_loader` yields `DataBlob` with `data` a 2-tuple
  and `meta["gt"]` keys; `to_device` moves both tensors.
- **Unit — `ClassificationMethod`**: `train_step` on toy logits + labels → finite
  `LossOutput`; `metric_key="macro_f1"` maximize; `evaluate` returns macro-F1/confusion.
- **Unit — `VOPairMethod`**: `train_step` returns finite `LossOutput` with
  `mce`/`vo_photo`; `evaluate` returns `{"mce": ...}`; `metric_key="mce"`
  minimize; `--vo_stage` `supervised` vs `photometric` select different losses.
- **Integration — VO e2e**: `BaseTrainer(method=vo_pair, pipeline=vo_pair)` over
  `length=8, size=(64,64)` trains 1 epoch, saves checkpoint, val `mce` finite,
  resume round-trip under new checkpoint schema (§7).
- **Integration — classification e2e**: `BaseTrainer(method=classification,
  pipeline=images)` over a tiny synthetic HF/ImageFolder dataset, trains 1 epoch,
  saves checkpoint, `best_macro_f1` written; resume works.
- **Integration — self-supervised e2e**: `BaseTrainer(method=simmim, pipeline=images)`
  over a tiny image folder, trains 1 epoch, `metric_key="loss"` written.

### 11.2 Existing tests that MUST change (inventory)
Complete blast radius from `grep -rl "genml_kit.pretrain\|genml_kit.training.vo\|genml_kit.training.train" tests/`:

| File | Why it breaks |
|---|---|
| `tests/test_trainer.py` | constructs `ClassificationTrainer`/`VOTrainer` (`train_vo.py:141`), checks `best_macro_f1`/`best_mce`; must construct `BaseTrainer(method=..., pipeline=...)` per §7 |
| `tests/test_vo_training.py` | imports `STAGES`/`vo_losses`/`photometric_residual` from `training/vo/train_vo.py` — `VOTrainer` deleted; keep `vo_losses`/`evaluate_vo` as module fns, import via `VOPairMethod` |
| `tests/test_vo_e2e.py` | `load_model("vo/npu-small")` + `evaluate_vo` — switch to pipeline + method form |
| `tests/test_vo_pairs.py` | `VOPairDataset` direct — unchanged (dataset contract stable) |
| `tests/test_pretrain_methods.py` | `get_method`/`list_methods`/`SimMIMMethod`/`make_mask` from `pretrain.methods` — imports move to `genml_kit.methods` |
| `tests/test_pretrain_smoke.py` | patched `genml_kit.pretrain.cli.main` (`:135`) + `from genml_kit.pretrain.cli import main` (`:135`), checks method-state restore — CLI deleted; rewrite against `genml-kit-train` |
| `tests/test_pretrain_vis.py` | `from genml_kit.pretrain.cli import log_validation_images` (`:11`) — needs new home (§12.Q10) |
| `tests/test_ensemble_collate.py` | `from genml_kit.pretrain.cli import build_pretrain_dataset, log_validation_images` (`:20`) — ditto |
| `tests/test_seed_utils.py` | `from genml_kit.pretrain.cli import parse_args` (5 sites: 144,153,166,189,201) — CLI deleted |
| `tests/test_signal_utils.py` | `import genml_kit.pretrain.cli` (`:114`) wiring test — CLI deleted |
| `tests/test_headless_models.py` | `from genml_kit.pretrain.methods.ijepa import _PatchEmbedder` (`:111`) — module renamed; needs a public home for `_PatchEmbedder` |
| `tests/test_transformer_encoder_init.py` | `from genml_kit.pretrain.methods.ijepa import _Predictor` (`:43`) — ditto |
| `tests/test_dino.py` | `pretrain.augmentations.multicrop` (`:8`), `pretrain.losses.dino` (`:9`), `pretrain.methods.get_method` (`:10`) — augmentations/losses stay, methods move |
| `tests/test_byol.py` | `pretrain.losses.byol` (`:9`), `pretrain.methods.get_method` (`:10`) |
| `tests/test_supcon_loss.py` | `pretrain.losses.contrastive` (`:6`) |
| `tests/test_pipeline.py` | dataset + preprocessing — keep (data remains stable) |

`pretrain/losses/` and `pretrain/augmentations/` survive (§9), so `test_dino.py`,
`test_byol.py`, `test_supcon_loss.py` only need the method-import path updated.

### 11.3 CLI smoke
- `genml-kit-train --pipeline vo_pair --method vo_pair --help`
- `genml-kit-train --pipeline images --method classification --help`
- `genml-kit-train --method simmim --pipeline images --help` (self-supervised path)

---

## 12. Open Questions (resolved + remaining)

1. **`--method` default** — **RESOLVED**: `classification` (matches old default shape).
2. **Move `PretrainMethod` → `Method`** — **RESOLVED**: move to `genml_kit/methods/`
   wholesale; delete `genml_kit/pretrain/methods/`. No BC alias.
3. **`--pipeline`↔`--method` pairing validation** — **RESOLVED**: no startup
   `fatal` on mismatch (per user decision), BUT the resolved pair is logged at
   `info` on startup (§5.1) so a wrong pairing (e.g.
   `--pipeline images --method vo_pair`) is diagnosable before the batch-0
   shape crash.  Known mismatches fail fast on the first forward anyway; the
   log makes the cause obvious.
4. **`ClassificationMethod` vs `ClassificationTrainer`** — **RESOLVED**: method fully
   owns loss+metric (`CombinedFocalLoss`/macro-F1/confusion); `ClassificationTrainer`
   deleted; no shim.
5. **`evaluate()` optional vs mandatory** — **RESOLVED**: optional.  Default
   averages `train_step` metrics over the loader (§3.1 shows a concrete default);
   only VO (`mce`) and classification (macro-F1/confusion) override.
6. **`genml-kit-pretrain` removal** — **RESOLVED**: remove entirely; delete
   `genml_kit/pretrain/cli.py`, the `genml-kit-pretrain` entry in `pyproject.toml`,
   and `PretrainTrainer`.
7. **Doc revision** — **RESOLVED (required)**: rewrite top-level `README.md`
   "Pre-Training Guide" and every `genml-kit-pretrain` example to unified
   `genml-kit-train --method X --pipeline Y`.  Verified full inventory (re-derive
   with `grep -rn "genml-kit-pretrain"` at implementation time):
     - `README.md`: **492, 518, 591, 632, 1061** (also update the flag tables that
       mention each binary, not just the invocation lines).
     - `scdiag/README.md`: **27, 57, 72** plus the `prepare_*.py` usage prints at
       `scdiag/scripts/prepare_isic.py:201` and `prepare_derm1m.py:14,216`.
     - `vo/README.md` contains **no CLI references** (`grep -rn pretrain vo/` is
       empty) — the earlier "`vo/README.md` references" clause was wrong and is
       dropped.
   No stale references may remain.  This is a **first-class Stage-4 deliverable**,
   not an afterthought.
8. **Unified binary semantics** — **RESOLVED**: ONE binary `genml-kit-train`;
   `genml-kit-pretrain` deleted; `genml-kit-infer` stays separate (consumes
   checkpoints).
9. **Grad-clip default under the unified loop** — **RESOLVED**: `args.grad_clip`
   wins when `> 0`; its shared default is already `1.0` (`utils/args.py:93,116`)
   so default runs are unchanged.  The only behavioral delta: pretrain today
   hardcodes `max_norm=1.0` (`cli.py:237/243`) and ignores `--grad_clip`, so a
   user passing `--grad_clip 0` (disable) or a value ≠ 1.0 will now be honored
   — an intentional unification, matching the supervised path.
10. **New home for moved pipeline helpers** — **RESOLVED**: `build_pretrain_dataset`
   + `log_validation_images` move into the `images` pipeline module
   (`genml_kit/pipelines/images.py`); `build_pretrain_transform` becomes
   `ImagesPipeline.build_transform` and the `Method.build_transform` default
   imports it from there (NOT from the deleted `pretrain/cli.py`).  This gives
   `test_pretrain_vis.py`/`test_ensemble_collate.py` a stable import target.
11. **`Method.add_args` decorator convention** — **RESOLVED**: instance methods
   everywhere; BYOL/DINO switch from `@classmethod` to instance (today
   `byol.py:20-21`, `dino.py:50` are classmethods; the rest instance).