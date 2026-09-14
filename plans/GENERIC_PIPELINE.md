# Generic Pipeline + Method Registry — Detailed Plan (v4)

Status: **DRAFT for review** (no code written yet)
Date: 2026-XX-XX (resumable tomorrow)
Scope: Introduce a **pipeline** abstraction (data only) and **reshape `PretrainMethod`
  into a general "method"** (objective: loss + model + metric), so the training loop
  is data- and objective-agnostic, and VO trains through the **same** harness as
  classification and pre-training.

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
  data = pipeline.to_device(blob.data, device)
  out = model(data)                      # model built by method.build()
  loss_out = method.loss_fn(out, blob)   # method owns the loss
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
method_cls().add_args(parser)
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
  - build_loader(args)           -> train_loader, val_loader
  - to_device(blob_data, device) -> device tensors

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
# the method's loss_fn needs (labels, GT similarity, ...).
DataBlob = collections.namedtuple("DataBlob", ["data", "meta"])

# What a model returns.  predictions is opaque to the loop; the method
# loss_fn decodes it.
ModelOutput = collections.namedtuple("ModelOutput", ["predictions"])

# What method.loss_fn returns.  metrics is dict[str, torch.Tensor] for logging.
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
  Does NOT build models or compute losses (that is Method's job).
  """

  name: str = ""

  @classmethod
  def add_args(cls, parser):
    """Register this pipeline's data CLI flags (mirrors PretrainMethod)."""

  @abc.abstractmethod
  def build_loader(self, args, mode="train"):
    """Return a DataLoader yielding DataBlob namedtuples for *mode*."""

  @abc.abstractmethod
  def to_device(self, data, device):
    """Move blob.data (opaque) onto device.  Return moved data."""
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
  name = "images"

  @classmethod
  def add_args(cls, parser):
    # Data flags moved from train.py: --dataset, --datasets (pretrain),
    # --label_column, --image_column, --image_size, --split, --cache_dir,
    # --sampler, ...
    ...

  def __init__(self, ...): ...

  def build_loader(self, args, mode="train"):
    # Moves build_data()/load_and_split_dataset()/build_pretrain_dataset()
    # from train.py / pretrain/cli.py verbatim.
    # Returns a loader whose items -> DataBlob(data=images, meta={"labels": ...}).
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
  name = "vo_pair"

  @classmethod
  def add_args(cls, parser):
    g = parser.add_argument_group("vo_pair pipeline")
    g.add_argument("--vo_length", type=int, default=20000)
    g.add_argument("--vo_val_length", type=int, default=2000)
    g.add_argument("--pitch_deg", type=float, default=45.0)

  def __init__(self, length, val_length, size, pitch_deg, seed): ...

  def build_loader(self, args, mode="train"):
    ds = VOPairDataset(length=self.length if mode == "train" else self.val_length,
                       size=self.size, seed=self.seed + (0 if mode == "train" else 1000),
                       pitch_deg=self.pitch_deg)
    shuffle = (mode == "train")
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, worker_init_fn=seed_worker)

  def to_device(self, data, device):
    # data is (image_a, image_b)
    return tuple(x.to(device, non_blocking=True) for x in data)
```

---

## 3. Method Registry — reshape `PretrainMethod` into a general "Method"

### 3.1 New base: `genml_kit/methods/base.py` (or reshape in place)
```python
"""Base class for all training methods (supervised + self-supervised)."""

import abc


class Method(abc.ABC):
  """Objective side of training: model + loss + metric.

  ``PretrainMethod`` reshapes into this: a method builds the model,
  computes the loss from (model_output, blob), and declares the metric
  contract — for supervised (classification, VO) and self-supervised
  (SimMIM, SupCon, DINO, BYOL, IJEPA, VO photometric) alike.
  """

  name: str = ""            # registry key (like PretrainMethod.NAME)

  metric_key: str = "loss"  # best-checkpoint metric field
  has_metric_improved = staticmethod(lambda old, new: new < old)

  @classmethod
  def add_args(cls, parser):
    """Register this method's objective/model CLI flags."""

  @abc.abstractmethod
  def build_model(self, args, device):
    """Return the model.  Model's forward(data) takes the blob.data shape."""

  @abc.abstractmethod
  def loss_fn(self, model_output, blob):
    """Return LossOutput(loss, metrics).  blob is understood by the method."""

  def evaluate(self, model, loader, device):
    """Optional: dict[str, float] of val metrics.

    Default: average loss_fn metrics.  Override for e.g. mce / macro-F1.
    """
    raise NotImplementedError
```

### 3.2 Registry — extend `genml_kit/pretrain/methods/registry.py`
```python
# Rename _METHODS' decorator semantics but keep API:
# register_method(cls) already does: name = cls.NAME; _METHODS[name] = cls.
# Add a builder mirroring classifiers:
def build_method(name, **kwargs):
  cls = get_method(name)
  return cls(**kwargs)
```
(No structural change to `register_method` — only the docstring + maybe rename
file to `genml_kit/methods/` if we want it task-agnostic; keep backward-compat
aliases.)

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

# after (VOPairMethod.loss_fn)
def loss_fn(self, model_output, blob):
  a, b = blob.data            # from VOPairPipeline (data-only)
  residual = photometric_residual(a, b, model_output.predictions["params"])
  return LossOutput(loss=residual.mean(), metrics={"vo_photo": residual.mean().detach()})
```

### 3.4 The `--method` flag now covers BOTH supervised and self-supervised
- `genml-kit-train --method classification --pipeline images ...`  (supervised)
- `genml-kit-train --method vo_pair    --pipeline vo_pair ...`    (supervised VO)
- `genml-kit-pretrain --method vo_pair --pipeline vo_pair ...`    (self-supervised)
- `genml-kit-pretrain --method simmim  --pipeline images ...`     (existing UX)

---

## 4. Generic BaseTrainer — single branchless loop

```python
def train_epoch(self, epoch, saver, step, monitor):
  total, batches = 0.0, 0
  for blob in self.pipeline.train_loader:
    data = self.pipeline.to_device(blob.data, self.device)
    out = self.model(data)                        # model built by method
    loss_out = self.method.loss_fn(out, blob)     # method owns loss
    loss = loss_out.loss / self.args.grad_accum_steps
    loss.backward()
    if (step + 1) % self.args.grad_accum_steps == 0:
      if self.args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
      self.optimization.optimizer.step()
      self.optimization.optimizer.zero_grad(set_to_none=True)
      step += 1
    total += loss.item(); batches += 1
  self.writer.add_scalar("train/loss", total / max(batches, 1), epoch)
  return total / max(batches, 1), step

def validate(self):
  metrics = self.method.evaluate(self.model, self.pipeline.val_loader, self.device)
  key = self.method.metric_key
  val = metrics[key]
  self.writer.add_scalar(f"val/{key}", val, self.epoch)
  return val
```

`BaseTrainer.__init__` becomes:
```python
def __init__(self, args, model, method, pipeline, optimization, device, writer, ...):
  ...
```

---

## 5. Unified CLIs

### 5.1 `genml-kit-train` (supervised + VO) — two-pass parse on BOTH registries
```python
def main(argv=None):
  parser = build_parser()          # generic: --pipeline, --method, --model, --
                                  # image_size, batch_size, grad_accum,
                                  # optimization/checkpoint/logging args...
  known, _ = parser.parse_known_args(argv)
  pipeline_cls = get_pipeline(known.pipeline)
  method_cls = get_method(known.method)
  pipeline_cls().add_args(parser)  # data flags (--dataset or --vo_length...)
  method_cls().add_args(parser)    # objective flags (--vo_stage, --loss weights)
  args = parser.parse_args(argv)

  device = resolve_device(args.device)
  seed_everything(resolve_seed(args.seed), deterministic=True)

  pipeline = pipeline_cls()
  method = method_cls()
  model = method.build_model(args, device)
  train_loader = pipeline.build_loader(args, mode="train")
  val_loader   = pipeline.build_loader(args, mode="val")

  optimization = build_optimization(args, model, device)
  writer = open_writer(args, ...)
  BaseTrainer(args, model, method, pipeline, optimization, device, writer).run()
```

`--pipeline images --method classification` are the defaults for `genml-kit-train`
(these replace the old `genml-kit-train` and `genml-kit-pretrain` entry points).

### 5.2 `genml-kit-pretrain` is **removed entirely** (no alias, no backward compat)
With v4, "train" vs "pretrain" are **configurations, not programs**:

| axis | train (classification) | pretrain (simmim) | pretrain (vo_pair) |
|---|---|---|---|
| pipeline | `images` | `images` | `vo_pair` |
| method | `classification` | `simmim` | `vo_pair` (stage=photometric) |
| model | headful | headless | `VOSimilarityNet` |
| loss | CE/focal | MIM | photometric residual |
| val | macro-F1 | none | mce |

Every difference is now a `--pipeline` + `--method` choice, so ONE binary
`genml-kit-train` covers all four quadrants; **`genml-kit-pretrain` is deleted**
(console script removed from `pyproject.toml`, `genml_kit/pretrain/cli.py` deleted,
`PretrainTrainer` deleted):

```bash
# supervised classification (formerly genml-kit-train)
genml-kit-train --method classification --pipeline images ...
# self-supervised SimMIM (formerly genml-kit-pretrain --method simmim)
genml-kit-train --method simmim --pipeline images --datasets ... --image_size 224 ...
# supervised VO
genml-kit-train --method vo_pair --pipeline vo_pair ...
# self-supervised VO (formerly genml-kit-pretrain --method vo_pair --vo_stage 1)
genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage photometric ...
```

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
  **method** does, via a `needs_labels` attribute on the pipeline construction
  (classification `needs_labels=True`, self-supervised `False`).  When `False`, the
  existing `FieldSectorDataset` label-stripping is applied; when `True`, labels flow
  into `DataBlob.meta["labels"]`.  This replaces the old
  `process=build_pretrain_dataset(needs_labels=method.needs_labels)` toggle.
- **Transform**: the pipeline owns the transform (`build_transforms`/`DictFieldTransform`
  with `fields=("image",)`), NOT the method.  `Method.build_transform` (used by
  SimMIM etc.) moves into the pipeline's data-transform responsibility.
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

Old checkpoints store `best_macro_f1` (classification, maximize) and `best_macro_f1`
(pretrain, renamed for shape-compat with a *negated* convention in some historic
versions). A generic `BaseTrainer` must **not** hardcode a metric key.

**New rule:** the checkpoint's best-metric key comes from the **method**:
`method.metric_key` (e.g. `macro_f1`, `mce`, `loss`), and the stored value uses a
direction-neutral name so resume is unambiguous:

- Non-maximizing metrics (VO `mce`, `loss`): stored as-is.
- Maximizing metrics (classification `macro_f1`): stored as `best_` + `metric_key`
  (e.g. `best_macro_f1`), and `has_metric_improved` (maximize) is honored on resume.

`BaseTrainer` reads/writes `best_{method.metric_key}` (and the method's
`has_metric_improved`) on load/save.  **Explicit decision:** old checkpoint files
whose schema differs (`best_macro_f1` for classification, `best_mce_negated` for VO)
are **not migrated** — the clean break means older checkpoints need re-saving under
the new convention.  (No backward compatibility, per project decision.)

**Do NOT store** `best_mce_negated` or any negated variant anymore; negation is gone
from the loop boundary (it lives only in `has_metric_improved`).

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
| `genml_kit/training/trainer.py` | `BaseTrainer(model, method, pipeline, ...)`; one loop |
| `genml_kit/training/train.py` | two-pass `--pipeline`+`--method`; move classification data → `images` pipeline, classification loss → `classification` method |
| `genml_kit/training/vo/train_vo.py` | keep `vo_losses`/`evaluate_vo`; delete `VOTrainer` (loss moves to `vo_pair` method) |
| `genml_kit/models/vo/vo_similar.py` | `forward(data)` tuple reshape |
| `genml_kit/training/infer.py` | **unchanged** (stays separate; consumes checkpoints) |
| `pyproject.toml` | remove `genml-kit-pretrain`; keep `genml-kit-train` + `genml-kit-infer` |
| tests | construct `BaseTrainer(model, method, pipeline, ...)` (see §11) |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Circular imports | `pipelines` imports `train_vo` lazily inside `loss_fn`/methods; `trainer` imports only `pipelines.base`/`methods.base` |
| Classification UX regression | `images` pipeline + `classification` method move code verbatim; defaults keep `--pipeline images --method classification` |
| `--method` backward compat | `register_method`/`get_method`/`list_methods` unchanged; only hooks reshaped |
| Namedtuple picklability | fine (already used) |
| Existing tests construct trainers | update to `BaseTrainer(model, method, pipeline, ...)` |
| Metric direction | `Method.metric_key` + `has_metric_improved` |

---

## 11. Test Plan & Test-Change Inventory

### 11.1 New tests
- **Unit — pipeline registry**: `register_pipeline` decorator; duplicate-name fatal;
  `get_pipeline` unknown fatal; `list_pipelines` sorted; `build_pipeline` instantiates.
- **Unit — method registry**: `register_method`/`build_method`/`get_method`/`list_methods`.
- **Unit — contracts**: namedtuple fields/defaults.
- **Unit — `ImagesPipeline`**: `build_loader` yields `DataBlob` with `meta["labels"]`
  (supervised) and no labels (self-supervised toggle).
- **Unit — `VOPairPipeline`**: `build_loader` yields `DataBlob` with `data` a 2-tuple
  and `meta["gt"]` keys; `to_device` moves both tensors.
- **Unit — `ClassificationMethod`**: `loss_fn` on toy logits + labels → finite
  `LossOutput`; `metric_key="macro_f1"` maximize; `evaluate` returns macro-F1/confusion.
- **Unit — `VOPairMethod`**: `loss_fn` finite `LossOutput` with `mce`/`vo_photo`;
  `evaluate` returns `{"mce": ...}`; `metric_key="mce"` minimize; `--vo_stage`
  supervised vs photometric differ.
- **Integration — VO e2e**: `BaseTrainer(method=vo_pair, pipeline=vo_pair)` over
  `length=8, size=(64,64)` trains 1 epoch, saves checkpoint, val `mce` finite,
  resume round-trip under new checkpoint schema (§7).
- **Integration — classification e2e**: `BaseTrainer(method=classification,
  pipeline=images)` over a tiny synthetic HF/ImageFolder dataset, trains 1 epoch,
  saves checkpoint, `best_macro_f1` written; resume works.
- **Integration — self-supervised e2e**: `BaseTrainer(method=simmim, pipeline=images)`
  over a tiny image folder, trains 1 epoch, `metric_key="loss"` written.

### 11.2 Existing tests that MUST change (inventory)
| File | Current | New |
|---|---|---|
| `tests/test_trainer.py` | constructs `ClassificationTrainer`/`VOTrainer`, checks `best_macro_f1`/`best_mce_negated` | construct `BaseTrainer(method=..., pipeline=...)`; check `best_macro_f1`/`best_mce` under §7 |
| `tests/test_vo_training.py` | `vo_losses`/`evaluate_vo` via `VOTrainer` | via `VOPairMethod` |
| `tests/test_vo_e2e.py` | `load_model("vo/npu-small")`, `evaluate_vo` | pipeline + method form |
| `tests/test_vo_pairs.py` | `VOPairDataset` direct | unchanged (dataset contract stable) |
| pretrain tests (`simmim`/`supcon`/`dino`/`byol`/`ijepa`) | `PretrainMethod` + `PretrainTrainer` | `Method` + `BaseTrainer` with `images` pipeline |
| `tests/test_pipeline.py` | dataset + preprocessing | keep (data remains stable) |

### 11.3 CLI smoke
- `genml-kit-train --pipeline vo_pair --method vo_pair --help`
- `genml-kit-train --pipeline images --method classification --help`
- `genml-kit-train --method simmim --pipeline images --help` (self-supervised path)

---

## 12. Open Questions (resolved + remaining)

1. **`--method` default** — **RESOLVED**: `classification` (matches old default shape).
2. **Move `PretrainMethod` → `Method`** — **RESOLVED**: move to `genml_kit/methods/`
   wholesale; delete `genml_kit/pretrain/methods/`. No BC alias.
3. **`--pipeline`↔`--method` pairing validation** — **RESOLVED**: none. Any combination
   allowed; no `fatal` on mismatch (per user decision).
4. **`ClassificationMethod` vs `ClassificationTrainer`** — **RESOLVED**: method fully
   owns loss+metric (`CombinedFocalLoss`/macro-F1/confusion); `ClassificationTrainer`
   deleted; no shim.
5. **`evaluate()` optional vs mandatory** — **RESOLVED**: optional.  Default averages
   `loss_fn` metrics over the loader; only VO (`mce`) and classification
   (macro-F1/confusion) override.
6. **`genml-kit-pretrain` removal** — **RESOLVED**: remove entirely; delete
   `genml_kit/pretrain/cli.py`, the `genml-kit-pretrain` entry in `pyproject.toml`,
   and `PretrainTrainer`.
7. **Doc revision** — **RESOLVED (required)**: rewrite top-level `README.md`
   "Pre-Training Guide" and every `genml-kit-pretrain` example
   (`README.md:492,518,632`, `scdiag/scripts/*.py` usage prints,
   `vo/README.md` references) to unified `genml-kit-train --method X --pipeline Y`.
   No stale references may remain.  This is a **first-class Stage-4 deliverable**,
   not an afterthought.
8. **Unified binary semantics** — **RESOLVED**: ONE binary `genml-kit-train`;
   `genml-kit-pretrain` deleted; `genml-kit-infer` stays separate (consumes
   checkpoints).