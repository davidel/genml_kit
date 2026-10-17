# Semantic Segmentation Support In genml_kit

Status: **Implemented** (see "Implementation status" below).
Owner: Davidel.
Last updated: this branch (see `git log`).

This document is the detailed implementation plan for adding **semantic
segmentation** to `genml_kit` as a first-class objective, while preserving
the invariants the toolkit already relies on (the "branchless trainer",
registries, method lifecycle hooks, and the two-phase transform contract).

It is written to be read top-to-bottom before any code is touched. Every
integration point below cites the exact file and lines it interacts with, so
the implementation phase should hold **no surprises**.

## Implementation status

The plan below has been implemented on this branch. Deviations from the
original sketch, worth noting for review:

- **Transform ownership.** The method (`SegmentationMethod.prepare_transforms`)
  builds the paired train/val transforms and the pipeline reads them back,
  rather than the pipeline composing them -- this matches the existing
  ``train_transforms`` / ``val_transforms`` contract used by classification.
- **CLI flags.** ``SegmentationPipeline.add_args`` declares only
  ``--mask_column``; ``--dataset`` / ``--image_column`` / ``--val_split`` /
  ``--cache_dir`` are owned by the always-registered ``images`` pipeline
  (argparse option strings are global to the single parser).
- **Shared spec parser.** ``parse_classifier_spec`` and ``parse_decoder_spec``
  both delegate to a new ``genml_kit.utils.cli.parse_inline_spec`` (avoids
  duplicating the ``name:key=value`` logic).
- **Class count inference.** ``wire_data`` infers ``num_classes`` by scanning
  the training masks (excluding the ignore index), so no ``--num_classes``
  flag is required; ``--seg_num_classes`` overrides it.
- **HF dispatch.** ``load_model`` gained a ``task="segmentation"`` parameter
  that selects ``AutoModelForSemanticSegmentation``.
- **Freeze semantics.** The repo's ``--freeze`` lists *trainable* patterns, so
  the CNN-encoder recipe is ``--freeze 'decoder.*'`` (not ``'encoder.*'``).

---

## 0. TL;DR

Add four orthogonal, additive surfaces — plus a small amount of glue —
without touching the driver's control flow:

1. **A new `Method`**: `SegmentationMethod` (`genml_kit/methods/segmentation.py`).
2. **A new `DataPipeline`**: `SegmentationPipeline` (`genml_kit/pipelines/segmentation.py`).
3. **A model-instantiation path**: `seg_wrapper` custom model + a `DECODERS`
   registry + `parse_decoder_spec` (mirrors `training/classifiers/`).
4. **Two data sources**: a `segfolder/<path>` dataset and an HF
   `--mask_column`, plus a `tv_tensors.Mask`-aware paired transform.

Plus: a Dice+CE loss, an mIoU/Dice metric, and tests mirroring the existing
VO e2e test.

The one genuinely new *concept* is the **encoder tap contract**
(`EncoderSpec`): how a decoder author discovers which encoder levels to skip
from. Section 7 is dedicated to it.

---

## 1. Why segmentation, and what "from scratch" means here

Recall the shape story (from the design discussion):

```
classification : (B, 3, H, W)            -> (B, num_classes)
vo_pair        : (B, 3, H, W) x2         -> (B, 8)
segmentation   : (B, 3, H, W)            -> (B, num_classes, H, W)   # per-pixel logits
```

- **Encoder**: a hierarchical backbone producing a feature pyramid
  (ResNet / ConvNeXt / Swin / MiT, or the SMP encoders already used by
  `genml_kit/models/uvito/model.py`).
- **Decoder**: upsamples back to full resolution, fusing deep semantics with
  shallow detail via **skip connections** — always randomly initialized and
  therefore **always trained from scratch**.

Therefore "from scratch" is a spectrum:

| Scenario | Encoder init | Decoder init | Realistic on RTX 4090 (24 GB)? |
|---|---|---|---|
| Fully from scratch | random | random | Yes for small U-Net; needs ~20k–100k masks |
| **Recommended** | ImageNet / SimMIM / DINO / IJEPA | random | Yes — best quality per GPU-hour |
| LoRA fine-tune | frozen + LoRA (transformer encoders only) | random | Yes — cheap |

**Key reuse insight**: the toolkit's existing self-supervised methods
(SimMIM, I-JEPA, DINO — see `genml_kit/methods/{simmim,ijepa,dino}.py`) are
a natural *encoder pretraining front-end* for segmentation. Their checkpoints
are loaded in phase 2 through the existing `--source_checkpoint` machinery
(`genml_kit/methods/base.py:271`, `genml_kit/io/checkpointing.py`). They never
predict segmentation themselves.

---

## 2. Current architecture (the integration surface)

This section records exactly what exists today, so the plan can be precise.

### 2.1 Registry layer

| Registry | File | Registration |
|---|---|---|
| `MODELS` / `PROCESSORS` | `genml_kit/models/registry.py` | `@MODELS.register` |
| `METHODS` | `genml_kit/methods/registry.py` | `@METHODS.register` |
| `PIPELINES` | `genml_kit/pipelines/registry.py` | `@PIPELINES.register` |
| `LOSSES` | `genml_kit/losses/registry.py` | `@LOSSES.register` |
| `CLASSIFIERS` | `genml_kit/training/classifiers/__init__.py` | `@CLASSIFIERS.register` |

All wrap the generic `Registry` in `genml_kit/utils/registry.py`:
`register` (bare / named / keyword), `contains`, `get`, `build`, `list_names`.
`Registry.get` calls `fatal(...)` on an unknown name (fail-fast convention).

### 2.2 Model dispatch

`genml_kit/models/registry.py::load_model(...)` dispatches in this order:

1. registered custom model (`MODELS.contains(parsed.model)`), passing
   `num_labels`, `id2label`, `label2id`, `image_size`, `device`,
   `checkpoint_path`, `backbone`;
2. external script (`load_extern(model_name, "create_model")`) for a `.py`
   path or HTTP(S) URL;
3. HuggingFace `AutoModelForImageClassification` (or `AutoModel` when
   `num_labels == 0`, the timm headless convention).

`parse_model_name` splits the colon syntax
`"model_name:hf_name"` -> `ParsedModelName(model, backbone, processor)`
(`registry.py:32-36`).

### 2.3 The classifier precedent (to be mirrored for decoders)

- `genml_kit/training/classifiers/base.py`: `BaseClassifier(nn.Module)` with
  `forward(hidden_states)` + `extract_features(hidden_states)`; receives
  `(B, N, D)` and does **not** own the backbone.
- `genml_kit/training/classifiers/mlp.py`: reference implementation, registered
  as `"mlp"`.
- `genml_kit/training/classifiers/__init__.py`:
  - `parse_classifier_spec("mlp:hidden=512,dropout=0.3")` ->
    `(name, {"hidden": "512", "dropout": "0.3"})`;
  - `build_classifier(spec, num_labels, hidden_size, **kwargs)` resolves a
    registered name, a `name:kwargs` spec, or a `.py` path/URL defining
    `create_classifier()`;
  - `_register_builtins()` imports bundled heads so their decorators fire.
- `genml_kit/models/cls_model_wrapper/{loader.py,model.py}`: wraps an HF
  backbone + a pluggable classifier; the loader parses `--model_arg classifier=...`.

**This is the exact pattern segmentation's decoder should follow**, and the
model wrapper (`SegModelWrapper`) should be the sibling of `ClsModelWrapper`.

### 2.4 Method contract

`genml_kit/methods/base.py` — `Method(abc.ABC)` declares:

- class attrs: `NAME`, `METRIC_KEY`, `METRIC_MINIMIZE`, `NEEDS_LABELS`,
  `initial_best_metric`;
- `@classmethod add_args(parser)`;
- `prepare_transforms(args, model, device)` — lifecycle position 1, before
  loaders are built; stores `self.train_transforms` / `self.val_transforms` /
  `self.tta_transform` on the method (never into `args`);
- `wire_data(args, pipeline)` — position 2;
- `build_model(args, device)` — position 3; internally calls `apply_extras`
  (grad-checkpointing, LoRA, `--source_checkpoint`, `param_init`, freeze);
- `train_step(model, blob, global_step, *, labels=None) -> LossOutput`;
- `eval_step(...)`, `evaluate(...)`, `has_metric_improved`, `post_train(...)`,
  `build_optimization(...)`, `get_checkpoint_state` / `load_checkpoint_state`.

`LossOutput = namedtuple("LossOutput", ["loss", "metrics", "td_errors"], defaults=[None])`
and `DataBlob = namedtuple("DataBlob", ["data", "meta"])`
(`genml_kit/pipelines/contracts.py`). **`ModelOutput` is deliberately NOT here**
— that name belongs to `models.registry.ModelOutput`.

The reference supervised method is
`genml_kit/methods/classification.py::ClassificationMethod`
(`NAME="classification"`, `METRIC_KEY="macro_f1"`, `NEEDS_LABELS=True`).

### 2.5 Pipeline contract

`genml_kit/pipelines/base.py::DataPipeline(abc.ABC)`:

- `NAME`, `build_loader(args, mode, *, needs_labels=None, **kwargs)` (caches
  `self.train_loader` / `self.val_loader`), `to_device(blob, device)`,
  `build_transform(args, method)`, `init_env(args)`.

`genml_kit/pipelines/images.py` is the reference:
- `add_args` adds the dataset group (`--dataset`, `--datasets`,
  `--label_column`, `--image_column`, `--split`, `--cache_dir`, `--sampler`);
- `build_transform` returns the *pipeline* phase of the two-phase contract;
  `build_loader` composes it with the method's `build_transform`
  (`images.py:296-312`);
- `to_device` moves `DataBlob.data` and `DataBlob.meta`.

### 2.6 Data sources today

- `parse_dataset_specs(names, ...)` (`genml_kit/datasets/factory.py`) maps
  `"imagefolder/<path>"` -> `{"source": "imagefolder"}`, an existing directory
  -> `{"source": "imagefolder"}`, everything else -> `{"source": "hf"}`.
- `ImageFolderDataset` (`genml_kit/datasets/image_folder.py`) — `Path.rglob`
  scan, `_IMAGE_EXTS`, `getitem_retry` wrapping.
- `HFDatasetProxy` (`genml_kit/datasets/hf_proxy.py`) — auto-detects image and
  label columns, `ClassLabel` normalization.
- `DictFieldTransform(ds, t, fields=(...))`
  (`genml_kit/datasets/transforms.py`) — applies one transform to several
  fields; the VO pair pipeline already uses `fields=("image_a","image_b")`.

**There is currently no paired image+mask loader and no mask-aware transform.**

### 2.7 Trainer / driver

- `genml_kit/training/trainer.py::BaseTrainer.run()` is objective-agnostic: it
  owns the loop, AMP, accumulation, checkpointing, signals.
- `genml_kit/training/train.py::main()` orchestrates the lifecycle in a fixed
  order: `prepare_transforms` -> `build_loader` -> `init_env` -> `wire_data`
  -> `build_model` -> resume -> `BaseTrainer(...).run()` -> `post_train`.
  **This order must not change.**

### 2.8 Dependencies & style

- `pyproject.toml` declares `segmentation_models_pytorch>=0.3.0` **but as the
  optional `uvito` extra** (`uvito = ["segmentation_models_pytorch>=0.3.0"]`),
  not in the default `dependencies`. So SMP-based decoders/encoders require
  `pip install genml_kit[uvito]` (or a new `seg` extra). This must be a guarded
  import — as `genml_kit/models/uvito/` already does — so importing the
  package never hard-fails when SMP is absent. The HF `seg_wrapper` path
  (`AutoModelForSemanticSegmentation`) needs no SMP at all.
- `torchvision.transforms.v2` is already used (`training/model_utils.py`,
  `pipelines/images.py`).
- Ruff config lives in `pyproject.toml`; yapf style in `.style.yapf`
  (`based_on_style = google`, `column_limit = 88`, `indent_width = 2`).

---

## 3. Goal & non-goals

### Goals
- Train semantic segmentation via `genml-kit-train --method segmentation`
  using any hierarchical encoder (timm / HF / ConvNeXt / SMP) with a pluggable
  decoder.
- Support HF reference segmentation models (`AutoModelForSemanticSegmentation`)
  so transformer LoRA works out of the box.
- Support local `images/` + `masks/` corpora and HF datasets that carry a mask
  column.
- Keep `--source_checkpoint` (load a SimMIM/DINO/IJEPA encoder), `--freeze`,
  `--lora`, `--grad_checkpoint`, AMP, and checkpoint/resume working unchanged.

### Non-goals (explicitly deferred)
- Instance / panoptic segmentation, Mask2Former query heads.
- Conv2d-LoRA (see §8 — deferred; CNN encoders use `--freeze` instead).
- COCO-JSON rasterization (`CocoSegDataset`) — sketched in §6.4, deferred.
- Distributed / FSDP training.
- A new inference CLI: reuse `training/infer.py` and add an optional
  mask-writing helper in a later PR.

---

## 4. Design principles (the invariants to preserve)

1. **Branchless trainer** — no edits to the `BaseTrainer` loop; the objective
   is expressed purely through `Method`.
2. **No new generic contract type** — do not add `ModelOutput` to
   `pipelines/contracts.py`; masks ride in `DataBlob.meta`.
3. **Orthogonal axes** — backbone, decoder, dataset source, and method are
   independent and each registered/parsed like its classification sibling.
4. **Two-phase transforms** — geometric vs photometric split (§6.2).
5. **Fail-fast** — bad names/specs go through `fatal()`.
6. **Method owns transforms** — never stuff transforms into `args`.
7. **`prepared via registries`** — adding a decoder/dataset never requires
   editing `main()`.

---

## 5. File-by-file change list

### New files

| Path | Purpose |
|---|---|
| `genml_kit/models/decoders/__init__.py` | `DECODERS` registry, `parse_decoder_spec`, `build_decoder`, `_register_builtins` |
| `genml_kit/models/decoders/base.py` | `BaseDecoder(nn.Module)` ABC + `EncoderSpec` protocol |
| `genml_kit/models/decoders/unet.py` | `unet` decoder (skip-based) |
| `genml_kit/models/decoders/fpn.py` | `fpn` decoder |
| `genml_kit/models/decoders/linear.py` | `linear`/`upernet` head (ViT-friendly, dense-predict) |
| `genml_kit/models/seg_model_wrapper/__init__.py` | registration hook |
| `genml_kit/models/seg_model_wrapper/model.py` | `SegModelWrapper(backbone, decoder)` |
| `genml_kit/models/seg_model_wrapper/loader.py` | `seg_wrapper` `@MODELS.register` loader |
| `genml_kit/methods/segmentation.py` | `SegmentationMethod` |
| `genml_kit/pipelines/segmentation.py` | `SegmentationPipeline` |
| `genml_kit/datasets/seg_folder.py` | `SegFolderDataset` (images/ + masks/) |
| `genml_kit/losses/segmentation.py` | `dice_loss`, `dice_ce_loss`, `combined` |
| `genml_kit/training/seg_metrics.py` | `iou_metrics` / confusion accumulation |

### Edited files

| Path | Edit |
|---|---|
| `genml_kit/datasets/factory.py` | recognise the `segfolder/<path>` prefix |
| `genml_kit/datasets/transforms.py` | add `PairedSegTransform` (geometric/photometric split) |
| `genml_kit/datasets/hf_proxy.py` | add `mask_column` parameter |
| `genml_kit/models/__init__.py` | import `seg_model_wrapper` so it registers |
| `genml_kit/models/registry.py` | add `AutoModelForSemanticSegmentation` dispatch |
| `genml_kit/methods/__init__.py` | add `segmentation` to the `_register_builtins()` import tuple |
| `genml_kit/pipelines/__init__.py` | add `segmentation` to the `_register_builtins()` import list |
| `genml_kit/losses/builtin.py` | register `"dice_ce"` in `LOSSES` |
| `genml_kit/losses/__init__.py` | export the new losses |
| `README.md` | document the method + package-layout row |
| `pyproject.toml` | (optional) add a `seg` extra or reuse `uvito` for `segmentation_models_pytorch` — see §2.8 |

---

## 6. Detailed component designs

### 6.1 `SegmentationMethod`

File: `genml_kit/methods/segmentation.py`.

```python
@METHODS.register
class SegmentationMethod(Method):
  NAME = "segmentation"
  METRIC_KEY = "miou"          # maximize
  METRIC_MINIMIZE = False
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self._num_classes = None
    self._ignore_index = 255
    self._id2label = None
    self._criterion = None

  @classmethod
  def add_args(cls, parser): ...
  def prepare_transforms(self, args, model, device): ...
  def build_model(self, args, device): ...
  def train_step(self, model, blob, global_step, *, labels=None): ...
  def eval_step(self, model, blob): ...
  def get_checkpoint_state(self, model, args): ...
  def has_metric_improved(self, new, best): return new > best
```

Key points:

- **`prepare_transforms`** resolves the normalization from the processor
  (via `genml_kit.models.load_processor`, exactly as
  `training/model_utils.py` does today) and stores a `PairedSegTransform`
  (§6.2) on `self.train_transforms` / `self.val_transforms`. It must not write
  into `args`.
- **`build_model`** calls `self.apply_extras(args, model, device)` (the inherited
  helper in `methods/base.py:236-295`) so LoRA/freeze/grad-ckpt/source-checkpoint
  all keep working. It then recovers `num_classes`/`id2label` from the model.
- **`train_step`** reads the *upsampled* logits from `blob`/model output,
  interpolates to the mask's `H,W` if the decoder emits a different grid, and
  computes `dice_ce_loss(logits, mask, ignore_index=...)`.
- **`eval_step`** accumulates a segmentation confusion matrix across the loader
  (see §6.5); `evaluate` (inherited) averages metrics. Metric key `miou`.
- **Mask tensor** lives in `blob.meta["mask"]` (dtype `long`), image in
  `blob.data`. This mirrors `vo_pair` (which puts GT in `meta`).

**Important**: `train_step`'s signature keeps `labels=None`; the mask is
retrieved from `blob.meta`, consistent with `DataBlob` semantics.

### 6.2 Paired transforms (the two-phase geometric/photometric split)

File: `genml_kit/datasets/transforms.py` (new class `PairedSegTransform`).

The rule:

- **Geometric** (resize, random crop, flip, rotate) -> applied to image **and**
  mask, pixel-aligned; masks use **nearest** interpolation.
- **Photometric** (color jitter, blur, normalization, random erasing) -> image
  **only**.

Implementation strategy (preferred): carry the mask as `tv_tensors.Mask` and
build the transform with `torchvision.transforms.v2`; v2 automatically routes
masks to nearest-neighbour and skips photometric kernels. Provide a fallback
path using `DictFieldTransform` with an explicit split of two `Compose` lists
for custom `.py` transforms that are not v2-aware.

Contract for a custom segmentation transform script (mirrors
`create_train_transform`):

```python
def create_seg_transform(image_size, is_train):
    # returns a callable: dict(row) -> dict(row) with "image" and "mask"
    ...
```

Use `--train_augmentation_script` (already a classification flag) or a new
`--seg_augmentation_script` — decision in §9.

### 6.3 `SegmentationPipeline`

File: `genml_kit/pipelines/segmentation.py`.

```python
@PIPELINES.register
class SegmentationPipeline(DataPipeline):
  NAME = "segmentation"

  @classmethod
  def add_args(cls, parser): ...        # --dataset / --mask_column / --seg_ignore_index
  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs): ...
  def to_device(self, blob, device): ...
```

- `build_loader` resolves sources via `parse_dataset_specs` (§6.4) and builds a
  `DataLoader`; the dataset yields `{"image": PIL, "mask": PIL/np.ndarray}`.
- Applies the pipeline transform phase (resize/normalize) then the method
  transform (`self.build_transform` + method's), per §6.2.
- `to_device` moves `blob.data` (image tensor) and `blob.meta["mask"]` to the
  device; subclass/move helper mirrors `VOPairPipeline._move_to_device`
  (`pipelines/vo_pair.py`).
- Collation: default collate works because each item is a dict of tensors —
  mirror `images.py`'s `default_collate`-based collate function.

### 6.4 Data sources

**a) `segfolder/<path>` (covers VOC / Cityscapes / Kvasir / DRIVE / BDD100K…).**

- Extend `parse_dataset_specs` (`datasets/factory.py`) to emit
  `{"source": "segfolder", "name": <path>}` when the name starts with
  `segfolder/`.
- `SegFolderDataset` (`datasets/seg_folder.py`):

```
<root>/images/<stem>.<ext>     or   <root>/<split>/images/...
<root>/masks/<stem>.png
```

  - discover via `Path.rglob` with `_IMAGE_EXTS` (import from `image_folder.py`
    to avoid duplication);
  - match stems case-insensitively; `fatal()` if counts disagree (fail-fast);
  - decode image as RGB, mask as `PIL` "L"/mode-agnostic then to `long`.

**b) HF datasets with a mask column** (e.g. `scene_parse_150`):

- Add `mask_column` to `HFDatasetProxy` (`datasets/hf_proxy.py`): keep the
  existing image/label autodetection untouched, add an optional second image
  field emitted as `"mask"`.
- Add `--mask_column` CLI flag in `SegmentationPipeline.add_args`.

**c) COCO-JSON (deferred).** `CocoSegDataset` rasterizing polygons/RLE from
`segments/*`; not in the first PR.

### 6.5 Losses and metrics

**Loss** (`genml_kit/losses/segmentation.py`), registered as `"dice_ce"` in
`losses/builtin.py`:

```python
def dice_loss(logits, target, ignore_index=255, eps=1.0): ...
def dice_ce_loss(logits, target, weight=1.0, ignore_index=255): ...
```

- `logits` shape `(B, C, H, W)`, `target` `(B, H, W)` long.
- CE ignores `ignore_index`; Dice computed per-class over one-hot, excluding
  ignored pixels.
- Exposed via the existing `load_loss(spec)` (`losses/registry.py`) and the
  global `--loss` / `--loss_args` flags (`genml_kit/utils/args.py:175`), so
  users can pass `--loss dice_ce` or a `--loss_script`-style `.py` path.

**Metric** (`genml_kit/training/seg_metrics.py`):

- Accumulate a `(num_classes, num_classes)` confusion matrix over the val
  loader via fast histogram (`bincount` on `target * C + pred`, masked by
  `ignore_index`), then compute per-class IoU and mIoU + pixel accuracy +
  Dice. Hold it in a small stateful helper `IoUMeter` reset per epoch.
- `SegmentationMethod.evaluate` overrides to use the meter; `METRIC_KEY="miou"`.

### 6.6 CLI surface (summary)

```
--method segmentation
--pipeline segmentation
--dataset segfolder:/data/voc2012          # or an HF repo name
--mask_column segmentations                # HF path
--seg_ignore_index 255
--model seg_wrapper:timm/convnext_tiny.fb_in22k
--model_arg decoder=unet:depth=4,channels=256
--loss dice_ce                            # resolved via the LOSSES registry
--freeze 'encoder.*'                       # CNN encoder adaptation
--lora --lora_r 16                         # transformer encoder adaptation
--source_checkpoint s3://.../simmim_enc.pt # reuse SSL-pretrained encoder
```

---

## 7. The encoder tap contract (`EncoderSpec`) — the one new concept

**Problem**: a decoder author cannot hard-code encoder tap shapes because the
encoder is a runtime string. This is the exact issue already documented in
`genml_kit/models/uvito/model.py:76-79` ("The dummy forward pass is needed
because `encoder_name` is a runtime string, so (c, h, w) cannot be known
statically").

**Solution**: the wrapper probes the encoder and injects concrete dims.

```python
# genml_kit/models/decoders/base.py
@dataclass
class EncoderTap:
  feature: torch.Tensor   # (B, C_i, H_i, W_i)
  stride: int             # 2 ** i

class EncoderSpec:           # implemented by each encoder family adapter
  channels: list[int]        # per-tap channel width, ordered
  out_indices: list[int]

  def features(self, x) -> list[EncoderTap]: ...
  @property
  def num_levels(self) -> int: return len(self.out_indices)
```

Discovery order (cheapest first):

1. **Declared metadata** — SMP `encoder.out_channels` (already read in
   `uvito/model.py:83-87`, where `0` means "not exposed"); timm
   `model.feature_info.channels()` / `reduction()`.
2. **Dummy probe** — forward `torch.randn(1,3,H,W)` in eval/no_grad, read
   `(C,H,W)` per level. Used when metadata is absent (custom encoders).
3. **Lazy modules** — `nn.LazyConv2d` if a decoder wants to avoid the probe.

Reconciliation: the wrapper maps the decoder's desired `num_levels` onto
available encoder levels (select / repeat / upsample). A decoder declares its
intent; the framework supplies dimensions. This is the segmentation analogue of
`parse_classifier_spec`.

**Two encoder families, two accessors** (documented explicitly):

- **Hierarchical** (ResNet/ConvNeXt/Swin/MiT, SMP): a real spatial pyramid;
  taps = reduction indices.
- **Single-scale** (plain ViT, `ConvViT`): no pyramid; taps =
  intermediate transformer layers via `output_hidden_states=True`, reshaped
  `(B,N,D) -> (B,D,h,w)`. Uses the `linear`/`upernet` decoder.

---

## 8. Decoder flexibility & LoRA reality

### 8.1 Pluggable decoders (mirror classifiers)

- `DECODERS` registry; built-ins `unet`, `fpn`, `linear`/`upernet`.
- `parse_decoder_spec("unet:depth=4,channels=256")` mirrors
  `parse_classifier_spec`.
- A `.py` path/URL defines `create_decoder(...)`:

```python
def create_decoder(num_classes, in_channels, image_size, image_mean=None,
                   image_std=None, **kwargs):
    # in_channels == [C_1, C_2, C_3, C_4] probed from the actual encoder
    return MyDecoder(in_channels=in_channels, num_classes=num_classes)
```

### 8.2 `SegModelWrapper`

File: `genml_kit/models/seg_model_wrapper/model.py`.

- Owns `backbone` + `decoder`; `forward(pixel_values) -> ModelOutput(logits)`
  where `logits` is `(B, C, H, W)`.
- Implements `extract_backbone_features` for parity with other wrappers.
- Exposes `num_labels`, `config.id2label` (protocol parity with
  `ClsModelWrapper`/`UVitoAdapter`).

### 8.3 LoRA / freeze

- **Transformer encoders** (SegFormer/ViT/Swin): `apply_lora`
  (`training/model_utils.py:152`) discovers `q_proj/k_proj/v_proj/...`; works
  out of the box. The decoder is **not** LoRA-wrapped (it is trained fully).
- **CNN encoders** (ResNet/EfficientNet): all `Conv2d`; PEFT's `LoraConfig`
  only touches `nn.Linear`, so the encoder gets no adapters. Use
  `--freeze 'encoder.*'` + train the decoder (`apply_freeze_patterns`,
  `training/train_compat.py`). **Conv2d-LoRA is explicitly deferred** (§3);
  if wanted later it is a self-contained addition to `apply_lora`.

---

## 9. Open decisions (need confirmation before coding)

1. **Transform flag**: reuse `--train_augmentation_script` or add
   `--seg_augmentation_script`? (Leaning: reuse, since the contract
   `create_train_transform` is already documented.)
2. **Decoder default**: `unet` for CNN encoders vs `linear` for ViT — should
   the wrapper pick a default based on encoder family, or require an explicit
   `--model_arg decoder=`?
3. **HF seg dispatch trigger**: key off `--task segmentation`, off
   `--method segmentation`, or off the resolved model class? (Leaning: the
   method sets a flag on `args`, but since transforms are prepared before the
   model, we may need an explicit `--task`/`--model_kind`.)
4. **`CocoSegDataset`** in the first PR or a follow-up? (Leaning: follow-up.)
5. **HF `AutoModelForSemanticSegmentation` output**: `.logits` is often
   `(B, C, H/4, W/4)`; confirm we always upsample in `train_step` vs in the
   wrapper. (Leaning: wrapper returns native; method upsamples to mask size.)

---

## 10. Testing plan

Mirror `tests/test_vo_e2e.py`, `tests/test_uvito.py`, `tests/test_classifiers.py`.

| Test file | Coverage |
|---|---|
| `tests/test_seg_transforms.py` | geometric applied to image+mask, **mask uses nearest**; photometric image-only; shapes stay aligned |
| `tests/test_seg_folder.py` | stem matching, RGB image / long mask, `fatal()` on mismatch |
| `tests/test_seg_decoders.py` | registry + `parse_decoder_spec`; `.py` custom decoder; `EncoderSpec` dummy-probe channel discovery |
| `tests/test_seg_losses.py` | Dice/CE on toy masks; `ignore_index` respected; perfect prediction -> IoU 1.0 |
| `tests/test_seg_metrics.py` | confusion accumulation vs sklearn on small arrays |
| `tests/test_seg_e2e.py` | tiny fake encoder+decoder, 2-3 steps, loss decreases, checkpoint round-trips; `--freeze` keeps encoder params unchanged; LoRA path builds (transformer fixture) |
| `tests/test_seg_wrapper.py` | `seg_wrapper` loader parses `--model_arg decoder=...`; logits shape `(B,C,H,W)` |

Mark convergence tests `@pytest.mark.slow` (CI runs `-m 'not slow'`; the repo
uses `--strict-markers` and `filterwarnings = error`).

**Sandbox note**: no GPU here; e2e tests must be CPU-only and tiny.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Mask/image misalignment after augmentation | `tv_tensors.Mask` + explicit geometric/photometric split; dedicated transform test |
| HF `.logits` at fractional resolution | Always upsample to mask size in `train_step`; document |
| Branchless-trainer creep | All logic in `Method`/`Pipeline`; **no** `trainer.py` edits |
| ViT has no pyramid | `linear`/`upernet` decoder using hidden-state taps; documented in §7 |
| LoRA silently adapting nothing (CNN) | Detect "no Linear targets" and `fatal()` with a hint to use `--freeze` |
| Memory on 4090 | AMP already supported; use `--grad_checkpoint`; batch size guidance in README |

---

## 12. Rollout order

1. Losses + metrics (+ tests). Self-contained, no integrations.
2. `segfolder` dataset + `parse_dataset_specs` prefix (+ tests).
3. `EncoderSpec` + decoders + `SegModelWrapper` + loader (+ tests).
4. `SegmentationPipeline` + `PairedSegTransform` (+ tests).
5. `SegmentationMethod` + e2e test.
6. HF `AutoModelForSemanticSegmentation` dispatch + `--mask_column`.
7. README + package-layout table update.
8. (Deferred) `CocoSegDataset`, Conv2d-LoRA, inference mask writer.

Each step is independently reviewable and leaves `main()` untouched.

---

## 13. References

- Repo README: "How It Works", "The Method Lifecycle", "Package Layout".
- `genml_kit/models/uvito/model.py` — SMP encoder feature pyramid (the tap
  reference) and the dummy-probe rationale.
- `genml_kit/models/cls_model_wrapper/` — the wrapper+head pattern to mirror.
- `genml_kit/training/classifiers/` — the spec parser to mirror.
- `genml_kit/pipelines/vo_pair.py` + `genml_kit/datasets/vo_pairs.py` — the
  paired-data precedent.
- Vo README: staged supervised objective (structural precedent for a
  non-classification method).
