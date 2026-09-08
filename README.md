# genml_kit

[![CI](https://github.com/davidel/genml_kit/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/genml_kit/actions/workflows/ci.yml)

A general-purpose toolkit for image classification. Supports self-supervised
pre-training, supervised fine-tuning, and XGBoost ensemble inference — all
from the command line.

`genml_kit` is domain-agnostic: point it at any HuggingFace dataset, local
`ImageFolder` tree, or timm/HuggingFace backbone. It grew out of a
skin-lesion classification project; the dermoscopy-specific workflow (dataset
preparation, tuned recipes) is documented separately in
[`scdiag/README.md`](scdiag/README.md).

## Contents

- [Why genml_kit?](#why-genml_kit)
- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pre-Training Guide](#pre-training-guide)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [Inference Guide](#inference-guide)
- [Tips & Pitfalls](#tips--pitfalls)
- [Gradient Monitor](#gradient-monitor)
- [Custom Models](#custom-models)
- [References and Further Reading](#references-and-further-reading)
- [Development](#development)
- [License](#license)

## Why genml_kit?

Practitioners face two recurring problems: **labeled data is scarce** and
**off-the-shelf models are not domain-specific**. A ViT pre-trained on
ImageNet can classify cats and dogs, but specialized imagery — medical,
satellite, industrial, scientific — looks nothing like natural photos, and
the feature distributions are fundamentally different.

genml_kit solves this with a two-stage pipeline:

1. **Pre-train** on large, often unlabeled image collections (or a labeled
   superset) to learn domain-appropriate visual features.
2. **Fine-tune** on your smaller labeled dataset, starting from those
   pre-trained features instead of random initialization.

This consistently outperforms training from scratch, especially when your
labeled dataset has fewer than ~5 000 images. The tool also supports ensemble
inference with XGBoost on top of the learned features, which can squeeze out
additional performance for deployment.

> **Concrete example:** the dermoscopy workflow this toolkit was extracted
> from — preparing HAM10000 / ISIC / Derm1M corpora, per-method
> hyperparameters, and a worked pre-train → fine-tune run — is documented in
> [`scdiag/README.md`](scdiag/README.md).

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pre-Training                             │
│  Unlabeled/labeled images                                       │
│  ──────────────────────────────────►  Encoder with learned      │
│  SimMIM / I-JEPA / SupCon              visual features          │
└────────────────────────────┬────────────────────────────────────┘
                             │  encoder weights
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Fine-Tuning                               │
│  Small labeled dataset  ──────────►  Trained classifier         │
│  + pre-trained encoder                for your task             │
└────────────────────────────┬────────────────────────────────────┘
                             │  backbone features
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     (Optional) Ensemble                         │
│  Backbone features  ──────────►  XGBoost on top of the         │
│                                   learned representations      │
└─────────────────────────────────────────────────────────────────┘
```

**Pre-training** teaches the model to understand the target imagery — textures,
boundaries, colour patterns, and spatial relationships. **Fine-tuning** adapts
that understanding to your specific classification task (e.g. disease vs.
healthy, defective vs. passing). **Ensemble inference** (optional) trains a
tree-based model on the same features, which sometimes generalises better
than a linear head for small datasets.

### A useful mental model

The encoder turns an image into a vector of features. During pre-training, we
choose an artificial task whose answer can be obtained from the images
(or, for SupCon, from their labels). The encoder learns parameters theta that
make this task easy. During fine-tuning, a classifier is attached to the
encoder and the whole model, or a selected part of it, is adapted to the real
labels:

```text
image x  ──► encoder f_theta(x)  ──► classifier g_phi  ──► class probabilities
                    │                         │
             reusable features          task-specific boundary
```

Pre-training and fine-tuning are not two names for the same job. Pre-training
shapes a useful representation; fine-tuning decides how that representation
should be used for the target labels. If the pre-training data is visually
related to the target data, this gives the classifier a much better starting
point than random initialization. If the domains are very different, use a
smaller learning rate for the backbone and validate carefully.

For further reading, see the original [SimMIM paper](https://arxiv.org/abs/2111.09886),
[I-JEPA paper](https://arxiv.org/abs/2301.08243), and
[Supervised Contrastive Learning paper](https://arxiv.org/abs/2004.11362).
The [timm documentation](https://huggingface.co/docs/timm/index) and the
[Hugging Face image classification guide](https://huggingface.co/docs/transformers/tasks/image_classification)
are useful references when selecting a backbone or processor.

## Package layout

| Module | Contents |
|---|---|
| `genml_kit.training` | `train.py` and `infer.py` CLI harnesses, `optim_factory.py` (optimizers, LLRD, schedulers), `model_utils.py` (loading, freezing, feature extraction), `param_align.py`, `eval.py` + `metrics.py`, `grad_monitor.py`, `train_reporting.py`, `tta.py`, `xgb_utils.py` + `xgb_pipeline.py`, `classifiers/` (pluggable heads) |
| `genml_kit.pretrain` | `cli.py` harness; `methods/` (SimMIM, I-JEPA, DINO, BYOL, SupCon via one registry); `losses/`; `augmentations/` (multi-crop, dual-view) |
| `genml_kit.models` | model/processor registry; `timm/`, `convvit/`, `uvito/`, `cls_model_wrapper/` backends; `processors/base.py` |
| `genml_kit.datasets` | `hf_proxy.py` (HuggingFace → PyTorch bridge), `image_folder.py`, `ensemble.py`, `field_dataset.py`, `balanced_sampler.py`, `weighted_sampler.py`, `retry.py` |
| `genml_kit.io` | `checkpointing.py` (atomic saves, LoRA state, remote fetch), `storage_utils.py` (S3 / GCS / R2) |
| `genml_kit.utils` | glog-style logging, CLI arg groups, seeding, signal handling, GPU info, tables, external `.py` script loading, `image_dump`, transformer init helpers |

## Installation

```bash
pip install genml_kit

# With timm model support:
pip install "genml_kit[timm]"

# With GCS checkpoint sync:
pip install "genml_kit[gcs]"

# With AWS S3 / Cloudflare R2 checkpoint sync (both use boto3):
pip install "genml_kit[s3]"

# With LoRA fine-tuning:
pip install "genml_kit[lora]"

# With UVito model support:
pip install "genml_kit[uvito]"

# Everything above in one shot (gcs, s3, lora, timm, uvito):
pip install "genml_kit[all]"
```

**Requirements:** Python ≥ 3.9, PyTorch, torchvision, transformers, datasets,
NumPy, scikit-learn ≥ 1.3, XGBoost ≥ 2.0, Pillow, tensorboard.

## Quick Start

The fastest way to get started:

```bash
# Fine-tune a ViT on an image-classification dataset (5 epochs, ~2 minutes on GPU)
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset cifar10 \
                --label_column label \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```

HuggingFace ViT backbones use fixed 224x224 position embeddings and do not
interpolate them, so they require `--image_size 224`.  The toolkit default of
448 targets size-flexible backbones such as ConvNeXt, timm EVA, or ConvViT,
which pool or interpolate the token sequence to the input size.

This downloads the model and dataset from HuggingFace, trains for 5 epochs,
and saves `genml_kit_latest.pt` and `genml_kit_best.pt`. See
[Pre-Training Guide](#pre-training-guide) below for the full pipeline
(starting with pre-training before fine-tuning).

---

## Pre-Training Guide

Pre-training learns general visual features from large datasets *before* you
fine-tune on your specific task. This is especially valuable wherever labeled
data is expensive to obtain but raw images are available in bulk — medical
imaging, remote sensing, industrial inspection, scientific imaging.

genml_kit supports three pre-training methods, each with different strengths:

### Choosing Your Method

| Method | Needs labels? | Best when… | Key idea |
|---|---|---|---|
| **SimMIM** | No | You have large unlabeled datasets; want a simple, proven approach | Mask 60% of image patches, train the model to reconstruct the raw pixels |
| **I-JEPA** | No | You want faster training and better downstream transfer than SimMIM | Predict *representations* of masked regions, not raw pixels — avoids learning noise |
| **SupCon** | Yes | You have labels and want representations that cluster by class | Pull same-class images together, push different classes apart in feature space |

### How Each Method Works

The three methods differ mainly in what they call a correct answer. SimMIM
asks for pixels, I-JEPA asks for features, and SupCon asks for relative
positions in feature space. That distinction matters: pixel reconstruction can
spend effort reproducing colour and high-frequency detail, while contrastive
learning spends effort making classes separable.

**SimMIM** (Masked Image Modelling): Randomly masks ~60% of image patches and
trains a lightweight decoder to reconstruct the original pixels. The encoder
must learn to understand textures, boundaries, and spatial context from just
40% of the image. Think of it as a "fill in the blanks" exercise for vision
models. Good default choice when you have lots of unlabeled images.

For an image split into patches x_1, ..., x_N, let M be the set of masked
patch indices and x_hat_i the decoder prediction. SimMIM minimizes mean
squared error over masked patches:

$$
L_{\text{MIM}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| \hat{x}_i - x_i \right\|_2^2
$$

Here x_i is the original patch, x_hat_i is the predicted patch, and |M| is
the number of masked patches. Only masked patches contribute to the loss;
otherwise copying visible pixels would make the task too easy. A higher
`--mask_ratio` supplies less context and creates a harder task, but an
excessively high ratio can make reconstruction ambiguous.

**I-JEPA** (Joint-Embedding Predictive Architecture): Also masks patches, but
instead of reconstructing pixels, it predicts the *latent representation* of
the masked region from the visible context. This avoids wasting capacity on
pixel-level noise (e.g. exact JPEG compression artifacts) and learns more
transferable features. Uses a teacher–student setup with EMA momentum ramping.

Let f_theta be the student encoder and f_xi the teacher encoder. The predictor
q_theta receives visible context and predicts a target representation
z_j = f_xi(x_j) for a masked region. The objective is representation-space
regression:

$$
L_{\text{IJEPA}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\| q_\theta(f_\theta(\text{context}))_i - \mathrm{stopgrad}(f_\xi(x_i)) \right\|_2^2
$$

`stopgrad` means that the teacher target is treated as fixed while updating
the student. The teacher is not optimized by backpropagation; it follows the
student with an exponential moving average:

$$
\xi \leftarrow m \, \xi + (1 - m) \, \theta
$$

Here m is `--teacher_momentum`. A high m changes the teacher slowly and gives
more stable targets. The asymmetric teacher update and masking are important:
two networks simply trained to copy each other could collapse to a constant
vector.

**SupCon** (Supervised Contrastive Learning): Uses labels to define "positive"
pairs (same class) and "negative" pairs (different classes). The loss pulls
features of same-class images together and pushes different-class features
apart. Produces a feature space where similar images naturally cluster.
Requires a `ContrastiveEncoder` (backbone + projection head) and balanced
batch sampling to ensure each batch has enough same-class pairs.

For normalized projections z_i = f_theta(x_i) / ||f_theta(x_i)||_2, the
similarity of examples i and j is their dot product divided by temperature tau:

$$
s_{ij} = \frac{z_i^\top z_j}{\tau}
$$

The positive set for anchor i is P(i) = {j: j != i and y_j = y_i}, where y_i
is the class label. SupCon averages the log-softmax probability assigned to
those positives:

$$
\mathcal{L}_i = -\frac{1}{|\mathcal{P}(i)|} \sum_{p \in \mathcal{P}(i)} \log \left( \frac{\exp(s_{ip})}{\sum_{a \neq i} \exp(s_{ia})} \right)
$$

$$
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_i
$$

B is the batch size. Lower temperature makes the distribution sharper: this
can help separate hard negatives but can also make optimization less stable.
`--samples_per_class` matters because an anchor needs another example of its
class to have a positive. A class represented once contributes no useful
SupCon term for that anchor.

### Typical Hyperparameters

These are reasonable starting points. Tune from here based on your dataset
size and GPU memory:

| Parameter | SimMIM | I-JEPA | SupCon |
|---|---|---|---|
| `--image_size` | 448 | 448 | 448 |
| `--batch_size` | 32 | 32 | 64 |
| `--lr` | 1e-4 | 1e-4 | 1e-4 |
| `--epochs` | 200 | 200 | 100 |
| `--scheduler` | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| `--amp_dtype` | bfloat16 | bfloat16 | bfloat16 |
| `--mask_ratio` | 0.6 | — | — |
| `--teacher_momentum` | — | 0.996→1.0 | — |
| `--temperature` | — | — | 0.07 |
| `--samples_per_class` | — | — | 16 |

**Tips:**
- Start with 200 epochs for SimMIM/I-JEPA. SupCon converges faster (~100).
- `--temperature 0.07` is the standard from the original SupCon paper.
  Lower = sharper contrastive distribution; try 0.05–0.1.
- `--samples_per_class 16` with `--batch_size 64` gives 4 classes per batch
  on a 7-class dataset. Adjust so batch_size is divisible by
  samples_per_class × num_classes.
- Use `--amp_dtype bfloat16` if your GPU supports it (Ampere+). Otherwise
  `float16` with GradScaler works too.

### Example: Full Pre-Training Pipeline

```bash
# Step 1: Pre-train with SimMIM on two large image collections
genml-kit-pretrain --method simmim \
                   --model convvit \
                   --datasets "imagefolder/raw-photos" "imagefolder/more-photos" \
                   --image_size 448 \
                   --batch_size 32 \
                   --epochs 200 \
                   --lr 1e-4 \
                   --scheduler CosineAnnealingLR \
                   --sched_arg T_max=200 --sched_arg eta_min=1e-6 \
                   --amp_dtype bfloat16 \
                   --checkpoint ./checkpoints/convvit_simmim

# Step 2: Fine-tune on your labeled dataset
genml-kit-train --model convvit \
                --dataset my-org/labeled-photos \
                --label_column category \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100 \
                --lr 3e-5 \
                --batch_size 32 \
                --amp_dtype bfloat16
```

### Example: Supervised Contrastive Pre-Training

```bash
genml-kit-pretrain --method supcon \
                   --model convvit \
                   --datasets my-org/labeled-photos \
                   --label_column category \
                   --image_size 448 \
                   --batch_size 64 \
                   --samples_per_class 16 \
                   --proj_dim 128 \
                   --temperature 0.07 \
                   --epochs 100 \
                   --lr 1e-4 \
                   --amp_dtype bfloat16 \
                   --checkpoint ./checkpoints/convvit_supcon

# Then fine-tune as above with --source_checkpoint ./checkpoints/convvit_supcon_latest.pt
```

### Pre-Training CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--method` | `simmim` | Pre-training method. Choices: `simmim`, `ijepa`, `supcon`. |
| `--model` | `convvit` | Model name registered in genml_kit or HuggingFace model ID. |
| `--datasets` | (required) | Space-separated dataset names or local paths. |
| `--cache_dir` | `None` | HuggingFace cache directory for downloads. |
| `--remote_checkpoint` | `None` | Remote URI for checkpoint sync (`gs://BUCKET/PREFIX`, `r2://BUCKET/PREFIX`, or `s3://BUCKET/PREFIX`). |
| `--hf_token` | `None` | HuggingFace token for gated datasets (or set `HF_TOKEN` env var). |
| `--image_column` | auto-detected | Explicit HF image column name. |
| `--label_column` | `None` | Explicit HF label column name. Required by `--method supcon` if non-standard. |
| `--strict_datasets` | `False` | Abort on first dataset-loading failure instead of skipping. |
| `--image_size` | `448` | Input image size (square). |
| `--batch_size` | `32` | Per-GPU batch size. |
| `--seed` | `42` | RNG seed for data shuffling, batch sampling, and dropout. Pass the same value to reproduce a run. See [Reproducibility](#reproducibility). |
| `--deterministic` | `False` | Enable deterministic algorithms (cuDNN deterministic mode, benchmark off). Costs throughput; ops without a deterministic CUDA kernel warn instead of failing. |
| `--epochs` | `200` | Total pre-training epochs. |
| `--lr` | `1e-4` | Peak learning rate for AdamW. |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16`. Omit to disable. |
| `--num_workers` | `4` | DataLoader worker processes. |
| `--device` | auto-detect | Device: `cpu`, `cuda`, or `cuda:INDEX`. |
| `--resume` | `True` | Auto-resume from latest checkpoint. Use `--no-resume` to disable. |
| `--state_save` | `opt,sched` | States to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched` | States to restore on resume: `opt`, `sched`, `amp`, `none`. |
| `--checkpoint` | (required) | Checkpoint path prefix (saves `_latest.pt` and `_best.pt`). |
| `--log_level` | `INFO` | Minimum logging level. |
| `--log_targets` | `STDERR` | Comma-separated log destinations. `STDERR` logs to standard error; any other entry is a log file path (appended). Example: `STDERR,/tmp/train.log` logs to both. |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps; `-1` disables. See [Gradient Monitor](#gradient-monitor). |
| `--norm_history` | `0` | Keep last N norm snapshots per parameter for trend analysis. |
| `--trend_top_n` | `10` | Show top N params in trend table by abs change %. `0` = show all. |
| `--grad_clip` | `1.0` | Maximum gradient norm for clipping. `0` disables. |
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. |
| `--llrd_decay` | `None` | Layer-wise learning rate decay factor. |
| `--vis_every` | `0` | Log reconstruction visualisation every N steps (SimMIM only). |
| `--save_every` | `500` | Save checkpoint every N optimizer steps. 0 disables. |
| `--model_arg` | `{}` | Override model configuration (repeatable). |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name or `.py` script path. |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name or `.py` script path. |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). |
| `--source_checkpoint` | `None` | Path to source checkpoint to absorb parameters from. |
| `--param_rename` | `None` | Regex-based key rename patterns (`SEARCH;REPLACE`). |
| `--grad_checkpoint` | `False` | Enable gradient checkpointing. Reduces activation memory by ~40-50% at the cost of ~25-35% more compute per step. Enables larger batch sizes. See [Gradient Checkpointing](#gradient-checkpointing). |

**SupCon-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--proj_dim` | `128` | Output dimensionality of the projection head. |
| `--proj_hidden` | `None` | Hidden layer size of the projection MLP. `None` = single linear layer. |
| `--temperature` | `0.07` | NT-Xent temperature. Lower = sharper contrastive distribution. |
| `--samples_per_class` | `16` | Samples per class in each batch. Batch size should be divisible by this. |

### Dataset Ensemble

`genml-kit-pretrain` stitches multiple datasets into a single pre-training
corpus. This is useful because no single dataset is large enough for
effective pre-training on its own.

Supported dataset types:
- **HuggingFace datasets** — any HF dataset ID that returns decoded image
  data (e.g. `cifar10`, `food101`). Gated datasets require `--hf_token` or
  `HF_TOKEN`.
- **Local image directories** — pass a path to a folder of images
  (ImageFolder format).

Datasets are loaded lazily (only when first accessed). By default, datasets
that fail to load are logged and skipped (best-effort mode). Use
`--strict_datasets` to abort on the first failure.

Images that cannot be decoded are skipped with a warning — this prevents a
single corrupted file from blocking an entire pre-training run.

#### Label Validation

When using `--method supcon` (or any future label-aware method), the ensemble
validates that every dataset supports labels. Datasets without a label column
cause a clear error *before* training begins, not a cryptic runtime failure
mid-epoch.

Labels are automatically remapped to a shared global label space across all
datasets, so mixing datasets with overlapping but differently-named classes
works transparently.

### Preparing Datasets

Some datasets store images inside zip archives or need custom preprocessing
before they can be used for pre-training. The toolkit consumes any local
ImageFolder directory, so a small preparation script is all it takes:

```bash
python my_prepare_script.py --output_dir ./prepared_images
```

Then use the extracted directory as a local dataset:

```bash
genml-kit-pretrain --datasets ./prepared_images ./other-images \
                   --image_size 448 --batch_size 32 ...
```

For a concrete worked example of such a preparation script, see
`scdiag/scripts/prepare_derm1m.py` and `scdiag/scripts/prepare_ham10000.py`
in the repository (dermoscopy corpora).

---

## Fine-Tuning Guide

After pre-training (or directly, if you skip pre-training), fine-tune a
classifier on your labeled dataset.

### What fine-tuning is changing

Suppose the encoder produces h = f_theta(x). A linear classification head
computes logits a = W h + b, and softmax turns them into probabilities:

$$
p(y=c \mid x) = \frac{\exp(a_c)}{\sum_k \exp(a_k)}
$$

Training minimizes cross-entropy, -log p(y | x), over labeled examples. The
new classifier head is normally initialized from scratch because its output
size depends on the target classes. When loading a pre-training checkpoint,
the useful part to transfer is the encoder; an unused SupCon projection head
should not be copied into the classifier.

The practical choice is how much of theta to update:

- **Full fine-tuning** updates encoder and head. It gives the model the most
  freedom, but needs enough data and a conservative learning rate.
- **Frozen-backbone training** updates only the head. It is a useful baseline
  for small datasets and shows how much information the representation holds.
- **LLRD** updates all layers but gives early layers smaller learning rates.
  This is often a good compromise when the pre-training domain differs from
  the fine-tuning domain (e.g. ImageNet to specialized imagery).
- **LoRA** freezes the original matrices and learns small low-rank updates.
  It is useful when GPU memory or labeled data is limited.

Compare these strategies on the same validation split. The lowest training
loss is not necessarily the best model: monitor macro-F1, balanced accuracy,
weighted F1, and per-class precision and recall, especially for minority
classes.

### Evaluation Metrics

Validation reports include:

- **Top-1 accuracy:** the percentage of validation images assigned the correct
  class by the highest-probability prediction.
- **Precision:** for a class, the fraction of images predicted as that class
  that truly belong to it: `TP / (TP + FP)`.
- **Recall:** for a class, the fraction of images belonging to that class that
  are predicted correctly: `TP / (TP + FN)`.
- **F1:** the harmonic mean of precision and recall:
  `2 * precision * recall / (precision + recall)`. When the denominator is
  zero, scikit-learn's `zero_division=0` behavior reports zero.
- **Macro F1:** the arithmetic mean of the per-class F1 scores. Every class
  contributes equally, regardless of its validation-set size. This is the
  metric used for best-checkpoint selection.
- **Weighted F1:** the mean of per-class F1 scores weighted by each class's
  validation support. It is therefore more influenced by common classes.
- **Balanced accuracy:** the arithmetic mean of per-class recall. It gives
  each class equal weight and is useful for imbalanced datasets.
- **Support:** the number of true validation examples for a class.

The confusion matrix uses rows for true classes and columns for predicted
classes. Diagonal entries are correct predictions. The compact confusion
summary reports each class's recall and its largest confusion destinations.


When TTA is enabled, validation loss is computed from the original image only,
while Top-1 accuracy, balanced accuracy, macro F1, weighted F1, and per-class
metrics use probabilities averaged over the original image and augmented
views. The log also reports original-view metrics and the change produced by
TTA.

Training metrics may be measured on an augmented or weighted-sampler stream.
They should not be compared directly with validation metrics unless they use
the same sampling and preprocessing scheme.

### Basic Fine-Tuning

```bash
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 448
```

### With a Custom Classifier Head

Replace the default linear head with a custom MLP or attention-based
classifier:

```bash
# Freeze backbone, train only the custom head
genml-kit-train --model cls_model_wrapper:google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --classifier mlp \
                --classifier_args hidden=512 dropout=0.3 \
                --freeze ".*\.(head|pool)"
```

### With LoRA (Parameter-Efficient Fine-Tuning)

Freeze the entire backbone and train only small low-rank adapter matrices.
Reduces trainable parameters by ~97% while often matching full fine-tuning:

```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --lora --lora_r 16 --lora_alpha 32 \
    --lora_target_modules "query,key,value" \
    --freeze "classifier\.(head|pool|encoder)" \
    --lr 3e-5 \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --epochs 20
```

### From a Pre-Trained Checkpoint

Load encoder weights from a pre-training run (SimMIM, I-JEPA, or SupCon):

```bash
genml-kit-train --model convvit \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100
```

The backbone weights are loaded automatically; the classifier head is
reinitialised (different `num_classes`). Use `--state_load none` to avoid
carrying over old optimizer/scheduler states.

### Layer-wise Learning Rate Decay (LLRD)

LLRD is a compromise between freezing the backbone and updating every layer at
the same speed. If layers are indexed from shallow 0 to deep L, a common
schedule is:

$$
\text{lr}(\text{layer}) = \text{lr}_{\text{base}} \cdot d^{\,L - \text{layer}}
$$

where d is `--llrd_decay`, usually between 0.8 and 1.0. The deepest layer
receives the base rate while earlier layers receive smaller updates. Early
layers tend to represent general edges and textures; later layers are more
task-specific. This is a useful heuristic, not a law, so validate it on your
dataset. A very small decay factor can effectively freeze the shallow network.

### Mixup, label smoothing, and imbalance

Mixup forms a virtual example from two training examples:

$$
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where lambda ~ Beta(alpha, alpha). The labels are probability vectors, not
class indices. Mixup smooths the decision boundary and can help on small
datasets, but strong Mixup can obscure fine-grained image details. Label
smoothing similarly replaces a one-hot label with a mostly-correct
distribution. Focal loss instead changes the emphasis: with predicted
probability p_t for the correct class, its basic form is
`L = -(1-p_t)^gamma log(p_t)`, so easy examples receive less weight. Use
these tools deliberately; combining every regularizer is not automatically
better.

### Hyperparameter Guidance

| Scenario | `--lr` | `--epochs` | `--batch_size` | Notes |
|---|---|---|---|---|
| Large dataset (>10k images) | 3e-5 | 20–50 | 32–64 | Standard fine-tuning |
| Small dataset (<1k images) | 1e-5 | 50–100 | 16–32 | Consider LoRA, stronger augmentation |
| From pre-trained checkpoint | 3e-5 | 50–100 | 32 | Lower LR than from scratch |
| Custom classifier head only | 1e-3 | 50–200 | 32 | Higher LR since only head trains |

**Tips:**
- Start with `--lr 3e-5` for full fine-tuning, `1e-3` for head-only training.
- Use `--mixup_alpha 0.2` for small datasets — it helps prevent overfitting.
- `--focal_gamma 2.0` down-weights easy examples, useful when classes are
  imbalanced.
- `--class_multipliers "rare_class=3.0"` increases the loss weight for
  safety-critical or otherwise priority classes.

### Reproducibility

Every run is seeded by default: `--seed 42` drives the train/val split,
DataLoader shuffling, per-worker augmentation randomness, mixup, dropout,
`BalancedBatchSampler` batch composition, and the XGBoost stage.  Two
runs with identical arguments follow identical RNG streams.  To compare
hyperparameters under a different random draw, pass a different seed.

For bit-exact reproducibility (e.g. debugging a numerically divergent
run), add `--deterministic`.  This enables cuDNN deterministic mode and
PyTorch's deterministic-algorithms mode:

```bash
genml-kit-train --model convvit --deterministic ...
```

Two caveats:

- Some CUDA ops have no deterministic kernel.  Instead of aborting a
  long-running job, genml_kit logs a warning and proceeds (the op falls
  back to a non-deterministic kernel).
- `float16` AMP with `GradScaler` involves non-associative reductions
  that can still differ run-to-run; use `bfloat16` (default on
  Ampere+) or disable AMP for strictly repeatable arithmetic.

Checkpointing is atomic: `_latest.pt` is written to a temporary file
and renamed into place, so an interrupted run never leaves a truncated
resume point.

### Fine-Tuning CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--model` | `google/vit-base-patch16-224` | HuggingFace model name, local path, or custom model (e.g. `convvit`, `timm:<name>`). |
| `--dataset` | required | HuggingFace dataset name or `imagefolder/PATH` for local data. |
| `--image_column` | auto-detected | Explicit HF image column name. |
| `--label_column` | auto-detected | Explicit HF label column name. |
| `--image_size` | `448` | Augmentation crop size (processor handles final resize). |
| `--epochs` | `5` | Number of training epochs. |
| `--batch_size` | `32` | Batch size. |
| `--lr` | `3e-5` | Peak learning rate. |
| `--weight_decay` | `0.01` | Weight decay. |
| `--label_smoothing` | `0.0` | Label smoothing factor. |
| `--focal_gamma` | `0.0` | Focal loss gamma (`0` = disabled). Down-weights easy examples. |
| `--class_multipliers` | `""` | Per-class priority multipliers. Example: `"cat=3.0,dog=1.0"`. |
| `--sampler` | `none` | Training sampler: `none` (shuffle) or `weighted` (WeightedRandomSampler for class imbalance). |
| `--sampler_weights` | `frequency` | Weight mode for `--sampler weighted`: `frequency` (inverse-freq), `multipliers` (--class_multipliers), or `combined` (freq × multipliers). |
| `--mixup_alpha` | `0.0` | Mixup alpha (`0` = disabled; recommended: `0.2`). |
| `--seed` | `42` | RNG seed for data shuffling, the train/val split, mixup, dropout, and the XGBoost stage. Pass the same value to reproduce a run. See [Reproducibility](#reproducibility). |
| `--deterministic` | `False` | Enable deterministic algorithms (cuDNN deterministic mode, benchmark off). Costs throughput; ops without a deterministic CUDA kernel warn instead of failing. |
| `--grad_accum_steps` | `1` | Gradient accumulation steps (effective batch = batch_size × steps). |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16`. |
| `--device` | auto-detect | Device: `cpu`, `cuda`, or `cuda:INDEX`. |
| `--lr_group` | `None` | Per-parameter-group learning rates (repeatable). Format: `"REGEX=LR"`. |
| `--llrd_decay` | `None` | Layer-wise LR decay factor per depth level. Example: `--llrd_decay 0.85`. |
| `--checkpoint` | `genml_kit` | Checkpoint base path (`_latest.pt` / `_best.pt` appended). |
| `--log_every` | `20` | Log every N steps. |
| `--grad_monitor` | `-1` | Log gradient statistics every N steps. See [Gradient Monitor](#gradient-monitor). |
| `--norm_history` | `0` | Keep last N norm snapshots for trend analysis. |
| `--trend_top_n` | `10` | Show top N params in trend table. `0` = show all. |
| `--grad_clip` | `1.0` | Max gradient norm for clipping. `0` disables. |
| `--save_every` | `500` | Save checkpoint every N optimizer steps. 0 disables. |
| `--num_workers` | `2` | DataLoader worker processes. |
| `--log_level` | `INFO` | Logging level. |
| `--log_targets` | `STDERR` | Comma-separated log destinations. `STDERR` logs to standard error; any other entry is a log file path (appended). Example: `STDERR,/tmp/train.log` logs to both. |
| `--log_dir` | `None` | TensorBoard log directory (default: `<checkpoint_dir>/logs`). Requires the `tensorboard` package (installed by the `dev` extra). |
| `--cache_dir` | `None` | HuggingFace cache directory. |
| `--remote_checkpoint` | `None` | Remote URI for checkpoint sync (`gs://BUCKET/PREFIX`, `r2://BUCKET/PREFIX`, or `s3://BUCKET/PREFIX`). |
| `--source_checkpoint` | `None` | Path to source checkpoint to absorb parameters from. |
| `--param_rename` | `None` | Regex-based key rename patterns (`SEARCH;REPLACE`). |
| `--classifier` | `None` | Classifier head spec: registered name (e.g. `mlp`) or `.py` path. |
| `--classifier_args` | `{}` | Extra classifier kwargs (repeatable). Example: `hidden=512 dropout=0.3`. |
| `--freeze` | `None` | Regex patterns for parameters to keep trainable. All others frozen. |
| `--lora` | `False` | Enable LoRA via PEFT. Requires `pip install "genml_kit[lora]"`. |
| `--lora_r` | `8` | LoRA rank. |
| `--lora_alpha` | `16` | LoRA alpha (scaling = `alpha / r`). |
| `--lora_dropout` | `0.0` | Dropout on LoRA layers. |
| `--lora_target_modules` | `None` | Comma-separated module names for LoRA (e.g. `"query,key,value"`). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name or `.py` script path. |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name or `.py` script path. |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). |
| `--state_save` | `opt,sched,amp` | States to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched,amp` | States to restore on resume. |
| `--xgboost_model` | `None` | Output path for XGBoost model (trains after PyTorch). |
| `--xgb_*` | various | XGBoost hyperparameters (see `--help` for full list). |
| `--model_arg` | `{}` | Override model configuration (repeatable). |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--train_augmentation_script` | `None` | Custom augmentation script. Must define `create_train_transform()`. |
| `--grad_checkpoint` | `False` | Enable gradient checkpointing. Reduces activation memory by ~40-50% at the cost of ~25-35% more compute per step. Enables larger batch sizes. See [Gradient Checkpointing](#gradient-checkpointing). |
| `--tta` | `None` | Test-Time Augmentation. `default` uses built-in 4-view transform (identity + flips). A path/URL loads an external script defining `create_tta_transform()`. Omit to disable. |

Training automatically resumes from an existing `_latest.pt` or `_best.pt`
checkpoint if one exists at the `--checkpoint` path.

### Remote Checkpoint Sync (GCS / R2 / S3)

`--remote_checkpoint` uploads each saved checkpoint to cloud storage.
Requires `pip install "genml_kit[s3]"` for `s3://` and `r2://` URIs (both
use boto3), or `genml_kit[gcs]` for `gs://`. Both `genml-kit-train` and
`genml-kit-pretrain` accept the flag.

The sync also works in reverse at startup: before auto-resume, any missing
`_latest.pt` / `_best.pt` is downloaded from the remote prefix (latest is
tried first, matching resume precedence). A locally present checkpoint is
never overwritten — the local copy always wins — and connection or
credential failures degrade to a warning instead of aborting startup.

**AWS S3** — credentials come from the standard environment variables:

```bash
%env AWS_ACCESS_KEY_ID=AKIA...
%env AWS_SECRET_ACCESS_KEY=...
%env AWS_SESSION_TOKEN=...        # only for temporary (STS/SSO) credentials
%env AWS_DEFAULT_REGION=us-east-1

--remote_checkpoint s3://my-bucket/genml_kit/convvit_ijepa
```

Permanent IAM-user keys need only the access/secret pair; the session
token is picked up automatically when present. When the key variables
are unset, boto3's default credential chain applies (IAM instance role,
`~/.aws/credentials`, SSO cache).

**Cloudflare R2** — the S3-compatible API with R2 credentials:

```bash
%env R2_ENDPOINT_URL=https://<account_id>.r2.cloudflarestorage.com
%env R2_ACCESS_KEY_ID=...
%env R2_SECRET_ACCESS_KEY=...

--remote_checkpoint r2://my-bucket/genml_kit/convvit_ijepa
```

### LoRA Details

Low-Rank Adaptation freezes the pre-trained backbone and injects small
trainable low-rank matrices into attention layers. The LoRA output is
`ΔW = (alpha/r) × B @ A`, where A and B are the low-rank matrices.

| r | alpha | alpha/r | Use case |
|---|---|---|---|
| 8 | 16 | 2.0 | Conservative, very few params |
| 16 | 32 | 2.0 | Good default for medium datasets |
| 16 | 64 | 4.0 | Larger updates (helpful under bfloat16) |
| 32 | 64 | 2.0 | More capacity, same scaling |

LoRA can be combined with a custom classifier:

```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --classifier cls_attention \
    --classifier_args 'num_encoder_layers=2' \
    --lora --lora_r 16 --lora_alpha 32 \
    --freeze 'classifier\.(head|pool|encoder)' \
    --lr_group 'backbone.*=1e-5' 'classifier.*=3e-4' \
    ...
```

### Cross-Dataset Resume

Switch from one dataset to another while keeping backbone weights:

```bash
genml-kit-train --model facebook/convnextv2-base-22k-224 \
                --dataset my-org/other-labeled-images \
                --label_column category \
                --checkpoint genml_kit \
                --state_load none \
                --epochs 10 \
                --batch_size 16 \
                --lr 3e-5 \
                --mixup_alpha 0.2 \
                --amp_dtype bfloat16
```

Backbone weights load via `strict=False`; the classifier head (different
`num_classes`) is reinitialised. `--state_load none` prevents carrying over
old optimizer/scheduler states.

### Gradient Checkpointing

Training deep vision transformers is memory-intensive: the attention maps for
EVA02-base at 448×448 resolution consume ~4.6 GB across 12 layers, which
limits the maximum batch size even on a 24 GB GPU. Gradient checkpointing
trades compute for memory by discarding intermediate activations during the
forward pass and recomputing them during the backward pass.

```bash
genml-kit-train \
    --model timm:eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --grad_checkpoint \
    --amp_dtype bfloat16 \
    ...
```

**What it does:** Each transformer block stores only its input and output.
The intra-block activations (attention maps, FFN intermediates) are
recomputed on-demand during the backward pass. This reduces peak activation
memory by ~40-50%.

**What it costs:** The recomputation adds ~25-35% more compute per training
step. Whether this translates to a net throughput gain or loss depends on
your GPU:
- **Memory-bound GPU** (VRAM-limited, compute underutilized): larger batches
  fill the GPU's compute capacity → net throughput **increase** of ~50-60%.
- **Compute-saturated GPU** (TFLOPS at 100%): every extra FLOP adds to step
  time → net throughput **decrease** of ~20-25%, though the larger batch can
  still improve training stability and final accuracy.

**Backend support:** Enabled automatically for all model backends — timm,
HuggingFace, ConvViT, and UVito — via native APIs or per-block
`torch.utils.checkpoint` in the transformer loops.

---

## Inference Guide

Run inference on individual images:

```bash
genml-kit-infer --model facebook/convnextv2-base-22k-224 \
                --checkpoint genml_kit_best.pt \
                path/to/image.jpg path/to/other_image.png
```

Output is JSON with per-class probabilities:

```json
{
  "source": "image.jpg",
  "predictions": [
    {"label": "golden_retriever", "probability": 0.435},
    {"label": "labrador", "probability": 0.281}
  ]
}
```

### XGBoost and test-time augmentation

The neural classifier makes decisions through its head. The XGBoost option
takes the encoder representation h = f_theta(x) instead and fits an ensemble
of decision trees to those vectors. A tree partitions feature space with rules
such as h_37 < t; boosting adds trees sequentially so each new tree focuses on
errors left by previous trees. This can work well when the dataset is small
and the representation is already useful, but it is not guaranteed to beat the
neural head. Use validation data to choose tree depth, number of rounds, and
ensemble weight.

Test-time augmentation (TTA) runs the same image through several plausible
views and averages the probability vectors:

$$
\bar{p}(y \mid x) = \frac{1}{K} \sum_{k=1}^{K} p(y \mid T_k(x))
$$

The transformations T_k should preserve the label semantics. Horizontal flips
are usually safer than orientation-specific crops; verify that an augmentation
does not erase the cue that distinguishes the classes.

### Inference CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace model name or custom model. |
| `--checkpoint` | (required) | Path to state dict or wrapped checkpoint. |
| `--top_k` | `None` | Show top-K predictions; omit for all classes. |
| `--output` | `None` | Write JSON results to file. |
| `--device` | `None` | Force PyTorch device (`cuda`, `cpu`). Auto-detected if omitted. |
| `--cache_dir` | `None` | HuggingFace cache directory. |
| `--xgboost_model` | `None` | XGBoost model path. If provided, runs XGBoost alongside PyTorch. |

Wrapped checkpoints (containing `model_state_dict` and metadata) are preferred
over raw state dictionaries, which produce a metadata warning.

### XGBoost Inference

When `--xgboost_model` is provided, the output includes both predictions:

```json
{
  "source": "image.jpg",
  "predictions": [
    {"label": "golden_retriever", "probability": 0.435}
  ],
  "xgboost_predictions": [
    {"label": "golden_retriever", "probability": 0.612}
  ]
}
```

---

## Tips & Pitfalls

### `--checkpoint` and `--source_checkpoint` are different

`--checkpoint` names the output prefix used for saving and resuming the current
training run. `--source_checkpoint` imports weights from another run before
the new optimizer is created. When moving a SupCon encoder into classification,
use:

```bash
genml-kit-train \
    --checkpoint /content/eva02_finetune \
    --source_checkpoint /content/eva02_supcon_latest.pt \
    --param_rename 'encoder\\.model\\.(.*);model.$1' \
    ...
```

The rename is needed because the pre-training wrapper stores backbone keys
under `encoder.model.*`, whereas the fine-tuning model stores them under
`model.*`. The SupCon `projection.*` keys are expected to remain unused, and
new classifier-head keys are expected to be missing from the source checkpoint.
Those messages indicate a successful partial transfer if the backbone keys
are matched.

### Interpreting a SupCon plateau

SupCon loss is not expected to approach zero. If an anchor has k-1 positives and
positives become much more similar than negatives, a useful reference value is
approximately `-log(k-1)`. This is only a diagnostic approximation: class
counts may vary, the sampler may repeat examples, and temperature affects
optimization. Judge the loss together with embedding quality and downstream
validation metrics. A plateau near this reference can mean convergence; a
plateau well above it can mean too few positives, a learning-rate problem, or
labels that are not being passed correctly.

### A practical debugging order

1. Confirm that images and labels are valid and that each SupCon batch has
   repeated classes.
2. Confirm that the loss changes when the learning rate changes and inspect
   `--grad_monitor` for zero or exploding gradients.
3. Check whether pre-trained weights loaded by reading the alignment report,
   not just the final epoch number.
4. Compare against a simple head-only or full-fine-tuning baseline before
   adding LLRD, LoRA, Mixup, focal loss, and class multipliers together.
5. Select the checkpoint using a validation metric appropriate to the
   objective, rather than training loss alone.

---

## Gradient Monitor

`--grad_monitor N` logs a per-parameter gradient report every *N* training
steps. This helps diagnose training instability (exploding / vanishing
gradients, imbalanced parameter updates) before it shows up in the loss.

### Summary Line

```
[Step 29400] Gradient Report: 202 params | grad_rms: mean=5.68e-01 max=3.41e+00 min=1.66e-05 | grad/param: mean=2.94e-01
```

All norms are **RMS** (root mean square): L2 norm divided by `sqrt(numel)`.
This makes them independent of tensor shape and directly comparable across
parameters of different sizes.

| Field | Meaning |
|---|---|
| `params` | Total number of trainable parameters. |
| `grad_rms: mean/max/min` | RMS of the gradient tensor for each parameter, then aggregated. `max` is the single most aggressive gradient — the one most likely to cause instability. |
| `grad/param: mean` | Average gradient-to-parameter ratio. Healthy: < 0.1. Concerning: > 1.0. |

### Per-Parameter Columns

| Column | Symbol | What to look for |
|---|---|---|
| **g_rms** | `‖∇L‖/√N` | Compare across params. One param with g_rms 100× higher is a problem. |
| **p_rms** | `‖W‖/√N` | Per-element scale context. With `std=0.02` init, expect ~0.02. |
| **g/p** | `‖∇L‖ / (‖W‖ + ε)` | **Most useful column.** Healthy: < 0.1. Concerning: > 1.0 (update overshoots). Dangerous: > 5.0. |
| **g_max** | `max|∇L|` | Highlights individual neurons with extreme gradients. |
| **sparse** | `% zero` | High sparsity (> 50%) = most neurons not receiving signal. |
| **status** | | `OK` = healthy. `STL` = stalled. `OVF` = exploding. `IMB` = imbalanced. `GPR` = high g/p ratio. |

### Reading the Report

**Healthy training:** g/p < 0.1, `grad/param: mean` in 0.01–0.1 range, g_rms within ~10× across layers.

**Exploding gradients:** One or more params with `OVF`, g/p > 5.0, `grad_rms: max` >> `mean`. Fix: lower LR, add `--grad_clip 1.0`, or use warmup.

**Vanishing gradients:** Many params with `STL`, g_rms near 1e-7, high sparsity. Fix: increase LR, check for dead neurons.

**Imbalanced updates:** Some params `IMB`, large g/p disparity between layers. Fix: use `--lr_group` for different rates, or freeze the dominant component.

### Norm Trend History

`--norm_history N` (requires `--grad_monitor`) keeps the last N snapshots
per parameter. A trend summary table is appended to each report showing
direction (`UP`/`DOWN`/`---`), percentage change, and min/max values.

---

## Custom Models

genml_kit supports any HuggingFace `AutoModelForImageClassification` model,
any timm model via `timm:<name>`, and custom architectures registered in
`genml_kit.models`.

### Built-in Custom Models

| Name | Description | `--model` value |
|---|---|---|
| timm | Any model from [timm](https://github.com/huggingface/pytorch-image-models) | `timm:<model_name>` |
| ConvViT | Multi-block conv stem + ViT encoder with CLS-guided attention pooling | `convvit` |
| UVito | Frozen SMP encoder + learnable patch projection + Transformer encoder | `uvito` |
| ClsModelWrapper | HuggingFace backbone + custom classifier head | `cls_model_wrapper:<hf_name>` |
| ContrastiveEncoder | Backbone + projection head for contrastive pre-training | `contrastive_encoder:<hf_name>` |

### Adding a Custom Model

1. Create `genml_kit/models/{name}/` with `model.py`, `processor.py`,
   `loader.py`, and `__init__.py`.
2. Add the import to `genml_kit/models/__init__.py`.
3. The model must expose `.forward(pixel_values=images)` → object with `.logits`,
   and `config.id2label` / `config.label2id`.
4. CLI overrides via `--model_arg KEY=VALUE` are forwarded to the loader.

### Custom Classifiers

`ClsModelWrapper` lets you replace the default HF classification head:

```python
class Classifier(nn.Module):
    def __init__(self, num_labels, hidden_size, **kwargs):
        super().__init__()
        self.head = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states):          # (B, N, D) tensor
        features = hidden_states[:, 0]         # CLS token
        return self.head(features)

    def extract_features(self, hidden_states): # (B, N, D) → (B, D)
        return hidden_states[:, 0]
```

The `extract_features` method is used by `--xgboost_model` for XGBoost
training on backbone features.

---

## References and Further Reading

- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).
  A useful comparison for masked-image pre-training.
- Peng et al., [Masked Image Modeling with Vision Transformers](https://arxiv.org/abs/2111.09886).
  The SimMIM paper.
- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243).
  The I-JEPA paper.
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).
  The DINO paper — self-distillation with an EMA teacher and multi-crop.
- Khosla et al., [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362).
  The SupCon objective and experiments.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
  The low-rank adaptation idea used by genml_kit.
- Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412).
  The Mixup augmentation strategy.
- The [PyTorch optimization documentation](https://pytorch.org/docs/stable/optim.html)
  explains AdamW, schedulers, and gradient clipping.
- The [scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
  is useful when choosing metrics for imbalanced classification.

## Development

```bash
git clone https://github.com/davidel/genml_kit
cd genml_kit
pip install -e ".[dev,all]"
pytest
```

Tests run with warnings-as-errors (`filterwarnings = ["error"]` in
`pyproject.toml`), so a new dependency deprecation warning fails the suite
instead of scrolling past.  Code is formatted with
[yapf](https://github.com/google/yapf) using the project's `.style.yapf`
(Google style, 2-space indent) and linted with
[Ruff](https://docs.astral.sh/ruff/) (`ruff check .`).

### Naming conventions for class members

- Anything that PyTorch registers — child `nn.Module`, `nn.Parameter`,
  registered buffer — is **never** underscore-prefixed, regardless of
  intended visibility.  A leading underscore would change `state_dict()`
  keys, optimizer param groups, and DDP behavior.
- Everything else that is not part of a class's public contract
  (internal state, helper hooks invoked only by the class itself) gets a
  leading underscore.  A member called from other classes, overridden as
  an extension point, or documented as API stays public.
- Extension-point methods that *callers* invoke (e.g.
  `PretrainMethod.build_transform`) are public; same-named hooks the
  base class *itself* dispatches (e.g. `BaseImageProcessor._build_transform`)
  are private.

## License

Apache-2.0
