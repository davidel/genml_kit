# genml_kit

[![CI](https://github.com/davidel/genml_kit/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/genml_kit/actions/workflows/ci.yml)

A general-purpose toolkit for image classification and decoder-only language
models. Supports self-supervised pre-training, supervised fine-tuning, LoRA,
causal-LM training / generation, preference alignment (DPO, reward models),
and XGBoost ensemble inference — all from the command line.

`genml_kit` is domain-agnostic: point it at any HuggingFace dataset, local
`ImageFolder` tree, or timm/HuggingFace backbone. It grew out of a
skin-lesion classification project; the dermoscopy-specific workflow (dataset
preparation, tuned recipes) is documented separately in
[`scdiag/README.md`](scdiag/README.md). The drone visual-odometry front-end
(similarity estimation between oblique aerial frames) is documented separately
in [`vo/README.md`](vo/README.md).

## Contents

- [Why genml_kit?](#why-genml_kit)
- [How It Works](#how-it-works)
  - [The Method Lifecycle](#the-method-lifecycle)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pre-Training Guide](#pre-training-guide)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [Inference Guide](#inference-guide)
- [Tips & Pitfalls](#tips--pitfalls)
- [Gradient Monitor](#gradient-monitor)
- [Custom Losses](#custom-losses)
- [Custom Models](#custom-models)
- [Semantic Segmentation](#semantic-segmentation)
- [Object Detection](#object-detection)
- [Visual Odometry](#visual-odometry)
- [Knowledge Distillation](#knowledge-distillation)
- [Large Language Models (causal LM)](#large-language-models-causal-lm)
- [Reinforcement Learning](#reinforcement-learning)
- [References And Further Reading](#references-and-further-reading)
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

## How It Works

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

### A Useful Mental Model

The encoder turns an image into a vector of features. During pre-training, we
choose an artificial task whose answer can be obtained from the images (or,
for SupCon, from their labels). The encoder learns parameters $`\theta`$ that
make this task easy. During fine-tuning, a classifier is attached to the
encoder and the whole model, or a selected part of it, is adapted to the real
labels:

```text
image x  ──► encoder f_theta(x)  ──► classifier g_phi  ──► class probabilities
                    │                         │
             reusable features          task-specific boundary
```

Here $`x`$ is the input image, $`f_\theta(x)`$ the feature vector computed by
the encoder with parameters $`\theta`$, and $`g_\phi`$ the classifier head with
parameters $`\phi`$.

Pre-training and fine-tuning are not two names for the same job. Pre-training
shapes a useful representation; fine-tuning decides how that representation
should be used for the target labels. If the pre-training data is visually
related to the target data, this gives the classifier a much better starting
point than random initialization. If the domains are very different, use a
smaller learning rate for the backbone and validate carefully.

The pre-training methods are introduced above with links to their papers (see
the [Pre-Training Guide](#pre-training-guide)). The
[timm documentation](https://huggingface.co/docs/timm/index) and the
[Hugging Face image classification guide](https://huggingface.co/docs/transformers/tasks/image_classification)
are useful references when selecting a backbone or processor.

### The Method Lifecycle

`genml-kit-train` is a *branchless* driver: it never inspects which
`--method` or `--pipeline` it is running.  Instead, every objective registers
itself through the `Method` registry and declares the same four lifecycle
hooks, which the driver calls in one fixed order:

| # | Hook | Purpose |
|---|---|---|
| 1 | `prepare_transforms(args, device)` | Resolve any transforms the method needs (e.g. the HF processor's normalization) **before** the loaders are built. |
| 2 | `wire_data(args, pipeline)` | Consume pipeline-built data attributes: label space, class weights, criterion. Runs **before** the model so the HF head can be sized from the label space. |
| 3 | `build_model(args, device)` | Construct the model. Every method applies LoRA (`--lora`), source-checkpoint weights (`--source_checkpoint`), freeze patterns (`--freeze`) and gradient checkpointing (`--grad_checkpoint`) through the same `_apply_model_extras` step — so all four flags work for **every** objective, not just classification. |
| 4 | `post_train(args, pipeline, device, result)` | Optional post-training stage, called after `BaseTrainer.run()` returns (skipped when the run was interrupted). |

Two ordering invariants are enforced by the hook order itself, not by
convention: transforms resolve before loaders consume them, and the label
space exists before the classification head is constructed.

Adding a new objective means implementing the hooks you need (only
`build_model` and `train_step` are abstract); the driver never changes.
`ClassificationMethod` additionally implements `post_train` to run the
optional XGBoost shallow-head stage; any future classifier-style method can
reuse the same hook.

## Package Layout

| Module | Contents |
|---|---|
| `genml_kit.training` | `train.py` and `infer.py` CLI harnesses, `optim_factory.py` (optimizers, LLRD, schedulers), `model_utils.py` (loading, freezing, feature extraction), `param_align.py`, `eval.py` + `metrics.py`, `grad_monitor.py`, `train_reporting.py`, `tta.py`, `xgb_utils.py` + `xgb_pipeline.py`, `text_collator.py` + `preference_collator.py` (text / preference batches), `llm_model.py` + `generate.py` + `llm_eval.py` (LLM inference, generation, eval), `classifiers/` (pluggable heads) |
| `genml_kit.methods` | objective-side registry (classification, segmentation, detection, vo_pair, SimMIM, I-JEPA, DINO, BYOL, SupCon, distillation, `causal_lm`, `dpo`, `reward_model`) |
| `genml_kit.losses` | shared loss library (focal, Dice+CE segmentation, NT-Xent contrastive, DINO self-distillation, BYOL, KD distillation, alignment: `sequence_logps` / `dpo_loss` / `bradley_terry_loss`) |
| `genml_kit.augmentations` | objective-level transforms (dual-view, multi-crop) |
| `genml_kit.models` | model/processor/tokenizer registry; `timm/`, `convvit/`, `uvito/`, `cls_model_wrapper/`, `seg_model_wrapper/` backends; `decoders/` (pluggable segmentation decoders); `generative.py` (`GenerativeLM` contract), `tokenizer.py` (`TOKENIZERS`), `processors/base.py` |
| `genml_kit.datasets` | `hf_proxy.py` (HuggingFace → PyTorch bridge), `image_folder.py`, `seg_folder.py` (paired image/mask), `det_folder.py` (COCO detection), `text_proxy.py` (HF text → items), `preference_proxy.py` (chosen/rejected pairs), `ensemble.py`, `field_dataset.py`, `balanced_sampler.py`, `weighted_sampler.py`, `retry.py`, `vo_pairs.py` (synthetic VO pairs) |
| `genml_kit.geometry` | `similarity.py` (2-D similarity algebra, batched differentiable Umeyama, backward-map warp) — the math behind the visual-odometry front-end lives in [`vo/README.md`](vo/README.md) |
| `genml_kit.io` | `checkpointing.py` (atomic saves, LoRA state, remote fetch), `storage_utils.py` (S3 / GCS / R2) |
| `genml_kit.utils` | glog-style logging, CLI arg groups + `shared_args.py` (globally shared `ARG_*` CLI flags), seeding, signal handling, GPU info, tables, external `.py` script loading, `image_dump`, transformer init helpers, `generation.py` (samplers + batched `generate`) |
| `genml_kit.llm_cli` | `genml-kit-llm` entry point (`generate` / `eval` subcommands) |

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

# With causal-LM fine-tuning (LoRA + transformers):
pip install "genml_kit[llm]"

# With a bare SentencePiece tokenizer.model (the 'spm' tokenizer entry):
pip install "genml_kit[llm-spm]"

# With UVito model support:
pip install "genml_kit[uvito]"

# Everything above in one shot (gcs, s3, lora, llm, llm-spm, timm, uvito):
pip install "genml_kit[all]"
```

**Requirements:** Python ≥ 3.9, PyTorch, torchvision, transformers, datasets,
NumPy, scikit-learn ≥ 1.3, XGBoost ≥ 2.0, Pillow, tensorboard.

## Quick Start

The fastest way to get started:

<!-- genml-cli-test
name: quick-start-cifar10
network: true
override:
  - --dataset {tmp}/img
  - -.--image_column
extra: --epochs 1 --max_items 16
-->
```bash
# Fine-tune a ViT on an image-classification dataset (5 epochs, ~2 minutes on GPU)
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset uoft-cs/cifar10 \
                --image_column img \
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

genml_kit supports five pre-training methods, each with different strengths:

### Choosing Your Method

| Method | Needs labels? | Best when… | Key idea |
|---|---|---|---|
| **SimMIM** | No | You have large unlabeled datasets; want a simple, proven approach | Mask 60% of image patches, train the model to reconstruct the raw pixels |
| **I-JEPA** | No | You want faster training and better downstream transfer than SimMIM | Predict *representations* of masked regions, not raw pixels — avoids learning noise |
| **SupCon** | Yes | You have labels and want representations that cluster by class | Pull same-class images together, push different classes apart in feature space |
| **DINO** | No | You want strong features from a modest amount of data; proven, robust default | Self-distillation: a student matches the output distribution of its own EMA copy (the teacher) |
| **BYOL** | No | Similar goals to DINO with a simpler pipeline (two views, no multi-crop) | Online network predicts the output of its own EMA target network |

### How Each Method Works

The methods differ mainly in what they call a correct answer. SimMIM asks for
pixels, I-JEPA asks for features, DINO and BYOL ask a network to predict the
output of its own slowly-moving copy, and SupCon asks for relative positions in
feature space. That distinction matters: pixel reconstruction can spend effort
reproducing colour and high-frequency detail, while the other methods spend
effort building invariant, separable representations.

**SimMIM** (Masked Image Modelling): Randomly masks ~60% of image patches and
trains a lightweight decoder to reconstruct the original pixels. The encoder
must learn to understand textures, boundaries, and spatial context from just
40% of the image. Think of it as a "fill in the blanks" exercise for vision
models. Good default choice when you have lots of unlabeled images.
Reference: Xie et al., [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886), CVPR 2022.

For an image split into patches $`x_1, \dots, x_N`$, let $`\mathcal{M}`$ be
the set of masked patch indices and $`\hat{x}_i`$ the decoder prediction.
SimMIM minimizes the mean squared reconstruction error over masked patches:

$$
\large
\mathcal{L}_{\text{MIM}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\\| \hat{x}_i - x_i \right\\|_2^2
$$

where:

- $`x_i`$ — the pixels of the $`i`$-th original patch;
- $`\hat{x}_i`$ — the decoder's reconstruction of that patch;
- $`\mathcal{M}`$ — the set of masked patch indices, and $`|\mathcal{M}|`$ its
  size (how many patches are masked);
- $`\lVert \cdot \rVert_2^2`$ — the squared Euclidean ($`L_2`$) norm, i.e. the
  sum of squared pixel differences over the patch.

Only masked patches contribute to the loss; otherwise copying visible pixels
would make the task too easy. A higher
`--mask_ratio` supplies less context and creates a harder task, but an
excessively high ratio can make reconstruction ambiguous.

**I-JEPA** (Joint-Embedding Predictive Architecture): Also masks patches, but
instead of reconstructing pixels, it predicts the *latent representation* of
the masked region from the visible context. This avoids wasting capacity on
pixel-level noise (e.g. exact JPEG compression artifacts) and learns more
transferable features. Uses a teacher–student setup with EMA momentum ramping.
Reference: Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.

Let $`f_\theta`$ be the student encoder and $`f_\xi`$ the teacher encoder;
$`\theta`$ and $`\xi`$ are their respective parameter vectors (the subscripts
name the network a symbol belongs to). The predictor $`q_\theta`$ receives the
visible context and predicts a target representation $`z_j = f_\xi(x_j)`$ for a
masked region. The objective is regression in representation space:

$$
\large
\mathcal{L}_{\text{IJEPA}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \left\\| q_\theta(f_\theta(\text{context}))_i - \mathop{\mathrm{stopgrad}}\left(f_\xi(x_i)\right) \right\\|_2^2
$$

where:

- $`x_i`$ — the $`i`$-th masked patch (the prediction target);
- $`f_\theta(\text{context})`$ — the student's encoding of the *visible* patches;
- $`q_\theta(\cdot)_i`$ — the predictor's output for the $`i`$-th masked location;
- $`f_\xi(x_i)`$ — the teacher's representation of the masked patch;
- $`\mathrm{stopgrad}(\cdot)`$ — treat the argument as a constant: gradients do
  not flow through it during the student update;
- $`\lVert \cdot \rVert_2^2`$ — squared Euclidean distance in representation space;
- $`\mathcal{M}`$ — the set of masked patch indices.

The teacher is not optimized by backpropagation; it follows the student with
an exponential moving average:

$$
\large
\xi \leftarrow m \\, \xi + (1 - m) \\, \theta
$$

where $`m`$ is the momentum (`--teacher_momentum`) and the arrow denotes
assignment: the new teacher parameters are a convex blend of the old teacher
and the current student. A high $`m`$ changes the teacher slowly and gives
more stable targets. The asymmetric teacher update and masking are important:
two networks simply trained to copy each other could collapse to a constant
vector (the teacher-follows-student trick and $`\mathrm{stopgrad}`$ come from
Grill et al., [BYOL](https://arxiv.org/abs/2006.07733), NeurIPS 2020).

Three mechanism details complete the picture:

- **Block masking.** Target regions are not scattered single patches: the mask
  is composed of a few contiguous rectangular blocks (4 blocks of 6×6 patches
  in the current implementation). Contiguous holes are much harder to fill in
  from context than isolated ones, so the task rewards spatial understanding
  rather than high-frequency texture interpolation.
- **Two crop scales.** The *context* (source) view is a small random crop (50%
  of the image side) and the *target* view is a larger one (85%). The predictor
  must infer representations of a region it sees only through a narrow window,
  which encourages semantic rather than layout-memorized features.
- **Momentum ramp.** $`m`$ moves from `--teacher_momentum` (0.996) toward
  `--teacher_final_momentum` (1.0) as training progresses: as the encoder
  matures, the teacher freezes progressively, stabilizing the targets late in
  training.

**SupCon** (Supervised Contrastive Learning): Uses labels to define "positive"
pairs (same class) and "negative" pairs (different classes). The loss pulls
features of same-class images together and pushes different-class features
apart. Produces a feature space where similar images naturally cluster.
Requires a `ContrastiveEncoder` (backbone + projection head) and balanced
batch sampling to ensure each batch has enough same-class pairs.
Reference: Khosla et al., [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), NeurIPS 2020.

An image $`x_i`$ is mapped by the encoder-and-projection-head
$`f_\theta`$ to a vector $`f_\theta(x_i)`$, which is rescaled to unit length
(normalized) so that comparisons are scale-invariant:

$$
\large
z_i = \frac{f_\theta(x_i)}{\lVert f_\theta(x_i) \rVert_2}
$$

where $`\lVert v \rVert_2 = \sqrt{\sum_d v_d^2}`$ is the Euclidean ($`L_2`$)
length of a vector $`v`$, so every $`z_i`$ lies on the unit hypersphere.
The similarity of two examples $`i`$ and $`j`$ is then the dot product of
their normalized projections, divided by the temperature $`\tau`$
(`--temperature`):

$$
\large
s_{ij} = \frac{z_i^\top z_j}{\tau}
$$

where $`z_i^\top z_j`$ is the dot product (cosine of the angle between the two
unit vectors) and $`\tau`$ rescales the similarities before the softmax.

Fix one example $`i`$, called the *anchor*. Its *positive set* is

$$
\large
\mathcal{P}(i) = \\{\\, j : j \neq i \text{ and } y_j = y_i \\,\\}
$$

i.e. all other examples in the batch that share the anchor's class label
$`y_i`$; every remaining example is a *negative*. SupCon averages the
log-softmax probability that the anchor assigns to each of its positives:

$$
\large
\mathcal{L}_i = -\frac{1}{|\mathcal{P}(i)|} \sum_{p \in \mathcal{P}(i)} \log \left( \frac{\exp(s_{ip})}{\sum_{a \neq i} \exp(s_{ia})} \right)
$$

$$
\large
\mathcal{L} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_i
$$

where:

- $`s_{ip}`$ — similarity between the anchor and positive $`p`$ (above);
- $`\sum_{a \neq i}`$ — sum over all other examples in the batch, positives
  *and* negatives (the softmax denominator);
- $`|\mathcal{P}(i)|`$ — the number of positives for the anchor;
- $`B`$ — the batch size; the total loss averages the per-anchor terms.

Minimizing $`\mathcal{L}`$ pushes $`s_{ip}`$ up for same-class pairs and down
for different-class pairs. A lower temperature makes the distribution sharper:
this
can help separate hard negatives but can also make optimization less stable.
`--samples_per_class` matters because an anchor needs another example of its
class to have a positive. A class represented once contributes no useful
SupCon term for that anchor.

#### Temperature

The division by $`\tau`$ (`--temperature`, default 0.07) above, and the two
temperatures used by DINO below, all do the same job. Softmax over logits
$`s`$ becomes

$$
\large
\mathrm{softmax}(s)_j = \frac{\exp(s_j / \tau)}{\sum_k \exp(s_k / \tau)}
$$

A small $`\tau`$ divides every logit by a tiny number, so exponentials grow
very fast and the distribution concentrates on the maximum (sharp); a large
$`\tau`$ flattens it toward uniform. Sharpening amplifies small differences in
similarity, which speeds learning, but too-small $`\tau`$ makes gradients
vanish for non-maximum classes and destabilizes training. SupCon uses one
moderate value; DINO uses a small teacher temperature (sharper *targets*,
they are meant to be confident) and a larger student temperature (softer
*predictions*, easier to move).

**DINO** (self-**di**stillation with **no** labels): Caron et al.,
[Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021.
One backbone is instantiated twice. The **student** $`f_\theta`$ is trained by
gradient descent; the **teacher** $`f_\xi`$ is the EMA copy of the student
($`\xi \leftarrow m\,\xi + (1-m)\,\theta`$, as in I-JEPA above, ramped from
`--dino_momentum` 0.996 to `--dino_final_momentum` 1.0). Each image is cut
into 2 **global** crops plus `--dino_local_num` (8) small **local** crops:
the teacher sees only global views, the student sees all of them, so the
student must produce globally consistent features from partial views.

Both towers end in an MLP projection head whose output is turned into a
probability vector over the projection dimension. The teacher's distribution
is *sharpened* by its small temperature and *centered* by a running mean
$`c \in \mathbb{R}^{D}`$ updated once per step,

$$
\large
c \leftarrow m_c \\, c + (1 - m_c) \cdot \frac{1}{B}\sum_{i=1}^{B} f_\xi(x_i)
$$

(`--dino_center_momentum` 0.9), and the loss is the cross-entropy between the
teacher's processed distribution $`p_\xi`$ and the student's $`p_\theta`$,
averaged over all (teacher global, student crop) pairings:

$$
\large
\mathcal{L} = - \sum_{j} p_{\xi,j} \\, \log p_{\theta,j}
$$

Centering and sharpening are the two anti-collapse mechanisms: centering
cancels any component the teacher produces for *every* input (a constant
output carries no information and would otherwise still lower the loss), and
sharpening makes the teacher confident, forcing the student to truly reproduce
its argmax rather than drifting toward a flat, uninformative distribution.
Because the targets come from the teacher and the teacher follows the student,
this is self-distillation — there are no external labels anywhere.

**BYOL** (**B**ootstrap **Y**our **O**wn **L**atent): Grill et al.,
[Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733), NeurIPS 2020.
The same two-tower idea as DINO but with two augmented views and no
multi-crop. The **online** network (encoder + projector + predictor MLP) and
the **target** network (encoder + projector, EMA copy) process the two views;
only the online tower receives gradients. With $`z_1, z_2`$ the target
projections of the two views and $`p_1, p_2`$ the online *predictions* of
those projections, the loss is the symmetric cosine regression

$$
\large
\mathcal{L} = 2 - \cos(p_1, z_2) - \cos(p_2, z_1)
$$

which is 0 when predictions perfectly match target projections and reaches 4
when they point in opposite directions; $`\cos(u,v) = u^\top v / (\lVert u \rVert_2 \lVert v \rVert_2)`$ is the cosine of the angle between the two
vectors. Two asymmetries keep this from collapsing to a constant: the
predictor exists only on the online side (predicting a target is easier than
being one), and the targets move slowly under the EMA — exactly the
teacher-follows-student trick noted under I-JEPA.

### Typical Hyperparameters

These are reasonable starting points. Tune from here based on your dataset
size and GPU memory:

| Parameter | SimMIM | I-JEPA | DINO | BYOL | SupCon |
|---|---|---|---|---|---|
| `--image_size` | 448 | 448 | 224 | 224 | 448 |
| `--batch_size` | 32 | 32 | 64 | 64 | 64 |
| `--lr` | 1e-4 | 1e-4 | 5e-4 | 5e-4 | 1e-4 |
| `--epochs` | 200 | 200 | 100 | 100 | 100 |
| `--scheduler` | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| `--amp_dtype` | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 |
| `--mask_ratio` | 0.6 | — | — | — | — |
| `--teacher_momentum` | — | 0.996→1.0 | — | — | — |
| `--dino_momentum` | — | — | 0.996→1.0 | — | — |
| `--byol_momentum` | — | — | — | 0.996→1.0 | — |
| `--temperature` | — | — | — | — | 0.07 |
| `--samples_per_class` | — | — | — | — | 16 |

**Tips:**
- Start with 200 epochs for SimMIM/I-JEPA. DINO, BYOL, and SupCon converge
  faster (~100).
- `--temperature 0.07` is the standard from the original SupCon paper.
  Lower = sharper contrastive distribution; try 0.05–0.1.
- DINO and BYOL train on smaller crops (224) with a larger LR because their
  loss acts on a small projection space rather than every pixel; the defaults
  above match the method-specific `--dino_*` / `--byol_*` flags.
- `--samples_per_class 16` with `--batch_size 64` gives 4 classes per batch
  on a 7-class dataset. batch_size need not be divisible by
  samples_per_class; the remainder is spread evenly across the per-class
  groups so each batch still contains exactly batch_size samples.
- Use `--amp_dtype bfloat16` if your GPU supports it (Ampere+). Otherwise
  `float16` with GradScaler works too.

### Example: Full Pre-Training Pipeline

<!-- genml-cli-test
name: pretrain-full-pipeline
network: true
override:
  - --datasets {tmp}/img
  - --dataset {tmp}/img
  - --label_column label
  - -.--source_checkpoint
extra: --epochs 1
-->
```bash
# Step 1: Pre-train with SimMIM on two large image collections.
# v4.2: pre-training is a *configuration* of the unified genml-kit-train
# CLI (--pipeline images + --method simmim), not a separate binary.
genml-kit-train --pipeline images --method simmim \
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

<!-- genml-cli-test
name: pretrain-supcon
network: true
override:
  - --datasets {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline images --method supcon \
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
| `--method` | `classification` | Objective. The pre-training examples below pass `--method simmim` explicitly. Choices: `classification`, `simmim`, `ijepa`, `dino`, `byol`, `supcon`, `vo_pair`. |
| `--model` | `convvit` | Model name registered in genml_kit or HuggingFace model ID. |
| `--datasets` | (required) | Space-separated dataset names or local paths. |
| `--cache_dir` | `None` | HuggingFace cache directory for downloads. |
| `--remote_checkpoint` | `None` | Remote URI for checkpoint sync (`gs://BUCKET/PREFIX`, `r2://BUCKET/PREFIX`, or `s3://BUCKET/PREFIX`). |
| `--hf_token` | `None` | HuggingFace token for gated datasets (or set `HF_TOKEN` env var). |
| `--image_column` | auto-detected | Explicit HF image column name. |
| `--label_column` | `None` | Explicit HF label column name. Required by `--method supcon` if non-standard. |
| `STRICT_DATASETS` (env) | `False` | Abort on first dataset-loading failure instead of skipping. |
| `--image_size` | `448` | Input image size (square). |
| `--batch_size` | `32` | Per-GPU batch size. |
| `--seed` | `None` | Explicit RNG seed: pins the shuffling and enables full determinism (cuDNN deterministic kernels, benchmark off). Omit (default) to keep runs fast — RNG streams are still seeded from `$GENML_KIT_SEED` (default 42). See [Reproducibility](#reproducibility). |
| `--epochs` | `5` | Number of epochs. The recipes below override this for pre-training. |
| `--lr` | `3e-5` | Peak learning rate (base). The recipes below override this. |
| `--amp_dtype` | `None` | Mixed precision: `float16` or `bfloat16`. Omit to disable. |
| `--num_workers` | `4` | DataLoader worker processes. |
| `--device` | auto-detect | Device: `cpu`, `cuda`, or `cuda:INDEX`. |
| `--resume` | `True` | Auto-resume from latest checkpoint. Use `--no-resume` to disable. |
| `--state_save` | `opt,sched,amp` | States to save: `opt`, `sched`, `amp`, `none`. |
| `--state_load` | `opt,sched,amp` | States to restore on resume: `opt`, `sched`, `amp`, `none`. |
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
| `--model_arg` | `{}` | Override model configuration (repeatable). For `--model cls_model_wrapper:<hf_name>` the classifier head travels here too: `--model_arg classifier=mlp:hidden=512,dropout=0.3`. |
| `--proc_arg` | `{}` | Override processor configuration (repeatable). |
| `--optimizer` | `AdamW` | `torch.optim` optimizer class name or `.py` script path. |
| `--opt_arg` | `{}` | Extra optimizer kwargs (repeatable). |
| `--scheduler` | `None` | `torch.optim.lr_scheduler` class name or `.py` script path. |
| `--sched_arg` | `{}` | Extra scheduler kwargs (repeatable). |
| `--source_checkpoint` | `None` | Path to source checkpoint to absorb parameters from. |
| `--param_rename` | `None` | Regex-based key rename patterns (`SEARCH;REPLACE`). |
| `--grad_checkpoint` | `False` | Enable gradient checkpointing. Reduces activation memory by ~40-50% at the cost of ~25-35% more compute per step. Enables larger batch sizes. See [Gradient Checkpointing](#gradient-checkpointing). |

**Defaults shown are the generic `genml-kit-train` defaults; the
pre-training recipes below override `--method`, `--epochs`, `--lr` for their
specific objective.** The pre-training examples invoke
`genml-kit-train --pipeline images --method simmim ...` explicitly.

**SupCon-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--proj_dim` | `128` | Output dimensionality of the projection head. |
| `--proj_hidden` | `None` | Hidden layer size of the projection MLP. `None` = single linear layer. |
| `--temperature` | `0.07` | NT-Xent temperature. Lower = sharper contrastive distribution. |
| `--samples_per_class` | `16` | Samples per class in each batch. Batch size need not be divisible by this; the remainder is spread evenly across the per-class groups. |

### Dataset Ensemble

With `--pipeline images`, `genml-kit-train` stitches multiple datasets into
a single pre-training corpus (via `--datasets`). This is useful because no
single dataset is large enough for effective pre-training on its own.

Supported dataset types:
- **HuggingFace datasets** — any HF dataset ID that returns decoded image
  data (e.g. `cifar10`, `food101`). Gated datasets require `--hf_token` or
  `HF_TOKEN`.
- **Local image directories** — pass a path to a folder of images
  (ImageFolder format).

Datasets are loaded lazily (only when first accessed). By default, datasets
that fail to load are logged and skipped (best-effort mode). Set the
`STRICT_DATASETS` environment variable to a truthy value (`1`, `true`,
`yes`, `on`) to abort on the first failure instead.

Images that cannot be decoded are skipped with a warning — this prevents a
single corrupted file from blocking an entire pre-training run.

#### Label Validation

When using a label-aware method (`--method supcon`), the ensemble validates
that every dataset supports labels. Datasets without a label column cause a
clear error *before* training begins, not a cryptic runtime failure mid-epoch.

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

<!-- genml-cli-test
name: pretrain-prepared-images
network: true
override:
  - --datasets {tmp}/img
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline images --method simmim \
                --model convvit \
                --datasets ./prepared_images ./other-images \
                --image_size 448 --batch_size 32 --epochs 20
```

For a concrete worked example of such a preparation script, see
`scdiag/scripts/prepare_derm1m.py` and `scdiag/scripts/prepare_ham10000.py`
in the repository (dermoscopy corpora).

---

## Fine-Tuning Guide

After pre-training (or directly, if you skip pre-training), fine-tune a
classifier on your labeled dataset.

### What Fine-Tuning Is Changing

Suppose the encoder produces the feature vector $`h = f_\theta(x)`$. A linear
classification head computes logits $`a = W h + b`$, and softmax turns them
into probabilities:

$$
\large
p(y=c \mid x) = \frac{\exp(a_c)}{\sum_k \exp(a_k)}
$$

where:

- $`h \in \mathbb{R}^{D}`$ — the encoder's feature vector for image $`x`$;
- $`W \in \mathbb{R}^{C \times D}`$, $`b \in \mathbb{R}^{C}`$ — the head's weight
  matrix and bias ($`C`$ = number of classes, $`D`$ = feature dimension);
- $`a_c`$ — the logit (raw score) for class $`c`$;
- $`p(y=c \mid x)`$ — the model's probability that the class of $`x`$ is
  $`c`$; the softmax exponentials make the scores positive and summing to one.

Training minimizes cross-entropy between the model's predicted distribution
$`p`$ and the target distribution $`q`$ over the $`C`$ classes,

$$
\large
\mathcal{L}_{\text{CE}}(p, q) = -\sum_{c=1}^{C} q_c \log p_c
$$

For a hard label the target $`q`$ is a one-hot vector, so only the term of the
true class survives and the loss reduces to the familiar $`-\log p(y \mid x)`$.
genml_kit generalizes this in three directions via `CombinedFocalLoss`:

- **Soft targets.** With Mixup (below) $`q`$ is the blend
  $`\lambda\, y_i + (1-\lambda)\, y_j`$, and the sum keeps both classes'
  terms weighted by $`\lambda`$ and $`1-\lambda`$.
- **Label smoothing** (via `--loss_args '{"label_smoothing": ...}'`):
  the one-hot target is replaced by
  $`q_c = 1 - \varepsilon + \varepsilon/C`$ for the true class and
  $`\varepsilon/C`$ for the others. The model is no longer asked to drive
  $`p_c \to 1`$ exactly, which reduces overconfidence and improves
  calibration on validation data.
- **Class weights** (`--class_multipliers`): each term of the sum is scaled by
  a per-class factor $`w_c`$, so rare classes contribute larger gradients.

The new classifier head is normally initialized from scratch because
its output
size depends on the target classes. When loading a pre-training checkpoint,
the useful part to transfer is the encoder; an unused SupCon projection head
should not be copied into the classifier.

The practical choice is how much of $`\theta`$ to update:

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
  that truly belong to it:
  $`\text{precision} = TP / (TP + FP)`$.
- **Recall:** for a class, the fraction of images belonging to that class that
  are predicted correctly: $`\text{recall} = TP / (TP + FN)`$.
  Here $`TP`$ (true positives) counts images of the class that were correctly
  predicted as that class, $`FP`$ (false positives) counts images of *other*
  classes wrongly predicted as this class, and $`FN`$ (false negatives) counts
  images of this class wrongly predicted as something else.
- **F1:** the harmonic mean of precision and recall:
  $`F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}`$.
  When the denominator is zero, scikit-learn's `zero_division=0` behavior
  reports zero.
- **Macro F1:** the arithmetic mean of the per-class F1 scores:
  $`F_{1}^{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F_{1,c}`$.
  Every class contributes equally, regardless of its validation-set size
  ($`C`$ is the number of classes, $`F_{1,c}`$ the F1 of class $`c`$).
  This is the metric used for best-checkpoint selection.
- **Weighted F1:** the mean of per-class F1 scores weighted by each class's
  validation support. It is therefore more influenced by common classes.
- **Balanced accuracy:** the arithmetic mean of per-class recall:
  $`\text{balanced accuracy} = \frac{1}{C} \sum_{c=1}^{C} \text{recall}_c`$.
  It gives each class equal weight and is useful for imbalanced datasets.
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

<!-- genml-cli-test
name: fine-tune-basic
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```

### With A Custom Classifier Head

Replace the default linear head with a custom MLP or attention-based
classifier:

<!-- genml-cli-test
name: fine-tune-custom-head
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
# Freeze backbone, train only the custom head
genml-kit-train --model cls_model_wrapper:google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --image_size 224 \
                --model_arg classifier=mlp:hidden=512,dropout=0.3 \
                --freeze ".*\.(head|pool)"
```

### With LoRA (Parameter-Efficient Fine-Tuning)

Freeze the entire backbone and train only small low-rank adapter matrices.
Reduces trainable parameters by ~97% while often matching full fine-tuning:

<!-- genml-cli-test
name: fine-tune-lora
network: true
extras: lora
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --model_arg classifier=mlp:hidden=512 \
    --lora --lora_r 16 --lora_alpha 32 \
    --lora_target_modules "query,key,value" \
    --freeze "classifier\.(head|pool|encoder)" \
    --lr 3e-5 \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --epochs 20
```

### From A Pre-Trained Checkpoint

Load encoder weights from a pre-training run (SimMIM, I-JEPA, or SupCon):

<!-- genml-cli-test
name: fine-tune-from-checkpoint
override:
  - --dataset {tmp}/img
  - --label_column label
  - -.--source_checkpoint
extra: --epochs 1
-->
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

### Layer-Wise Learning Rate Decay (LLRD)

LLRD is a compromise between freezing the backbone and updating every layer at
the same speed. If layers are indexed from shallow 0 to deep $`L`$, a common
schedule is (introduced by Howard & Ruder, [Universal Language Model Fine-tuning](https://arxiv.org/abs/1801.06146), ACL 2018):

$$
\large
\text{lr}(\text{layer}) = \text{lr}_{\text{base}} \cdot d^{\\,L - \text{layer}}
$$

where:

- $`\text{lr}_{\text{base}}`$ — the `--lr` you pass on the command line;
- $`d`$ — the decay factor (`--llrd_decay`), usually between 0.8 and 1.0;
- $`L`$ — the index of the deepest layer, $`\text{layer}`$ the current layer's
  index, both counted from 0 at the input side.

The deepest layer
receives the base rate while earlier layers receive smaller updates. Early
layers tend to represent general edges and textures; later layers are more
task-specific. This is a useful heuristic, not a law, so validate it on your
dataset. A very small decay factor can effectively freeze the shallow network.

### Cosine Annealing

`--scheduler CosineAnnealingLR` (the default in every example above) decays
the learning rate from its initial value $`\eta_{\max}`$ toward $`\eta_{\min}`$
(0 by default) following a half cosine wave over `--epochs` epochs $`T`$:

$$
\large
\eta(t) = \eta_{\min} + \frac{1}{2}\left(\eta_{\max} - \eta_{\min}\right)
\left(1 + \cos\left(\frac{t \pi}{T}\right)\right)
$$

where $`t`$ is the current epoch. The cosine shape spends more time at high
learning rates than a linear decay — useful early, when large steps explore
the loss surface — and slows down smoothly as it approaches zero, which lets
the model settle into a minimum instead of bouncing around it. Late-training
low learning rates are also what make the EMA teacher of DINO/BYOL
effective: the teacher tracks a student that changes gently.

### Mixup, Label Smoothing, And Imbalance

Mixup forms a virtual example from two training examples.
Reference: Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412), ICLR 2018.

$$
\large
\tilde{x} = \lambda x_i + (1 - \lambda) x_j
$$

$$
\large
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
$$

where $`(x_i, y_i)`$ and $`(x_j, y_j)`$ are two randomly paired training
examples and the mixing coefficient $`\lambda`$ is drawn from the Beta
distribution, $`\lambda \sim \mathrm{Beta}(\alpha, \alpha)`$, with $`\alpha`$
set by `--mixup_alpha` (typical values 0.2–0.4). The parameter $`\alpha`$
shapes the distribution: small $`\alpha`$ puts most weight near 0 or 1,
large $`\alpha`$ near 1/2. The labels are probability vectors, not
class indices. Mixup smooths the decision boundary and can help on small
datasets, but strong Mixup can obscure fine-grained image details. Label
smoothing similarly replaces a one-hot label with a mostly-correct
distribution.

#### Focal Loss

Focal loss instead changes the emphasis of cross-entropy. Reference: Lin et
al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), ICCV 2017.
With a hard label and $`p_t`$ the predicted probability of the correct class,
its basic form is

$$
\large
\mathcal{L}_{\text{focal}} = -(1 - p_t)^{\gamma} \log(p_t)
$$

where $`\gamma`$ (via `--loss_args '{"gamma": ...}'`) down-weights easy examples: when
$`p_t \to 1`$ the factor $`(1-p_t)^{\gamma} \to 0`$, so already-confident,
easy examples receive almost no weight and training focuses on hard ones.
With $`\gamma = 0`$ the factor is 1 for every example and the loss is plain
cross-entropy; typical values are 1–3, and 2.0 is the usual default.

genml_kit implements the general form (`CombinedFocalLoss` in
`genml_kit/losses/focal.py`), which composes cleanly with the other
tools in this section. Let $`q`$ be the target distribution from the previous
step (one-hot, Mixup-blended, or label-smoothed), $`w_c`$ the per-class
multipliers (`--class_multipliers`), and $`p`$ the model's softmax output.
The loss for one example is

$$
\large
\mathcal{L} = \left(1 - p_t\right)^{\gamma} \sum_{c=1}^{C} w_c\\, q_c
\left(-\log p_c\right), \qquad p_t = \sum_{c=1}^{C} q_c\\, p_c
$$

Two details matter for the reader combining these flags:

- **Unified $`p_t`$.** The modulating factor uses the *expected* probability
  under the target distribution, $`p_t = \sum_c q_c p_c`$, rather than the
  probability of a single class. For a one-hot target $`q`$ this reduces
  exactly to the basic form above ($`p_t`$ becomes the correct-class
  probability); with Mixup soft targets it blends the two classes'
  probabilities, giving one smooth modulation per example instead of an
  arbitrary single-class choice.
- **One modulation per example.** The factor $`(1-p_t)^{\gamma}`$ scales the
  whole weighted cross-entropy sum — weights and soft labels act *inside* the
  sum, focal acts *outside* it. Each term is therefore attenuated
  proportionally to how confident the model already is about that example, no
  matter which class the probability mass sits on.

Use these tools deliberately; combining every regularizer is not
automatically better.

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
- `--loss_args '{"gamma": 2.0}'` down-weights easy examples, useful when
  classes are imbalanced.
- `--class_multipliers "rare_class=3.0"` increases the loss weight for
  safety-critical or otherwise priority classes.

### Reproducibility

RNG streams are always seeded.  By default the seed comes from the
`GENML_KIT_SEED` environment variable (default `42`), and it drives
DataLoader shuffling, per-worker augmentation randomness, mixup, and
dropout — the user not caring about determinism does not mean the
toolkit runs on entropy.  To compare hyperparameters under a different
random draw, change `GENML_KIT_SEED` or pass an explicit seed.

Two levels of determinism:

| Invocation | Seeding | CUDA kernels | Reproducible? |
|---|---|---|---|
| default (no `--seed`) | `GENML_KIT_SEED` or 42 | `cudnn.benchmark=True` (fastest algorithms) | RNG streams identical; arithmetic not bit-exact |
| `--seed 42` | 42 everywhere | cuDNN deterministic mode, benchmark off, deterministic algorithms | bit-exact |

Passing an explicit `--seed` therefore buys full determinism at a
throughput cost (typically 10–20%); omit it for maximum speed:

```bash
genml-kit-train --model convvit --seed 42 ...
```

The train/val split and the XGBoost stage follow the user's seed: with
`--seed` given they are pinned; without it they draw fresh entropy each
run (HF `train_test_split(seed=None)`, XGBoost `random_state=None`).

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
| `--loss` | `None` | Loss spec: registered name (`focal`, `supcon`, or the RL bundles `ppo`/`sac`/`dqn`) or a path/URL to a `.py` script defining `build_loss(**kwargs)`. `None` (default) selects the method's built-in loss. |
| `--loss_args` | `{}` | JSON kwargs forwarded to the loss builder. Example: `'{"gamma": 1.0, "label_smoothing": 0.1}'`. |
| `--class_multipliers` | `""` | Per-class priority multipliers. Example: `"cat=3.0,dog=1.0"`. |
| `--sampler` | `none` | Training sampler: `none` (shuffle), `weighted` (WeightedRandomSampler for class imbalance), or `balanced` (equal samples per class per batch; batch_size need not divide evenly). |
| `--sampler_weights` | `frequency` | Weight mode for `--sampler weighted`: `frequency` (inverse-freq), `multipliers` (--class_multipliers), or `combined` (freq × multipliers). |
| `--mixup_alpha` | `0.0` | Mixup alpha (`0` = disabled; recommended: `0.2`). |
| `--seed` | `None` | Explicit RNG seed: pins the train/val split and the XGBoost stage and enables full determinism (cuDNN deterministic kernels, benchmark off). Omit (default) to keep runs fast — shuffling/mixup/dropout are still seeded from `$GENML_KIT_SEED` (default 42). See [Reproducibility](#reproducibility). |
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
use boto3), or `genml_kit[gcs]` for `gs://`. The unified `genml-kit-train`
CLI accepts the flag for every pipeline/method configuration.

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
trainable low-rank matrices into attention layers.
Reference: Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.

Instead of updating a weight matrix $`W`$ directly, LoRA keeps it frozen and
adds a low-rank product to its output:

$$
\large
W' = W + \Delta W, \qquad \Delta W = \frac{\alpha}{r} \cdot B A
$$

where:

- $`W \in \mathbb{R}^{d \times k}`$ — the frozen pre-trained weight matrix;
- $`A \in \mathbb{R}^{r \times k}`$, $`B \in \mathbb{R}^{d \times r}`$ — the two
  small trainable matrices ($`BA`$ is a matrix product);
- $`r`$ — the rank (`--lora_r`), much smaller than $`d`$ or $`k`$, which is
  what makes $`\Delta W`$ cheap: instead of $`d \times k`$ weights it trains
  only $`r(d + k)`$;
- $`\alpha`$ — the scale factor (`--lora_alpha`); the effective step size of
  the update is $`\alpha / r`$.

Only $`A`$, $`B`$ (and the classifier head) receive gradients.

| r | alpha | alpha/r | Use case |
|---|---|---|---|
| 8 | 16 | 2.0 | Conservative, very few params |
| 16 | 32 | 2.0 | Good default for medium datasets |
| 16 | 64 | 4.0 | Larger updates (helpful under bfloat16) |
| 32 | 64 | 2.0 | More capacity, same scaling |

LoRA can be combined with a custom classifier:

<!-- genml-cli-test
name: lora-custom-classifier
network: true
extras: lora
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --model_arg classifier=cls_attention:num_encoder_layers=2 \
    --lora --lora_r 16 --lora_alpha 32 \
    --freeze 'classifier\.(head|pool|encoder)' \
    --lr_group 'backbone.*=1e-5' 'classifier.*=3e-4' \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --epochs 20
```

### Cross-Dataset Resume

Switch from one dataset to another while keeping backbone weights:

<!-- genml-cli-test
name: cross-dataset-resume
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
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

<!-- genml-cli-test
name: gradient-checkpointing
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train \
    --model timm:eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --grad_checkpoint \
    --amp_dtype bfloat16 \
    --epochs 20
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

<!-- genml-cli-test
name: inference-guide
network: true
override:
  - --checkpoint {tmp}/chk_infer/model_best.pt
  - --model convvit --image_size 32
-->
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

### XGBoost And Test-Time Augmentation

The neural classifier makes decisions through its head. The XGBoost option
takes the encoder representation $`h = f_\theta(x)`$ instead and fits an
ensemble of decision trees to those vectors.
Reference: Chen & Guestrin, [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754), KDD 2016.
A tree partitions feature space with rules such as $`h_{37} < t`$ (comparing
the 37th feature of $`h`$ against a learned threshold $`t`$); boosting adds
trees sequentially so each new tree focuses on errors left by previous trees.
This can work well when the dataset is small
and the representation is already useful, but it is not guaranteed to beat the
neural head. Use validation data to choose tree depth, number of rounds, and
ensemble weight.

Test-time augmentation (TTA) runs the same image through several plausible
views and averages the probability vectors:

$$
\large
\bar{p}(y \mid x) = \frac{1}{K} \sum_{k=1}^{K} p(y \mid T_k(x))
$$

where $`T_1, \dots, T_K`$ are the $`K`$ test-time transformations (e.g. the
original image plus flipped copies) and $`\bar{p}`$ is their averaged
probability vector: each transformed copy is classified independently, and the
final prediction averages the per-view class probabilities. The
transformations should preserve the label semantics. Horizontal flips
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

### `--checkpoint` And `--source_checkpoint` Are Different

`--checkpoint` names the output prefix used for saving and resuming the current
training run. `--source_checkpoint` imports weights from another run before
the new optimizer is created. When moving a SupCon encoder into classification,
use:

<!-- genml-cli-test
name: param-rename
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
  - --checkpoint {tmp}/ckpt_param_rename
  - -.--source_checkpoint
extra: --epochs 1
-->
```bash
genml-kit-train \
    --model google/vit-base-patch16-224 \
    --image_size 224 \
    --checkpoint /content/eva02_finetune \
    --source_checkpoint /content/eva02_supcon_latest.pt \
    --param_rename 'encoder\\.model\\.(.*);model.$1' \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --epochs 20
```

The rename is needed because the pre-training wrapper stores backbone keys
under `encoder.model.*`, whereas the fine-tuning model stores them under
`model.*`. The SupCon `projection.*` keys are expected to remain unused, and
new classifier-head keys are expected to be missing from the source checkpoint.
Those messages indicate a successful partial transfer if the backbone keys
are matched.

### Interpreting A SupCon Plateau

SupCon loss is not expected to approach zero. If an anchor has k-1 positives and
positives become much more similar than negatives, a useful reference value is
approximately `-log(k-1)`. This is only a diagnostic approximation: class
counts may vary, the sampler may repeat examples, and temperature affects
optimization. Judge the loss together with embedding quality and downstream
validation metrics. A plateau near this reference can mean convergence; a
plateau well above it can mean too few positives, a learning-rate problem, or
labels that are not being passed correctly.

### A Practical Debugging Order

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

### Low GPU Utilization (`util=0` In The Training Log)

The periodic `GPU:` line reports memory in MiB and SM utilization as a
percentage sampled at log time (`mem=allocated/totalMiB (pct) res=…MiB
util=…%`). `mem` counts live tensors, `res` is the caching allocator's
pool, and `util` is one instant — not an average — so an isolated 0 is
normal. A *persistent* pattern of `util=0` at every log line, with
`mem`/`res` byte-identical between samples, means the GPU finished the
last batch and is idle waiting for the next one: the run is **input
bound**, not compute bound. The throughput line (`img/s=`) is the ground
truth for step time.

Common causes and fixes, in order of likelihood:

- **Too few DataLoader workers.** JPEG decode + augmentation is CPU work;
  with `--num_workers 4` a 4090 can easily outrun the pipeline at
  448². Check `nproc` on the machine and raise `--num_workers` toward
  the core count — `img/s` should scale until the GPU saturates
  (a `util` that now sits consistently above ~90% means you are done).
- **Expensive decode.** Full-resolution JPEGs dominate decode cost.
  Pre-resizing the dataset offline (e.g. ~512px long side) cuts decode
  work several-fold and often doubles throughput on small VRAM budgets.
- **Slow storage.** A network volume (S3/R2 mount, NFS) adds latency per
  image; prefer local NVMe for the dataset.
- **`--seed`.** It enables cuDNN deterministic mode and disables
  benchmarking; expect slower steps by design.  Omit it when you do not
  need bit-exact runs.

Fine-tuning runs on frozen-backbone models need very little GPU memory
(`mem` is typically a few hundred MiB), so a low `mem` number is normal
there and not a sign of a problem. Loader prefetch (`prefetch_factor=4`
per worker) and `cudnn.benchmark=True` (default; disabled by an explicit
`--seed`) already help absorb jitter and speed up the GPU side.

---

## Gradient Monitor

`--grad_monitor N` logs a per-parameter gradient report every *N* training
steps. This helps diagnose training instability (exploding / vanishing
gradients, imbalanced parameter updates) before it shows up in the loss.

### Summary Line

```
[Step 29400] Gradient Report: 202 params | grad_rms: mean=5.68e-01 max=3.41e+00 min=1.66e-05 | grad/param: mean=2.94e-01
```

All norms are **RMS** (root mean square): the $`L_2`$ norm of the tensor
divided by $`\sqrt{N}`$, where $`N`$ is the number of elements (`numel`):

$$
\large
\mathrm{RMS}(v) = \frac{\lVert v \rVert_2}{\sqrt{N}} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} v_n^2}
$$

Here $`v`$ is the tensor (a parameter or its gradient) flattened to $`N`$
values $`v_1, \dots, v_N`$, and $`\lVert v \rVert_2 = \sqrt{\sum_n v_n^2}`$ is
its Euclidean norm. Dividing by $`\sqrt{N}`$ makes the measure independent of
tensor shape and directly comparable across parameters of different sizes.

| Field | Meaning |
|---|---|
| `params` | Total number of trainable parameters. |
| `grad_rms: mean/max/min` | RMS of the gradient tensor for each parameter, then aggregated. `max` is the single most aggressive gradient — the one most likely to cause instability. |
| `grad/param: mean` | Average gradient-to-parameter ratio. Healthy: < 0.1. Concerning: > 1.0. |

### Per-Parameter Columns

| Column | Symbol | What to look for |
|---|---|---|
| **g_rms** | $`\mathrm{RMS}(\nabla_W \mathcal{L})`$ | Compare across params. One param with g_rms 100× higher is a problem. |
| **p_rms** | $`\mathrm{RMS}(W)`$ | Per-element scale context. With `std=0.02` init, expect ~0.02. |
| **g/p** | $`\lVert \nabla_W \mathcal{L} \rVert_2 / (\lVert W \rVert_2 + \varepsilon)`$ | **Most useful column.** Healthy: < 0.1. Concerning: > 1.0 (update overshoots). Dangerous: > 5.0. |
| **g_max** | $`\max_n \lvert \partial \mathcal{L} / \partial W_n \rvert`$ | Highlights individual neurons with extreme gradients. |
| **sparse** | `% zero` | High sparsity (> 50%) = most neurons not receiving signal. |
| **status** | | `OK` = healthy. `STL` = stalled. `OVF` = exploding. `IMB` = imbalanced. `GPR` = high g/p ratio. |

In this table $`W`$ denotes one parameter tensor, $`\nabla_W \mathcal{L}`$ its
gradient (the vector of partial derivatives of the loss with respect to each
element of $`W`$), and $`\varepsilon`$ a tiny constant that only prevents
division by zero for parameters that are exactly zero.

### Reading The Report

**Healthy training:** g/p < 0.1, `grad/param: mean` in 0.01–0.1 range, g_rms within ~10× across layers.

**Exploding gradients:** One or more params with `OVF`, g/p > 5.0, `grad_rms: max` >> `mean`. Fix: lower LR, add `--grad_clip 1.0`, or use warmup.

**Vanishing gradients:** Many params with `STL`, g_rms near 1e-7, high sparsity. Fix: increase LR, check for dead neurons.

**Imbalanced updates:** Some params `IMB`, large g/p disparity between layers. Fix: use `--lr_group` for different rates, or freeze the dominant component.

### Norm Trend History

`--norm_history N` (requires `--grad_monitor`) keeps the last N snapshots
per parameter. A trend summary table is appended to each report showing
direction (`UP`/`DOWN`/`---`), percentage change, and min/max values.

---

## Custom Losses

Losses are resolved through the `LOSSES` registry
(`genml_kit.losses.registry.load_loss`).  A loss spec is either a
registered name (`focal`, `supcon`, or the RL bundles `ppo`, `sac`,
`dqn`) or a path/URL to a `.py` script defining `build_loss(**kwargs)`:

```python
# my_loss.py
import torch.nn as nn

class MyLoss(nn.Module):
  def __init__(self, alpha=1.0):
    super().__init__()
    self.alpha = alpha

  def forward(self, logits, targets):
    return self.alpha * nn.functional.cross_entropy(logits, targets)

def build_loss(alpha=1.0, **kwargs):
  """Protocol: return a callable (logits, targets) -> tensor."""
  return MyLoss(alpha=alpha)
```

<!-- genml-cli-test
name: custom-loss
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train --model google/vit-base-patch16-224 \
  --dataset my-org/my-labeled-images --label_column category \
  --image_size 224 \
  --loss ./my_loss.py --loss_args '{"alpha": 2.0}' \
  --epochs 20
```

**RL losses** work the same way but return a *bundle* (dict of
callables); the script's entries are merged over the method's built-in
bundle (the method passes its own name as the merge base: `ppo`, `sac`,
or `dqn`), so a script may override just one loss and inherit the rest:

```python
# my_ppo_entropy.py
def build_loss(**kwargs):
  return {"entropy": lambda ent: ent.mean() * 2.0}  # override only entropy
```

<!-- genml-cli-test
name: rl-custom-loss
extras: rl
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline rl --method ppo --env_id CartPole-v1 \
  --loss ./my_ppo_entropy.py
```

The built-in bundles are merged over (`ppo`: `policy`/`value`/`entropy`;
`sac`: `q`/`policy`/`alpha`; `dqn`: `td`).  The `--loss_args` JSON
kwargs are forwarded to `build_loss(**kwargs)`.

## Custom Models

genml_kit supports any HuggingFace `AutoModelForImageClassification` model,
any timm model via `timm:<name>`, and custom architectures registered in
`genml_kit.models`.

### Built-In Custom Models

| Name | Description | `--model` value |
|---|---|---|
| timm | Any model from [timm](https://github.com/huggingface/pytorch-image-models) | `timm:<model_name>` |
| ConvViT | Multi-block conv stem + ViT encoder with CLS-guided attention pooling | `convvit` |
| UVito | Frozen SMP encoder + learnable patch projection + Transformer encoder | `uvito` |
| ClsModelWrapper | HuggingFace backbone + custom classifier head | `cls_model_wrapper:<hf_name>` |
| ContrastiveEncoder | Backbone + projection head for contrastive pre-training | `contrastive_encoder:<hf_name>` |

### Adding A Custom Model

1. Create `genml_kit/models/{name}/` with `model.py`, `processor.py`,
   `loader.py`, and `__init__.py`.
2. Add the import to `genml_kit/models/__init__.py`.
3. The model must expose `.forward(pixel_values=images)` → object with `.logits`,
   and `config.id2label` / `config.label2id`.
4. CLI overrides via `--model_arg KEY=VALUE` are forwarded to the loader.

### External Model Scripts

Instead of registering a model inside the source tree, you can point
`--model` at a `.py` file (or HTTP/HTTPS URL).  The script must define
`create_model`:

```python
# my_model.py
import torch.nn as nn

class Tiny(nn.Module):
  def __init__(self, num_labels, hidden=128):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(784, hidden), nn.ReLU(),
                             nn.Linear(hidden, num_labels))

  def forward(self, pixel_values):
    return pixel_values

def create_model(num_labels, image_size=224, id2label=None, label2id=None,
                 checkpoint_path=None, device="cpu", **kwargs):
  """Protocol: return a torch.nn.Module."""
  return Tiny(num_labels=num_labels, hidden=kwargs.get("hidden", 128))
```

<!-- genml-cli-test
name: custom-model
network: true
override:
  - --dataset {tmp}/img
  - --label_column label
extra: --epochs 1
-->
```bash
genml-kit-train --model ./my_model.py --loss focal \
  --dataset my-org/my-labeled-images --label_column category \
  --epochs 20
```

The script is **fully responsible for weight loading**: when
`checkpoint_path` is provided (inference), call
`genml_kit.io.checkpointing.load_checkpoint_weights(checkpoint_path, model, ...)`
inside `create_model`, mirroring the ConvViT/UViTO loaders.  In training
`checkpoint_path` is `None` (random init) and the global `--source_checkpoint`
/ `--resume` paths apply uniformly after `create_model` returns.

An optional `create_processor(image_size=224, **kwargs)` may be defined
for the processor; it must return an object exposing `image_mean` /
`image_std`.

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

The head is selected through `--model_arg`, not a dedicated flag:

```bash
--model cls_model_wrapper:<hf_name> --model_arg classifier=mlp:hidden=512,dropout=0.3
```

The `classifier` value is an inline spec: the head name (a registered
built-in \u2014 `mlp`, `cls_attention` \u2014 or a path to a `.py` file defining a
`create_classifier(num_labels, hidden_size, **kwargs)` function), optionally followed by `:key=value,...` kwargs. Values
are parsed as int/float/bool, and bracketed lists such as
`cls_slice=(0, 1)` work. A head is required: `cls_model_wrapper` fails fast
with a clear message when `classifier` is missing. Models that embed their
own head (`convvit`, `uvito`, `timm:<name>`) do not accept this kwarg — for
a custom head on a HuggingFace backbone use `cls_model_wrapper` instead.
`genml-kit-infer` needs the same `--model_arg classifier=...` to rebuild the
head before loading the checkpoint weights.

---

## Semantic Segmentation

Segmentation predicts a class for **every pixel**: an image `(B, 3, H, W)`
goes in and per-pixel logits `(B, num_classes, H, W)` come out.  `genml_kit`
models this as `encoder + decoder`: the encoder is any hierarchical timm
backbone (ResNet / ConvNeXt / Swin / MiT) emitting a feature pyramid, and
the decoder fuses that pyramid back up to full resolution (with skip
connections for `unet`/`fpn`).

### Quick start

<!-- genml-cli-test
name: segmentation-quick-start
network: true
override:
  - --dataset segfolder/{tmp}/seg
extra: --epochs 1
-->
```bash
genml-kit-train \
  --pipeline segmentation --method segmentation \
  --dataset segfolder//data/voc2012 \
  --model seg_wrapper:timm/resnet34 \
  --model_arg decoder=unet:depth=4 \
  --loss dice_ce \
  --image_size 512 --batch_size 8 --epochs 40
```

The layout expected by `segfolder/` is a paired tree:

```
/data/voc2012/images/2007_000032.jpg
/data/voc2012/masks/2007_000032.png
```

or, for a corpus with named splits, `train/`, `val/` and `test/`
sub-directories each containing `images/` + `masks/`.  Image and mask are
matched on the file stem (case-insensitive) and a mismatch fails fast.

HuggingFace datasets that carry a mask feature are used with `--mask_column`:

<!-- genml-cli-test
name: segmentation-mask-column
network: true
override:
  - --dataset {tmp}/hf_seg
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline segmentation --method segmentation \
  --dataset scene_parse_150 --mask_column annotation \
  --model seg_wrapper:timm/convnext_tiny --model_arg decoder=fpn
```

### Class-id compaction

Masks are expected to carry integer class ids, but the ids do **not** have
to be dense.  The pipeline scans every mask (train **and** val, so a class
that only appears in validation is still mapped) and compacts whatever ids
it finds into a dense `[0, C)` range:

| Source id | Compact id |
|---|---|
| 12 | 0 |
| 17 | 1 |
| 255 (= ignore) | dropped |

- **segfolder path:** the mapping is rebuilt on every run (a full scan)
  and remapped through `apply_mask_map` on every access.
- **HuggingFace path:** the `{raw → compact}` map is cached to disk under
  `<cache_dir>/seg_class_maps/segclassmap__<hash>.json`, or
  `~/.cache/genml_kit/seg_class_maps/` when no `--cache_dir` is given
  (falling back to a temp dir if that is unwritable).  The cache is keyed
  on the dataset spec **and** a content hash of the masks themselves, so
  editing a mask invalidates the entry even if the spec string is
  unchanged.

The class count reported by training (and the default for
`--seg_num_classes`) is the **compact** count, i.e. the number of
distinct ids found after dropping `--seg_ignore_index` (`len(mask_map)`).
(The count falls back to `max_label + 1` over a sample of loader masks
only when the pipeline did not build a mask map, e.g. a method wired
without the segmentation pipeline.)

### Data augmentation

The default transforms for the `segmentation` method (in
`genml_kit/datasets/transforms.py`) are split into a geometric stage and
a photometric stage:

- **Geometric stage** — applied to the image **and** the mask together,
  so they always stay pixel-aligned:
  - `v2.RandomResizedCrop(image_size, scale=(0.5, 2.0),
    ratio=(0.75, 4/3))`
  - `v2.RandomHorizontalFlip(p=0.5)`
- **Photometric stage** — applied to the **image only** (masks contain
  class ids, never color):
  - `v2.RandomApply([v2.ColorJitter(brightness=0.4, contrast=0.6,
    saturation=0.8, hue=0.1)], p=0.8)`

Validation is deterministic: `v2.Resize(image_size, BICUBIC)` then
`CenterCrop(image_size)`.

The geometric/photometric split is a correctness contract: geometric ops
go through `tv_tensors.Mask`, so torchvision issues one random draw per
sample and the mask gets the same crop/translation as the image; color
ops never see the mask.

To take full control, pass `--seg_augmentation_script <path>.py` pointing
at a module that defines

```python
def create_seg_transform(image_size: int, *, is_train: bool):
  ...
```

When set, the script **owns the whole paired transform** (image + mask),
including normalization (the model processor's `image_mean`/`image_std`
are *not* applied automatically in this mode), and is loaded lazily so a
plain run never imports it.

### Encoder + decoder

`--model seg_wrapper:<backbone>` selects the encoder; the decoder is chosen
with `--model_arg decoder=<spec>`.  Built-in decoders:

| Decoder | Skip connections | Notes |
|---|---|---|
| `unet` | yes (all levels) | Classic U-Net top-down fusion; `depth=N` selects the deepest N encoder levels |
| `fpn` | yes (additive) | Feature-pyramid fusion into one high-resolution map |
| `linear` | no | 1×1 conv + bilinear upsample; good baseline, ViT-friendly |

The decoder spec accepts `:key=value` kwargs (`decoder=unet:depth=4,decoder_channels=256`)
and, like classifier heads, a path to a `.py` file defining
`create_decoder(in_channels, num_classes, **kwargs)` where `in_channels` is
the probed per-level encoder channel list.

### Training from scratch vs. fine-tuning

The decoder is always randomly initialized and trained from scratch; the
encoder is what you initialise.  Options, in increasing order of GPU cost:

- **Pretrained encoder (recommended):** `seg_wrapper` loads timm ImageNet
  weights by default (`pretrained=False` disables this).
- **Reuse a self-supervised checkpoint:** pre-train with SimMIM / I-JEPA /
  DINO, then load the encoder with `--source_checkpoint <ckpt>` (see
  [From A Pre-Trained Checkpoint](#from-a-pre-trained-checkpoint)).
- **CNN encoder with a frozen backbone:** `--freeze 'decoder.*'` keeps the
  decoder trainable and freezes everything else.
- **Transformer encoder with LoRA:** `--lora` (PEFT discovers the
  `q_proj`/`k_proj`/`v_proj` projections automatically).  CNN encoders are
  almost all `Conv2d`, so LoRA does not apply to them — use `--freeze`.

### Loss and metric

The default criterion is `dice_ce` (Dice + cross-entropy), selectable via
`--loss dice_ce`; the validation metric is **mean IoU** (`miou`, maximize,
also reporting `mdice` and `pixel_acc`).  The class count is inferred from
the training masks (ignoring `--seg_ignore_index`, default 255) unless
`--seg_num_classes` overrides it.  Pixels equal to `--seg_ignore_index` are
excluded from both the loss and the metric.

### Validation overlays

Set `--vis_every N` to have the trainer call the method's
`log_validation` every N epochs, which logs a `seg/overlay` TensorBoard
grid of **image / ground-truth mask / predicted mask** rows (8 samples by
default) into the run's log dir (default `<checkpoint_dir>/logs`, see
`--log_dir`).  The masks are rendered with a fixed 20-hue palette; the
ignore class (`--seg_ignore_index`, or 255) renders black.  Overlays use
the *current* model state, so they double as a quick training-progress
check (e.g. whether the decoder is learning the class layout before the
mIoU catches up).

Note that some backbones (e.g. plain ViT, Swin) require their native input
resolution (224) and reject other sizes.

### Inference

`genml-kit-infer` is classification-only today: it predicts a top-K label
for each image and does **not** emit per-pixel masks.  For segmentation,
use the validation loop (mIoU + overlays above) or the checkpoint's
weights directly.

---

## Object Detection

Detection predicts a set of **bounding boxes**, each with a class label, for
an image.  `genml_kit` ships an anchor-free FCOS head over the same encoder
feature pyramid that segmentation uses, wired as the `detection` pipeline and
the `detection` method.

### How It Works

The model is `det_wrapper:<backbone>` = **encoder + FCOS head**:

* The encoder is any timm backbone loaded with `features_only=True`, exactly
  as for segmentation, so it yields a feature pyramid (finest level first).
* The head runs a small convolutional tower on **every** pyramid level and
  predicts, per spatial location, a `num_classes`-way class score, four
  positive box distances `(l, t, r, b)` in input pixels, and a centerness
  score.  The distances are `exp()`-activated and scaled by the level's
  stride, so a single location grid spans all scales.

Training uses the FCOS objective (`--loss fcos` by default), a sum of three
terms:

* **Focal loss** over all locations (the class imbalance between the few
  positives and the many background locations is large).
* **GIoU loss** over the positive locations only.
* **Centerness BCE**, which teaches each positive location how central it is
  inside its target box, so NMS can rank central boxes higher.

The **assignment** is purely geometric and needs no Hungarian matcher: a
location is a *candidate* for a box when it lies inside the box, and a
*positive* when it additionally lies inside the box's centre region
(`--loss_args center_sampling_radius=1.5`).  When several boxes cover a
location, the **smallest-area** box wins, so small objects are not starved of
positives by large ones.

At validation time the dense predictions are decoded to `xyxy` boxes,
per-class NMS is applied (`--det_nms_thresh`), and the result feeds a
COCO-style **mAP** meter (`--det_max_detections`, default 100).  The reported
metric is `map` (COCO `mAP@[.5:.95]`, maximize), so the best checkpoint is
stored under `best_map`; `map_50` and `map_75` are logged alongside it.

### Dataset Format

The `detection` pipeline reads **COCO-JSON** corpora via the `detfolder/`
spec:

```
<root>/images/<file_name>
<root>/annotations.json
```

or, with named splits:

```
<root>/train/images/<file_name>
<root>/train/annotations.json
<root>/val/images/<file_name>
<root>/val/annotations.json
```

`annotations.json` is the standard COCO object-detection schema (`images`,
`annotations` with `[x, y, width, height]` `bbox`, and `categories`).  The
loader converts `bbox` to `xyxy` and **compacts** the (1-based, sparse) COCO
category ids into a dense `[0, C)` label space, so the head width equals the
number of categories actually present.  Images with no annotations are legal
(pure-background samples) and still contribute negatives to the loss.  The
annotation file name and image sub-directory are configurable via
`--det_annotation_file` and `--det_image_subdir`.

To generate a throwaway corpus for a CPU smoke run without downloading COCO,
use the bundled generator (`scripts/make_tiny_coco.py`), then point the
pipeline at it:

<!-- genml-cli-test
name: detection-dataset-format
network: true
override:
  - --dataset detfolder/{tmp}/det
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline detection --method detection \
  --dataset detfolder/tiny_coco --model det_wrapper:timm/resnet18 \
  --image_size 640 --batch_size 4 --epochs 20
```

### Example

<!-- genml-cli-test
name: detection-example
network: true
override:
  - --dataset detfolder/{tmp}/det
extra: --epochs 1
-->
```bash
genml-kit-train \
  --pipeline detection --method detection \
  --dataset detfolder//data/coco_tiny \
  --model det_wrapper:timm/resnet50 \
  --model_arg head=fcos_head:num_convs=2 \
  --image_size 640 --batch_size 4 \
  --epochs 20 --lr 1e-4
```

Override the inferred class count with `--det_num_classes`; tune the decode
with `--det_score_thresh` and `--det_nms_thresh`; override the criterion with
`--loss` / `--loss_args` (e.g. `--loss_args center_sampling_radius=1.0`).

### Transforms

Detection does **not** reuse the segmentation transform stack: `torchvision`
remaps masks automatically through geometric ops but leaves boxes untouched,
so every resize / flip must move the coordinates itself.  The default
pipeline is therefore a hand-written **letterbox** (aspect-preserving resize
then pad, which keeps boxes rectangular) followed by a box-aware horizontal
flip (train only) and a `(C, H, W)` float conversion.  Boxes stay in absolute
`xyxy` pixels throughout.

---

## Visual Odometry

Visual odometry (VO) estimates the 2-D **similarity transform** — rotation,
uniform scale, and translation — that maps one image onto another taken a
fraction of a second later.  `genml_kit` ships a learned VO front-end plus a
synthetic pair generator, wired as the `vo_pair` pipeline and the `vo_pair`
method.

The full mathematics (coordinate frames, the rotation matrix, Umeyama's
closed-form solution, the photometric objective, the `mce` metric, and a
failure-mode appendix) lives in [`vo/README.md`](vo/README.md) — a
from-the-ground-up tutorial.  This section is the practical *usage* guide.

### Quick start

<!-- genml-cli-test
name: vo-quick-start
extra: --epochs 1 --vo_length 16 --vo_val_length 8
-->
```bash
# Supervised VO on synthetic pairs (default npu-small profile)
genml-kit-train \
  --pipeline vo_pair --method vo_pair \
  --model vo/npu-small \
  --image_size 64 --batch_size 32 --epochs 10

# Photometric self-supervised stage (adds the photometric residual term)
genml-kit-train \
  --pipeline vo_pair --method vo_pair \
  --vo_stage photometric --epochs 10
```

### Model profiles

The VO net is selected through the model registry as `vo/<profile>`:

| `--model` | Profile | Encoder widths |
|---|---|---|
| `vo/npu-small` | default | `(16, 32, 64, 128)` |
| `vo/edge-mid` | mid-size | see `genml_kit/models/vo/vo_similar.py` |
| `vo/station` | large | see `genml_kit/models/vo/vo_similar.py` |

### Pipeline flags (`--pipeline vo_pair`)

| Flag | Default | Meaning |
|---|---|---|
| `--vo_length` | `20000` | number of synthetic training pairs |
| `--vo_val_length` | `2000` | number of synthetic validation pairs |
| `--pitch_deg` | `45.0` | camera pitch for the ground-plane render |
| `--terrain` | `field` | terrain profile name |
| `--seeded_pairs` | `0` | seed the pair generator for reproducible pairs |

### Method flags (`--method vo_pair`)

| Flag | Default | Meaning |
|---|---|---|
| `--vo_stage` | `supervised` | `supervised` (mce + similarity deltas) or `photometric` (adds the photometric residual) |
| `--vo_profile` | `npu-small` | VO model profile (also selectable via `--model vo/<profile>`) |
| `--vo_loss_cfg` | — | comma-separated loss weights `w_log_s,w_theta,w_conf,w_photo` |

The reported metric is `mce` (mean corner error, in pixels), minimized — the
best checkpoint is stored under `best_mce`.  Set `--image_size 64` for the
synthetic pairs, whose default resolution is 64×64.

---

## Knowledge Distillation

Knowledge distillation trains a small **student** to mimic a large **teacher**.
`genml_kit` implements the three families of distillation (response / logit
matching, feature / FitNets, attention, and relation / RKD) and both teacher
sources: a separately-loaded frozen model (`--teacher_kind external`) and an
exponential-moving-average copy of the student itself
(`--teacher_kind ema`, i.e. self-distillation).  It is wired as the
`distillation` method over the standard `images` pipeline.

The full mathematics (temperature and the soft-target insight, the KL↔CE
collapse and the `T²` scaling, attention-transfer normalisation, RKD, and the
EMA-teacher variance argument) lives in
[`knowledge_distillation/README.md`](knowledge_distillation/README.md) — a
from-the-ground-up tutorial.  This section is the practical *usage* guide.

### Quick start

<!-- genml-cli-test
name: distillation-quick-start
network: true
override:
  - --dataset {tmp}/img
  - -.--image_column
  - --teacher_model google/vit-base-patch16-224
  - --teacher_checkpoint {tmp}/teacher/model_best.pt
extra: --epochs 1
-->
```bash
# Self-distillation: teacher is an EMA of the student (no external model)
genml-kit-train \
  --pipeline images --method distillation \
  --model google/vit-base-patch16-224 \
  --dataset uoft-cs/cifar10 --image_column img --label_column label \
  --teacher_kind ema --kd_temperature 4.0 --kd_alpha 1.0 \
  --epochs 10

# External teacher: frozen checkpoint hands over its soft targets
genml-kit-train \
  --pipeline images --method distillation \
  --model google/vit-base-patch16-224 \
  --dataset uoft-cs/cifar10 --image_column img --label_column label \
  --teacher_kind external \
  --teacher_model google/vit-large-patch16-224 \
  --teacher_checkpoint /path/to/teacher.pt \
  --kd_temperature 4.0 --kd_alpha 1.0 --epochs 10
```

### Method flags (`--method distillation`)

| Flag | Default | Meaning |
|---|---|---|
| `--teacher_kind` | `external` | `external` loads a frozen teacher; `ema` self-distils from an EMA of the student |
| `--teacher_model` | — | teacher model spec (required for `--teacher_kind external`) |
| `--teacher_checkpoint` | — | checkpoint holding the external teacher's weights |
| `--kd_temperature` | `4.0` | distillation temperature $`T`$ |
| `--kd_alpha` | `1.0` | weight of the response (logit) KD term (`0` disables it) |
| `--kd_feature_alpha` | `0.0` | weight of the FitNets feature KD term (`0` disables it) |
| `--kd_attention` / `--no-kd_attention` | off | use attention transfer on the matched layers instead of raw feature MSE |
| `--kd_relation_alpha` | `0.0` | weight of the relational (RKD) KD term (`0` disables it) |
| `--kd_layer_strategy` | `uniform` | how to pair teacher/student layers: `by_index`, `uniform`, `by_name` |
| `--kd_layer_pattern` | — | regex selecting the feature layers to pair (feature KD) |
| `--kd_proj_hidden` | `512` | hidden width of the FitNets projection |
| `--kd_ema_momentum` | `0.996` | initial EMA momentum for `--teacher_kind ema` |
| `--kd_ema_final_momentum` | `1.0` | final EMA momentum for `--teacher_kind ema` |

Enable one family at a time and watch the validation metric after each addition
(see Appendix B of the distillation tutorial for the pitfalls).  A labeled
`--dataset` is required: the ground-truth cross-entropy branch anchors the
student on examples where the teacher is confidently wrong.

---

## Large Language Models (causal LM)

genml_kit can fine-tune decoder-only language models through the same
branchless driver: a new `text` pipeline (token collation) pairs with a new
`causal_lm` method (shifted next-token cross-entropy, perplexity metric).

### Quick start

<!-- genml-cli-test
name: causal-lm-quick-start
network: true
extras: lora
override:
  - --dataset {tmp}/text
extra: --epochs 1
-->
```bash
# Fine-tune a tiny GPT-2 on a local text corpus
genml-kit-train --pipeline text \
                --method causal_lm \
                --model sshleifer/tiny-gpt2 \
                --dataset ./my_text_corpus \
                --text_column text \
                --max_seq_len 256 \
                --lora
```

The `--dataset` value may be a HuggingFace dataset name, or a directory saved
with `datasets.Dataset.save_to_disk` (containing `train` / `validation`
splits). With `--val_split` a held-out fraction of the training split is used
for validation instead.

### Pipeline flags (`--pipeline text`)

| Flag | Default | Meaning |
|---|---|---|
| `--text_column` | — | column holding a raw LM sequence (plain-LM mode) |
| `--instruction_column` | — | prompt column (instruction mode, with `--response_column`) |
| `--response_column` | — | response column (instruction mode) |
| `--max_seq_len` | `512` | token sequence length (truncation / packing window) |
| `--mask_prompt` | off | mask prompt tokens out of the loss (instruction tuning) |
| `--packing` | off | pack sequences into fixed `max_seq_len` windows |
| `--pad_to_multiple_of` | — | round padded length up to a multiple (e.g. `8`) |
| `--max_items` | — | cap the number of loaded items (smoke tests) |

`--dataset`, `--split`, `--val_split`, `--cache_dir`, `--image_column`,
`--max_seq_len`, `--mask_prompt`, `--pad_to_multiple_of` and `--max_items` are
**globally shared** CLI arguments: they are declared once by the harness
(`genml_kit/utils/shared_args.py`), so every pipeline and method can read
`args.<flag>` directly instead of guarding with `getattr(...)`.

### Method flags (`--method causal_lm`)

| Flag | Default | Meaning |
|---|---|---|
| `--tokenizer` | `--model` | tokenizer name/path (defaults to the model) |
| `--trust_remote_code` | off | allow custom code from the model hub |
| `--attn_implementation` | — | attention backend (`sdpa`, `eager`, `flash_attention_2`) |

The metric is `ppl` (perplexity, lower is better), so the best checkpoint is
stored under `best_ppl`. `--lora`, `--grad_checkpoint`, `--freeze`,
`--source_checkpoint` and `--resume` all work for causal LMs through the same
shared model-extras path every other objective uses.

### Efficient fine-tuning (LoRA, gradient checkpointing, bf16)

LoRA is the recommended way to fine-tune a causal LM. When
`--lora_target_modules` is omitted, genml_kit **resolves the target modules
from the model's own structure** rather than from a hardcoded name list, so it
works across architectures without configuration:

| Model family | Auto-resolved targets |
|---|---|
| GPT-2 (HF `Conv1D`) | `c_attn`, `attn.c_proj` |
| Llama / Mistral / Qwen2 / HF ViT | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| BERT | `query`, `key`, `value`, `attention.output.dense` |
| timm ViT | `qkv`, `attn.proj` |
| No attention (plain MLP backbone) | every `nn.Linear` (with a warning) |

The resolver detects attention blocks by class/structure, keeps only their
projection layers, and emits the shortest dotted suffix that does **not** also
match a non-attention layer (so GPT-2's `attn.c_proj` is not confused with
`mlp.c_proj`). Pass `--lora_target_modules` explicitly to override.

<!-- genml-cli-test
name: causal-lm-lora
network: true
extras: lora
override:
  - --dataset {tmp}/text
  - --model sshleifer/tiny-gpt2
-->
```bash
# LoRA + gradient checkpointing + bf16 on a small LM
genml-kit-train --pipeline text \
                --method causal_lm \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --dataset ./my_text_corpus \
                --text_column text \
                --max_seq_len 1024 \
                --lora --lora_r 16 --lora_alpha 32 \
                --grad_checkpoint \
                --amp_dtype bfloat16
```

`--lora`, `--lora_r`, `--lora_alpha`, `--lora_dropout` and
`--lora_target_modules` are the shared LoRA flags; `--grad_checkpoint` enables
per-block activation checkpointing (HF `gradient_checkpointing_enable` for
transformers models, native APIs for timm/custom backbones); `--amp_dtype
bfloat16` selects the autocast dtype.

When the model is a PEFT `PeftModel`, checkpoints also embed a **portable
adapter blob** (`lora_state_blob`: a tar of `adapter_model.safetensors` +
`adapter_config.json`), so the adapters can be loaded with stock PEFT or
published to the Hub, while `--resume` continues to restore the raw tensor
state exactly.

### Tokenizers

`load_tokenizer(name)` mirrors the model/processor split. Normal model IDs
resolve through `transformers.AutoTokenizer`, which already handles both
SentencePiece and tiktoken Hub vocabularies. Two registry entries cover the
gaps: `byte` (a dependency-free tokenizer for offline tests) and
`spm` (a bare SentencePiece `tokenizer.model`, behind the optional
`sentencepiece` dependency).

### Generation

`genml-kit-llm generate` turns prompts into completions. Generation is a
**model contract**, not a HuggingFace-only path: a model opts in by
subclassing `GenerativeLM` (`genml_kit/models/generative.py`), whose default
`generate` runs a portable token loop over the architecture-agnostic
`forward(...) -> .logits` contract that `causal_lm` already trains against.
HuggingFace models are *not* wrapped (their `state_dict` keys must stay intact
for `--resume`); the dispatcher calls their own `generate` directly. So custom
models registered in `MODELS` generate exactly like HF ones.

<!-- genml-cli-test
name: llm-generate
network: true
override:
  - --prompts {tmp}/prompts/prompts.txt
  - --dataset {tmp}/prompt_ds --split test
  - --model sshleifer/tiny-gpt2
  - --checkpoint {tmp}/chk_infer/model_best.pt
  - -.--lora_adapter
-->
```bash
# Prompts from a file (one per line, or JSONL with --prompt_column)
genml-kit-llm generate --model sshleifer/tiny-gpt2 \
                       --prompts prompts.txt \
                       --max_new_tokens 64 \
                       --do_sample --temperature 0.8 --top_p 0.95 \
                       --output completions.jsonl

# Prompts from a HuggingFace dataset column, with a fine-tuned checkpoint
genml-kit-llm generate --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                       --dataset my-org/prompts --split test \
                       --prompt_column question \
                       --checkpoint ./checkpoints/llm_best.pt \
                       --lora_adapter ./adapters/run1
```

| Argument | Default | Description |
|---|---|---|
| `--model` | (required) | HuggingFace id or registered custom model. |
| `--tokenizer` | `--model` | Tokenizer name/path override. |
| `--checkpoint` | `None` | State-dict `.pt`, or a `<root>` resolving to `<root>_best.pt` / `<root>_latest.pt`. |
| `--lora_adapter` | `None` | PEFT adapter directory (`save_pretrained` layout). |
| `--prompts` / `--dataset` | — | Prompt source (file/JSONL, or HF dataset + `--prompt_column`). |
| `--max_new_tokens` | `64` | Tokens to generate. |
| `--do_sample` | off | Sample instead of greedy decoding. |
| `--temperature` / `--top_k` / `--top_p` | `1.0` / `0` / `1.0` | Sampling knobs. |
| `--repetition_penalty` | `1.0` | Down-weight already-seen tokens (`1.0` disables). |
| `--num_beams` | `1` | Beam count (portable backend; `1` = greedy). |
| `--num_return_sequences` | `1` | Completions per prompt. |
| `--no_chat_template` | off | Skip the tokenizer's chat template. |

When the tokenizer exposes a chat template (most instruct models do),
prompts are automatically wrapped as a single user turn; `--no_chat_template`
keeps the raw prompt for base-model completion.

### Evaluation

`genml-kit-llm eval` reports quality metrics for a labeled dataset:

- **perplexity** — teacher-forced (`exp` of the mean shifted cross-entropy),
  over the `--text_column`; no generation involved.
- **exact-match** — fraction of generations matching the reference exactly.
- **token-F1** — SQuAD-style token-overlap F1 against the reference.
- **pass@k** — optional and pluggable: `--pass_at_k K --scorer score.py`
  invokes `score(prompt, completion, reference) -> float` `K` times per item
  and reports the pass rate. No code execution is built in.

<!-- genml-cli-test
name: llm-eval
network: true
override:
  - --dataset {tmp}/eval
  - --model sshleifer/tiny-gpt2
-->
```bash
genml-kit-llm eval --model sshleifer/tiny-gpt2 \
                   --dataset ./my_eval_corpus --split test \
                   --text_column text --prompt_column prompt \
                   --reference_column reference \
                   --metrics ppl exact_match token_f1 \
                   --output report.json
```

Both subcommands load their model through the same registry the trainer uses,
so a registered custom model generates and evaluates with no code changes.

### Alignment (DPO and reward models)

Preference-based post-training runs through the same branchless driver: a new
`preference` pipeline (chosen/rejected pairs) pairs with the `dpo` and
`reward_model` methods. The typical recipe is SFT → DPO → (optional) reward
model:

<!-- genml-cli-test
name: preference-chain
network: true
extras: lora
extra: --epochs 1
-->
```bash
# 1. Supervised fine-tune (masking the prompt out of the loss)
genml-kit-train --pipeline text --method causal_lm \
                --model sshleifer/tiny-gpt2 --dataset ./sft_corpus \
                --instruction_column prompt --response_column response \
                --mask_prompt --checkpoint ./sft/model

# 2. DPO against a frozen reference (defaults to --model)
genml-kit-train --pipeline preference --method dpo \
                --model sshleifer/tiny-gpt2 \
                --reference_checkpoint ./sft/model_best.pt \
                --dataset ./preferences --chosen_column chosen \
                --rejected_column rejected --prompt_column prompt \
                --mask_prompt --beta 0.1 --lora

# 3. Train a reward model on the same pairs
genml-kit-train --pipeline preference --method reward_model \
                --model sshleifer/tiny-gpt2 --dataset ./preferences \
                --chosen_column chosen --rejected_column rejected
```

**DPO** compares the policy against a **frozen reference** and pushes the
chosen response's implicit reward above the rejected one. The reference is
loaded explicitly (`--reference_model`, defaulting to `--model`, plus
`--reference_checkpoint`) and held as a plain attribute — it is never written
to `state_dict` and is rebuilt from args on resume. The sequence score is the
mean per-token log-probability of the response tokens, and `--beta` (default
`0.1`) sets the KL strength. Validation reports pairwise preference accuracy.

**Reward model** attaches a scalar `nn.Linear` head directly to the backbone
as `model.score` — no wrapper class, so the backbone's `state_dict` keys stay
unprefixed and `--source_checkpoint` matches them exactly (only `score.*` is
new). It scores the last non-pad hidden state and trains with the
Bradley–Terry ranking loss; validation reports pairwise ranking accuracy. The
hidden size is read from the config (`hidden_size` / `n_embd` / `d_model`),
falling back to a one-token probe.

The backbone must expose last-layer hidden states (accept
`output_hidden_states=True`); HuggingFace models do, custom models opt in —
the same contract used by `GenerativeLM`.

---

## Reinforcement Learning

genml_kit ships a self-contained reinforcement-learning (RL) toolkit built on
Gymnasium.  The companion document
[`rl/README.md`](rl/README.md) is a from-the-ground-up *mathematics* tutorial
(Bellman equations, TD, Q-learning, GAE, PPO, SAC); this section is the
practical *usage* guide.

### Installation

RL support is an optional extra:

```
pip install 'genml_kit[rl]'
```

### Supported algorithms

| `--method` | Algorithm | Action space | Key files |
|---|---|---|---|
| `dqn` | Double-DQN with epsilon-greedy, optional dueling + n-step + PER | Discrete | `genml_kit/methods/rl_dqn.py` |
| `ppo` | On-policy PPO with GAE, clipped surrogate, entropy bonus | Discrete or continuous | `genml_kit/methods/rl_ppo.py` |
| `sac` | Soft Actor-Critic, twin critics, reparameterized policy, auto-alpha | Continuous only | `genml_kit/methods/rl_sac.py` |

### Quick start

<!-- genml-cli-test
name: rl-quick-start
extras: rl
extra: --epochs 1
-->
```bash
# DQN on CartPole (discrete)
genml-kit-train --pipeline rl --method dqn --env_id CartPole-v1 \
    --epochs 20 --batch_size 64

# PPO on CartPole (discrete)
genml-kit-train --pipeline rl --method ppo --env_id CartPole-v1 \
    --epochs 20 --ppo_rollout_len 2048

# SAC on a continuous env (e.g. LunarLanderContinuous-v3)
genml-kit-train --pipeline rl --method sac --env_id LunarLanderContinuous-v3 \
    --epochs 20
```

All RL methods use `eval_return` (mean undiscounted return over
`--eval_episodes` episodes, default 5) as the checkpoint-selection metric.
The trainer bypasses the DataLoader: it steps the environment directly and
samples from a replay buffer (DQN/SAC) or re-uses a rollout buffer (PPO).

### Custom RL models

Every RL method resolves its network through the same model registry as the
rest of the framework: `--model` selects a registered factory and
`--model_arg key=value` forwards constructor kwargs.

<!-- genml-cli-test
name: rl-custom-models
extras: rl
extra: --epochs 1
-->
```bash
# DQN: dueling Q-head (also the default for --dqn_dueling)
genml-kit-train --pipeline rl --method dqn --model rl/qnet_dueling

# DQN: configure the MLP width via --model_arg
genml-kit-train --pipeline rl --method dqn \
    --model rl/qnet --model_arg hidden_dims=[512,256]

# PPO: wider actor-critic trunk
genml-kit-train --pipeline rl --method ppo \
    --model rl/actor_critic --model_arg hidden_dims=[64,64]

# SAC: critic head width (the actor defaults to rl/actor_critic)
genml-kit-train --pipeline rl --method sac --env_id Pendulum-v1 \
    --model rl/sac_critic --model_arg hidden_dims=[256,256]
```

Registered RL factories (auto-discovered; see `MODELS.list_names()`):

| Name | Returns | Notes |
|---|---|---|
| `rl/qnet` | `QNetwork` (flat Q-head) | DQN default |
| `rl/qnet_dueling` | `QNetwork` (dueling Q-head) | `--dqn_dueling` shortcut |
| `rl/actor_critic` | `ActorCritic` (shared backbone) | PPO default |
| `rl/sac_critic` | `SACCritic` (single critic head) | SAC default |

The methods pass a `SpaceSpec` (obs/action sizes derived from the env) to
the factory, so a custom registered factory can accept the standard scalar
kwargs (`obs_dim`, `n_actions`, `action_dim`, `hidden_dims`, ...) or the
`space=` spec directly.  SAC additionally supports a *factory-return
contract*: a factory may return a plain dict `{"actor": ..., "critic": ...}`
to supply both networks (the method composes the twin critic via a deep
copy).  As elsewhere in the framework there is **no** model-validation
layer: an incompatible network fails loudly at first use (`act()` /
`forward()`), and an unknown `--model` name falls through `load_model` to
Hugging Face and fails at load time.

### Evaluation videos

Pass `--record_eval_video` to record **one video per evaluation episode**
during validation (all RL methods: DQN / PPO / SAC).
`<checkpoint_dir>/videos/` as `eval_episode_000.mp4` etc., or as
`eval_episode_000.gif` when the MP4 writer is unavailable (e.g. no
`imageio[ffmpeg]`).  Recording is **only** attempted when the environment can
produce `rgb_array` frames: scripted/custom environments without a renderer
are detected at validation time and skipped without crashing.

### How a PPO training epoch works

PPO is *on-policy*: every update must be computed from transitions
collected by the **current** policy.  To keep that guarantee, the rollout
buffer is **refilled from scratch each epoch** — nothing survives from the
previous epoch.

**One training epoch** (the `=== Epoch N/500 ===` line in the logs) is
exactly:

```
1.  rollout.reset()                        # clear the on-policy buffer
2.  collect rollout_len env steps          # --ppo_rollout_len (default 2048)
3.  GAE (gamma, lam) over the rollout      # advantages + returns
4.  normalize advantages once              # over the whole rollout
5.  ppo_epochs SGD passes over it          # --ppo_epochs (default 4)
    each pass: rollout_len / mini_batch_size minibatches
6.  next epoch -> back to 1
```

So the arithmetic of a run is:

| Quantity | Formula | Default (Pendulum) |
|---|---|---|
| env steps per epoch | `rollout_len` | 2048 |
| gradient steps per epoch | `ppo_epochs * (rollout_len / mini_batch_size)` | `4 * 2048/64 = 128` |
| episodes per epoch | `rollout_len / episode_len` | `2048/200 ≈ 10` |
| total env steps | `epochs * rollout_len` | 500 × 2048 = 1.02M |

The **two "epochs"** that can be confused:

- **Training epoch** (`--epochs`) — one rollout + its updates.  This is
  what the log lines and checkpoints count.
- **PPO SGD epoch** (`--ppo_epochs`) — the number of *passes over the
  same* rollout during the update phase.  Raising it increases data
  reuse of a single rollout; lowering it (or raising `rollout_len`)
  increases data freshness.

The buffer itself is wiped cheaply: `rollout.reset()` just rewrites the
pointer, so the tensor storage is reused and the per-epoch cost is one
`rollout_len` collection, not an allocation.  On resume, the rollout
buffer state is saved/restored with the checkpoint and the previous
epoch's last observation carries over, so an interrupted run continues
mid-environment rather than restarting the episode.

Changing `--ppo_rollout_len` therefore changes the *epoch length in
steps*, not the memory footprint (memory is `O(rollout_len)` but tiny —
~115 KB for Pendulum at 2048) and not the total step budget
(`epochs × rollout_len`).  A larger rollout gives the GAE a longer
horizon before bootstrap and more data per update, but also means more
gradient steps per rollout: at 50000 you'd run `4 × 50000/64 = 3125`
updates on one batch, which pushes PPO toward overfitting a single
rollout.

### PPO flag reference

| Flag | Default | Meaning |
|---|---|---|
| `--ppo_gamma` | 0.99 | Discount factor for GAE / returns |
| `--ppo_lam` | 0.95 | GAE trace-decay (bias-variance tradeoff) |
| `--ppo_clip_eps` | 0.2 | Clipping epsilon of the clipped surrogate |
| `--ppo_epochs` | 4 | SGD passes over each rollout |
| `--ppo_mini_batch_size` | 64 | Mini-batch size for PPO updates |
| `--ppo_entropy_coef` | 0.01 | Entropy bonus coefficient (see SB3-tricks section) |
| `--ppo_value_coef` | 0.5 | Value-loss coefficient |
| `--ppo_vf_clip_eps` | None | Value-function clipping (None = unclipped) |
| `--ppo_target_kl` | None | Early-stop a PPO epoch when approx-KL exceeds this (see SB3-tricks section) |
| `--ppo_rollout_len` | 2048 | Env steps collected per training epoch |
| `--ppo_discrete` / `--ppo_continuous` | discrete | Action-space selection |
| `--ppo_log_std_init` | log(0.5) ≈ −0.693 | Initial log_std of the continuous Gaussian policy (σ₀ = 0.5) |
| `--ppo_log_std_min` | −10.0 | Lower safety clamp of log_std (σ floor ≈ 4.5e-5).  Kept far below a converged policy's operating range so the entropy gradient never dies at the boundary (the old −2.0 default froze exploration at σ = 0.135) |
| `--ppo_log_std_max` | 2.0 | Upper safety clamp of log_std (σ ceiling ≈ 7.4; bounds entropy) |

The `log_std_*` knobs control the **continuous** Gaussian policy's
standard deviation.  `log_std_init` sets where exploration starts;
`log_std_min`/`log_std_max` are wide safety rails (SOTA pattern: SB3
unbounded, rsl_rl `std_range=(1e-6, 1e6)`, jaxrl `LOG_STD_MIN=-10`) that
prevent σ collapse/explosion while leaving entropy freely trainable.

### Stable-Baselines3-style tricks

The PPO implementation supports three opt-in techniques that
Stable-Baselines3 bakes into its default algorithms.  All three are
**off by default** — every existing flag combination behaves exactly as
before — and each is documented with *when it helps* and *when it
doesn't*.

#### Reward normalization (`--reward_normalize`)

Normalizes rewards by the running standard deviation of discounted
returns, matching SB3 `VecNormalize(norm_reward=True)`.  For each step,
`running_return = gamma * running_return + reward` is fed to a Welford
`RunningMeanStd`; the reward is then divided by `sqrt(var + eps)` and
clamped to `[-reward_norm_clip, reward_norm_clip]` (default 10.0).

- **When it helps:** environments with large or unnormalized reward
  scales (Pendulum-v1, MuJoCo) where an O(1000) value loss otherwise
  dominates the total loss and swamps the entropy/policy signal.
- **When it doesn't:** rewards already O(1) (CartPole, most toy envs) —
  normalization is harmless but unnecessary.
- **Eval always uses raw rewards** (SB3 `VecNormalize.train(False)`):
  the running-return stats are not updated during evaluation, so
  `eval_return` stays comparable across runs.  The running stats travel
  with the checkpoint and are restored on resume.

#### Orthogonal initialization (`--param_init ortho`)

Applies SB3-style orthogonal initialization to the model after
construction: gain `sqrt(2)` for policy/hidden layers, `1.0` for
value/head layers, zero biases — matching the original PPO
implementation.  `log_std` is left untouched.  Works for any method
(PPO, SAC, and non-RL MLPs alike).

- **When it helps:** continuous RL policies, where the default PyTorch
  init can leave the policy near-degenerate at start and slow the
  critical early phase.
- **When it doesn't:** fine-tuned / checkpoint-loaded models (init is
  skipped when weights are loaded from a checkpoint) or discrete
  policies that already train fine with defaults.

#### `target_kl` early stopping (`--ppo_target_kl <float>`)

SB3 stops a PPO epoch early when the approximate KL divergence (the
Schulman **k1** estimator, `0.5 * mean(log_ratio^2)`, always >= 0)
exceeds `target_kl`.  This guards against destructive updates on
high-variance minibatches.

- **Suggested starting point:** `--ppo_target_kl 0.03`.
- **Too low** (e.g. < 0.005) stalls learning; **too high** (e.g. > 0.1)
  disables the safety.  `None` (default) disables early stopping.
- Only affects PPO epochs after the first rollout, so it is a pure
  safety valve for the on-policy update phase.

#### Entropy coefficient (`--ppo_entropy_coef <float>`)

Coefficient of the entropy bonus (default 0.01).  SB3 uses **0.0** for
continuous PPO, relying on init + normalization instead; genml_kit keeps
a small positive default as light regularization.

- **Higher** (0.05–0.1) encourages broader exploration — may help hard
  exploration tasks.
- **Lower** (toward 0.0) sharpens the policy — may help fine-tuning late
  in training.
- No single value is best for all tasks: sweep 0.0–0.1 on your task.

A typical "serious" continuous-PPO setup (Pendulum-style) looks like:

<!-- genml-cli-test
name: rl-ppo-serious
extras: rl
extra: --epochs 1
-->
```bash
genml-kit-train --pipeline rl --method ppo --env_id Pendulum-v1 \
    --ppo_continuous --obs_normalize --reward_normalize --param_init ortho \
    --ppo_target_kl 0.03 --epochs 500
```

### Custom environments

Pass `--env_script /path/to/env.py` to load a custom environment.  The
script must expose a `make_env()` factory returning a Gymnasium-style env
(`reset()`, `step(action)`, `close()`; `observation_space` and
`action_space` attributes, which the pipeline introspects to size the
networks).  `make_env()` may take a single `seed` keyword argument if your
environment wants reproducible initialization.

### Environment & replay options

Relevant flags:
  - `--env_id`, `--obs_dim`, `--env_script` -- environment selection.
  - `--replay_capacity`, `--batch_size`, `--n_step`, `--prioritized`,
    `--per_alpha`, `--per_beta_start`, `--per_beta_frames` -- replay tuning.
  - **Discount factor** -- there is intentionally **no** global `--gamma`: each
    method owns its own because RL objectives are not active simultaneously and
    tune it differently. DQN uses `--dqn_gamma`, PPO `--ppo_gamma`, SAC
    `--sac_gamma` (all default `0.99`). The n-step returns computed by the
    shared replay buffer use the discount of the **selected method**; the
    method installs it into the buffer during `wire_data`, so a single run never
    mixes discounts.
  - `--obs_normalize`, `--obs_norm_clip` -- running observation
    normalization (Welford).  Both train and eval observations are
    normalized with the same RMS statistics.
  - `--reward_normalize`, `--reward_norm_clip` -- running-return reward
    normalization (SB3 `VecNormalize(norm_reward=True)`).  See
    "Stable-Baselines3-style tricks" below.
  - `--param_init ortho` -- SB3-style orthogonal parameter initialization.
    See "Stable-Baselines3-style tricks" below.
  - `--ppo_gamma`, `--ppo_lam`, `--ppo_clip_eps`, `--ppo_epochs`,
    `--ppo_mini_batch_size`, `--ppo_value_coef`, `--ppo_vf_clip_eps`,
    `--ppo_rollout_len`, `--ppo_discrete`/`--ppo_continuous`,
    `--ppo_log_std_init/min/max` -- PPO update and policy knobs.  See the
    "PPO flag reference" table above.
  - `--ppo_target_kl` -- early-stop a PPO epoch when the k1 approximate-KL
    exceeds this threshold (SB3 `target_kl`).  See below.
  - `--ppo_entropy_coef` -- entropy bonus coefficient (default 0.01).
    See below.
  - `--eval_episodes`, `--env_seed` -- evaluation.
  - `--record_eval_video` -- record one MP4 (or GIF fallback) per evaluation
    episode under `<checkpoint_dir>/videos/`; only when the environment
    supports rendering (see "Evaluation videos" below).

### Episode-end semantics (terminated vs truncated)

Gymnasium distinguishes *terminated* (the MDP truly ended, e.g. the pole
fell) from *truncated* (the episode was cut short, e.g. by a time limit).
genml_kit threads the true `terminated` flag through the replay buffer,
rollout buffer, GAE, and the TD targets: a truncated final step still
bootstraps `γ V(s′)`, while a terminated one does not.  Environments using
the legacy 4-tuple gym API (no `terminated`/`truncated` split) fall back to
treating `done` as `terminated`.

## References And Further Reading

- Xie et al., [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886), CVPR 2022.
- Assran et al., [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243), CVPR 2023.
- Grill et al., [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733), NeurIPS 2020.
  The EMA teacher and stop-gradient design used by I-JEPA-style methods.
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021.
  The DINO paper — self-distillation with an EMA teacher and multi-crop.
  The centering + sharpening trick to avoid representation collapse.
- Khosla et al., [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362), NeurIPS 2020.
- He et al., [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR 2022.
  A useful comparison for masked-image pre-training.
- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), ICLR 2022.
- Howard & Ruder, [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146), ACL 2018.
  Introduced layer-wise learning rate decay.
- Zhang et al., [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412), ICLR 2018.
- Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), ICCV 2017.
  Introduced focal loss and the $`(1-p_t)^\gamma`$ easy-example down-weighting.
- Chen & Guestrin, [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754), KDD 2016.
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

### Verifying README commands

Command blocks preceded by a `<!-- genml-cli-test ... -->` HTML comment are
executed by `tests/test_readme_cli.py`.  The comment is invisible on
GitHub; its YAML body controls limiting args, network/token markers, and
substitutions.  Run them with `pytest -m network` (needs internet, no
credentials) — the offline suite excludes them by default.

### Naming Conventions For Class Members

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

