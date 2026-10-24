# Knowledge Distillation & Model Export — A Self-Contained Tutorial

This document teaches, from the ground up, the ideas behind **knowledge
distillation** and how they should be woven into `genml_kit`: how a large,
accurate *teacher* network can hand its behaviour to a small, cheap *student*
network, how the objective that drives that transfer is built, how an
exponential-moving-average (EMA) teacher turns distillation into
*self-distillation*, and how the finished student is exported to a portable
runtime (ONNX).

It is written for the same reader as [`vo/README.md`](../vo/README.md): someone
who knows how to train a classifier but wants the *why* behind every line they
are about to add to the harness. No result in this document depends on
`genml_kit` internals to be understood — the mathematics is developed from first
principles and only *then* mapped onto the repository's `Method`, `Pipeline`
and `LOSSES` abstractions.

> **Scope.** This is a design + tutorial document, not implemented code. It
> specifies the modules we intend to add, the losses they will use, the CLI
> they will expose, and — equally important — the **shared utilities that must
> be extracted first** so that no distillation code is cut-and-pasted from the
> existing self-supervised methods.

---

## How To Read This Document

- **Read order.** Part 1 frames the problem. Parts 2–4 build the three families
  of distillation loss (response, feature, relation). Part 5 explains where the
  teacher comes from (fixed vs. EMA). Part 6 maps all of this onto the
  `genml_kit` architecture. Part 7 covers ONNX export. Part 8 is the refactoring
  contract — *read it even if you only care about the math*, because it is what
  keeps the module honest.
- **Math notation.** Inline maths appears as $`x`$; display maths appears
  centred. The rendering rules (escaping, operators, the emphasis hazard) are
  centralised in [`docs/MARKDOWN_LATEX.txt`](../docs/MARKDOWN_LATEX.txt); every
  formula below obeys them.
- **Checkpoints.** Boxes labelled **Checkpoint** pose a question with the answer
  immediately after. **Pitfall** boxes record the mistakes this document exists
  to prevent.

---

## Contents

- [Part 1 — What Distillation Is, And Why We Want It](#part-1--what-distillation-is-and-why-we-want-it)
  - [1. The Soft-Target Insight](#1-the-soft-target-insight)
  - [2. The Three Families Of Knowledge](#2-the-three-families-of-knowledge)
  - [3. Dictionary Learning Versus Function Mimicry](#3-dictionary-learning-versus-function-mimicry)
- [Part 2 — Response-Based Distillation (Logit Matching)](#part-2--response-based-distillation-logit-matching)
  - [4. Temperature And The Softmax](#4-temperature-and-the-softmax)
  - [5. The Kullback–Leibler Divergence Objective](#5-the-kullbackleibler-divergence-objective)
  - [6. The Gradient Of The KD Loss](#6-the-gradient-of-the-kd-loss)
  - [7. Why The $`T^2`$ Factor, And Why It Is Not Cosmetic](#7-why-the-t2-factor-and-why-it-is-not-cosmetic)
  - [8. Combining With The Ground-Truth Cross-Entropy](#8-combining-with-the-ground-truth-cross-entropy)
- [Part 3 — Feature-Based Distillation](#part-3--feature-based-distillation)
  - [9. Hint Layers And The FitNets Projection](#9-hint-layers-and-the-fitnets-projection)
  - [10. Attention Transfer](#10-attention-transfer)
- [Part 4 — Relation-Based Distillation](#part-4--relation-based-distillation)
  - [11. Distilling The Similarity Structure](#11-distilling-the-similarity-structure)
- [Part 5 — Where Does The Teacher Come From?](#part-5--where-does-the-teacher-come-from)
  - [12. Fixed Teachers And Offline Distillation](#12-fixed-teachers-and-offline-distillation)
  - [13. Self-Distillation And The EMA Teacher](#13-self-distillation-and-the-ema-teacher)
  - [14. Momentum Scheduling](#14-momentum-scheduling)
- [Part 6 — Mapping Onto The genml_kit Architecture](#part-6--mapping-onto-the-genml_kit-architecture)
  - [15. The Method / Pipeline Split](#15-the-method--pipeline-split)
  - [16. The Loss Registry Entry](#16-the-loss-registry-entry)
  - [17. The Teacher As A Loaded Model](#17-the-teacher-as-a-loaded-model)
  - [18. The CLI Surface](#18-the-cli-surface)
  - [19. Checkpointing](#19-checkpointing)
- [Part 7 — ONNX Export](#part-7--onnx-export)
  - [20. What ONNX Actually Freezes](#20-what-onnx-actually-freezes)
  - [21. Dynamic Axes And The Tracing Contract](#21-dynamic-axes-and-the-tracing-contract)
  - [22. Post-Training Quantization And Verification](#22-post-training-quantization-and-verification)
- [Part 8 — Refactoring Contract: No Cut-And-Paste](#part-8--refactoring-contract-no-cut-and-paste)
  - [23. The Duplication That Already Exists](#23-the-duplication-that-already-exists)
  - [24. Extract The EMA Teacher](#24-extract-the-ema-teacher)
  - [25. Extract The Feature-Extraction And Backbone-Dim Probes](#25-extract-the-feature-extraction-and-backbone-dim-probes)
  - [26. Extract The Layer-Matching Helper](#26-extract-the-layer-matching-helper)
  - [27. Acceptance Criteria For The Refactor](#27-acceptance-criteria-for-the-refactor)
- [Appendices](#appendices)
  - [Appendix A: Symbol Table](#appendix-a-symbol-table)
  - [Appendix B: Failure Modes And Pitfalls](#appendix-b-failure-modes-and-pitfalls)
  - [Appendix C: Reading List](#appendix-c-reading-list)
  - [Appendix D: Math Rendering Reference](#appendix-d-math-rendering-reference)

---

# Part 1 — What Distillation Is, And Why We Want It

## 1. The Soft-Target Insight

A trained classifier does not merely emit a *label*; it emits a full
probability distribution over classes. Suppose a network is shown a picture of
a car. It might output

$$
\large
p(y \mid x) = \big[\;\text{car}: 0.90,\quad \text{truck}: 0.08,\quad
\text{bicycle}: 0.01,\quad \text{cat}: 0.0001,\;\dots\big].
$$

The hard label "car" throws away almost all of that vector. But the ratios
`truck : car = 0.08 : 0.90` are not noise — they encode the teacher's belief
that a car is *much more like a truck than like a cat*. This is the
**dark knowledge** of Hinton et al. (2015): the small probabilities carry a
similarity structure learned from data that is invisible in a one-hot target.

Distillation therefore asks the student not "*what is the answer?*" but "*what
would you have answered, with all your uncertainty?*". Formally, if we write
the teacher and student logits as $`z^{(t)}`$ and $`z^{(s)}`$, the goal is

$$
\large
\min_{\theta_s}\ \mathbb{E}_{x}\ \mathcal{D}\!\left(
q_t(x)\,\middle\|\,q_s(x)\right),
$$

where $`q_t`$ and $`q_s`$ are tempered softmaxes (Part 2) and $`\mathcal{D}`$
is a divergence — almost always the Kullback–Leibler one.

**Checkpoint.** Why would matching *wrong* teacher probabilities help a
student? *Answer: they are not "wrong", they are a dense regression target. A
one-hot target supervises one number per example; the soft target supervises
$`C-1`$ numbers per example, and those numbers are mutually consistent because
they all came from the same function. The student gets a far richer gradient
signal per sample.*

## 2. The Three Families Of Knowledge

Distillation methods are conventionally grouped by *where* the transfer
happens (Gou et al., 2021):

| Family | What is matched | Canonical reference |
|---|---|---|
| **Response-based** | the final softmax / logits | Hinton et al., 2015 |
| **Feature-based** | intermediate activations of hidden layers | Romero et al. (FitNets), 2015 |
| **Relation-based** | pairwise/triplet structure between samples or layers | Park et al. (RKD), 2019 |

Parts 2, 3 and 4 develop one loss per family. They compose: a single training
run may use all three, with weights $`\alpha_{\text{resp}}, \alpha_{\text{feat}},
\alpha_{\text{rel}}`$.

## 3. Dictionary Learning Versus Function Mimicry

Hinton's original framing casts the teacher as a *soft label generator*: the
student is trained on a dataset of `(inputs, teacher-distributions)` pairs. A
more modern framing calls distillation **function matching**: we are
approximating the teacher function $`f_t`$ with a student $`f_s`$ under a
chosen metric.

The two framings differ in the limit:

- With the KL response loss and a perfectly matched student capacity,
  distillation recovers the teacher's conditional distribution $`p_t(y \mid x)`$.
- With enough data, the *Bayes-optimal* student trained on hard labels also
  recovers a conditional distribution — but a *degenerate* one (near one-hot),
  because hard labels carry no soft information. **Distillation is what makes a
  finite student able to represent the teacher's uncertainty.**

This matters downstream: an exported student with calibrated soft outputs is
usable as a *dense* component (a logit or feature source inside another model),
not just as a top-1 classifier. That is exactly why genml_kit wants it — the
`Method`s in the harness are composable, and a distilled backbone is a cheap
drop-in encoder.

---

# Part 2 — Response-Based Distillation (Logit Matching)

## 4. Temperature And The Softmax

The tempered softmax of a logit vector $`z \in \mathbb{R}^{C}`$ at
temperature $`T > 0`$ is

$$
\large
q_i(T) = \frac{\exp(z_i / T)}{\sum_{j=1}^{C} \exp(z_j / T)}.
$$

At $`T = 1`$ this is the ordinary softmax. As $`T \to \infty`$ the distribution
flattens toward uniform; as $`T \to 0^{+}`$ it sharpens toward one-hot. The
temperature is the knob that **exposes the dark knowledge**: at $`T = 1`$ a
confidence of $`0.08`$ may underflow to something indistinguishable from
$`0`$ under log, but at $`T = 4`$ the same logit ratio is amplified before the
softmax compresses it again.

**Pitfall.** Temperature is *not* applied to the ground-truth cross-entropy
branch (Part 6). It lives only on the distillation branch. Applying it to both
silently rescales the supervised signal and is one of the most common
implementation bugs.

## 5. The Kullback–Leibler Divergence Objective

The response loss is the forward KL from teacher to student, taken over the
batch:

$$
\large
\mathcal{L}_{\text{KD}}(x)
= T^{2}\, \mathrm{KL}\!\left( q_t(T) \,\middle\|\, q_s(T) \right)
= T^{2} \sum_{i=1}^{C} q_{t,i}(T)\,\log\!\frac{q_{t,i}(T)}{q_{s,i}(T)}.
$$

Because $`q_t`$ is treated as a constant (its gradient is zero), the entropy
term $`-\sum_i q_{t,i}\log q_{t,i}`$ drops out of the gradient, and the loss
reduces, up to constants, to the cross-entropy

$$
\large
\mathcal{L}_{\text{KD}}(x)
\;\equiv\; -\,T^{2}\sum_{i=1}^{C} q_{t,i}(T)\,\log q_{s,i}(T)
\qquad (\text{teacher detached}).
$$

Implementations almost always compute it as
`KLDivLoss(log_softmax(student/T), softmax(teacher/T)) * T**2`, which is
numerically cleaner than forming the ratio explicitly.

## 6. The Gradient Of The KD Loss

Let $`q_s = \mathrm{softmax}(z_s / T)`$. Differentiating the cross-entropy form
with respect to a student logit $`z_{s,k}`$ gives the familiar softmax Jacobian
contraction:

$$
\large
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{s,k}}
= T\,\big(q_{s,k}(T) - q_{t,k}(T)\big).
$$

Two things are visible in this formula.

1. The gradient is a **difference of distributions**, so it vanishes exactly
   when the student matches the teacher — the correct fixed point.
2. The factor $`T`$ (before the $`T^2`$ scaling in the loss) is what the $`T^2`$
   is meant to counteract.

**Checkpoint.** What happens if you forget the $`T^2`$? *Answer: the gradient
magnitude of the distillation term scales like $`1/T`$ relative to the
cross-entropy term. At the common $`T = 4`$ the distillation signal is silently
diluted by $`4\times`$, and the effective $`\alpha`$ you set on the CLI is a
lie. This is the single most cited "why is my distilled model worse" cause.*

## 7. Why The $`T^2`$ Factor, And Why It Is Not Cosmetic

From §6, the raw gradient carries a factor $`T`$ from the division inside the
softmax. Multiplying the loss by $`T^2`$ restores the gradient to
$`O(1)`$ in $`T`$, so that the balance between the distillation term and the
ground-truth term is governed *only* by the mixing weight $`\alpha`$ and not by
the temperature. With the $`T^2`$ in place, changing $`T`$ changes *which
examples contribute* (the softness) but not *how loudly* the branch speaks.

## 8. Combining With The Ground-Truth Cross-Entropy

The total objective for plain logit distillation is the convex combination

$$
\large
\mathcal{L}
= (1 - \alpha)\,\underbrace{\mathcal{L}_{\text{CE}}(z_s, y)}_{\text{hard labels}}
\;+\; \alpha\,\underbrace{T^{2}\,\mathrm{KL}\!\left(q_t \middle\| q_s\right)}_{\text{soft labels}},
$$

with $`\alpha \in [0, 1]`$. If ground-truth labels are unavailable (unlabeled
data, or a pure function-matching run), set $`\alpha = 1`$ and drop the CE
branch — the loss is then a pure distribution-matching objective, which is
exactly the regime of self-distillation (Part 5).

---

# Part 3 — Feature-Based Distillation

## 9. Hint Layers And The FitNets Projection

Response distillation only constrains the *output*. A student with a small
backbone may never reach the teacher's accuracy because its intermediate
representation is too narrow — it can match the answers on the training set
without ever forming the teacher's internal features. **FitNets** (Romero et
al., 2015) fixes this by matching a hidden activation of the teacher (the
*hint*) to a hidden activation of the student (the *guided* layer).

Let $`h_t \in \mathbb{R}^{H_t}`$ and $`h_s \in \mathbb{R}^{H_s}`$ be the two
activation vectors for the same input. When $`H_s \ne H_t`$ we insert a
learned linear map $`W_r`$ (with bias) and minimise

$$
\large
\mathcal{L}_{\text{feat}}
= \big\|\, r(h_s) - h_t \,\big\|_2^{2},
\qquad r(h_s) = W_r\,h_s + b_r,
$$

where the student-side projection $`r`$ is trained jointly and *discarded*
afterwards — exactly analogous to the projection head in the repository's
`ContrastiveEncoder` (`genml_kit/models/contrastive.py`), which is dropped
after pre-training.

**Design note for genml_kit.** Feature distillation should reuse the existing
`ProjectionHead`-style MLP rather than re-implement it, and it must reuse the
existing `encode_with_backbone` / `detect_backbone_dim` helpers
(`genml_kit/models/encoder_utils.py`) to obtain $`h_s`$ and $`h_t`$ in the first
place. This is developed in Part 8.

## 10. Attention Transfer

When the intermediate tensors are *spatial* ($`(C, H, W)`$, as in a
convolutional or ViT feature map), an L2 match on the raw tensor is brittle:
it is sensitive to channel permutation and scale. **Attention transfer** (Zagoruyko
& Komodakis, 2017) instead matches a *summary* of each map. For a feature map
$`A \in \mathbb{R}^{C \times H \times W}`$ define the attention map

$$
\large
\mathcal{A}(A) = \sum_{c=1}^{C} \lvert A_{c} \rvert^{p},
\qquad p = 2,
$$

an $`(H, W)`$ tensor, then normalise by its $`\ell_2`$ norm so that maps of
different magnitude are compared by *shape* rather than scale:

$$
\large
\tilde{\mathcal{A}}(A) = \frac{\mathcal{A}(A)}
{\big\|\,\mathrm{vec}\,\mathcal{A}(A)\,\big\|_2}.
$$

The transfer loss is $`\mathcal{L}_{\text{AT}} = \big\| \tilde{\mathcal{A}}(h_s)
- \tilde{\mathcal{A}}(h_t) \big\|_2^{2}`$. Attention transfer is the natural
feature loss for the ViT-based `convvit` and timm backbones already bundled in
the repository, because it is agnostic to the number of channels and only
requires spatial alignment.

---

# Part 4 — Relation-Based Distillation

## 11. Distilling The Similarity Structure

The previous two families match *a sample to itself*. Relation distillation
matches the *geometry between samples*. For a batch $`\{x_1, \dots, x_B\}`$ let
$`\psi(h_i, h_j)`$ be a similarity between representations, and let
$`d_t(i,j), d_s(i,j)`$ be distance metrics derived from teacher and student
features. **RKD** (Park et al., 2019) minimises the Huber difference of the
angle and distance structures:

$$
\large
\mathcal{L}_{\text{rel}}
= \sum_{i \ne j}\ \ell_\delta\!\Big(\phi\big(d_s(i,j)\big),\ \phi\big(d_t(i,j)\big)\Big)
\;+\; \sum_{(i,j,k)} \ell_\delta\!\big(\text{angle}_s, \text{angle}_t\big),
$$

where $`\ell_\delta`$ is the Huber loss and $`\phi`$ a normalisation. The key
property is that these targets are **translation and permutation invariant**:
the student need not reproduce the teacher's *coordinate frame*, only its
*relational structure*.

For genml_kit, relation distillation is the cheapest to add because the
repository already owns the machinery: the supervised-contrastive loss
(`genml_kit/losses/contrastive.py`) is precisely a pairwise similarity over a
batch. A relation-KD loss is a sibling that compares *two* such matrices (one
per network) instead of using labels.

---

# Part 5 — Where Does The Teacher Come From?

## 12. Fixed Teachers And Offline Distillation

The simplest regime: train (or load) a teacher, freeze it, and train the
student against its outputs. If the teacher's responses are precomputed and
cached to disk, this is **offline distillation** — the student never even holds
the teacher in memory, and each epoch is a normal supervised run over cached
soft targets. Offline distillation is attractive for genml_kit because it maps
onto the existing data pipeline: the "label" simply becomes a stored
vector instead of an integer.

**Checkpoint.** What is the cost of offline distillation? *Answer: storage.
Each example now carries $`C`$ floats instead of one int, and the teacher must
be re-run whenever the input augmentation changes. Online (teacher-in-the-loop)
distillation avoids the storage but pays teacher forward-pass compute every
step.*

## 13. Self-Distillation And The EMA Teacher

Self-distillation removes the external teacher entirely: the teacher is an
**exponential moving average** of the student's own weights. If the student is
$`\theta_s`$, the teacher $`\theta_t`$ is updated, every optimizer step, by

$$
\large
\theta_t \leftarrow m\,\theta_t + (1 - m)\,\theta_s,
\qquad m \in (0, 1).
$$

This is exactly the mechanism already used by three methods in the repository —
**DINO**, **BYOL** and **I-JEPA** — and it is why the refactoring in Part 8 is
not optional housekeeping but a prerequisite: distillation would be the
*fourth* copy of the same EMA loop if written naively.

The teacher parameters are never updated by gradients (`requires_grad = False`),
so only the student branch receives a backward pass. The forward pass, however,
runs through both networks, and the teacher sees the *same* input under a
(usually different) augmentation — the "two views" idea that `DualViewTransform`
already provides.

## 14. Momentum Scheduling

A constant $`m`$ works, but the literature (BYOL, DINO, I-JEPA) ramps it. A
representative schedule is a cosine ramp from an initial value to a final value:

$$
\large
m(k) = m_{\text{start}} + (m_{\text{end}} - m_{\text{start}})\,
\frac{1 - \cos\!\big(\pi\, k / K\big)}{2},
$$

where $`k`$ is the current optimizer step and $`K`$ the step budget. Early in
training the teacher tracks the student closely (small $`m`$, teacher adapts
fast); later it becomes a slow, stable target (large $`m`$). The step budget
$`K`$ is computed at `wire_data` time from the loader length and the epoch
count — the same pattern the BYOL method already uses to size its momentum ramp.

---

# Part 6 — Mapping Onto The genml_kit Architecture

The repository separates the **data side** (`DataPipeline`) from the
**objective side** (`Method`), with a `LOSSES` registry for reusable criteria
and a `MODELS` registry for loaders. Distillation enters each layer through
well-defined seams. Nothing here should add a new control-flow branch to
`train.py`; the whole point of the v4.2 harness is that a new capability is a
new *registration*, not a new *program*.

## 15. The Method / Pipeline Split

**Proposed module.** `genml_kit/methods/distillation.py`, registering a
`DistillationMethod(Method)` under the name `"distillation"`. It follows the
same contract as `classification.py` and `supcon.py`:

- `add_args(group)` — declares `--teacher_model`, `--kd_temperature`,
  `--kd_alpha`, `--kd_feat_weight`, `--kd_attn_weight`, and the EMA flags.
- `build_model(args, device)` — loads the **student** via the existing
  `load_model` (`genml_kit.models`) and loads the **teacher**, then freezes it.
  This mirrors `BYOLMethod.build_model`, which applies LoRA/freeze to the online
  encoder *before* constructing the composite.
- `build_transform(args, image_size)` — for self-distillation, returns a
  `DualViewTransform`; for fixed-teacher logit KD, returns the ordinary
  classification transform.
- `train_step(model, blob, global_step)` — runs student forward, teacher
  forward (under `torch.no_grad()`), and returns a `LossOutput`.
- `wire_data` — computes the optimizer-step budget $`K`$ when an EMA schedule
  is active.

**Pipeline.** No new pipeline is strictly required for response distillation:
the existing `images` pipeline already emits `DataBlob(data=images,
meta={"labels": ...})`, and the teacher's soft targets are produced inside
`train_step`. This is a deliberate reuse — adding a pipeline is only justified
if we later wish to *cache* teacher targets (offline mode).

## 16. The Loss Registry Entry

The reusable objectives belong in the `LOSSES` registry
(`genml_kit/losses/registry.py`), *not* inlined in the method, following the
existing convention where `focal`, `supcon`, `ppo`, `sac` and `dqn` bundles are
registered builders:

- `"kd_response"` → `(student_logits, teacher_logits, T, reduction) -> Tensor`
- `"kd_feature"` → `(student_feat, teacher_feat, projector=None) -> Tensor`
- `"kd_attention"` → `(student_map, teacher_map, p=2) -> Tensor`

Each is a small pure function in `genml_kit/losses/distillation.py`, exported
from the package `__init__` like every other loss. Registering them means an
*external script* can override a single key via the existing
`load_loss(spec, ...)` merge logic, with no changes to the core.

## 17. The Teacher As A Loaded Model

The teacher is a model, so it should be loaded through the *same* registry that
loads students. To keep `DistillationMethod` agnostic, load the teacher with
`load_model(args.teacher_model, ...)` and immediately:

1. call `model.eval()` and set `requires_grad = False` on every parameter;
2. build a dedicated validation transform from the teacher's own processor
   (`load_processor(args.teacher_model, ...)`), because the teacher's expected
   normalisation may differ from the student's.

**Pitfall.** Never share one `processor` between teacher and student unless the
two models genuinely use the same `image_mean` / `image_std`. Feeding the
student's-normalised tensor to a teacher trained under different statistics
produces confident, wrong soft targets — the distillation then teaches the
student a bug.

## 18. The CLI Surface

All flags register through `add_args`, so `build_parser` picks them up
automatically and `--help` documents them. The intended surface:

| Flag | Meaning | Default |
|---|---|---|
| `--teacher_model` | registry name / HF id of the frozen teacher | *(required)* |
| `--kd_temperature` | softmax temperature $`T`$ (§4) | `4.0` |
| `--kd_alpha` | weight of the soft-target branch (§8) | `0.7` |
| `--kd_feat_weight` | weight of the FitNets feature loss (§9) | `0.0` |
| `--kd_attn_weight` | weight of the attention-transfer loss (§10) | `0.0` |
| `--kd_ema` | enable an EMA (self-distillation) teacher (§13) | `false` |
| `--kd_ema_momentum` | schedule start $`m_{\text{start}}`$ | `0.996` |
| `--kd_ema_final_momentum` | schedule end $`m_{\text{end}}`$ | `1.0` |

The last three flags delegate to the shared EMA utility extracted in Part 8.

## 19. Checkpointing

`DistillationMethod` implements `get_checkpoint_state` / `load_checkpoint_state`
exactly as `BYOLMethod` and `DINOMethod` do: the EMA momentum and the current
teacher step are stored so a resumed run continues the same schedule. The
teacher *weights* need not be stored when the teacher is EMA (they are
reconstructible), but they *must* be stored when the teacher is external and
its identity would otherwise be lost. This decision is recorded in the method's
checkpoint state dict under a key such as `"teacher_kind": "ema" | "external"`.

---

# Part 7 — ONNX Export

Distillation's payoff is a small, fast model. That payoff is only realised if
the student can leave PyTorch. ONNX is the interchange format; a new console
script `genml-kit-export` (registered in `pyproject.toml` next to
`genml-kit-train` and `genml-kit-infer`) produces it.

## 20. What ONNX Actually Freezes

`torch.onnx.export` runs the model once in **trace** mode by default: it records
the tensor operations executed for a dummy input and emits a static graph. This
has three consequences worth stating explicitly:

1. **Python control flow depending on tensor *values* is lost.** A branch like
   `if x.max() > 0:` is traced once for the dummy input. Distillation is
   unaffected in the common case, but any loss-side branch must not leak into
   the exported forward.
2. **Parameters are baked as constants** (weights and buffers), so the exported
   file is self-contained.
3. **Input/output names** should be set explicitly (`input_names=["input"]`,
   `output_names=["logits"]`) for downstream tooling.

## 21. Dynamic Axes And The Tracing Contract

The exported graph must accept a variable batch size and, for segmentation-style
students, variable spatial size. Declare this with dynamic axes:

```python
torch.onnx.export(
    model.eval(),
    dummy_input,
    path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "logits": {0: "batch"},
    },
    opset_version=17,
)
```

**Pitfall.** The custom heads (attention pooling, `convvit`) contain operations
whose opset support varies; pin `opset_version` explicitly and gate the export
extra in `pyproject.toml` so that a core install is unaffected. Verify the ONNX
graph numerically against PyTorch (§22) before trusting it.

## 22. Post-Training Quantization And Verification

Two-step hand-off to a runtime:

1. **Dynamic quantization** of the exported model weights reduces size with no
   calibration data:
   `onnxruntime.quantization.quantize_dynamic(model, {MatMul: QInt8})`.
2. **Numerical verification.** Run a fixed batch through both the PyTorch model
   and the ONNX model and assert the maximum absolute difference is below a
   tolerance (e.g. $`10^{-4}`$ in float32). A distilled student that fails this
   check is a bug in the tracing contract, not in the distillation.

**Checkpoint.** Why verify against PyTorch rather than a held-out metric?
*Answer: a metric hides small per-logit errors behind an average; an elementwise
`allclose` catches the single off-by-a-swap operation that flips one class.
Verify the function, then trust the metric.*

---

# Part 8 — Refactoring Contract: No Cut-And-Paste

This part is binding. The distillation module may **not** be written by copying
existing code. Where the repository already contains the logic we need, it is
extracted into a shared utility and all call sites — old and new — are migrated
to it. The guideline is the one the repository already follows: when three
self-supervised wrappers shared feature extraction, `encoder_utils.py` was born
so the logic would not be triplicated. Distillation makes it four.

## 23. The Duplication That Already Exists

A survey of the current tree shows the same two blocks repeated:

**Block A — EMA teacher construction and update** (identical shape in
`models/byol.py`, `models/dino.py`, `methods/ijepa.py`, and again in the RL
target networks `models/rl/qnetwork.py`, `models/rl/sac_critic.py`):

```python
teacher = copy.deepcopy(student)
for p in teacher.parameters():
    p.requires_grad = False
# ... and, at every step:
for tp, sp in zip(teacher.parameters(), student.parameters()):
    tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)
```

**Block B — backbone feature extraction / dimension probing**
(`models/encoder_utils.py` already holds the canonical version, but the
composite wrappers each re-derive `backbone_dim` by hand).

Additionally, the three methods `BYOLMethod`, `DINOMethod`, `IJEPAMethod`
each carry a near-identical **cosine momentum ramp over a step budget computed
in `wire_data`**.

| Duplicated concern | Current copies | Target |
|---|---|---|
| EMA teacher build + step | `byol`, `dino`, `ijepa`, `rl/qnetwork`, `rl/sac_critic` | one `EMAModel` utility |
| Cosine momentum schedule | `byol`, `dino`, `ijepa` | one `cosine_momentum` helper |
| Backbone-dim probe / encode | `encoder_utils` + ad-hoc re-derivation | call `encoder_utils` everywhere |
| Layer matching by name/depth | *does not exist yet* (needed by FitNets) | new `match_layers` helper |

## 24. Extract The EMA Teacher

Introduce a single small module — proposed `genml_kit/models/ema.py` — exposing:

- `class EMAModel(nn.Module)` — wraps a target copy, freezes it in the
  constructor, and provides `update(momentum)` implementing Block A, plus
  `state_dict` / `load_state_dict` that skip the frozen params.
- `def cosine_momentum(step, total_steps, start, end)` — the schedule of §14.

`BYOL`, `DINO`, `I-JEPA` and the two RL target networks are then refactored to
*hold* an `EMAModel` instead of each re-implementing deepcopy + freeze + blend.
`DistillationMethod` becomes the next caller, not a new copy. The RL networks
use a *hard* (periodic copy) rather than soft update; `EMAModel` therefore
exposes both `update_soft(alpha)` and `copy_hard()` so a single utility serves
every site.

**Why this is the right seam.** The EMA update is a pure, parameter-order–local
operation with no knowledge of what the network computes. It belongs next to
`encoder_utils.py` at the `models/` layer, not inside any method.

## 25. Extract The Feature-Extraction And Backbone-Dim Probes

`models/encoder_utils.py` already provides `detect_backbone_dim` and
`encode_with_backbone`. The refactor is to make them the **only** path:

- `BYOL._EncoderWithProjection`, `DINO._EncoderWithHead`, and
  `ContrastiveEncoder` stop re-deriving `backbone_dim` and call
  `detect_backbone_dim` uniformly.
- Feature distillation calls `encode_with_backbone(encoder, images)` to obtain
  $`h_s`$ and $`h_t`$, so the HF-output handling (`.pooler_output`,
  `.last_hidden_state`, `.logits`) is written once.

**Pitfall to avoid during the refactor.** `encode_with_backbone` currently ends
with a bare `return raw` fallback. Before feature distillation relies on it,
document and test that contract: a teacher and student whose backbones return
*differently shaped* outputs must fail loudly, not silently produce a
shape-mismatched L2 loss that broadcasts.

## 26. Extract The Layer-Matching Helper

FitNets and attention transfer need to pair a *teacher* layer $`\ell`$ with a
*student* layer $`\ell'`$. Today no such helper exists. Instead of hard-coding
indices inside the method, add a small, testable function — proposed
`genml_kit/models/layer_match.py`:

- `def match_layers(teacher_layers, student_layers, strategy) -> list[tuple]`,
  where `strategy` is one of `"by_index"`, `"by_name"` (matched on a regex of
  parameter names, reusing the regex style already present in
  `io/checkpointing.py`) or `"uniform"` (evenly spaced selection when the two
  networks have different depths).

This keeps the *policy* of which layers to distil out of `train_step`, so the
method reads as orchestration and the matching logic is unit-testable in
isolation.

## 27. Acceptance Criteria For The Refactor

The distillation PR is only mergeable when all of the following hold:

1. **No new copy** of the EMA loop, the cosine ramp, or the backbone-dim probe
   exists anywhere in the tree; every site calls the shared utility.
2. The refactored `BYOL`, `DINO`, `I-JEPA` and the two RL target networks pass
   their existing tests **unchanged in behaviour** — refactoring must not alter
   numerics. Any intentional numeric change is called out separately.
3. New utilities (`EMAModel`, `match_layers`) have unit tests that do not
   require a GPU or a real dataset.
4. `ruff` is clean and every file is formatted with `yapf` **2-space indent**
   per `.style.yapf`. **No `noqa` annotations are added to silence linters
   without explicit sign-off.**
5. The README (`knowledge_distillation/README.md`, this file) is updated to
   match the shipped module names and flags.

---

# Appendices

## Appendix A: Symbol Table

| Symbol | Meaning | Section |
|---|---|---|
| $`z^{(t)}, z^{(s)}`$ | teacher, student logits | §1, §4 |
| $`T`$ | distillation temperature | §4 |
| $`q_t, q_s`$ | tempered softmaxes of teacher / student | §4 |
| $`\alpha`$ | weight on the soft-target branch | §8 |
| $`h_t, h_s`$ | teacher / student hidden activations | §9 |
| $`W_r, b_r`$ | FitNets projection weight / bias | §9 |
| $`\mathcal{A}(A)`$ | attention map of a feature tensor | §10 |
| $`\psi, d`$ | similarity / distance used by relation KD | §11 |
| $`m`$ | EMA momentum | §13 |
| $`k, K`$ | current step, total step budget | §14 |
| $`\theta_s, \theta_t`$ | student / teacher parameters | §13 |

## Appendix B: Failure Modes And Pitfalls

- **§4** Applying temperature to the ground-truth CE branch as well as the KD
  branch.
- **§6** Forgetting the $`T^2`$ factor, silently scaling the KD gradient by
  $`1/T`$.
- **§9** Matching raw feature tensors with an L2 loss across an unaligned
  channel order; use attention transfer (§10) for spatial maps.
- **§17** Sharing one `processor` between teacher and student with different
  normalisation statistics.
- **§19** Not recording whether a checkpoint's teacher was `ema` or `external`,
  making resume ambiguous.
- **§21** Tracing a Python branch whose condition depends on tensor *values*;
  it is frozen for the dummy input.
- **§25** Relying on `encode_with_backbone`'s bare-`raw` fallback for feature
  distillation without a shape check.
- **§27** Adding `# noqa` to silence `ruff` instead of fixing the root cause.

## Appendix C: Reading List

**Distillation**

- G. Hinton, O. Vinyals, J. Dean, *Distilling the Knowledge in a Neural
  Network*, arXiv:1503.02531, 2015 — soft targets, temperature, the $`T^2`$
  scaling (§4–§8).
- A. Romero et al., *FitNets: Hints for Thin Deep Nets*, ICLR 2015 — hint
  layers and the guided projection (§9).
- S. Zagoruyko, N. Komodakis, *Paying More Attention to Attention*, ICLR 2017 —
  attention transfer for spatial maps (§10).
- W. Park et al., *Relational Knowledge Distillation*, CVPR 2019 — relation
  objectives and the Huber loss on angles/distances (§11).
- J. Gou et al., *Knowledge Distillation: A Survey*, IJCV 2021 — the
  response/feature/relation taxonomy used in Part 1 (§2).

**EMA teachers / self-distillation**

- J.-B. Grill et al., *Bootstrap Your Own Latent*, NeurIPS 2020 — online/target
  EMA (§13). Implemented in `genml_kit/models/byol.py`.
- M. Caron et al., *Emerging Properties in Self-Supervised Vision
  Transformers*, ICCV 2021 — self-distillation with a momentum teacher and
  centering (§13). Implemented in `genml_kit/models/dino.py`.
- M. Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture*, CVPR 2023 — EMA target encoder (§13). Implemented
  in `genml_kit/methods/ijepa.py`.

**Export / runtime**

- ONNX documentation — `torch.onnx.export`, opset versioning, dynamic axes
  (§20–§21).
- ONNX Runtime — `quantize_dynamic` and the numerical verification workflow
  (§22).
- PyTorch docs — `torch.nn.KLDivLoss` (the `log_target` semantics of §5) and
  `torch.autograd` (the softmax Jacobian of §6).

**Repository-internal**

- `genml_kit/models/encoder_utils.py` — the canonical backbone-dim probe and
  feature extractor (§25).
- `genml_kit/models/contrastive.py` — the `ProjectionHead` reused by FitNets
  (§9).
- `genml_kit/losses/contrastive.py` — the pairwise-similarity pattern that the
  relation-KD loss parallels (§11).
- `genml_kit/io/checkpointing.py` — the regex parameter-renaming style reused by
  `match_layers` (§26).

## Appendix D: Math Rendering Reference

The Markdown + KaTeX rendering rules for this document (inline math, display
math, escaping, operators, pitfalls) are centralised in
[`docs/MARKDOWN_LATEX.txt`](../docs/MARKDOWN_LATEX.txt).
