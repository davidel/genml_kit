# Knowledge Distillation — A Self-Contained Tutorial

This document teaches, from the ground up, the ideas behind **knowledge
distillation**: how a large, accurate *teacher* network can hand its behaviour to
a small, cheap *student* network, how the objective that drives that transfer is
built, and how an exponential-moving-average (EMA) teacher turns distillation
into *self-distillation*.

It is written for the same reader as [`vo/README.md`](../vo/README.md): someone
who knows how to train a classifier but wants the *why* behind every idea. No
result in this document depends on `genml_kit` internals to be understood — the
mathematics is developed from first principles, with derivations and proofs
worked out in full.

> **Scope.** This is a self-study tutorial, not an implementation guide. It
> develops the mathematics of distillation — the losses, their gradients, and the
> theory behind each design choice — and leaves the engineering to the code.

---

## How To Read This Document

- **Read order.** Part 1 frames the problem and the three families of knowledge.
  Parts 2–4 build one loss per family (response, feature, relation), each with
  its derivation. Part 5 explains where the teacher comes from (fixed vs. EMA)
  and why an averaged teacher is a better target. Part 6 collects the pieces and
  composes them into a single objective with a worked example.
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
- [Part 6 — Putting It Together](#part-6--putting-it-together)
  - [15. Composing The Three Losses](#15-composing-the-three-losses)
  - [16. A Worked Example](#16-a-worked-example)
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
p(y \mid x) = \big[\\;\text{car}: 0.90,\quad \text{truck}: 0.08,\quad
\text{bicycle}: 0.01,\quad \text{cat}: 0.0001,\\;\\dots\big].
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
\min_{\theta_s}\ \mathbb{E}_{x}\ \mathcal{D}\\!\left(
q_t(x)\\,\middle\\|\\,q_s(x)\right),
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
run may use all three, with weights $`\alpha_{\text{resp}}, \alpha_{\text{feat}}`$
and $`\alpha_{\text{rel}}`$ (Part 6).

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

This matters downstream: a student with calibrated soft outputs is usable as a
*dense* component (a logit or feature source inside another model), not just as
a top-1 classifier.

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

To see this quantitatively, consider the *logit difference* $`\Delta = z_i - z_j`$
between two classes. Their tempered log-ratio is

$$
\large
\log \frac{q_i(T)}{q_j(T)} = \frac{z_i - z_j}{T} = \frac{\Delta}{T}.
$$

So the temperature **divides** the separation between classes. A small,
informative gap $`\Delta`$ that is squashed to a ratio near $`1`$ at $`T = 1`$
becomes a visible, trainable ratio at large $`T`$. The student learns the
*ordering* of the non-target classes — exactly the dark knowledge — instead of
only the identity of the argmax.

**Pitfall.** Temperature is *not* applied to the ground-truth cross-entropy
branch (§8). It lives only on the distillation branch. Applying it to both
silently rescales the supervised signal and is one of the most common
implementation bugs.

## 5. The Kullback–Leibler Divergence Objective

The response loss is the forward KL from teacher to student, taken over the
batch:

$$
\large
\mathcal{L}_{\text{KD}}(x)
= T^{2}\\, \mathrm{KL}\\!\left( q_t(T) \\,\middle\\|\\, q_s(T) \right)
= T^{2} \sum_{i=1}^{C} q_{t,i}(T)\\,\log\\!\frac{q_{t,i}(T)}{q_{s,i}(T)}.
$$

### What the KL divergence means

The objective above is the Kullback–Leibler (KL) divergence between the two
tempered softmaxes, but the symbol deserves unpacking before we push it around
algebraically. For two distributions $`p`$ and $`q`$ over the same $`C`$ classes,
the *forward* KL is

$$
\large
\mathrm{KL}(p \\| q) = \sum_{i=1}^{C} p_i \log \frac{p_i}{q_i}.
$$

It is a **directed** quantity: in general
$`\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)`$, it is **non-negative**, and it
is **zero if and only if** $`p = q`$. For these reasons it is a *divergence* and
not a distance — it breaks both the symmetry and the triangle inequality that a
metric requires. (The logarithm is natural, so the units are *nats*, exactly as
everywhere else in this document.)

The two ingredients it is built from are the **entropy** and the
**cross-entropy**. The entropy

$$
\large
H(p) = -\sum_{i=1}^{C} p_i \log p_i
$$

is the expected *surprise* $`-\log p_i`$ of a draw from $`p`$ — equivalently, the
smallest average number of nats needed to encode samples drawn from $`p`$. The
cross-entropy

$$
\large
H(p, q) = -\sum_{i=1}^{C} p_i \log q_i
$$

is the average cost of coding those same samples with a code built for $`q`$
instead. Splitting the logarithm of the KL sum shows that the divergence is
exactly the excess of that cost over the entropy:

$$
\large
\mathrm{KL}(p \\| q) = \sum_{i=1}^{C} p_i \log p_i - \sum_{i=1}^{C} p_i \log q_i
= H(p, q) - H(p)
= \mathbb{E}_{p}\left[\log p - \log q\right].
$$

The expectation form is the most illuminating: KL is the average gap between the
surprise $`-\log q`$ assigned by the model and the true surprise $`-\log p`$ —
the number of nats wasted, on average, by believing $`q`$ when the truth is
$`p`$. When the two distributions agree, nothing is wasted and the gap collapses
to zero.

**Why it is non-negative (Gibbs' inequality).** The elementary bound
$`\log x \le x - 1`$ (with equality only at $`x = 1`$) gives

$$
\large
\mathrm{KL}(p \\| q) = -\sum_i p_i \log \frac{q_i}{p_i}
\ge -\sum_i p_i \left(\frac{q_i}{p_i} - 1\right)
= -\left(\sum_i q_i - \sum_i p_i\right) = 0,
$$

with equality precisely when $`q_i = p_i`$ for every $`i`$. This is what makes KL
a legitimate training objective: the loss has a unique minimum at the target and
cannot go negative.

**Forward versus reverse.** Because KL is asymmetric, the *direction* matters.
The forward divergence $`\mathrm{KL}(p \| q)`$ penalises every class where $`p`$
is non-zero but $`q`$ is small, so it is *mass-covering* (mean-seeking): the
minimiser spreads itself out to cover all of $`p`$'s support. The reverse
divergence $`\mathrm{KL}(q \| p)`$ instead penalises classes where $`q`$ is
non-zero but $`p`$ is small, which is *mode-seeking* (zero-forcing): it
concentrates on a single mode of $`p`$ and ignores the rest. Distillation uses
the forward direction $`\mathrm{KL}(q_t \| q_s)`$ — teacher first — so the
student is pushed to cover the teacher's entire soft distribution, dark knowledge
and all.

**Summing up.** In one sentence: KL is the average *coding overhead* of using the
student's distribution to describe samples drawn from the teacher's; minimising it
turns the student into the cheapest possible description of the teacher.

**A worked example (in nats).** Reusing the three-class distributions of §16,
with $`q_t \approx [0.766,\ 0.171,\ 0.063]`$ and
$`q_s \approx [0.506,\ 0.307,\ 0.186]`$, the teacher's entropy is
$`H(q_t) \approx 0.680`$ and the cross-entropy to the student is
$`H(q_t, q_s) \approx 0.829`$. Hence

$$
\large
\mathrm{KL}(q_t \\| q_s) = H(q_t, q_s) - H(q_t)
\approx 0.829 - 0.680 = 0.149 \text{ nats}.
$$

For comparison, the reverse direction gives
$`\mathrm{KL}(q_s \| q_t) \approx 0.173`$ nats — the flip side of the asymmetry,
and the reason the forward divergence above is the one we minimise.

### Why KL collapses to cross-entropy

Split the logarithm and separate the two terms:

$$
\large
\mathrm{KL}(q_t \\| q_s)
= \sum_i q_{t,i} \log q_{t,i} - \sum_i q_{t,i} \log q_{s,i}
= -H(q_t) + H(q_t, q_s),
$$

where $`H(q_t) = -\sum_i q_{t,i}\log q_{t,i}`$ is the **entropy** of the teacher
and $`H(q_t, q_s) = -\sum_i q_{t,i}\log q_{s,i}`$ is the **cross-entropy**
between teacher and student.

Because $`q_t`$ is produced by the frozen teacher, it does not depend on the
student parameters $`\theta_s`$. Its entropy $`H(q_t)`$ is therefore a
**constant** with respect to the optimisation, and

$$
\large
\arg\min_{\theta_s}\ \mathrm{KL}(q_t \\| q_s)
= \arg\min_{\theta_s}\ H(q_t, q_s).
$$

This is the key simplification: **minimising the KL divergence is exactly
minimising the cross-entropy to the teacher's soft targets.** The loss reduces,
up to the constant $`H(q_t)`$, to

$$
\large
\mathcal{L}_{\text{KD}}(x)
\\;\equiv\\; -\\,T^{2}\sum_{i=1}^{C} q_{t,i}(T)\\,\log q_{s,i}(T)
\qquad (\text{teacher detached}).
$$

Implementations almost always compute it as
`KLDivLoss(log_softmax(student/T), softmax(teacher/T)) * T**2`, which is
numerically cleaner than forming the ratio explicitly, and which automatically
drops the constant.

## 6. The Gradient Of The KD Loss

Let $`q_s = \mathrm{softmax}(z_s / T)`$. We differentiate the cross-entropy
form of §5 with respect to a student logit $`z_{s,k}`$. The derivation has two
ingredients.

**Step 1 — the softmax Jacobian.** For the tempered softmax,

$$
\large
\frac{\partial q_{s,i}}{\partial z_{s,k}}
= \frac{1}{T}\\, q_{s,i}\big(\delta_{ik} - q_{s,k}\big),
$$

where $`\delta_{ik}`$ is the Kronecker delta. This follows from the quotient
rule: the numerator differentiates to $`q_{s,i}`$ when $`i = k`$ and to zero
otherwise, while the normaliser differentiates to $`q_{s,i} q_{s,k}`$.

**Step 2 — chain rule through the logarithm.** Since
$`\partial \log q_{s,i} / \partial z_{s,k} = (1/q_{s,i})\,\partial q_{s,i}/\partial z_{s,k}`$,

$$
\large
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{s,k}}
= -T^{2} \sum_i q_{t,i}\\, \frac{1}{q_{s,i}}\\, \frac{\partial q_{s,i}}{\partial z_{s,k}}
= -T \sum_i q_{t,i}\big(\delta_{ik} - q_{s,k}\big).
$$

The two sums then collapse because $`\sum_i q_{t,i} = 1`$:

$$
\large
\frac{\partial \mathcal{L}_{\text{KD}}}{\partial z_{s,k}}
= -T\big(q_{t,k} - q_{s,k}\big)
= T\\,\big(q_{s,k}(T) - q_{t,k}(T)\big).
$$

Two things are visible in this formula.

1. The gradient is a **difference of distributions**, so it vanishes exactly
   when the student matches the teacher — the correct fixed point.
2. The factor $`T`$ (before the $`T^2`$ scaling in the loss) is what the $`T^2`$
   is meant to counteract.

**Checkpoint.** What happens if you forget the $`T^2`$? *Answer: the gradient
magnitude of the distillation term scales like $`1/T`$ relative to the
cross-entropy term. At the common $`T = 4`$ the distillation signal is silently
diluted by $`4\times`$, and the effective $`\alpha`$ you set is a lie. This is
the single most cited "why is my distilled model worse" cause.*

## 7. Why The $`T^2`$ Factor, And Why It Is Not Cosmetic

From §6, the raw gradient carries a factor $`T`$ from the division inside the
softmax. Multiplying the loss by $`T^2`$ makes the *total* gradient of the
distillation branch

$$
\large
T^{2} \cdot \frac{1}{T}\big(q_{s,k} - q_{t,k}\big) \cdot T
= T^{2}\big(q_{s,k} - q_{t,k}\big) \cdot \frac{1}{1},
$$

i.e. $`O(1)`$ in $`T`$ once the $`1/T`$ from the loss and the $`T`$ from the
softmax Jacobian are accounted for. Concretely, the gradient of the
cross-entropy-to-soft-targets term is

$$
\large
\frac{\partial}{\partial z_{s,k}}\Big[-T^{2}\sum_i q_{t,i}\log q_{s,i}\Big]
= T\big(q_{s,k} - q_{t,k}\big)\cdot T = T^{2}\big(q_{s,k} - q_{t,k}\big),
$$

so the balance between the distillation term and the ground-truth term is
governed *only* by the mixing weight $`\alpha`$ and **not** by the temperature.
With the $`T^2`$ in place, changing $`T`$ changes *which examples contribute*
(the softness) but not *how loudly* the branch speaks.

**Checkpoint.** Why is this called "not cosmetic"? *Answer: without it, tuning
$`T`$ silently re-tunes the loss balance. Two runs with different temperatures
but the same $`\alpha`$ would be incomparable, and the "best" temperature on a
grid search would really be encoding an accidental learning-rate change. The
$`T^2`$ factor decouples the two knobs.*

## 8. Combining With The Ground-Truth Cross-Entropy

The total objective for plain logit distillation is the convex combination

$$
\large
\mathcal{L}
= (1 - \alpha)\\,\underbrace{\mathcal{L}_{\text{CE}}(z_s, y)}_{\text{hard labels}}
\\;+\\; \alpha\\,\underbrace{T^{2}\\,\mathrm{KL}\\!\left(q_t \middle\\| q_s\right)}_{\text{soft labels}},
$$

with $`\alpha \in [0, 1]`$. If ground-truth labels are unavailable (unlabeled
data, or a pure function-matching run), set $`\alpha = 1`$ and drop the CE
branch — the loss is then a pure distribution-matching objective, which is
exactly the regime of self-distillation (Part 5).

**Checkpoint.** Why keep the hard-label term at all? *Answer: the soft targets
are a rich but imperfect teacher. On examples where the teacher is confidently
wrong, the CE term anchors the student to the truth. The convex combination lets
the two signals disagree in a controlled way: $`\alpha`$ is the trust you place
in the teacher versus the labels.*

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
= \big\\|\\, r(h_s) - h_t \\,\big\\|_2^{2},
\qquad r(h_s) = W_r\\,h_s + b_r,
$$

where the student-side projection $`r`$ is trained jointly and *discarded*
afterwards — exactly analogous to the projection head used in self-supervised
pre-training, which is dropped once the encoder is fine-tuned.

### Why the projection is needed, and why it is discarded

The projection exists for one reason: **dimension mismatch**. If
$`H_s = H_t`$ the loss is simply $`\|h_s - h_t\|_2^2`$ and no projection is
required. When the student is *narrower* than the teacher, there is no way to
compare the two vectors directly, so we learn the best linear map into the
teacher's space. Discarding $`r`$ afterwards is deliberate: it is a *scaffold*,
not part of the model. Keeping it would let the student satisfy the feature
match in a rotated subspace that is useless to its own classifier head.

**Checkpoint.** Does the projection hurt if the student is *wider* than the
teacher? *Answer: no — $`W_r`$ then maps a larger space into a smaller one, which
is a normal linear projection. The only cost is a few extra parameters during
training, removed at export time.*

**Checkpoint.** Where should the hint layer be? *Answer: FitNets uses an early-
to-middle layer. Early layers transfer generic structure (edges, textures) that
generalises; late layers encode teacher-specific decisions that a smaller
student cannot represent. Matching too deep tends to fight the response loss.*

## 10. Attention Transfer

When the intermediate tensors are *spatial* ($`(C, H, W)`$, as in a
convolutional or ViT feature map), an L2 match on the raw tensor is brittle:
it is sensitive to channel permutation and scale. **Attention transfer**
(Zagoruyko & Komodakis, 2017) instead matches a *summary* of each map. For a
feature map $`A \in \mathbb{R}^{C \times H \times W}`$ define the attention map

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
{\big\\|\\,\mathrm{vec}\\,\mathcal{A}(A)\\,\big\\|_2}.
$$

### The normalisation is scale-invariant (proof)

For any scalar $`c > 0`$:

$$
\large
\mathcal{A}(cA) = \sum_c \lvert cA_c\rvert^2 = c^2 \sum_c \lvert A_c\rvert^2 = c^2 \mathcal{A}(A),
$$

and therefore
$`\|\mathrm{vec}\,\mathcal{A}(cA)\|_2 = c^2 \|\mathrm{vec}\,\mathcal{A}(A)\|_2`$.
The factors of $`c^2`$ cancel in the ratio, so

$$
\large
\tilde{\mathcal{A}}(cA) = \frac{c^2 \mathcal{A}(A)}{c^2 \\|\mathrm{vec}\\,\mathcal{A}(A)\\|_2}
= \tilde{\mathcal{A}}(A).
$$

**Normalisation removes the scale of the activation and leaves only its spatial
shape.** Two students whose feature maps differ only by a constant factor get the
same loss, which is exactly what we want: the teacher's *where* matters, not its
*how loud*.

The transfer loss is

$$
\large
\mathcal{L}_{\text{AT}} = \big\\| \tilde{\mathcal{A}}(h_s) - \tilde{\mathcal{A}}(h_t) \big\\|_2^{2}.
$$

Attention transfer is the natural feature loss for ViT-based and timm backbones,
because it is agnostic to the number of channels (the sum over $`C`$ removes it)
and only requires spatial alignment.

**Pitfall.** $`\mathcal{A}`$ is a sum of squares, so it discards *sign*. Two maps
that are negatives of each other have identical attention. This is fine when the
channel ordering is arbitrary (a CNN or a ViT whose channels are learned), but it
means attention transfer cannot, on its own, pin down the orientation of a
feature map. Pair it with a response loss when the sign matters.

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
= \sum_{i \ne j}\ \ell_\delta\\!\Big(\phi\big(d_s(i,j)\big),\ \phi\big(d_t(i,j)\big)\Big)
\\;+\\; \sum_{(i,j,k)} \ell_\delta\\!\big(\text{angle}_s, \text{angle}_t\big),
$$

where $`\ell_\delta`$ is the Huber loss and $`\phi`$ a normalisation.

### The two terms, written out

**Distance-wise.** Let $`\mu_s = \tfrac{1}{|B|}\sum_i \|h_i - h_j\|`$ be the mean
pairwise distance in a batch, and define the normalised distance
$`\phi(d_s(i,j)) = d_s(i,j) / \mu_s`$. The distance term penalises disagreement
in the *relative* spacing of representations. Dividing by the batch mean makes
the term invariant to the overall scale of the feature space.

**Angle-wise.** For a triplet $`(i, j, k)`$ with $`j`$ the anchor, the angle is

$$
\large
\text{angle}(i, j, k) = \Big\langle \frac{h_i - h_j}{\\|h_i - h_j\\|_2},\
\frac{h_k - h_j}{\\|h_k - h_j\\|_2} \Big\rangle,
$$

the cosine of the angle subtended at the anchor $`j`$. The angle term penalises
disagreement in the *direction* structure — the shape of the constellation of
points, independent of rotation and reflection.

### Why this is translation- and permutation-invariant

Both terms are built from *differences* $`h_i - h_j`$ and inner products of
normalised differences. Subtracting a constant vector from every $`h_i`$ (a
translation) leaves every difference unchanged; reordering the batch permutes
the summation indices without changing the sum. So the student need not
reproduce the teacher's *coordinate frame*, only its *relational structure* —
which is why RKD transfers well across architectures whose feature spaces are
not directly comparable.

**Checkpoint.** When is relation distillation the *only* viable choice? *Answer:
when teacher and student have different output spaces (e.g. different label
sets, or teacher = ensemble of models) so that no element-wise feature or logit
match exists. Relation losses compare *within-network* structure, so they never
need the two spaces to align.*

**Pitfall.** The angle term is $`O(B^3)`$ in the batch size because it sums over
triplets. In practice one samples a fixed number of triplets per batch rather
than enumerating all of them.

---

# Part 5 — Where Does The Teacher Come From?

## 12. Fixed Teachers And Offline Distillation

The simplest regime: train (or load) a teacher, freeze it, and train the
student against its outputs. If the teacher's responses are precomputed and
cached to disk, this is **offline distillation** — the student never even holds
the teacher in memory, and each epoch is a normal supervised run over cached
soft targets.

**Checkpoint.** What is the cost of offline distillation? *Answer: storage.
Each example now carries $`C`$ floats instead of one int, and the teacher must
be re-run whenever the input augmentation changes. Online (teacher-in-the-loop)
distillation avoids the storage but pays teacher forward-pass compute every
step.*

**Checkpoint.** Why is the teacher frozen? *Answer: the distillation target must
be stationary. If the teacher's parameters moved with the student, the student
would be chasing a moving target and the objective would have no fixed point.*

## 13. Self-Distillation And The EMA Teacher

Self-distillation removes the external teacher entirely: the teacher is an
**exponential moving average** of the student's own weights. If the student is
$`\theta_s`$, the teacher $`\theta_t`$ is updated, every optimizer step, by

$$
\large
\theta_t \leftarrow m\\,\theta_t + (1 - m)\\,\theta_s,
\qquad m \in (0, 1).
$$

### Why an average is a better target

Unrolling the recursion shows that $`\theta_t`$ is a geometric average of the
student's past weights:

$$
\large
\theta_t^{(k)} = (1 - m)\sum_{\tau = 0}^{k-1} m^{\tau}\ \theta_s^{(k-\tau)}
\\;+\\; m^{k}\theta_t^{(0)}.
$$

The weights $`(1-m) m^{\tau}`$ sum to $`1 - m^k \to 1`$, so asymptotically
$`\theta_t`$ is a convex combination of all past students, with exponentially
decaying weight on older steps. Now suppose each $`\theta_s^{(k)}`$ is the true
optimum plus zero-mean noise: $`\theta_s^{(k)} = \theta^\star + \epsilon_k`$
with $`\mathrm{Var}(\epsilon_k) = \sigma^2`$. If the noise is roughly
independent across steps, the variance of the average is

$$
\large
\mathrm{Var}(\theta_t) \approx \sigma^2\\,(1-m)^2 \sum_{\tau=0}^{\infty} m^{2\tau}
= \sigma^2\\,(1-m)^2 \cdot \frac{1}{1 - m^2}
= \sigma^2\\, \frac{1 - m}{1 + m}.
$$

For $`m`$ close to $`1`$ this is a **large variance reduction**: at $`m = 0.996`$
the factor is $`(1-m)/(1+m) \approx 0.002`$, i.e. the teacher is roughly
$`500\times`$ less noisy than a single student checkpoint. A lower-variance
target is a better regression target, which is the theoretical reason EMA
teachers (BYOL, DINO, I-JEPA) work so well.

### Gradient flow

The teacher parameters are never updated by gradients
(`requires_grad = False`), so only the student branch receives a backward pass.
The forward pass, however, runs through both networks, and the teacher sees the
*same* input under a (usually different) augmentation — the "two views" idea.

**Checkpoint.** If the teacher is a copy of the student, why is this not
degenerate? *Answer: because the teacher is **lagged and averaged**. At any
instant it differs from the student (it is a slower, smoother version), so the
student is pulled toward its own recent consensus. This is a form of
self-consistency regularisation, not a tautology.*

## 14. Momentum Scheduling

A constant $`m`$ works, but the literature (BYOL, DINO, I-JEPA) ramps it. The
common schedule is a **linear** ramp from an initial value to a final value:

$$
\large
m(k) = m_{\text{start}} + (m_{\text{end}} - m_{\text{start}})\\,
\min\\!\left(\frac{k}{K},\\, 1\right),
$$

where $`k`$ is the current step and $`K`$ the step budget.

### Why ramp

Early in training the student is far from any good solution, so its instantaneous
weights are poor. A teacher that tracked them closely (small $`m`$) would simply
echo the noise. But early on we *want* the teacher to adapt fast, because the
student is still exploring and a stale teacher would give irrelevant targets. The
ramp resolves this tension:

- **Early (small $`m`$):** the teacher tracks the student closely, so it adapts
  quickly and never lags behind a rapidly-improving student.
- **Late (large $`m`$):** the teacher becomes a slow, stable target. Its
  variance has dropped (see §13), so it provides clean, low-noise targets that
  let the student converge.

**Checkpoint.** What does the step budget $`K`$ depend on? *Answer: the number of
optimizer steps in the whole run — the loader length times the epoch count. It is
fixed at the start of training so the schedule is deterministic and resumable.*

**Checkpoint.** Why not a cosine ramp? *Answer: it is a valid alternative, but it
changes the training dynamics (the momentum would not reach its maximum until
half-way through). A linear ramp is the simplest schedule that is monotone and
reaches $`m_{\text{end}}`$ exactly at the end.*

---

# Part 6 — Putting It Together

## 15. Composing The Three Losses

The three families are orthogonal and compose additively. A general distillation
objective is

$$
\large
\mathcal{L} = \mathcal{L}_{\text{task}}
\\;+\\; \alpha_{\text{resp}}\\,\mathcal{L}_{\text{KD}}
\\;+\\; \alpha_{\text{feat}}\\,\mathcal{L}_{\text{feat}}
\\;+\\; \alpha_{\text{rel}}\\,\mathcal{L}_{\text{rel}},
$$

where $`\mathcal{L}_{\text{task}}`$ is the ordinary supervised loss (or zero, for
a purely self-supervised run), and the three $`\alpha`$ weights select which
families are active. Setting a weight to zero disables that branch cleanly, so the
same code path covers logit-only KD ($`\alpha_{\text{feat}} = \alpha_{\text{rel}} = 0`$),
feature-only KD, and everything in between.

The pieces interact in ways worth stating explicitly:

- **Response + feature** is the classic FitNets configuration: match the final
  softmax *and* a hint layer, so the student is constrained at two depths.
- **Response + relation** is common for heterogeneous teachers: match the logits
  where the output spaces align, and the relational structure where they do not.
- **All three** is the "full" distillation; it is the most expensive and the most
  sensitive to weight tuning, because three gradients now compete for the same
  student parameters.

**Checkpoint.** What is the risk of turning everything on? *Answer: gradient
conflict. If the feature and response gradients point in opposite directions, the
student can stall. Start with response-only, add one family at a time, and watch
the validation metric after each addition.*

## 16. A Worked Example

Consider a three-class problem ($`C = 3`$) with a teacher whose logits for one
example are $`z^{(t)} = [4.0,\ 1.0,\ -1.0]`$ and a student whose logits are
$`z^{(s)} = [2.0,\ 1.0,\ 0.0]`$. Use $`T = 2`$.

**Step 1 — tempered softmaxes.**

$$
\large
q_t(2) = \mathrm{softmax}\\!\left(\tfrac{1}{2}[4, 1, -1]\right)
= \mathrm{softmax}([2, 0.5, -0.5]),
$$

$$
\large
q_s(2) = \mathrm{softmax}\\!\left(\tfrac{1}{2}[2, 1, 0]\right)
= \mathrm{softmax}([1, 0.5, 0]).
$$

Numerically, $`q_t(2) \approx [0.766,\ 0.171,\ 0.063]`$ and
$`q_s(2) \approx [0.506,\ 0.307,\ 0.186]`$. Notice how much softer the teacher is
at $`T = 2`$ than at $`T = 1`$ (where it would be $`[0.946,\ 0.047,\ 0.006]`$):
the dark knowledge is visible.

**Step 2 — the KL term.** The cross-entropy to the soft targets is

$$
\large
H(q_t, q_s) = -\sum_i q_{t,i}\log q_{s,i}
\approx -(0.766\log 0.506 + 0.171\log 0.307 + 0.063\log 0.186)
\approx 0.829.
$$

**Step 3 — the $`T^2`$ scaling.** With $`T = 2`$ the response loss contribution is
$`T^2 \cdot H(q_t, q_s) \approx 4 \times 0.829 = 3.316`$.

**Step 4 — the gradient.** By §6 the gradient with respect to the student logit
$`z_{s,1}`$ (class 1) is

$$
\large
T\big(q_{s,1} - q_{t,1}\big) = 2\\,(0.506 - 0.766) = -0.520.
$$

The negative sign says: *increase* $`z_{s,1}`$ to move the student toward the
teacher's more confident class-1 probability. The magnitude is the (temperature-
scaled) disagreement between the two distributions.

**Checkpoint.** What would the gradient be without the $`T^2`$? *Answer: the
$`T`$ factor in the gradient would be absent, so the signal would be
$`(q_{s,1} - q_{t,1}) = -0.260`$ — half as large at $`T = 2`$. The student would
learn more slowly and the effective weight of the distillation term would depend
on the temperature, which is exactly the bug §7 warns against.*

---

# Appendices

## Appendix A: Symbol Table

| Symbol | Meaning | Section |
|---|---|---|
| $`z^{(t)}, z^{(s)}`$ | teacher, student logits | §1, §4 |
| $`T`$ | distillation temperature | §4 |
| $`q_t, q_s`$ | tempered softmaxes of teacher / student | §4 |
| $`\alpha`$ | weight on the soft-target branch | §8 |
| $`h_t, h_s`$ | teacher / student activation vectors | §9 |
| $`W_r, b_r`$ | FitNets projection weights / bias | §9 |
| $`A`$ | a spatial feature map, $`(C, H, W)`$ | §10 |
| $`\mathcal{A}(A)`$ | attention map (sum of squared channels) | §10 |
| $`\tilde{\mathcal{A}}(A)`$ | scale-normalised attention map | §10 |
| $`\ell_\delta`$ | Huber loss with threshold $`\delta`$ | §11 |
| $`d_s, d_t`$ | student / teacher pairwise distances | §11 |
| $`\theta_s, \theta_t`$ | student / teacher parameters | §13 |
| $`m`$ | EMA momentum (teacher decay) | §13 |
| $`K`$ | total optimizer-step budget for the ramp | §14 |
| $`\alpha_{\text{resp}}, \alpha_{\text{feat}}, \alpha_{\text{rel}}`$ | per-family loss weights | §15 |

## Appendix B: Failure Modes And Pitfalls

- **§4** Applying temperature to the ground-truth CE branch as well as the KD
  branch.
- **§6** Forgetting the $`T^2`$ factor, silently scaling the KD gradient by
  $`1/T`$.
- **§9** Matching raw feature tensors with an L2 loss across an unaligned
  channel order; use attention transfer (§10) for spatial maps.
- **§10** Relying on attention alone to fix the *sign* of a feature map — the sum
  of squares discards it.
- **§11** Enumerating all triplets for the angle term; sample them instead, or
  the cost grows as $`O(B^3)`$.
- **§12** Caching teacher targets once and then changing the augmentation
  pipeline; the cache silently becomes stale.
- **§13** Updating the teacher with gradients (it must be `no_grad`) or letting
  the student and teacher share parameters (they must be distinct copies).
- **§15** Turning on all three losses at once and mis-attributing a regression to
  the wrong branch; add one family at a time.

## Appendix C: Reading List

- Hinton, Vinyals, Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), 2015 — the soft-target idea and the $`T^2`$ factor.
- Romero et al., [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550), 2015 — hint layers and the projection.
- Zagoruyko, Komodakis, [Paying More Attention to Attention](https://arxiv.org/abs/1612.03928), 2017 — attention transfer.
- Park et al., [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068), 2019 — distance- and angle-wise losses.
- Gou et al., [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525), 2021 — the taxonomy of the three families.
- Grill et al., [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733), 2020 — EMA teacher and momentum ramp.
- He et al., [Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722), 2020 — the momentum-encoder formulation.
- Caron et al., [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://arxiv.org/abs/2104.14294), 2021 — self-distillation with an EMA teacher and multi-crop.

## Appendix D: Math Rendering Reference

The Markdown + KaTeX rendering rules for this document (inline math, display
math, escaping, operators, pitfalls) are centralised in
[`docs/MARKDOWN_LATEX.txt`](../docs/MARKDOWN_LATEX.txt).
