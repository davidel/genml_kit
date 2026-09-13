# Visual Odometry Front-End — A Self-Contained Tutorial

This document teaches, from the ground up, the mathematics behind the
visual-odometry (VO) front-end of this repository: how two images taken a
split second apart by a camera on a flying drone can be turned into a single,
well-defined number — the *similarity transform* that maps one image onto the
other — together with a *confidence* in that answer.

We assume only what an engineering undergraduate knows by their third year:

- linear algebra (vectors, matrices, eigenvalues, and the singular value
  decomposition at the level of "what it does", not a full proof);
- basic probability and statistics (mean, variance, the normal distribution,
  what "Gaussian noise" means);
- basic signal processing (what a Fourier transform is, what a convolution
  does);
- a working idea of what a convolutional neural network is, and of what
  *supervised training* and *loss functions* do.

Everything else — image coordinates, pinhole cameras, homographies, warping,
phase correlation, Lucas–Kanade alignment, and Kalman-filter consumption — is
developed here from zero, always with the *why* before the *what*.

This document is a tutorial, not a reference card.  We will not assume you
have seen any of this before.  Every formula is motivated, most are derived,
and every abstract idea is anchored to a concrete, worked example — usually
with real numbers you can check by hand.  Where a piece of this repository
implements an idea, the implementation is pointed to in an *Implementation
note*, so you can see the math alive in code; but you never need to read the
code to understand the math.

---

## How to read this document

- **Read order.**  The chapters are ordered so that each one builds on the
  previous.  Part 1 defines the thing the system must output (a similarity
  transform) and the metric used to judge it.  Part 2 explains where the
  training data comes from and why the real world refuses to behave exactly
  like a similarity.  Part 3 shows how a neural network can be taught to
  *estimate* that similarity end-to-end.  Part 4 defines the losses used to
  train it.  Part 5 covers the classical, non-learned algorithms that serve
  as ground-truth oracles.  Part 6 shows how the output is consumed by the
  downstream state estimator (an extended Kalman filter).
- **If you only have 20 minutes**, read Parts 1 and 6 and skim the worked
  90-degree example in §3.  That gives you the "what" and the "why it
  matters".
- **Boxes.**  Callouts labelled *Implementation note* point at the
  corresponding code/tests in this repo &mdash; useful when you want to see
  the math in action.  Callouts labelled *Check your understanding* pose a
  short question whose answer is given immediately; use them to make sure the
  paragraph you just read actually landed.
- **Notation.**  All angles are in radians unless explicitly stated.  Vectors
  are column vectors.  Image coordinates are `(x, y)` with `x` growing to the
  right and `y` growing *downward* (the standard for images).  A list of every
  symbol, with a plain-English meaning and the section where it first appears,
  is in [Appendix A](#appendix-a-symbol-table).

---

## Contents

- **Part 1 — The thing we compute: the 2-D similarity transform**
  - [1. What problem are we solving?](#1-what-problem-are-we-solving)
  - [2. The similarity transform, taught from scratch](#2-the-similarity-transform-taught-from-scratch)
  - [3. A fully-worked example: the 90-degree turn](#3-a-fully-worked-example-the-90-degree-turn)
  - [4. How we store the parameters (log scale, wrapped angle)](#4-how-we-store-the-parameters-log-scale-wrapped-angle)
  - [5. Reading the parameters back from a matrix](#5-reading-the-parameters-back-from-a-matrix)
  - [6. The metric that judges everything: mean corner error](#6-the-metric-that-judges-everything-mean-corner-error)
- **Part 2 — Where the data comes from: synthetic pair generation**
  - [7. The oblique pinhole camera](#7-the-oblique-pinhole-camera)
  - [8. The ground-to-image homography](#8-the-ground-to-image-homography)
  - [9. Ground truth: fitting a similarity to a homography](#9-ground-truth-fitting-a-similarity-to-a-homography)
  - [10. The irreducible residual](#10-the-irreducible-residual)
  - [11. Warping: how frame B is actually rendered](#11-warping-how-frame-b-is-actually-rendered)
- **Part 3 — The learned estimator**
  - [12. The correlation cost volume](#12-the-correlation-cost-volume)
  - [13. The Umeyama closed-form fit](#13-the-umeyama-closed-form-fit)
  - [14. Why we regress corners, not parameters](#14-why-we-regress-corners-not-parameters)
  - [15. Differentiable warping with grid_sample](#15-differentiable-warping-with-grid_sample)
- **Part 4 — Losses and metrics**
  - [16. The supervised loss, term by term](#16-the-supervised-loss-term-by-term)
  - [17. The staged photometric auxiliary](#17-the-staged-photometric-auxiliary)
  - [18. Metrics and how they are sliced](#18-metrics-and-how-they-are-sliced)
- **Part 5 — Classical baselines (the oracles)**
  - [19. Phase correlation (translation only)](#19-phase-correlation-translation-only)
    - [20. Fourier–Mellin (rotation + scale)](#20-fouriermellin-rotation--scale)
    - [21. Lucas–Kanade and ECC (dense refinement)](#21-lucaskanade-and-ecc-dense-refinement)
    - [22. Keypoints + RANSAC (the strongest classical oracle)](#22-keypoints--ransac-the-strongest-classical-oracle)
- **Part 6 — Consumption: the EKF contract**
  - [23. A minimal Kalman-filter recap](#23-a-minimal-kalman-filter-recap)
  - [24. Confidence as measurement noise](#24-confidence-as-measurement-noise)
  - [25. Update and gating](#25-update-and-gating)
- **Appendices**
  - [Appendix A: Symbol table](#appendix-a-symbol-table)
  - [Appendix B: Failure modes and shortcuts](#appendix-b-failure-modes-and-shortcuts)
  - [Appendix C: Reading list](#appendix-c-reading-list)
  - [Appendix D: Math rendering notes](#appendix-d-math-rendering-notes)

---

# Part 1 — The thing we compute: the 2-D similarity transform

## 1. What problem are we solving?

A flying drone carries a camera.  The camera is mounted on the body, pointing
**obliquely** at the ground at roughly 45 degrees, and it is **not** on a
gimbal: pitch, yaw, and height all change as the drone flies, and the camera
moves with the body.  The camera takes pictures at a fixed rate.  Between two
consecutive pictures &mdash; call them frame A and frame B &mdash; the drone
has moved: it has advanced, maybe turned, maybe climbed or descended a little.

The visual-odometry (VO) front-end is the piece of software that looks at the
pair (A, B) and answers one question:

> **What rigid motion of the camera, expressed in the image plane, turns frame
> A into frame B?**

"Rigid motion of the camera, expressed in the image plane" deserves an
unpacking.  If the camera only *translates* (slides sideways or up/down in the
image), then frame B is frame A shifted by some number of pixels: a pure
**translation**.  If the drone *yaws* (turns left/right), the whole scene
rotates in the image: on top of the translation we get a **rotation**.  And if
the drone climbs or descends, or moves closer to or farther from the ground
features it is looking at, everything in the image appears to grow or shrink
uniformly: a **uniform scale**.

So the natural mathematical object is the **2-D similarity**:

$$
\large
x' = s \cdot R_\theta \cdot x + t
$$

which says: *to find where a pixel at position `x` in frame A ended up in
frame B, rotate it by angle `θ`, scale it by `s`, and translate it by `t`.*
Four numbers: one scale, one angle, two translation components.  That is
everything the front-end must produce — plus a **confidence** score saying how
much to trust it, for reasons we will get to in [Part 6](#part-6--consumption-the-ekf-contract).

Why a similarity and not something more general?  A drone's camera is a rigid
body; between two frames a few tens of milliseconds apart, the world's
appearance changes by exactly this family — unless the ground is not flat, or
the camera dives steeply, which is the subject of [Part 2](#part-2--where-the-data-comes-from-synthetic-pair-generation).
A more general family (affine: 6 numbers; homography: 8 numbers) could express
more, but would be harder to learn, harder to fit, and would not correspond to
the physical reality that a rigid body undergoing `(yaw, scale, translate)`
produces.  Four numbers is the *right* answer because it is the smallest
family that contains the true motion — and the truth is almost, but not
exactly, inside it.

**Check your understanding.**  Why is a pure "zoom in" (camera moves straight
toward a flat ground plane, staying perpendicular to it) a similarity and not,
say, an affine transform?  *Answer: zooming scales every image distance from
the image center by the same factor — that is exactly `s > 0` with `θ = 0`
and `t = 0`, applied about the origin.  This falls inside the similarity
family.  Affine would be needed only if horizontal and vertical scales
differed — which a rigid camera cannot produce on a flat ground.*

---

## 2. The similarity transform, taught from scratch

### 2.1 Coordinate frames and the image plane

Before any formula, settle the coordinates.  Images are arrays of pixels.
Pixel `(x, y)` in an image of width `W` and height `H` has `x ∈ [0, W−1]`
growing to the right and `y ∈ [0, H−1]` growing *down*.  We call this the
**image coordinate frame**.  All the pixel positions in this document live in
this frame: the top-left pixel is `(0, 0)`, the bottom-right is `(W−1, H−1)`.

The image frame is not a "mathematical" frame (where `y` grows up) — it is the
frame that code and data actually use.  Getting the `y`-direction wrong is the
single most common source of sign errors in all of image geometry.  We will
return to this in the worked example of §3, where the sign of a rotation is
pinned down once and for all.

### 2.2 The rotation matrix, from first principles

A **rotation by `θ` radians counter-clockwise** in the usual mathematical
frame maps a point `(x, y)` to `(x', y')`.  Where do the sine and cosine come
from?  Write a point in polar form: `x = r cos φ`, `y = r sin φ`.  Rotating by
`θ` adds `θ` to the angle:

$$
\large
x' = r \cos(\phi + \theta) = r (\cos\phi \cos\theta - \sin\phi \sin\theta)
   = x \cos\theta - y \sin\theta
$$

$$
\large
y' = r \sin(\phi + \theta) = r (\sin\phi \cos\theta + \cos\phi \sin\theta)
   = x \sin\theta + y \cos\theta
$$

which is exactly

$$
\large
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix},
\qquad
\begin{bmatrix} x' \\\\ y' \end{bmatrix} = R_\theta \begin{bmatrix} x \\\\ y \end{bmatrix}.
$$

Two properties worth internalizing because they will matter later:

- **Rotations preserve lengths and angles**: `det R_θ = cos²θ + sin²θ = 1`,
  and `R_θᵀ R_θ = I`.  This is what "rigid" means.
- **Rotations compose multiplicatively**: rotating by `θ` then by `φ` is
  rotating by `θ + φ`, which is the matrix product `R_φ R_θ`.  The product of
  two rotation matrices is a rotation matrix.
- **Order matters for scale.**  In the similarity `s·R·x`, the scale is
  applied *after* the rotation, but a uniform scale commutes with rotation
  (`s·R = R·s` as matrices?  Check: `(sR)ᵢⱼ = s·Rᵢⱼ = Rᵢⱼ·s` — yes, a scalar
  commutes with everything).  So the order scale-then-rotate vs rotate-then-
  scale genuinely does not matter here — which is a special property of
  *uniform* scale.  It is why we can even talk about "the" similarity
  unambiguously.

### 2.3 The full transform, and homogeneous coordinates

Putting rotation, scale and translation together:

$$
\large
x' = s\,R_\theta\,x + t,
\qquad
s > 0,\quad \theta \in (-\pi, \pi],\quad t = (t_x, t_y).
$$

The translation `t` is applied *after* scale and rotation.  Why this order?
Think of a drone that turns and advances: the image first rotates about the
origin by the yaw (and the zoom by the height change), and *then* the whole
rotated-and-scaled picture slides to its final place.  If you applied the
translation first, the rotation would swing the translation vector around —
changing its direction — which is not what a rigid body does.  The family of
transforms "first linear, then translate" is the standard.

**Homogeneous coordinates** are a bookkeeping trick that turns "linear part
plus translation" into a single matrix product.  Append a constant 1 to every
point and write:

$$
\large
\begin{bmatrix} x' \\\\ y' \\\\ 1 \end{bmatrix} = M \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix}, \qquad M =
\begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\\\ s\sin\theta & s\cos\theta & t_y \\\\ 0 & 0 & 1 \end{bmatrix}.
$$

Why bother?  Three reasons, all of which we use later:

1. **Composition becomes multiplication.**  Applying transform `M₁` then `M₂`
   is the single matrix `M₂·M₁`.  The `compose_similarity` helper in this repo
   is literally just a `3×3` matrix product, and `test_compose_matches_matrix_product`
   verifies the two agree.
2. **Uniform formulas.**  Fitting, warping, and rendering can all be written
   once against `3×3` matrices instead of special-casing "linear" and "plus
   translation".
3. **The identity is obvious**: the identity matrix means "no motion", i.e.
   `s = 1, θ = 0, t = (0,0)`.

**The inverse transform, derived.**  Because the whole pipeline inverts `M`
(every warp uses $`M^{-1}`$, §11; the EKF consumes increments, §24), the
inverse matrix is worth deriving once.  Write `M` in block form

$$
\large
M = \begin{bmatrix} A & t \\\\ 0 & 1 \end{bmatrix},
\qquad
A = s R_\theta = \begin{bmatrix} s\cos\theta & -s\sin\theta \\\\ s\sin\theta & s\cos\theta \end{bmatrix}.
$$

We want $`M^{-1}`$, i.e. a matrix satisfying $`M^{-1} (x', \, 1)^\top = (x, \, 1)^\top`$.
Solve $`x' = A x + t`$ for `x`:

$$
\large
x' = A x + t
\Longrightarrow
A x = x' - t
\Longrightarrow
x = A^{-1}(x' - t) = A^{-1} x' - A^{-1} t.
$$

In homogeneous coordinates, $`x = A^{-1}x' - A^{-1}t`$ becomes

$$
\large
M^{-1} = \begin{bmatrix} A^{-1} & -A^{-1} t \\\\ 0 & 1 \end{bmatrix}.
$$

Because $`A = s R_{one}theta`$ and rotations are orthogonal ($`R^{{-1}} = R^{{top}}`$), the inverse of the linear part is

$$
\large
A^{-1} = (s R_\theta)^{-1} = \frac{1}{s} R_\theta^{top}
= \frac{1}{s} \begin{bmatrix} \cos\theta & \sin\theta \\\\ -\sin\theta & \cos\theta \end{bmatrix},
$$

which says: *to undo scale-and-rotate, scale down by `1/s` and rotate the
other way* — exactly what undo should do.  The `0`-row/`1`-corner structure
of `M^{-1}` falls out of the algebra, not out of a guess.  This derivation
is the reason every `warp(A, M)` amounts to `grid_sample(A, M^{-1})` with no
special-casing of `t`.

*Implementation note.* All the algebra of this section lives in
`genml_kit/geometry/similarity.py`: `params_to_matrix` builds `M` from
`(log_s, theta, t)`, `params_from_matrix` does the reverse (§5), and
`compose_similarity` multiplies the matrices.  The test
`test_params_matrix_round_trip` checks that converting parameters → matrix →
parameters returns the original values.

### 2.4 Why "uniform" scale and "similarity" specifically

A *similarity* is the most general transform that maps every shape to a
**similar** shape — same angles, same relative proportions, scaled by one
number.  It has 4 degrees of freedom (DOF): 1 scale, 1 rotation, 2
translations.  For comparison:

| Family | DOF | What it can express | What it cannot |
|---|---|---|---|
| Translation | 2 | pure shift | rotation, scale |
| Rigid (rotation+translation) | 3 | shift + rotation | changing size |
| **Similarity** | **4** | shift + rotation + *uniform* scale | different scales on x vs y, shear, perspective |
| Affine | 6 | any linear map + translation | perspective foreshortening |
| Homography (projective) | 8 | anything a planar world can produce | — (but ill-posed from few points) |

A drone between two near frames: yaw → rotation, height change / approach →
uniform scale, ground drift → translation.  Exactly the similarity's DOF.
The *next* thing a moving rigid camera can do — perspective tilt because the
ground is not perpendicular to the optical axis — is outside the family and
becomes the "residual" of [Part 2](#part-2--where-the-data-comes-from-synthetic-pair-generation).

---

## 3. A fully-worked example: the 90-degree turn

This is the most important example in the whole document, because it pins
down, with integers, every sign convention we use — and because the same
example, encoded as a unit test, is what prevents those conventions from ever
silently flipping.

Take a tiny image and rotate it by 90° counter-clockwise.  Let the rotation
origin be the image center, and use a 2×2 image with corners at

$$
\large
(0,0) \quad (1,0) \quad (0,1) \quad (1,1)
$$

(small enough that every multiplication is an integer).  What does the rotation by π/2, whose matrix is
$`R_{\pi/2} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}`$, do to each corner?

$$
\large
\begin{bmatrix}0 & -1 \\\\ 1 & 0\end{bmatrix} \begin{bmatrix}1 \\\\ 0\end{bmatrix}
= \begin{bmatrix}0 \\\\ 1\end{bmatrix},
\qquad
\begin{bmatrix}0 & -1 \\\\ 1 & 0\end{bmatrix} \begin{bmatrix}0 \\\\ 1\end{bmatrix}
= \begin{bmatrix}-1 \\\\ 0\end{bmatrix}
$$

and of course `(0,0) → (0,0)`, `(1,1) → (−1,1)`.  So the point that was one
step to the *right* of the origin ends up one step *up*; the point one step
*down* ends up one step to the *left*.  That is, visually, exactly a
counter-clockwise quarter turn: **right → up → left → down**.

But remember the image frame from §2.1: `y` grows *down*.  "Up" in the image
is *negative* `y`.  A counter-clockwise rotation of the *scene* in an image
coordinate frame is a **positive** `θ` in our parameterization — but a
*negative* rotation in a math-textbook's y-up frame.  This is the classic
sign trap, and the remedy is:

> **We adopt the image frame `y`-down as the single source of truth.**
> "Counter-clockwise when viewed in the image frame" means the same thing the
> code means, the tests mean, and the drone's yaw means (a left turn).  The
> matrix `R_θ` above, with `θ = +π/2`, is the one used everywhere.

Concretely, in the test `test_quarter_turn_sign_convention`, the assertion is
recorded once: a quarter turn maps the reference corner as shown, and every
future consumer of `params_to_matrix` inherits that sign convention from this
one worked case.  If the drone turns 90° *left* (counter-clockwise in the
image frame), the front-end must report `θ ≈ +π/2`; a 90° *right* turn
reports `θ ≈ −π/2`.  What "left" and "right" mean in the *camera's* mounting
orientation is a calibration fact of the physical vehicle — but once the sign
is determined (by flying one hand-computed 90° turn and checking the sign), it
is locked in by this test and never re-derived.

**Worked with the second corner.**  Take the point `(1,1)` (bottom-right in
image coords).  $`R_{\pi/2}(1,1) = (-1, \, 1)`$.  In image coords that is one left,
one down — the bottom-right corner of the 2×2 image has moved to the
bottom-*left*.  Draw the 2×2 square, rotate it 90° CCW about its center, and
you will see exactly this: the corner that was rightmost becomes topmost, etc.
The numbers agree with the picture.

*Implementation note.* `tests/test_geometry_similarity.py::test_quarter_turn_sign_convention`
is the encoded version of this paragraph.  If you change a sign anywhere in
the algebra, this test breaks loudly — which is the entire point.

---

## 4. How we store the parameters (log scale, wrapped angle)

The math of §2 uses `(s, θ, t)`.  The code uses `(log s, θ, t)`.  Two of the
four numbers are stored differently, and each difference exists because it
makes the *learning problem* better behaved.

### 4.1 Scale in log space

We store `log s` (natural logarithm) and recover $`s = e^{log s}`$.

**Why?** Three independent reasons, all of which matter in a trained network:

1. **Scale must never be negative.**  $`s = e^{log s}`$ is positive for every
   real value of `log s`.  If a network regressed `s` directly, it could
   output `s ≤ 0` — a mirror image, a different transform family that is
   physically impossible for a rigid camera.  In log space, *any* real-number
   output is a valid positive scale.  The network literally cannot produce an
   invalid scale.
2. **Multiplications become additions.**  Two successive frames with scales
   `s₁` and `s₂` compose multiplicatively (`s_total = s₁·s₂`), but in log
   space: `log s_total = log s₁ + log s₂`.  A network that predicts "how much
   did the scale change" from frame to frame is predicting an *additive*
   increment — the native arithmetic of a linear regression head.  There is
   nothing to "learn to multiply".
3. **Symmetric errors.**  "10% too big" and "10% too small" are equally bad
   perception errors, but in linear scale `s` they are `+0.1` and `−0.09`
   (asymmetric).  In log space they are `±log(1.1)` — symmetric.  Any loss
   on `log s` automatically treats over- and under-estimation fairly.

### 4.2 Angle in radians, wrapped

We store `θ` in **radians**, and we always keep it in `(−π, π]`: that is, we
identify angles that differ by `2π`.  The wrap operation is

$$
\large
\mathrm{wrap}(\delta) = (\delta + \pi) \bmod 2\pi - \pi,
$$

so `wrap(3.9)` equals `3.9 − 2π ≈ −2.38` (in radians; `3.9` rad is more than
`π`).  In degrees for intuition: `wrap(370°) = 10°`, `wrap(−5°) = −5°`,
`wrap(359°) = −1°` (not `359°`).

**Why?**  A rotation by 370° is the same physical rotation as 10°.  If a
network predicted 370° while the ground truth was 10°, a *naive* loss
`|370 − 10|` would scream "360 degrees of error!" and backpropagate a giant,
wrong gradient — even though the prediction is perfect.  Everything
downstream (losses in §16, concordance checks, Kalman innovation in §25) must
operate on *wrapped angle differences*, or the `2π` boundary becomes a
discontinuity the network has to learn to jump across.  Wrapping removes the
discontinuity by construction.

**Check your understanding.**  What is `wrap(π + 0.1)` and why does it matter
for a network predicting angles near the boundary?  *Answer: `wrap(π+0.1) ≈
−(π−0.1)`.  The angle `π` (a half turn) and `−π` (also a half turn, the other
way around) are the same physical rotation; wrapping puts the prediction on
the same side of the cut as the target, so the loss is small where the physical
error is small.*

*Implementation note.* `genml_kit/geometry/similarity.py::wrap_angle` is the
wrapped subtraction used everywhere; `test_wrap_angle_branch_cut` checks the
boundary behavior.  §16 will show the wrapped difference inside the angle
loss.

### 4.3 Translation in pixels

`t = (t_x, t_y)` is stored in **pixels** of the *full-resolution* image.  One
convenience: pixels are what the metric of §6 uses, so no unit conversion is
ever needed between prediction and evaluation.  One caveat: networks operate
at reduced resolution (the encoder of §12 works at `1/8` scale), so the
translation a network predicts must be **rescaled by the resolution ratio** at
the network boundary.  In this codebase the corner-based formulation of §14
sidesteps most of this by working in a *normalized* corner coordinate space
and rescaling once, at the end — more on that in §14.

---

## 5. Reading the parameters back from a matrix

Given a `3×3` similarity matrix with linear part

$$
\large
A = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix},
$$

we want to recover `(s, θ, t)`.  From §2.3,

$$
\large
A = s \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix}
  = \begin{bmatrix} s\cos\theta & -s\sin\theta \\\\ s\sin\theta & s\cos\theta \end{bmatrix},
$$

so reading column by column:  the first column is `(a, c) = (s cos θ, s sin θ)`.
Its length is `s`:

$$
\large
s = \sqrt{a^2 + c^2},
\qquad
\theta = \mathrm{atan2}(c, a),
$$

because `atan2(sin, cos)` recovers the angle whose sine and cosine are
`c/s` and `a/s`.  The translation is simply the third column.

**The reflection trap.**  `atan2` only gives the right `θ` when `det A > 0`,
i.e. when `A` is a true rotation-scaling.  If `det A = ad − bc < 0`, the
linear part contains a *reflection* (a mirror), and `atan2(c, a)` returns an
angle that is off by a sign — because a reflected frame is not a rotation.
A rigid drone camera can never produce a mirror, so `det A < 0` is a *bug
signal*: it means the matrix was built wrong, or the estimate has collapsed.
The code checks the determinant and refuses to interpret a reflected matrix
as a similarity; `test_umeyama_rejects_reflection` pins that behavior.

---

## 6. The metric that judges everything: mean corner error

We will spend a lot of effort estimating `(s, θ, t)`.  How do we say one
estimate is better than another?  The answer this project uses, everywhere, is
the **mean corner error (MCE)**: take the four image corners, transform them
with the *estimate*, transform them with the *ground truth*, and average the
pixel distances between the two results.

$$
\large
\mathrm{MCE}(\hat{M}, M) = \frac{1}{4} \sum_{i=1}^{4}
\lVert \hat{M}\,u_i - M\,u_i \rVert_2,
$$

where `u_i` are the four corners.

Why corners, why the mean, why pixels?

- **Corners are the extreme points.**  Any error in `s` or `θ` grows with
  distance from the rotation center; the corners are farthest from the center,
  so they expose the *largest* possible effect of a scale/rotation error.
  A metric on corners therefore cannot be fooled by a transform that is right
  in the center but diverges at the edges.
- **All four parameters are folded into one number, with natural units.**
  A 0.5&nbsp;px error at the image border — depending on the transform —
  corresponds to roughly 0.5% in scale, 0.3° in rotation, or 0.5 px in
  translation.  The MCE blends them in exactly the way the downstream Kalman
  filter consumes them (§24), because it measures *the consequence* the
  consumer cares about: "if I trust this transform, where did the image
  content actually end up relative to where I think it did?"
- **It is interpretable.**  With pixels as units, "MCE = 1.3 px" means "corners
  are on average 1.3 pixels from where they should be" — you can see that on a
  screen.

**Why corners are enough: the only proof you need.**  Why does grading on the four
corners fully determine whether the *entire* transform is right?  The honest
answer has two parts.

*(a) Exactness: MCE = 0 if and only if the transforms agree everywhere.*  A
similarity has four degrees of freedom `(log s, θ, t_x, t_y)`.  A corner point
`u_i` mapped by the transform contributes two scalar equations (`u_i ↦ M u_i`
has an x- and a y-coordinate).  Four corners therefore give eight equations in
four unknowns — an overdetermined system, but one whose *minimal* content is
exact: if `M` and `M&#770;` agree on four non-degenerate corners, they agree
everywhere.  Proof: the difference `ΔM = M&#770; − M` satisfies `ΔM u_i = 0` for
`i = 1..4`.  With the four corners of a rectangle (non-degenerate), the only
similarity whose linear part kills all four vertices is the identity: `ΔM` maps two linearly independent vectors (a corner's horizontal and vertical
edges) to zero, so its linear part is the zero matrix, and then the translation
`Δt` must also vanish.  Hence MCE = 0 ⇒ `M = M&#770;` everywhere.  The metric is
not a sample of quality — it is an exact test of equality, made continuous by
the mean.

*(b) Continuity: a small MCE means a small error everywhere.*  For any pixel `x`
in the image, write the per-pixel displacement as `E(x) = M&#770;x − Mx = ΔA x + Δt`
with `ΔA` the difference of the linear parts.  The triangle inequality and the
definition of the operator norm give

$$
\large
\lVert E(x) \rVert_2 = \lVert \Delta A\,x + \Delta t \rVert_2
\;\le\; \lVert \Delta A \rVert_2 \, \lVert x \rVert_2 + \lVert \Delta t \rVert_2.
$$

The norm `x ↦ ‖x‖` is a convex function, and the image (a rectangle) is the convex
hull of its four corners, so `‖x‖ ≤ max_i ‖u_i‖` for every pixel.  Hence the worst
per-pixel error over the whole image satisfies

$$
\large
\max_{x} \lVert E(x) \rVert \;\le\; \lVert \Delta A \rVert_2 \, \max_i \lVert u_i \rVert + \lVert \Delta t \rVert_2 .
$$

Both terms on the right are controlled by the MCE.  The set of similarities with
`MCE ≤ m` is compact (the MCE is a continuous, coercive function of `(log s, θ, t)`
and `m` bounds it), and on that compact set both `‖ΔA‖` and `‖Δt‖` attain maxima;
therefore there is a constant `C` (depending only on image size, not on the
particular transform) with

$$
\large
\max_{x} \lVert E(x) \rVert \;\le\; C \cdot \mathrm{MCE}(\hat{M}, M) .
$$

In words: *a small mean corner error implies a small maximum error at every
pixel.*  The two parts together are exactly what a grading metric must be:
zero if and only if the answer is right everywhere, and continuous near zero.
This is special to similarities — an affine or homography has interior
degrees of freedom that can vanish at the four corners yet blow up in the
middle, so this corner-based guarantee would fail for them.  That is one more
reason the 4-DOF similarity family is the right model for a rigid oblique
camera.


*Implementation note.* `genml_kit/geometry/similarity.py::corner_residual` is
exactly this definition; `test_corner_residual_matches_manual` recomputes it
by brute force and checks equality.  §16 uses MCE as the *primary* supervised
loss; §9 uses the same corner idea to define a "residual" for ground truth.

---

# Part 2 — Where the data comes from: synthetic pair generation

Before we can train a network to *estimate* similarities (Part 3), we need
labeled training data: thousands of pairs `(frame A, frame B)` together with
the exact similarity that takes A to B.  Real flight footage is scarce, is
expensive to label, and its labels are noisy.  This project instead **renders
synthetic pairs**: it defines a virtual drone with a virtual oblique camera,
flies it over a procedurally-textured ground plane, renders both frames, and
*computes* the ground-truth similarity directly from the geometry — with no
human labeler in the loop.

Two facts make the labels exact rather than approximate:

1. The camera model and the motion are known to full precision (they were
   chosen by us), so the mapping from "world" to "pixels" is a *derivable*
   formula, not a measurement.
2. That formula is a **homography** (a projective transform), because the
   ground is modeled as a plane.  The ground-truth similarity is then defined
   as the *best similarity fit to that homography* — and the unavoidable gap
   between the two becomes the "irreducible residual", a number every dataset
   item carries because the network's *confidence* head is trained against it
   (§16, §24).

This part teaches the geometry behind those labels.  All of it lives in
`genml_kit/datasets/vo_pairs.py`.

## 7. The oblique pinhole camera

### 7.1 The pinhole projection model

The simplest true model of a camera is a **pinhole**: every light ray reaching
the sensor passes through a single point (the *center of projection*), so a
3-D world point `p` produces exactly one image point.  Two parameters describe
the camera:

- its **position** `c` (a 3-D world point);
- its **orientation** `R_cw` — a `3×3` rotation matrix whose *rows* are the
  camera's three axes expressed in world coordinates.  The convention "axes
  as rows, world→camera" means: to get from world coordinates to camera
  coordinates, multiply by `R_cw`.

A world point `p` is first expressed in camera coordinates,

$$
\large
\tilde{p}_{cam} = R_{cw}\,(p - c),
$$

and then projected to a pixel by the **intrinsic matrix** `K`:

$$
\large
u \sim K\,\tilde{p}_{cam},
\qquad
K = \begin{bmatrix} f_x & 0 & c_x \\\\ 0 & f_y & c_y \\\\ 0 & 0 & 1 \end{bmatrix},
$$

where `∼` means "up to a scale factor" (homogeneous coordinates): the actual
pixel is `u = (u_x/u_z, u_y/u_z)` with `u_z` the third component of
`K p̃_cam`.  The meaning of the parameters:

- `f_x, f_y` — **focal length in pixels** along each axis.  This is the
  physical focal length (mm) times the sensor density (pixels/mm), and it is
  the single number that connects "world angles" to "pixel separations".
  A camera with `f_x = 1000` on a `2000`-pixel-wide sensor sees a horizontal
  field of view of `2·atan(2000 / (2·1000)) = 90°`.  In general
  `FOV_h = 2 atan(W / (2 f_x))`.
- `c_x, c_y` — the **principal point**: the pixel where the optical axis
  lands (usually the image center for a well-built camera).


**Where the projection and focal length come from: a derivation.**  The pinhole
model is not an arbitrary formula — it is the geometry of *similar triangles*.
Place the pinhole at the origin with the optical axis along `+z` and the sensor
plane at `z = f` (a distance `f` behind the pinhole; for real lenses, `f` is
the focal length).  A world point `P = (X, Y, Z)` with `Z > 0` in front of the
camera emits a ray through the pinhole, which lands on the sensor at the point
where `z = f` intersects the line from `P` through the origin.

Parametrize that line: `(tX, tY, tZ)`.  It hits the sensor when `tZ = f`, i.e.
`t = f/Z`, so the sensor coordinate is

$$
\large
x = tX = f\,\frac{X}{Z},\qquad y = tY = f\,\frac{Y}{Z}.
$$

This pair `(x, y) = (fX/Z, fY/Z)` *is* the pinhole projection.  It says: the
image coordinate is the focal length times the *ratio* of the world coordinate
to the depth.  Three consequences that drive everything later:

- **Inversion.**  The `1/Z` makes near objects appear large and far objects
  small — the geometric origin of perspective and of the uniform-scale
  behavior a drone sees when it climbs (§1).
- **The focal length is the angle-to-pixels conversion factor.**  A world ray
  at angle `α` to the optical axis satisfies `tan α = X/Z`, so
  `x = f tan α`: pixels are the *tangent* of the viewing angle, scaled by
  `f`.  That is why `f` carries the units "pixels per radian at small
  angles".
- **Field of view follows.**  A sensor of width `W` (in pixels) subtends the
  angle `2α_max` where the edge `x = W/2` is hit:

$$
\large
\frac{W}{2} = f\,\tan\alpha_{\max}\;\Longrightarrow\; \alpha_{\max} = \arctan\!\left(\frac{W}{2f}\right),\qquad \mathrm{FOV}_h = 2\,\arctan\!\left(\frac{W}{2f}\right).
$$

For `f_x = 1000`, `W = 2000`: `FOV_h = 2 arctan(1) = 90°` — the worked
numbers in the bullet above, now derived rather than asserted.

The **intrinsic matrix** `K` packages this projection for both axes at once
(including the principal-point offset `c_x, c_y`), which is why the
homogeneous form of §8 reads `u ∼ K p̃`.

**Why does the ground plane matter here?**  A pinhole camera and a *plane*
world (our ground: `z = 0`) combine into a particularly simple map — which is
the subject of §8.  This simplification is the engine of the whole synthetic
generator.

### 7.2 The oblique mounting

The drone's camera is mounted pointing down at roughly 45° — *oblique*, not
nadir.  In the generator, the camera's look direction is characterized by two
angles:

- **pitch `φ`** below the horizontal (so `φ = 45°` looks halfway between the
  horizon and straight down; `φ = 90°` looks straight down = nadir);
- **yaw `ψ`** around the vertical.

A useful way to build the orientation `R_cw` is to construct its three rows
directly, as unit vectors:

1. the **forward** axis `f` — the look direction, computed from pitch and yaw
   (in the generator: `(cos ψ cos φ, sin ψ cos φ, −sin φ)`, i.e. mostly
   horizontal with a downward tilt);
2. the **right** axis `r = normalize(f × (0,0,1)ᵀ)` (the cross product of the
   look direction with world-"up" gives a horizontal vector perpendicular to
   the look — a "right" direction);
3. the **down** axis `d = f × r` (perpendicular to both; the third axis,
   pointing roughly down).

Stacked as rows into `R_cw`, these three unit vectors carry the full camera
attitude.  The important intuition: **`r` and `d` span the ground-plane
directions that the two image axes see, and `f` is what the optical axis
points along** — the structure that makes §8's factorization natural.

*Implementation note.* `genml_kit/datasets/vo_pairs.py::look_at_ground_h`
builds exactly this `R_cw` (pitch fixed at 45°, yaw drawn per sample) and
returns the homography `H` — the very object of §8.

## 8. The ground-to-image homography

Projecting a whole *plane* (the ground, `z = 0`) through a pinhole produces a
map that is not just "some function" — it is a **homography**: a `3×3`
invertible matrix acting on homogeneous coordinates, i.e. a *projective* (but
in general not affine, not similarity) transform.  Let us see why.

Camera coordinates of a ground point `p = (x, y, 0)` are

$$
\large
\tilde{p}_{cam} = R_{cw}\,p + t,
\qquad
t = -R_{cw}\,c,
$$

The rotation-into-camera is a linear function of `p`; since ground points have
`z = 0`, only the first two *columns* of `R_cw` ever touch `(x, y)`.  The map
"ground `(x, y)` → pixel" is therefore the composition of

- a linear map from `(x, y)`-ground to camera coordinates (2 columns of
  `R_cw` plus the fixed offset `t`), and
- the intrinsic projection `K`,

which is, in homogeneous form,

$$
\large
H = K \begin{bmatrix} r_x & d_x & t_x \\\\ r_y & d_y & t_y \\\\ f_x & f_y & t_z \end{bmatrix},
\qquad
t = -R_{cw}\,c,
$$

where `r` and `d` are precisely the "right" and "down" axes of §7.2: the
camera's orientation columns that pick out how ground displacements along the
two sideways directions turn into image motion.  Every ground point `(x, y)`
maps to a pixel `u ∼ H (x, y, 1)ᵀ`.

**Why a plane maps to a 3×3 matrix: the derivation.**  The claim "a pinhole
camera viewing a plane produces a homography" deserves a proof, because it is
the workhorse behind the whole generator.  Start from the pinhole projection of
§7 in homogeneous form and substitute the camera–ground relation.

A ground point `(x, y)` at height `z = 0` maps to camera coordinates by
§7’s rigid motion: `p_cam = R_cw (p − c)`.  The point is 3-D, but because the
world point always sits on the plane `z = 0`, the expression is *linear* in the
2-D ground coordinates: the `x`- and `y`-ground components enter through
columns 1 and 2 of `R_cw`, and the constant `−R_cw c` adds the offset.  So

$$
\large
p_{cam}(x, y) = C \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix},
\qquad
C = \begin{bmatrix} r_x & d_x & t_x \\\\ r_y & d_y & t_y \\\\ f_x & f_y & t_z \end{bmatrix},
$$

a 3×3 matrix `C` applied to the lifted ground point.  (The rows of `C` are the
transposed right/down axes of §7.2 plus the offset `t = −R_cw c` — check the
first row: `r_x x + d_x y + t_x = r · (x, y, 0) + t_x`, which is exactly the
x-coordinate of `R_cw (p − c)`.)

Now apply the intrinsic projection.  The pixel is homogeneous `u ∼ K p_cam`,
and the `∼` means "divide by the third coordinate afterwards".  *Here is the
key step:* the division by `p_cam_z` is already *absorbed* by the matrix
product if we work projectively, because

$$
\large
u = \frac{K\,C\, (x, y, 1)^\top}{\text{third coordinate of } KC(x,y,1)^\top}.
$$

The right-hand side is *exactly* the definition of the projective action of the
product `K C`: compute `KC (x,y,1)ᵀ`, get a 3-vector, divide by its last
entry, and read the first two as the pixel.  Letting `H = K C`, every ground
point maps to a pixel by

$$
\large
u \sim H \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix},
\qquad
H = K C,
$$

which is the *definition* of a homography: a 3×3 matrix acting on homogeneous
coordinates, modulo scale.

**Why 8 degrees of freedom (not 9).**  A 3×3 matrix has 9 entries, but two
matrices that differ by a global scale act identically (dividing by the third
coordinate cancels any overall factor).  So the homography has `9 − 1 = 8`
effective parameters — which is why four point correspondences (8 scalar
equations) generically determine it, and why fitting it from points is a
well-posed problem.  By contrast, a similarity has 4 DOF; the gap `8 − 4 = 4`
is precisely the foreshortening/perspective content that a similarity cannot
express — the seed of §10’s residual.

**Why is this the right object for us?**  Two reasons:

1. **It is exact, not approximate.**  As long as the world really is a plane,
   a pinhole camera's view of it is *exactly* a homography.  There is no
   lens-distortion fudge factor in the model.
2. **It is composable.**  Frame A is a view of the ground through camera A
   (`H_A`), frame B through camera B (`H_B`).  The image-to-image map from A
   to B is simply `H_B · H_A⁻¹` — the homography that takes pixel locations of
   A to pixel locations of B.  Composability is exactly what lets the
   generator place *one* textured ground tile and render both frames from two
   camera poses.

**The crucial consequence.**  `H_B · H_A⁻¹` is a homography, and a generic
homography is **not** a similarity: it can tilt, foreshorten, and shear text
as it moves.  But our front-end only outputs similarities (§2.4).  So there
is a built-in gap between "what the true image motion is" (homography) and
"what our output family can express" (similarity).  That gap is the topic of
§10 — and, in disguise, of the *entire* confidence mechanism.

## 9. Ground truth: fitting a similarity to a homography

Given a generated pair, what exactly is the ground-truth similarity?  Answer:
the similarity `(s, θ, t)` that best approximates the image-to-image
homography `H_B · H_A⁻¹`, in the MCE sense of §6 — *averaged pixel error of
the four image corners*.

The procedure (`homography_to_similarity` in `vo_pairs.py`):

1. Take the four corners `u_i` of frame A;
2. map them through the true image-to-image homography:
   `q_i = (H_B · H_A⁻¹) u_i` — these are the exact destinations of the corners;
3. fit the similarity that maps `u_i → q_i` as well as possible; because a
   similarity has 4 DOF and we have 8 constraints (4 corners × 2
   coordinates), the fit is *over-determined* — there is a closed-form
   least-squares answer, the **Umeyama** solve of §13;
4. report the residual `ρ = MCE between the fitted similarity's corners and
   the true homography's corners`.

Two facts about step 3 make it the *right* definition of ground truth:

- **It is exactly what the network is graded on.**  The training loss (§16)
  and the headline metric (§18) are the MCE against this same ground truth.
  The labels and the evaluation measure the identical quantity.
- **It is the best any similarity can do** for that pair, by construction.
  This is what makes the residual *irreducible*: a perfect network — one that
  recovered the ground-truth similarity exactly — would still have an MCE of
  `ρ` against the true homography-induced image motion.

*Implementation note.* `test_gt_residual_zero_for_pure_similarity_motion`
asserts that when the camera motion really is a similarity (see §10 for which
motions those are), the fitted similarity is exact and `ρ = 0`.  This is the
"canonical self-check" that the whole pipeline is consistent: generated GT,
`homography_to_similarity`, `umeyama_similarity`, and the network head all
agree on the same numbers when the world behaves.

## 10. The irreducible residual

### 10.1 When is a homography a similarity?

Not every camera motion produces an image map inside the similarity family.
The image-to-image homography `H_B · H_A⁻¹` is a genuine similarity exactly
when the camera's motion is one of:

- **rotation about the optical axis** (pure yaw when looking straight at a
  perpendicular plane) — the image spins;
- **translation perpendicular to the ground, with the optical axis
  perpendicular to the ground** (pure nadir approach) — the image zooms
  uniformly;
- **translation parallel to a ground plane, with the optical axis
  perpendicular to it** — the image shifts.

These are the *pure similarity motions*: yaw, zoom, pan, in the ideal
geometry.  The generator's 90° sign-convention example of §3 is literally one
of them.

### 10.2 What breaks it

As soon as the camera is *oblique* (our 45° mounting, by design!), the
optical axis is **not** perpendicular to the ground.  Then:

- a height change (or any motion with a component along the look axis) moves
  the camera closer to / farther from the ground points along the look
  direction, and because the view is oblique, near points and far points
  change apparent scale at different rates → **foreshortening**;
- content far from the image center lies along a more oblique viewing angle
  and diverges more.

The residual `ρ` grows with:

- **the off-nadir angle** (how far the optical axis tilts from perpendicular
  to the ground) — zero at nadir, maximal near the horizon;
- **the height delta** between frames (which drives the foreshortening rate
  when oblique);
- **the distance of content from the image center** (where the obliquity
  bites hardest).

### 10.3 The degenerate case

At exactly `φ = 0` (a *horizontal* camera), the ground plane passes through
the optical axis: every ground point is at an angle of 90° from the look
direction, the homography `H` collapses (`det H = 0`), and the residual is
undefined.  The generator simply never samples this regime — the flight
envelope keeps `φ` safely above 0.

### 10.4 Why we *emit* the residual

The residual is not a nuisance to be ignored: it is a **label**.  The network
is trained to predict it as the confidence head (§16), because the residual
is by definition "how much of this pair's apparent motion can never be
captured by the output family".  A network that sees a large predicted
residual knows to be humble; the downstream filter (§24) turns exactly this
number into measurement noise.  There would be no honest way to get a
calibrated confidence without this quantity — the residual is the ground
truth that makes the confidence *learnable* rather than decorative.

## 11. Warping: how frame B is actually rendered

Rendering `B = warp(A, M)` sounds trivial — "just move every pixel" — but the
details of *which mapping* and *which interpolation* are load-bearing: they
determine both what the network sees during training and which shortcuts it
can exploit.  The generator's choices are **recorded in the dataset
metadata** and kept identical between training and deployment.

### 11.1 Forward vs backward mapping

- **Forward mapping** iterates over *source* pixels and writes each to its
  destination $`M p`$.  Problem: several source pixels can land on the same
  destination, others on none — leaving **holes** — and sub-pixel destinations
  force messy decisions (which pixel wins?).  Aliasing and seams follow.
- **Backward mapping** iterates over *output* pixels and samples the source at
  $`M^{-1} p`$: *for each output pixel, where did its content come from?*
  Every output pixel gets exactly one value, no holes, no rasterizer needed,
  and each output sample is an independent query — which also makes the
  operation embarrassingly parallel and, crucially for us, *differentiable*
  per output pixel (§15).

All warps in this project — the generator's rendering and the network's
warped frame for the photometric loss — use the **backward map**.

### 11.2 Interpolation kernels

$`M^{-1} p`$ lands on a *continuous* position between pixels; the kernel decides
which source pixel values contribute to the output sample:

| Kernel | What it does | Where it belongs |
|---|---|---|
| Nearest | rounds to the closest source pixel; returns its value | fastest; but *quantizes* sub-pixel motion into whole-pixel steps — the rendered ground truth would have visible stair-step artifacts that a smooth regressor must then try to "unlearn". |
| **Bilinear** | weighted average of the 4 surrounding pixels, weights ∝ area overlap | **the default**: mild low-pass blur, which is literally what a real resampled photo looks like; smooth ground truth for a smooth regressor. |
| Bicubic | `4×4 = 16` neighbors with a cubic weighting | slightly sharper edges than bilinear, at higher cost; less commonly the default in generative pipelines. |

The right frame for a training generator is the one that (a) makes the ground
truth as *typical* as possible (what a real camera's resampling produces) and
(b) does not inject quantization that the network cannot model.  Bilinear
satisfies both; that is the default here.

### 11.3 Padding policy

Sampling $`M^{-1} p`$ near the image border asks for pixels outside the image.
The padding policy — `reflect`, `replicate`, `zeros`, `border` — decides what
those samples return.  This is not a cosmetic detail:

- **zero padding** creates a hard artificial border that the network can learn
  to *keypoint off of* — a shortcut that does not exist in real data (or
  worse, that overfits the generator);
- the **photometric loss of §17 subtracts a mask of exactly the non-padded
  overlap**, so the padding policy and the loss mask must agree or the mask
  is silently wrong.

### 11.4 The `align_corners` convention (a classic half-pixel bug)

Normalized sampling coordinates are not the same across tools.  PyTorch's
`grid_sample` with `align_corners=True` maps **pixel centers** to integer
coordinates:

$$
\large
x_n = \frac{2x}{W - 1} - 1,
$$

so pixel `0` sits at `x_n = −1` and pixel `W−1` at `x_n = +1`.  With
`align_corners=False`, the mapping is

$$
\large
x_n = \frac{2x}{W} - 1,
$$

which aligns the pixel *edges* — pixel `0` samples from `−1 + 1/W`, and the
outermost samples land *outside* the data range.  Mixing the two conventions
between the generator and the warper is a silent **0.5-pixel translation
error** — invisible by eye, fatal at the precision this system targets.

The policy, therefore, is a single line in the design: **fix one convention
per dataset, record it in the metadata, and unit-test the ground-truth
round-trip** — the 90° worked example of §3 is the canonical such test.  When
the GT `MCE` of a fully-correct pipeline is 0.5 px *by construction* because
of a convention mismatch, you find out in CI, not in flight.

*Implementation note.* `test_warp_identity_round_trip` (identity transform →
the image comes back exactly) and `test_warp_similarity_matches_point_map`
(sampling the warped image at $`M p`$ reproduces the source content at `p`, up
to the interpolation of two bilinear stages) pin down both the backward-map
semantics and the convention choice.

---

# Part 3 — The learned estimator

Part 3 answers the core question: *given frames A and B, how does a neural
network produce `(s, θ, t)`?*  The design has four pieces, each solving a
distinct sub-problem:

1. **§12 — learn where things are.**  A shared convolutional encoder turns
   each frame into a stack of feature maps; correlation between the two
   stacks, at every small displacement, builds a *cost volume* saying "how
   likely is it that content at (y,x) in A moved to (y+δy, x+δx) in B?".
2. **§13 — extract a transform from matches.**  Given a set of
   corresponding points (here: the image corners and their predicted
   destinations), the **Umeyama** algorithm computes the exact least-squares
   similarity in closed form.
3. **§14 — what the network actually regresses.**  Instead of guessing
   `(log s, θ, t)` directly (a non-linearly-constrained target), the head
   predicts *corner offsets*, and the Umeyama solve converts them into a
   guaranteed-valid similarity.
4. **§15 — a differentiable warp.**  For the *photometric* auxiliary loss
   (§17), the network must be able to warp frame B by its *own* predicted
   transform and compare pixels — which requires the warp itself to be
   differentiable.  `grid_sample` provides exactly that.

## 12. The correlation cost volume

### 12.1 From pixels to features

Do not correlate raw pixel intensities.  A raw pixel value says almost
nothing about "where is the same content now?" — lighting, exposure, and
camera noise change intensities between frames.  Instead, a **shared
convolutional encoder** (the same network applied to both frames, *Siamese*
style) maps each frame to a stack of feature maps:

$$
\large
F_a, F_b \in \mathbb{R}^{C \times h \times w}
$$

where `h = H/8`, `w = W/8` (the encoder stride is 8 — the default `cost_scale`
in this codebase) and `C` is the feature width.  Each of the `C` channels at
position `(y, x)` is a learned "signature" of the local image content at that
grid cell: a high-dimensional descriptor encoding edges, textures, and
corners in a way that is *invariant* to minor photometric changes and
reasonably robust to small geometric ones.

Two properties of this representation matter:

- **It is trained for the task.**  The features are not hand-crafted; the
  encoder is trained end-to-end (§16) so that the signatures are exactly what
  the cost-volume matching needs them to be.
- **It is *shared*.**  The same weights produce `F_a` and `F_b`, so the
  signatures are directly comparable — there is no "calibration" between the
  two frames.

### 12.2 The score

The core matching primitive is the **inner product** (dot product) between a
feature vector at a location in `F_a` and a feature vector at a *displaced*
location in `F_b`:

$$
\large
\mathrm{score}(y, x; \delta_y, \delta_x) = \sum_{c=1}^{C} F_a[c, y, x]\; F_b[c, y + \delta_y, x + \delta_x].
$$

Why an inner product?  For *unit*-length vectors, the inner product is the
cosine of the angle between them: `1` if identical, `0` if orthogonal,
`−1` if opposite.  So the score answers exactly "how similar is the content
here to the content there?" — high where the two feature vectors point the
same way in feature space.

The displacements run over a bounded window:

$$
\large
|\delta_y|, |\delta_x| \le r,
$$

with `r` the configured radius (default `6` at `1/8` resolution).  So we ask,
for every location, at most `(2·6+1)² = 169` displacement questions.  The
whole stack of scores is the **cost volume**:

$$
\large
\mathrm{cost\_volume} \in \mathbb{R}^{(2r+1)^2 \times h \times w},
$$

with channel ordering "dy-major": displacement `(δy, δx)` sits at channel

$$
\large
k = (\delta_y + r)\,(2r + 1) + (\delta_x + r).
$$

### 12.3 Why these design numbers?

- **Why correlate features, not pixels?**  Discussed in §12.1 — robustness.
- **Why a *bounded* displacement window?**  The drone's frame-to-frame motion
  is small at `1/8` resolution: a displacement of a few feature pixels covers
  tens of image pixels, which is the realistic envelope.  A full (global)
  correlation volume would be quadratic in resolution and mostly wasted
  compute on displacements that never occur.
- **Why 1/8 resolution?**  Three-way trade-off.  *Cost*: the volume is
  `(2r+1)² · h · w`; halving the resolution quarters the volume.  *Precision*:
  the head must recover sub-pixel accuracy from a `1/8`-scale grid — which is
  why §13/§14 fit a *continuous* similarity to the discrete grid rather than
  simply "argmax the volume".
- **Why dy-major ordering?**  It is a pure bookkeeping convention, but a fixed
  one: every consumer of the volume (the head, the visualizers, the tests)
  must agree on "where does displacement `(δy, δx)` live?" or features would
  be silently swapped.  The code documents it once and `test_corr_volume_layout`
  (if present) pins it.

### 12.4 From volume to transform

The volume tells the network *where content moved*.  It does not, by itself,
yield `(s, θ, t)`.  The bridge is a small **head** network that reads the
volume (plus the frame-A features — the "what is here to match?" signal) and
produces the network's output.  Which output?  Not the parameters directly —
that is the topic of §13–§14.

## 13. The Umeyama closed-form fit

Given `N` corresponding points `p_i` (in frame A) and `q_i` (where they
landed in frame B), the least-squares similarity is the `(s, R, t)`
minimizing

$$
\large
\sum_{i=1}^{N} \lVert q_i - (s R\, p_i + t) \rVert^2.
$$

This is **Umeyama's problem**, and it has a closed-form solution — no
iterative optimization, no local minima.  The algorithm (batched in
`genml_kit.geometry.similarity::umeyama_similarity`):

**Step 1 — center the points.**  Compute the centroids
`μ_p = mean(p)`, `μ_q = mean(q)` and the centered sets `p̂_i = p_i − μ_p`,
`q̂_i = q_i − μ_q`.  Centering decouples `t` from `(s, R)` — the optimal
translation is whatever remains after the rotation-scaling is applied.

**Step 2 — the covariance and its SVD.**  Form the `2×2` cross-covariance

$$
\large
H = \sum_i \hat{q}_i\, \hat{p}_i^{\top},
$$

and take its singular value decomposition `H = U Σ Vᵀ`.  The SVD of a `2×2`
matrix is cheap, exact, and numerically stable — and it is the heart of the
whole solve.

**Step 3 — rotation (with the reflection guard).**  The optimal rotation is

$$
\large
R = U \begin{bmatrix} 1 & 0 \\\\ 0 & d \end{bmatrix} V^{\top},
\qquad
d = \det(U V^{\top}),
$$

where `d = ±1` corrects for the fact that a covariance matrix's SVD alone
does not pin down whether the best fit is a rotation or a *reflection*.
If `d = −1` the classical SVD answer would silently produce a mirrored fit;
the guard flips the sign to keep `R` a genuine rotation.  We only ever want
rotations — a drone cannot mirror its own view — so the guard is not a
cosmetic detail but a correctness requirement.  This is the same `det < 0`
reflection trap seen in §5, surfacing here as a *protection* rather than a
detection.

**Step 4 — scale and translation.**  With the rotation fixed, the optimal
uniform scale and translation are

$$
\large
s = \frac{\sigma_1 + d\,\sigma_2}{\sum_i w_i \lVert \hat{p}_i \rVert^2},
\qquad
t = \mu_q - s\,R\, \mu_p,
$$

where `σ₁, σ₂` are the singular values of `H` and `d` is the same
reflection guard.  The scale is the ratio of "how much the points spread in
the target" to "how much they spread in the source", and the translation
re-centers the rotated-scaled source cloud onto the target centroid.

**Derivation: why these four steps solve the problem.**  The algorithm above is
compact, which is good for code and opaque for learning.  Here is where every
step comes from, starting from the objective and nothing else.

**Step 1 (derived).**  Fix `(s, R)` and minimize over `t` alone.  The objective
is

$$
\large
E(s, R, t) = \sum_i \lVert q_i - s R\, p_i - t \rVert^2.
$$

The gradient with respect to `t` is

$$
\large
\nabla_t E = -2 \sum_i (q_i - s R\, p_i - t),
$$

and setting it to zero gives exactly

$$
\large
\sum_i q_i - s R \sum_i p_i - N t = 0
\;\Longrightarrow\; t = \mu_q - s R\, \mu_p,
$$

which is Step 4's translation formula *before* the rotation is even known.
Substituting `t` back cancels the linear terms: with `p̂_i, q̂_i` the centered
points, the objective becomes

$$
\large
E = \sum_i \lVert \hat{q}_i - s R\, \hat{p}_i \rVert^2.
$$

This is the real content of "centering decouples `t`": the translation has been
*eliminated by substitution*, leaving only `(s, R)`.

**Step 2/3 (derived).**  Expand the centered objective:

$$
\large
E = \sum_i \lVert \hat{q}_i \rVert^2
+ s^2 \sum_i \lVert \hat{p}_i \rVert^2
- 2s \sum_i \hat{q}_i^{\top} R\, \hat{p}_i .
$$

The first two sums are *constants* in `R`; only the cross term depends on the
rotation.  Because `q̂_iᵀ R p̂_i` is a scalar, it equals its own trace, and the
trace is cyclic:

$$
\large
\sum_i \hat{q}_i^{\top} R\, \hat{p}_i
= \mathrm{tr}\Bigl(\sum_i \hat{q}_i \hat{p}_i^{\top} R\Bigr)
= \mathrm{tr}(H R),
$$

with `H = \sum_i \hat{q}_i \hat{p}_i^{\top}` the cross-covariance of Step 2.
So minimizing `E` means *maximizing `tr(HR)` over orthogonal `R`*.  Now write
`H = U \Sigma V^{\top}` (its SVD):

$$
\large
\mathrm{tr}(H R) = \mathrm{tr}(U \Sigma V^{\top} R)
= \mathrm{tr}(\Sigma \, V^{\top} R U)
= \sigma_1 [V^{\top} R U]_{11} + \sigma_2 [V^{\top} R U]_{22}.
$$

The matrix `W = V^{\top} R U` is orthogonal (product of orthogonals), so its
diagonal entries satisfy `|W_ii| \le 1`.  With `\sigma_1, \sigma_2 \ge 0`, the
sum is maximized by picking `W = I` — the identity — which forces
`V^{\top} R U = I`, i.e.

$$
\large
R = U V^{\top}.
$$

That is the whole derivation of Step 3's rotation, and it is where the
**reflection guard** enters: `U V^{\top}` is a rotation only if
`det(UV^{\top}) = +1`.  If the determinant comes out `−1`, then `W = I` is not
an orthogonal matrix with determinant +1 (it would be a reflection); the best
*rotation* is `W = \mathrm{diag}(1, -1)`, giving
`R = U \,\mathrm{diag}(1, d)\, V^{\top}` with `d = det(UV^{\top})` — exactly
Step 3.  (Pedagogical bonus: the proof never used the *values* of the singular
values except their non-negativity, which is why the same argument works in
any dimension.)

**Step 4 (derived).**  With `R` fixed, the objective is a quadratic in `s`:

$$
\large
E(s) = \sum_i \lVert \hat{q}_i \rVert^2
+ s^2 \sum_i \lVert \hat{p}_i \rVert^2 - 2s \, \mathrm{tr}(H R).
$$

Differentiate and set to zero:

$$
\large
0 = 2s \sum_i \lVert \hat{p}_i \rVert^2 - 2\, \mathrm{tr}(H R)
\;\Longrightarrow\; s = \frac{\mathrm{tr}(H R)}{\sum_i \lVert \hat{p}_i \rVert^2}.
$$

With `R = U\,\mathrm{diag}(1, d)\,V^{\top}`, the trace is
`tr(HR) = tr(U\Sigma V^{\top} U\,\mathrm{diag}(1,d)\,V^{\top}) = \sigma_1 + d\,\sigma_2`,
so the numerator is exactly `\sigma_1 + d\,\sigma_2` — Step 4's scale, now
derived rather than stated.  (The `w_i` weights appear in the actual
implementation for the generalized weighted version; the unweighted derivation
is the `w_i = 1` case.)

Every formula in the four steps has now been *derived* from the least-squares
objective: centering eliminates `t`, the trace trick reduces `R` to an
orthogonal maximization solved by `W = I`, and one derivative of a quadratic
gives `s`.  Nothing was pulled from a hat.

**Why does this matter for a neural network?**  Two reasons:

1. **It is exact and closed-form.**  No iterative loop in the middle of the
   network — training is fast and there are no convergence pathologies.
2. **It is differentiable.**  Gradients flow through the (batched) SVD, so
   `umeyama_similarity` can sit *inside* the training graph: the error signal
   on the final similarity back-propagates through the SVD into the corner
   offsets, and from there into the encoder.  `test_umeyama_gradients_flow`
   checks that a forward call under `torch.autograd` actually carries
   gradients.

*Implementation note.* `umeyama_similarity` is implemented in
`genml_kit/geometry/similarity.py`; `test_umeyama_recovers_exact_transform`
verifies the closed form recovers a planted similarity, and
`test_umeyama_matches_brute_force_least_squares` compares it against an
independent dense least-squares solve.

## 14. Why we regress corners, not parameters

The final head could regress `(log s, θ, t)` directly — a 4-vector — and be
done.  It does **not**, and the reason is worth understanding because it is
a recurring pattern in geometric deep learning: *some outputs are easier to
regress than others, and the difference is about geometry, not network
capacity.*

The network instead predicts a **delta for each of the four image corners**:

$$
\large
\Delta_i = \text{(offset of corner } u_i \text{ from where it should land)},
\qquad i = 1, 2, 3, 4,
$$

then builds `q_i = u_i + Δ_i` and hands the *four point correspondences*
`(u_i, q_i)` to the Umeyama solve of §13, producing a **guaranteed-valid**
similarity.

Why is this dramatically better than regressing parameters?

1. **Validity for free.**  Any four deltas produce *a* similarity (via
   Umeyama) with `s > 0` and a genuine rotation.  A direct 4-vector regression
   must learn — from data, imperfectly — not to emit `s ≤ 0` or `θ` outside
   `(−π, π]`.  Here validity is guaranteed *by construction*, for every
   possible input.
2. **Interpretable units.**  Corner deltas are pixels — the same unit the
   metric uses (§6) and the same unit the incremental EKF consumes (§24).
   Learning "move these corners by this many pixels" is a well-scaled,
   well-conditioned regression.  Learning `log s ∈ [−1.2, 2.0]`, `θ ∈ [−π, π]`
   (a quasi-circular target), and `t ∈ [−50, 50]` is three different scales of
   problem glued together, with a *discontinuity* in the angle at the wrap
   boundary (§4.2).  Robust uniform regression of that 4-vector is harder than
   it looks.
3. **The wrap discontinuity disappears.**  Because the network never predicts
   an angle — it predicts pixel offsets, and the angle only appears later,
   inside the closed-form solve — there is no `2π` boundary to learn to jump.
   The discontinuity is *not a loss function problem*; it has been moved out
   of the network's output space entirely.
4. **Ill-conditioning is the network's friend, turned around.**  Directly
   predicting scale/rotation near the identity is ill-conditioned (a small
   change in `(s, θ)` — especially near `s≈1, θ≈0` — is a tiny change in
   corners).  Regressing corners makes the representation *exactly as
   well-conditioned as the measurement is*: the thing the network predicts is
   the thing the data actually tells it.

**The conditioning argument, made precise.**  Claim 4 ("ill-conditioning is
turned around") deserves the linear algebra that backs it, because it is the
deepest of the four.  Let `f: (log s, θ, t) ↦ (corners)` be the map from
parameters to the four image corners they produce.  The *condition number* of
this map at a point is, roughly, how much the corners' output error
amplifies back into parameter-error sensitivity: if the Jacobian
`J = ∂f/∂(log s, θ, t)` has a small singular value, then a unit change in the
corresponding parameter direction produces almost *no* change in the corners —
so the corners simply do not *contain* the information to estimate that
direction reliably.

Compute the Jacobian's singular values near the identity
(`log s = 0, θ = 0, t = 0`).  For a corner displaced by `u` from the center,
the corner position as a function of the parameters is

$$
\large
f(u) = e^{\log s} R_\theta u + t
\;\approx\; (1 + \log s)(I + \theta G)\,u + t
\;\approx\; u + (\log s)\, u + \theta G u + t ,
$$

where `G` is the `2×2` rotation-generator matrix.  The three parameter
directions therefore act on the corners with:

$$
\large
\frac{\partial f}{\partial \log s} = u,
\qquad
\frac{\partial f}{\partial \theta} = G u,
\qquad
\frac{\partial f}{\partial t} = I .
$$

The first two columns *grow linearly with the corner's distance from the
center* `‖u‖`, while the translation column is `I` (unit).  For corners far
from the center, `‖u‖` is of order the image half-size — so the scale/rotation
columns are *large* compared with the translation column.  The singular
values of `J_f` are therefore spread: some are `O(‖u‖)` and some are `O(1)`,
and the **condition number** `κ = σ_max / σ_min` is `O(‖u‖)` — i.e. *the
parameter-to-corner map is as ill-conditioned as the image is large*.

Now the two design choices in sequence:

1. **Regressing corners directly** means predicting `f`'s *output* — where the
   network operates on the well-scaled, unit-level pixel deltas `Δ_i`
   (§14's claim 2).  The data carries the information in native units.
2. **The Umeyama solve** then inverts `f` exactly and *differentiably*.  The
   potentially ill-conditioned inversion `f^{-1}` is done by the closed-form
   SVD (whose condition number is exactly what it is — but now *geometry, not
   learning*, is responsible for it).

So the network never has to *learn* the ill-conditioned inverse map; it learns
the well-scaled forward output, and the closed-form solve handles the rest.
That is the precise content of "make the representation exactly as
well-conditioned as the measurement is."

The reference corners are *normalized* to a fixed canonical box (the four
corners of the feature grid, scaled to a `[-1, 1]`-style range) and `Δ_i` are
predicted in that same normalized space — so the same head works regardless
of image size; the final pixel-space similarity is recovered by rescaling
once, at the end.

*Implementation note.* In `genml_kit/models/vo/vo_similar.py`, the head is a
two-layer MLP `corner_mlp` whose final linear layer maps the pooled feature
vector (width = the encoder's final stage width, 128 for the default
`npu-small` profile) to a flat vector of 10 values: the first 8 are corner
deltas, the last 2 the confidence pair.  The head produces
`deltas = corner_mlp(pooled)[:, :8].view(-1, 4, 2)`, and
`params = umeyama_similarity(src, src + deltas)` — exactly the
correspondence-then-solve of this section.  The confidence head
`conf = corner_mlp(pooled)[:, 8:10]` shares the MLP but reads off the last two
outputs, trained against the residual (§16, §24).

## 15. Differentiable warping with grid_sample

§17 adds a loss that compares *pixels*: "frame B, warped by the predicted
transform, should look like frame A."  Evaluating that loss requires warping
frame B by the *predicted* transform — which means the warp must be
**differentiable with respect to the prediction**, or no gradient can flow
from the photometric loss back into the network.

The backward-map machinery of §11.1 makes this natural.  `F.grid_sample`
takes a **sampling grid** `G` (one `(x_n, y_n)` per output pixel, in
normalized coordinates) and, for each output pixel, samples the source image
at `G` using bilinear interpolation:

- if `G` came from the *predicted* transform `M⁻¹`, then
  `warped_B = grid_sample(B, G(M))` is differentiable with respect to
  `M`'s parameters — the chain rule flows from the output pixels through the
  interpolation weights into the normalized coordinates and then into
  `(s, θ, t)`.
- the invariant behind the tests: sampling the warped image at $`M p`$
  reproduces the source content at `p`, up to the interpolation of two
  bilinear stages (`test_warp_similarity_matches_point_map`).

The convention specifics that make the warp *correct*:

- pixel centers sit at integer coordinates (the `align_corners=True` choice of
  §11.4);
- normalized coordinates are `x_n = 2x / (W−1) − 1` — again §11.4;
- the grid is generated backward (output pixel → source position), so every
  output pixel is well-defined and no forward-mapping holes appear (§11.1).

*Implementation note.* The photometric branch of the training loss
(`genml_kit/training/vo/train_vo.py`) computes `warped = grid_sample(...)`
with the network's predicted similarity and applies it to frame B; the masked
zNCC of §17 then compares it with frame A over the valid overlap.

---

# Part 4 — Losses and metrics

Now that the estimator's architecture is clear, Part 4 defines the numbers
that drive training and evaluation.  The philosophy: **every term in the loss
exists to teach the network one specific thing**, and every metric exists to
measure one specific consequence.  Nothing is decorative.

## 16. The supervised loss, term by term

The primary loss supervises the network *directly against the ground-truth
similarity* (the fitted one from §9):

$$
\large
\mathcal{L}_{\mathrm{sup}} = \mathcal{L}_{\mathrm{mce}} + \lambda_s \lVert \log \hat{s} - \log s \rVert_1 + \lambda_\theta \bigl\lvert \mathrm{wrap}(\hat{\theta} - \theta) \bigr\rvert + \lambda_c\, \mathrm{SmoothL1}(\hat{r}, \rho)
$$

where the hatted quantities are the network's predictions.  Each term:

### 16.1 `L_mce` — the headline term

`L_mce` is the mean corner error of §6: transform the four corners with the
prediction, transform them with the truth, average the pixel distances.  It
is the primary term for a reason: it measures *exactly* the quantity the
system is graded on and the EKF consumes

- MCE is in pixels, so it is directly interpretable;
- it folds all four parameters into one number with the *weighting the
  consumer wants* (border corners amplify scale/rotation errors precisely as
  they will be seen downstream);
- it is computed via `corner_residual`, so it is automatically consistent
  with the evaluation metric — training and grading speak the same language.

### 16.2 `λ_s · |log ŝ − log s|₁` — scale in log space

Why L1?  Because L1 (mean absolute error) is robust to outliers and does not
over-penalize occasional large scale errors the way L2 would.  Why *log*
space?  §4.1: symmetric relative errors, additive composition, and no `s ≤ 0`
ever.  A "10% too big" error has the same absolute value in log space as a
"10% too small" error — L1 on `log s` treats them identically, which matches
how the physical error is perceived.

**Why L1 is robust, derived.**  The claim "L1 does not over-penalize outliers
the way L2 does" is quantitative, and the quantity is the *influence* of one
large error on the gradient.  Let the true residual be `e = log ŝ − log s`
and consider the contribution of a *single* sample to the total loss (the
`λ_s` factor is a constant and drops out):

$$
\large
\ell_2(e) = e^2,
\qquad
\ell_1(e) = |e| .
$$

The gradient magnitudes with respect to `e` are

$$
\large
\left| \frac{d}{de} \ell_2 \right| = 2 |e|,
\qquad
\left| \frac{d}{de} \ell_1 \right| = 1 .
$$

As `|e|` grows, the L2 gradient grows *linearly* — an outlier ten times larger
than a typical error contributes ten times the gradient push, and a network
will distort the whole estimate to shrink that one outlier.  The L1 gradient
is *bounded* by 1 for every sample, large or small: an outlier can never
dominate the update more than an ordinary sample.  That boundedness is the
mathematical content of "robust".  (At `e = 0`, L1 is not differentiable in
the classical sense, but its *subgradient* — any number in `[−1, 1]` — makes
the update well-defined; this is what the code's `l1_loss` uses.)

**Why SmoothL1 blends them.**  L1's constant gradient is bad near zero (it
never decays, causing slow convergence on small residuals), so the practical
compromise — SmoothL1 — is L2 for `|e| ≤ 1` and L1 for `|e| > 1`:

$$
\large
\mathrm{SmoothL1}(e) =
\begin{cases} \tfrac12 e^2 & |e| \le 1 \\\\ |e| - \tfrac12 & |e| > 1 \end{cases}.
$$

Its gradient is `e` for `|e| ≤ 1` and `±1` beyond — continuous at the
crossover, bounded everywhere: smooth like L2 for small errors, robust like L1
for outliers.  This is the derivation behind §16.4's choice of SmoothL1 for
the confidence residual.

### 16.3 `λ_θ · |wrap(θ̂ − θ)|` — the wrapped angle loss

The angle error is wrapped (§4.2) **before** taking the absolute value, so a
prediction of `θ̂ = 370°` against a truth of `θ = 10°` contributes `|wrap(10° − 370°)|`?  Careful with the order: `wrap(θ̂ − θ)`, i.e. `wrap(370° − 10°) = wrap(360°) = 0`.  The two angles are the same physical rotation, so the loss is (correctly) zero — no `2π`-boundary spike, no gradient misdirection.  This single wrap makes the angle loss everywhere-continuous, which is exactly what a gradient-based trainer needs.

**Why "wrapped" buys differentiability: the driving derivation.**  The claim
that wrapping removes the `2π` discontinuity deserves proof.  Define the
unwrapped loss `L_raw(d) = |d|` on the raw angular difference `d = θ̂ − θ`.
As `d` crosses `π` (the two angles are now the same physical rotation, e.g.
`185°` vs `−175°`), the *physical* error is tiny but the raw loss jumps from
`π` down to near 0 — the loss function is discontinuous there, and its
gradient is a delta: backpropagation would push the network *huge*, wrong
updates at exactly the boundary.

Wrapping replaces the input by `d' = wrap(d) = (d + π) mod 2π − π`, i.e.
it folds the difference into `(−π, π]`, and the loss becomes
`L_wrap(d) = |wrap(d)|`.  Away from the fold, `wrap` is a pure translation
`d ↦ d` (or `d − 2π`), so the derivative is the same `±1` as before; the
only special point is `d = ±π`, where the physical error is *maximal* `π` and
the loss is genuinely maximal too — there is no jump to create a delta.  The
wrapped loss is continuous everywhere and its subgradient is bounded by 1 at
every point:

$$
\large
\frac{d}{dd}\, \mathrm{wrap}(d) = 1 \ \text{(a.e. in the interior)},
\qquad
L_\mathrm{wrap}\ \text{is everywhere-continuous with} \ |\partial L_\mathrm{wrap}| \le 1 .
$$

So "wrap first, then absolute value" is not a heuristic: it is the operation
that converts a *discontinuous* loss into one whose gradient never explodes
and never points the wrong way across the boundary — exactly what a
gradient-based trainer requires.

### 16.4 `λ_c · SmoothL1(ρ̂, ρ)` — the confidence term

The network also predicts a residual `ρ̂` (the confidence head of §14), and
this term supervises it against the *actual* residual `ρ` of the pair (§9,
§10).  Why does the confidence get its own supervised term?

- Because the whole point of confidence is to be **calibrated**: the number
  must mean "how much of this pair can no similarity explain?", and the only
  way to make that number meaningful is to train it against the true
  residual.  A confidence that is not supervised against `ρ` would be
  decorative — it could be anything.
- `SmoothL1` is chosen as a robust regression loss: it behaves like L2 near
  zero (smooth gradient) and like L1 far from zero (robust to outlier
  residuals), which fits a quantity that is mostly small but occasionally
  large.

### 16.5 The weights `λ_s, λ_θ, λ_c`

The `λ`s are hyperparameters (set in `_LossCfg` / `args.vo_loss_cfg`, with
defaults `w_log_s = 1.0`, `w_theta = 1.0`, `w_conf = 0.5`, `w_photo = 0.1`).
They balance the terms *in their own units*: log-units, wrapped-radians,
pixels, and residual-pixels are not comparable, so the weights are the
relative *priorities*.  The typical setup keeps `w_log_s` and `w_theta` at 1
(the two geometry terms get equal say), `w_conf` lower (~0.5) because the
confidence is a secondary output, and `w_photo` lowest (0.1) because it is an
auxiliary (next section).

**Check your understanding.**  Why is there no `λ_t` for the translation?
*Answer: translation is not a separate term — it is implicitly part of the
MCE, which includes all four parameters.  A dedicated `|Δt|` term would
double-count translation and add a unit-scaling choice.  The MCE + log-scale +
wrapped-angle split exists only because scale and angle benefit from their
special parametrizations; translation needs nothing extra.*

## 17. The staged photometric auxiliary

Once the supervised loss has converged, a *photometric* term is staged in:

$$
\large
\mathcal{L}_{\mathrm{photo}} = 1 - \mathrm{zNCC}\bigl(I_a,\ I_b \circ M^{-1}\bigr),
$$

where `I_b ∘ M⁻¹` is frame B resampled by the backward map of the *predicted*
similarity (§15), and `zNCC` is **zero-normalized cross-correlation** over the
valid overlap.

### 17.1 What zNCC measures

For two image patches `x` and `y` (here: frame A's values and the warped
frame B's values, both taken over the same pixel sites, after subtracting
their means):

$$
\large
\mathrm{zNCC} = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}
{\sqrt{\sum_i (x_i - \bar{x})^2 \; \sum_i (y_i - \bar{y})^2}}.
$$

This is the **cosine of the angle between the two zero-centered vectors** of
pixel values — `+1` if the two images are identical up to a constant,
`0` if uncorrelated, `−1` if anti-correlated.  Because each image is
*centered* (its mean subtracted) before the correlation, zNCC is invariant to
per-frame *gain and bias*: a frame that is uniformly brighter or dimmer, or
uniformly shifted in exposure, still yields zNCC = 1 for perfectly aligned
content.  That is the *zero-normalized* part, and it is what makes the
photometric loss robust to the very exposure/contrast differences that break
raw pixel comparisons.

### 17.2 Why "staged"?

The photometric term is added only *after* the supervised loss has converged
(the `STAGES["supervised"]` → `STAGES["photometric"]` schedule in the
training code).  Two reasons:

1. **It is a weaker signal.**  The photometric loss is blind to uniform scale
   about the correlation center, and it is a *consistency* term, not a truth
   term — it can be satisfied by any transform that aligns the images, even a
   wrong one that is photometrically consistent.  Bootstrapping with the
   supervised loss first avoids letting this weaker signal dominate early.
2. **It is a *replacement* against shortcuts.**  Once the network has learned
   the supervised task, the photometric term acts as an almost-free
   *self-supervised* regularizer: it keeps the predictions honest on *unlabeled*
   structure (real photometric alignment), without needing more labels.

### 17.3 The masking detail

The zNCC is computed **only over the valid overlap** — the non-zero-padded
reliable region of `I_b ∘ M⁻¹`.  The reason is a direct consequence of §11.3:
the warp may push content off the frame, and zero-padded samples are
*not* content; including them would teach the network to align with the zero
border instead of the scene.  Technically, a mask is computed over
`(I_a > 0) ∧ (warped > 0)` and the zNCC is evaluated only on those pixels.
This is the "padding policy and loss mask must agree" rule of §11.3, enforced
in code.

## 18. Metrics and how they are sliced

Evaluation collects the **MCE** (mean corner error) per pair, plus the
**confidence** outputs, and — critically — does not report a single global
number.  The metrics are **sliced** along the axes that create different
regimes:

- **per terrain class** (the procedural terrain the generator used: different
  textures have different densities of matches, so MCE varies systematically);
- **per motion-range bin** (the generator's `range_bin` — how *far* the drone
  moved between frames; large motions are harder than small ones);
- **per augmentation level**;
- **per AGL configuration** (height above ground).

Why slice?  Because a global mean hides *where* the system is failing.  An
MCE of 0.5 px overall could be 0.1 px on flat grass and 3 px on featureless
terrain at high AGL — two completely different failure stories that the
global number erases.  Sliced evaluation is what makes a real diagnosis
possible, and it is the same philosophy that drives the *confidence* mechanism:
know *where* you are uncertain, not just how uncertain you are on average.

*Implementation note.* `evaluate_vo` (in `genml_kit/training/vo/train_vo.py`)
computes per-pair MCE and delegates the slices to `evaluate_sliced`, which
produces per-terrain / per-bin / per-AGL breakdowns.  The validation loop
writes the overall `VO/mce_val` and the sliced tables to the training
reporter/log.

---

# Part 5 — Classical baselines (the oracles)

Before deep learning, image alignment was solved by *classical* signal-
processing and computer-vision methods.  They still matter here for two
reasons:

1. **They are the oracles.**  Some of them (phase correlation, Lucas–Kanade /
   ECC) are accurate enough, on clean synthetic data, to be treated as
   ground truth for validating the learned front-end.
2. **They are the teachers.**  Every trick the neural estimator uses has a
   classical ancestor.  Cost volumes are correlation; corner heads are
   keypoint pipelines; confidence is, in part, a measured residual.  Seeing
   the classical versions is the fastest way to *understand* the learned
   ones.

This part develops each method from scratch.  Enough math is given to
reproduce each oracle; the implementation, where one exists, lives in
OpenCV/PyTorch rather than in this repo, because the network is the product
and the classical methods are the sanity check.

## 19. Phase correlation (translation only)

### 19.1 The shift theorem

Start with the simplest possible alignment problem: frame B is frame A
shifted by `(Δx, Δy)`:

$$
\large
I_b(x, y) = I_a(x - \Delta_x, y - \Delta_y).
$$

The Fourier transform turns shifts into *phase*: a shift in the spatial
domain is a multiplication by a complex exponential in the frequency domain:

$$
\large
\mathcal{F}\{I_b\}(\omega) = \mathcal{F}\{I_a\}(\omega) \cdot e^{-2\pi i (\omega_x \Delta_x + \omega_y \Delta_y)}.
$$

The *magnitudes* of the two transforms are identical — only the **phase**
carries the shift information.  This is the **shift theorem**, and it is the
entire basis of phase correlation.

**Derivation: proving the shift theorem from the DFT definition.**  A theorem
stated is half a theorem; this one deserves its two-line proof.  The 1-D DFT
and its inverse are

$$
\large
F(\omega) = \sum_{x=0}^{N-1} I(x)\, e^{-2\pi i\, \omega x / N},
\qquad
I(x) = \frac{1}{N} \sum_{\omega=0}^{N-1} F(\omega)\, e^{+2\pi i\, \omega x / N}.
$$

Take a shifted signal `I_b(x) = I_a(x - \Delta)` (indices modulo `N`, from the
cyclic convention of §19.4; the same computation works in 2-D as a product).
Its DFT is

$$
\large
\mathcal{F}\{I_b\}(\omega)
= \sum_x I_a(x - \Delta)\, e^{-2\pi i\, \omega x / N}.
$$

Now substitute `u = x - \Delta` (equivalently sum over `u = 0..N-1`, since the
index set is the same modulo `N`):

$$
\large
= \sum_{u} I_a(u)\, e^{-2\pi i\, \omega (u + \Delta) / N}
= \Bigl( \sum_{u} I_a(u)\, e^{-2\pi i\, \omega u / N} \Bigr)
\; \cdot \; e^{-2\pi i\, \omega \Delta / N}.
$$

The last equality uses that `e^{-2\pi i \omega (u+\Delta)/N} =
e^{-2\pi i \omega u/N} \cdot e^{-2\pi i \omega \Delta/N}` — the exponential
*factors*, which is precisely why "shift in space = multiply in frequency".
The final result is the clean statement

$$
\large
\mathcal{F}\{I_b\}(\omega) = e^{-2\pi i \omega \Delta / N}\; \mathcal{F}\{I_a\}(\omega),
$$

which is the 1-D shift theorem; the 2-D version in §19.1 follows by applying
the same argument to the two axes independently, giving the factor
`e^{-2\pi i(\omega_x \Delta_x + \omega_y \Delta_y)}`.  Notice the proof used
nothing beyond the definition of the DFT and the factorization of the
exponential — there is no hidden assumption other than the cyclic (mod `N`)
indexing, which is the same assumption the whole method inherits (§19.4).

### 19.2 The normalized cross-power spectrum

If we take the ratio of the two transforms, the magnitudes cancel and only
the phase difference remains:

$$
\large
R(\omega) = \frac{F_a(\omega)\, \overline{F_b(\omega)}}{|F_a(\omega)\, \overline{F_b(\omega)}|}
= e^{2\pi i (\omega_x \Delta_x + \omega_y \Delta_y)},
$$

a *pure* complex exponential whose frequency *is* the shift `(Δx, Δy)`.
Inverse-transforming `R` gives a single **impulse** (peak) located exactly at
`(Δx, Δy)`:

$$
\large
(\Delta_x, \Delta_y) = \arg\max\ \mathcal{F}^{-1}\left[ \frac{F_a\, \overline{F_b}}{|F_a\, \overline{F_b}|} \right].
$$

*Why is the normalization so important?*  Without dividing by the
magnitudes, the cross-power spectrum would be dominated by the *loudest*
frequencies (usually the lowest), producing a broad, smeared peak.  The
normalization whitens the spectrum so that every frequency votes equally for
the shift — which is what turns a fuzzy blob into a sharp spike.  This
"peaking" is also why phase correlation is *photometrically robust*: scaling
the image (a gain change) multiplies its spectrum by a constant, which
cancels in the normalized ratio.

### 19.3 Worked intuition with a 1-D sinusoid

A 1-D signal $`I_a(x) = cos(2\pi f x)`$ shifted by $`\Delta`$ becomes
$`\cos(2\pi f (x-\Delta)) = \cos(2\pi f x - 2\pi f \Delta)`$: the *same* sinusoid with a phase
offset $`2\pi f \Delta`$.  The normalized cross-power of the two is
$`e^{2\pi i f \Delta}`$, whose inverse transform is a spike at $`\Delta`$.  Now imagine every
frequency of a *real* image doing this simultaneously — each contributing a
spike at the *same* $`\Delta`$ — and you see why the sum piles up into one sharp
peak.  The peak location is the shift.  That is phase correlation.

### 19.4 The cyclic (wrap-around) assumption

The DFT implicitly assumes the image is *periodic*: content leaving the right
edge re-enters from the left (the image is a torus).  A translation of `−5`
pixels therefore appears at `N−5` — the shift is recovered modulo `N`.  For
small shifts relative to the image size this is harmless; for shifts that
approach the image size it becomes ambiguous.  The practical fix is to
window/taper the images before the transform (and to only *trust* the method
for shifts well inside the image).

### 19.5 What phase correlation alone can and cannot do

- **Can:** find pure translations, robustly, photometrically-insensitively,
  at any magnitude up to the wrap-around limit, with a single FFT pair.
- **Cannot:** handle rotation or scale.  Those *change the frequencies
  themselves* rather than just their phases — which is exactly the problem
  Fourier–Mellin (§20) is built to solve.

*Implementation note.* OpenCV exposes this as `cv2.phaseCorrelate`; it is the
standard quick translation estimate and the natural first stage of the
cascade in §21.3.

## 20. Fourier–Mellin (rotation + scale)

### 20.1 The log-polar trick

Rotation and scale are *not* shifts in the image domain, but they *become*
shifts in a cleverly chosen domain: **log-polar coordinates**.

- Rotating the image by `α` rotates its Fourier *magnitude* spectrum by `α`
  (rotation about the center is rotation in frequency space too).
- Scaling the image by `s` scales its Fourier magnitude spectrum by `1/s`
  *along each frequency axis* (a property of the 2-D Fourier transform).

Now take the frequency-plane axes `(u, v)` and write them in **polar** form
`(r, φ)` with `r = log √(u² + v²)`.  Then:

- rotation by `α` shifts the polar angle `φ` by `α` — a *shift in φ*;
- scaling by `s` shifts the log-radius `r` by `log s` — because the radial
  frequency axis in log units is `log(ρ/s) = log ρ − log s`, a *shift in
  log-radius*.

Both rotation and scale have become **pure translations** in the
log-polar domain.  And phase correlation (§19) is the exact tool for
translations!  That is the whole idea:

1. Take both frames' Fourier magnitudes;
2. resample them in log-polar coordinates;
3. phase-correlate the log-polar spectra → get the pair
   `(Δ log-radius, Δ angle)`;
4. convert back: $`s = e^{\Delta \log r}`$, $`\theta = \Delta \phi`$;
5. remove the estimated rotation/scale from frame B (via §11's warping) and
   phase-correlate *again* in the spatial domain to recover the residual
   translation.

**Derivation: why scale and rotation become shifts in log-polar frequency.**  The
two bullets above are the entire engine of Fourier–Mellin, so they deserve
proofs, not assertions.

**Scale theorem.**  Work in 1-D for clarity (2-D is the same argument per
axis).  Let `g(x) = I(x / s)` be the image scaled by `s` (a change of
variable with `s > 0`).  Its Fourier transform is

$$
\large
\mathcal{F}\{g\}(\omega)
= \int_{-\infty}^{\infty} I(x / s)\, e^{-2\pi i \omega x}\, dx .
$$

Substitute `u = x / s`, so `x = s u` and `dx = s du`:

$$
\large
= s \int_{-\infty}^{\infty} I(u)\, e^{-2\pi i \omega s u}\, du
= s\, \mathcal{F}\{I\}(s \omega).
$$

The transform of a signal scaled by `s` is a *reshaped* copy of the original
spectrum: `F_g(ω) = s·F_I(sω)` — compressed by a factor `s` in the frequency
axis.  (With the continuous Fourier convention there is a `1/s` factor as
well; what matters is the *axis rescaling*.)  So *scaling the image rescales
the frequency axis*, which is exactly the second bullet.

**Rotation property.**  A rotation of the image by `α` about the origin is a
rotation of its Fourier transform by `α`.  Why?  The Fourier transform is a
*linear* map that commutes with orthogonal coordinate changes: rotating the
argument of `I` before integrating is the same as rotating the output
coordinates, because `e^{-2πi ω·x}` is unchanged by a joint rotation of `ω`
and `x` (the dot product is rotation-invariant).  Hence

$$
\large
\mathcal{F}\{I(R_\alpha x)\}(\omega) = \mathcal{F}\{I\}(R_\alpha^{-1} \omega)
= \mathcal{F}\{I\}(R_\alpha^{\top} \omega),
$$

and `R_α` inverts to `R_{−α} = R_αᵀ` — a rotation of the frequency plane by
`α`, precisely the first bullet.

**From these to log-polar shifts.**  Write the frequency-plane coordinates in
polar form `(ρ, θ)` with `ρ = √(u² + v²)`.  A rotation by `α` sends
`θ → θ + α`: a shift *along the angle axis*.  A scaling by `s` sends
`ρ → ρ/s`, so after taking the logarithm,

$$
\large
\log \rho \;\xrightarrow{\;s\;}
 \log(\rho / s) = \log \rho - \log s,
$$

a pure shift *along the log-radius axis*.  Both operations are now
translations in the `(log ρ, θ)` plane — the domain where phase correlation
(§19) is the exact tool.  This is the whole content of the log-polar trick,
derived from the two frequency-domain facts above.

### 20.2 Why it works despite the "sampling" cost

The price of the trick is that log-polar resampling is an *interpolation*
(§11): the log-polar grid samples the frequency plane non-uniformly, and the
interpolation introduces a mild blur — the recovered `(s, θ)` is accurate but
not *ultra*-precise.  That is fine, because Fourier–Mellin is never the final
answer in this system: it is the **coarse initializer** that brings
`(s, θ)` into the convergence basin of a *fine* refinement, which is Lucas–
Kanade via ECC (§21).  The cascade is the standard engineering pattern:
*coarse-but-global, then fine-but-local*.

### 20.3 What Fourier–Mellin can and cannot do

- **Can:** estimate rotation and uniform scale against arbitrary magnitude
  (subject to interpolation blur), robustly.
- **Cannot:** do so to sub-pixel precision alone; and being Fourier-based, it
  inherits the *torus* periodic assumption — fine for small in-frame motions,
  a source of error near large ones.

## 21. Lucas–Kanade and ECC (dense refinement)

### 21.1 The brightness-constancy assumption

The other classical family is *dense, iterative, gradient-based* alignment.
Start from the **brightness constancy** assumption: the same physical point
appears with the same intensity in both frames — up to motion, the pixels are
constant:

$$
\large
I_b(x + \Delta x,\ y + \Delta y) \approx I_a(x, y).
$$

Taylor-expand `I_b` around `(x, y)` (small motion!):

$$
\large
I_b(x, y) + \nabla I_b(x, y) \cdot (\Delta x,\ \Delta y) \approx I_a(x, y),
$$

so

$$
\large
\nabla I_b(x, y) \cdot (\Delta x,\ \Delta y) \approx I_a(x, y) - I_b(x, y) =: \delta I(x, y)
$$

— a *linear* equation in the shift `(Δx, Δy)` at every pixel, with the
right-hand side being the frame difference and the coefficients being the
image gradient.  Stack all pixels into one least-squares problem (the
"normal equations"):

$$
\large
J^{\top} J\, \Delta p = J^{\top} (I_a - I_b),
\qquad
J = \nabla I_b,
$$

and solve for `Δp`.  Iterate (re-warp, re-differentiate) — that is the
**Lucas–Kanade** iteration.  Because this is gradient-descent on aligned
brightness, it refines a *good* initial guess to sub-pixel accuracy.

**Derivation: from Taylor to the normal equations.**  The jump from one pixel's
equation to the matrix equation "`JᵀJ Δp = Jᵀ(I_a − I_b)`" is the heart of LK,
so let me lay out every step.

Each pixel `(x, y)` gives one *linear* equation in the two unknown components
of `Δp = (Δx, Δy)`:

$$
\large
\begin{bmatrix} I_x & I_y \end{bmatrix} \begin{bmatrix} \Delta x \\\\ \Delta y \end{bmatrix}
= \delta I(x, y),
\qquad
I_x = \frac{\partial I_b}{\partial x},\quad I_y = \frac{\partial I_b}{\partial y},
$$

where `δI = I_a − I_b` is the frame difference.  Stack all `N` pixels
*vertically*: the left sides line up into a matrix `J` (the Jacobian, one row
per pixel) times the unknown `Δp`, and the right sides stack into the vector
`r = I_a − I_b`:

$$
\large
\begin{bmatrix} \nabla I_b^\top(x_1) \\\\ \nabla I_b^\top(x_2) \\\\ \vdots \\\\ \nabla I_b^\top(x_N) \end{bmatrix}\,\Delta p = \begin{bmatrix} \delta I(x_1) \\\\ \delta I(x_2) \\\\ \vdots \\\\ \delta I(x_N) \end{bmatrix}.
$$

The left stack is the Jacobian `J`; the right stack is the
frame-difference vector `r`.  So the display reads exactly `J · Δp = r`.

This is an overdetermined `N×2` system (`N ≫ 2` pixels).  There is generally no
exact solution, so we seek the least-squares fit: minimize
`‖J Δp − r‖²` over `Δp`.  Expand:

$$
\large
\lVert J\,\Delta p - r \rVert^2
= \Delta p^\top J^\top J\, \Delta p - 2\, r^\top J\, \Delta p + r^\top r .
$$

Differentiate with respect to `Δp` and set to zero:

$$
\large
0 = 2 J^\top J\, \Delta p - 2 J^\top r
\;\Longrightarrow\; J^\top J\, \Delta p = J^\top r,
$$

which is exactly the normal equation of §21.1.  The matrix `JᵀJ` is `2×2`
(small!) and `Jᵀr` is a 2-vector; solving it costs nothing once the gradients
are computed.  That is all LK does per iteration: build `J` from the gradient
of the warped frame, form `JᵀJ` and `Jᵀr`, solve `2×2`, and re-warp.

**Worked 1-D example (why the aperture problem is a rank statement).**  Take a
signal with a *constant* gradient: `I_b(x) = 2x`, and `δI(x) = -4` everywhere
(a hypothesized shift of `+2` pixels).  The single-pixel equation is
`2·Δx = −4`, so `Δx = −2` — recovered exactly: a ramp has enough gradient
structure to determine motion uniquely.  Now take `I_b(x) = 5` (constant,
zero gradient): every pixel equation is `0·Δx = δI`, contributing *no*
information.  Stacked, `J` is the zero matrix, `JᵀJ = 0`, and the normal
equation `0 = 0` is degenerate: `Δx` is completely unconstrained.  This is the
aperture problem in 1-D: **`JᵀJ` is rank-deficient exactly when the image
gradient does not span the directions of motion.**  In 2-D, the same thing
happens on a straight edge (gradients all parallel → `JᵀJ` has rank 1): motion
*along* the edge is invisible, exactly as §21.3 says — now with the rank
failure derived rather than asserted.

### 21.2 Adding scale and rotation, photometrically robust: ECC

LK's raw form assumes *pure translation* and *exact brightness equality*.
Both assumptions are too strong for us, and the **Enhanced Correlation
Coefficient (ECC)** generalization fixes them in one stroke:

$$
\large
\mathrm{ECC} = \mathrm{zNCC}\bigl(I_a,\ I_b \circ M^{-1}\bigr),
$$

i.e. it maximizes **zero-normalized cross-correlation** (§17) over the valid
overlap between frame A and the frame-B-warped-by-the-current-`M`.  The
benefits, compared to raw LK:

- **4-DOF similarity.**  The warp $`M^{-1}`$ carries `(s, θ, t)` — the whole
  transform, not just translation.
- **Photometric robustness.**  zNCC is invariant to per-frame gain/bias (§17.1),
  so exposure/contrast differences between frames do not corrupt the
  gradient.  This is the "enhanced" part.
- **Sub-pixel accuracy.**  Because the objective is smooth (bilinear warp),
  the iterative refinement converges to a *fraction* of a pixel — the
  precision that phase correlation (§19) and Fourier–Mellin (§20) cannot
  reach alone.

OpenCV's `cv2.findTransformECC` minimizes exactly this objective and is the
classical accuracy gold standard for the 4-DOF refine.

### 21.3 Practical properties (why the cascade exists)

- **Small convergence basin.**  LK/ECC converge only for *small* `Δp` — a few
  degrees of rotation, a few percent of scale.  Outside that basin the
  linearization is wrong and the iteration *diverges*.  Coarse-to-fine
  (pyramid) enlarges the basin; seeding with the Fourier–Mellin estimate
  (§20) gives the same benefit with one pyramid level.  The full classical
  pipeline is therefore:

  **phase correlation (global translation) → Fourier–Mellin (coarse
  rotation/scale) → ECC (fine 4-DOF)**.

- **The aperture problem, quantified.**  Pixels with zero gradient contribute
  *no* equation (`∇I_b = 0 ⇒ 0·Δp = 0` — nothing learned).  In texture-poor
  regions the normal matrix `JᵀJ` becomes rank-deficient and the solve is
  under-determined: the method literally cannot see motion along the
  direction in which the image is flat.  This is why LK needs **corners and
  edges** — gradients along two independent directions — which is exactly
  what the keypoint pipeline of §22 and the learned confidence head rely on.
- **Photometric robustness is why the scheme is phase-correlation → ECC:**
  both tolerate exposure differences, and ECC finishes the job the log-polar
  init starts.

## 22. Keypoints + RANSAC (the strongest classical oracle)

The strongest classical-modern pipeline does not align dense pixels at all.
It finds a sparse set of *keypoints* in both frames, *matches* them, throws
away bad matches, and fits the similarity to the survivors.  The full
oracle used here: **SuperPoint** keypoints, **LightGlue** matches, a ratio
test, **RANSAC** over similarity fits, and a final **Umeyama** on inliers.

### 22.1 The components, in teaching order

**Keypoint detector.**  A *saliency map* `S(x)` over the image scores how
"matchable" each location is; local maxima above a threshold are keypoints.
- *Classical:* **Harris** (a corner is where the image's second-moment matrix
  `M = Σ ∇I ∇Iᵀ` has two *large* eigenvalues — i.e. strong gradient energy in
  two independent directions, the same condition that saves LK from the
  aperture problem), and **SIFT** (difference-of-Gaussians extrema, with
  scale-space pyramids).
- *Learned:* **SuperPoint** — a CNN *trained* (mainly self-supervised) to
  produce repeatable keypoints and descriptors directly from the image.
  The learned version is more repeatable under the appearance changes our
  oblique drone view produces.

**Matcher.**  For each keypoint in A, find the best-matching keypoint in B
by descriptor distance (SIFT uses L2 on its histograms; LightGlue uses a
learned attention-based match score).  *Ratio test (Lowe's):* keep a match
only if the best distance is *much* smaller than the second-best
(`ratio < 0.8`).  This purges ambiguous matches — a keypoint with two
plausible destinations is not a reliable correspondence.

**RANSAC (Random Sample Consensus).**  The matches still contain *outliers*
(wrong correspondences).  RANSAC is the algorithm that fits a model robust
to them:

1. sample the **minimum number of matches needed**.  A similarity has 4 DOF
   and each point correspondence gives 2 scalar constraints, so the minimal
   all-inlier sample is `k = 2` matches (4 constraints — exactly enough);
2. fit the similarity (Umeyama, §13) to the sample;
3. count the **inliers** (matches whose reprojection error under that
   similarity is below a threshold);
4. repeat, keeping the fit with the most inliers;
5. re-fit (Umeyama) on *all* inliers for the final answer.

The "randomness" is the robustness engine: as long as *some* sample of
`k` matches is all-inlier, RANSAC will find the true model.  The
`threshold` is precisely the MCE-style tolerance of §6.

**Derivation: how many iterations does RANSAC need?**  The robustness claim —
"as long as *some* all-inlier sample is drawn, RANSAC finds the model" — can be
made quantitative, and the resulting formula is what sets the iteration count
in practice.

Let `w` be the fraction of inliers among the candidate matches, so a randomly
drawn match is an inlier with probability `w`.  A single sample of `k = 2`
matches is all-inlier with probability `w^k` (drawing `k` inlier matches, by
independence).  The probability that one sample is *not* all-inlier is
therefore `1 − w^k`.  After `m` independent samples, the probability that
*every* sample failed to be all-inlier is `(1 − w^k)^m`.  Hence the
probability that at least one sample is all-inlier — i.e. that RANSAC finds a
good fit — is

$$
\large
p_{\mathrm{success}} = 1 - (1 - w^k)^m .
$$

Solve for the number of iterations to achieve a target success probability
`p`:

$$
\large
1 - p = (1 - w^k)^m
\;\Longrightarrow\;
m = \frac{\ln(1 - p)}{\ln(1 - w^k)} .
$$

Two worked numbers make the formula concrete.  With `w = 0.5` (half the
matches are inliers) and `k = 2`, a single sample is all-inlier with
probability `0.25`; to get `p = 0.99` one needs
`m = ln(0.01)/ln(0.75) ≈ 16` iterations — cheap.  With `w = 0.2` (only a
fifth of matches good), `w² = 0.04`, and `m = ln(0.01)/ln(0.96) ≈ 113`
iterations — still cheap.  This is why RANSAC "works"; the formula quantifies
exactly how the inlier fraction and the sample size trade against the run
time, and it is the reason the pipeline's default iteration count is a small
constant rather than a guess.

**Why this is the strongest oracle.**  Sparse robust matching (SuperPoint +
LightGlue + RANSAC) is routinely at (or beyond) the accuracy of dense
classical methods on textured scenes, and it degrades *gracefully* on weak
texture (fewer keypoints → fewer matches → RANSAC reports low confidence).
It is the natural gold standard for evaluating the learned front-end, because
it is a *different* family of methods than the neural regressor being
measured.

*Implementation note.* The learned front-end's cost volume (§12) and corner
head (§14) are, in a very real sense, a *learned, differentiable, dense
analog* of this entire sparse pipeline: features ≈ descriptors, cost volume ≈
matching, corner offsets via Umeyama ≈ RANSAC + re-fit.  Understanding §22 is
the single most useful way to understand what the network has learned to
mimic.

---

# Part 6 — Consumption: the EKF contract

The front-end does not operate in a vacuum.  Its `(s, θ, t)` estimate is one
measurement in a larger *state-estimation* loop — typically an extended
Kalman filter (EKF) that fuses the VO increment with the drone's inertial
(gyro/accelerometer) readings to maintain a continuous pose estimate.  This
part defines the exact *contract* the front-end fulfills for that filter:
what it outputs, in what units, and — crucially — how its **confidence**
becomes *measurement noise*.

## 23. A minimal Kalman-filter recap

A Kalman filter estimates the state `x_k` of a system from two sources:
a *prediction* (the state-transition model) and *measurements* (the
observation model), both of which are noisy and both of which are assumed —
in the linear case — to be corrupted by zero-mean Gaussian noise.  The
linearized (EKF) version used here is:

**Prediction (transition):**

$$
\large
x_{k+1} = F_k\, x_k + w_k,
\qquad
w_k \sim \mathcal{N}(0, Q_k),
$$

**Measurement (observation):**

$$
\large
z_k = H_k\, x_k + v_k,
\qquad
v_k \sim \mathcal{N}(0, R_k),
$$

where

- `x_k` — the filter state (for us: drone pose — position, orientation,
  height — though the *instantiation* is up to the flight stack; this
  document pins the interface, not the state's contents);
- `F_k` — the **state-transition matrix**, integrated over the frame
  interval.  Between VO measurements, gyro/accelerometer data propagate the
  state forward ("dead-reckoning"); that is the `F_k x_k` term;
- `Q_k` — **process noise**, the covariance of `w_k`.  It dominates *when the
  platform maneuvers*, i.e. when the prediction is uncertain;
- `z_k` — the **measurement vector**.  This is where the VO front-end plugs
  in: the inter-frame increment `(θ, s, t)` is mapped into components of
  `z_k`;
- `H_k` — the **measurement Jacobian**, `H_k = ∂h/∂x` evaluated at the
  current estimate `x̂_k`: it says "given the state, what measurement would
  we expect?" and it makes VO's nonlinear observation locally linear;
- `R_k` — the **measurement noise covariance**.  The crucial design fact of
  this contract: **R_k is derived from the VO confidence at every step** — it
  is *not* a constant.

The filter alternates *predict* (apply `F_k` with growing `Q_k`-induced
uncertainty) and *update* (blend the prediction with the measurement,
weighted by how sure each source is — the Kalman gain `K_k`).  The output is
a posterior estimate `x̂_k` and its covariance `P_k` — the "how sure are we"
answer that navigation actually needs.

**Why "extended"?**  Because the VO observation is nonlinear: `(s, θ, t)` as
a function of pose involves sines, cosines, and perspective (Parts 1–2).  The
EKF *linearizes* `h` about the current estimate once per step (the `H_k`
above) and runs the standard linear mechanics — the same philosophy as the
LK linearization of §21, applied to the filter.

**Derivation: where the update and the Kalman gain come from.**  The
"weighted blend" in the paragraph above is not a heuristic — it is the
algebra of *multivariate Gaussians*, and the single most instructive
derivation in the whole filter.  If we ignore the time indices, the update
step is this: we hold a prior belief `x ~ N(μ, P)` (the prediction from
`F_k`, with `P = P_k⁻`) and receive a measurement `z = Hx + v` with
`v ~ N(0, R)`.  What is the best posterior belief `x | z`?

Bayes' rule says the posterior density is the prior times the likelihood.
Both are Gaussian, so the product is again Gaussian, and the exponent of a
Gaussian is a *quadratic*: add the prior quadratic and the measurement
quadratic,

$$
\large
(x - \mu)^\top P^{-1} (x - \mu)
\;+\;
(z - H x)^\top R^{-1} (z - H x) ,
$$

and complete the square in `x`.  Expanding the second term,

$$
\large
(z - Hx)^\top R^{-1} (z - Hx)
= z^\top R^{-1} z - 2 x^\top H^\top R^{-1} z + x^\top H^\top R^{-1} H x .
$$

The total exponent is `x^\top (P^{-1} + H^\top R^{-1} H)\, x - 2 x^\top (P^{-1} \mu + H^\top R^{-1} z) + (x\text{-independent terms})`.  Matching to a Gaussian with mean `μ⁺` and covariance `P⁺`:

$$
\large
(P^{+})^{-1} = P^{-1} + H^\top R^{-1} H,
\qquad
(P^{+})^{-1} \mu^{+} = P^{-1}\mu + H^\top R^{-1} z .
$$

Now *define* the Kalman gain `K = P H^\top (H P H^\top + R)^{-1}` and apply
the Woodbury matrix identity to the first line:

$$
\large
P^{+} = P - K H P,\qquad
\mu^{+} = \mu + K (z - H \mu) .
$$

The second line **is the update equation of §25.2**: `μ⁺ = μ + K·(innovation)`,
with the gain `K` measuring exactly the relative trust between `P` (how
uncertain the prediction is) and `R` (how noisy the measurement is).  If
`R` is tiny (confident VO), then `K H ≈ I` and `μ⁺ ≈ z`: the measurement
dominates; if `R` is huge (doubtful VO), `K ≈ 0` and `μ⁺ ≈ μ`: the filter
ignores the measurement and dead-reckons — which is precisely the confidence
ladder of §24.3, now *derived* from the Gaussian product rather than asserted.

So the whole update step is "add two quadratics and complete the square."
That is all the Kalman filter does, and why it is both optimal (for Gaussians)
and simple (it is just posterior-Gaussian algebra).

## 24. Confidence as measurement noise

### 24.1 What "confidence" means here

In this system, the confidence output is *not* a vague "I feel confident"
number; it is the network's prediction `ρ̂` of the **irreducible residual**
`ρ` (§9, §10) — "how many pixels of corner error would remain even if my
similarity were perfect?"  It is supervised to be exactly that (§16.4).
Therefore:

- a small `ρ̂` means: "this pair is well explained by a similarity (flat
  ground, gentle motion) — trust my `(s, θ, t)`.";
- a large `ρ̂` means: "this pair has significant foreshortening/obliquity
  that no similarity can absorb — be cautious."

### 24.2 The mapping to `R_k`

The filter's measurement-noise covariance must be small when the measurement
is good and large when it is not.  Since MCE and `ρ̂` are both in *pixels*,
the mapping is direct:

$$
\large
R_k = \mathrm{diag}\bigl(\sigma_\theta^2(\hat{\rho}),\; \sigma_s^2(\hat{\rho}),\; \sigma_t^2(\hat{\rho})\bigr),
$$

where each `σ²` is an *increasing* function of the predicted residual —
e.g. `σ = α₀ + α₁·ρ̂` with per-parameter scales `αᵢ` — so a confident
(low-`ρ̂`) VO measurement gets a small `R_k` and therefore *dominates* the
filter update, while a doubtful (high-`ρ̂`) one gets a large `R_k` and is
mostly ignored *as the prior physics carries the state forward*.

Two deliberate design consequences:

- **The units work out.**  `ρ̂` is in pixels (the same basis as the MCE), so
  converting it into per-parameter noise is a *scaling* question, not a
  "which units do we invent?" question.
- **The confidence is *calibrated by construction*.**  Because `ρ̂` is trained
  against the true residual (§16.4), a confident estimate is one that *really
  is* more accurate — the filter trusts it *because it should*.  The whole
  ladder of §24.3 rests on this calibration, which is why the supervised
  residual head is not optional.

### 24.3 The failure ladder, in filter terms

The system degrades gracefully along an explicit ladder, expressed here in
the filter's own language:

1. **Confident** (`ρ̂` small): the VO increment is a *strong* measurement;
   `R_k` small; the EKF update pulls the state hard toward the VO estimate.
2. **Marginal** (`ρ̂` medium): `R_k` grows; the filter blends VO more weakly
   with the inertial prediction.
3. **Low confidence / degenerate** (`ρ̂` large, or gated out — §25): the
   measurement is *rejected*; the filter **dead-reckons** — it integrates
   `F_k` alone (`w_k`-driven) until a trustworthy measurement reappears.

This ladder *is* the graceful-degradation story that makes a single bad frame
(untextured ground, sudden occlusion) a non-event instead of a jump in the
pose estimate.

## 25. Update and gating

### 25.1 The innovation

The **innovation** `y_k` is the difference between what we measured and what
the prediction expected:

$$
\large
y_k = z_k - h(\hat{x}_k^-),
$$

and its covariance $`S_k = H_k P_k^- H_k^\top + R_k`$ is a measure of "how
surprising is any deviation, given both the prediction and the measurement
noise?"  The innovation is *zero-mean* when the system is healthy; a
persistently large innovation is a red flag.

### 25.2 The Kalman gain

The update blends prediction and measurement according to their relative
certainties:

$$
\large
K_k = P_k^- H_k^\top S_k^{-1},
\qquad
\hat{x}_k^+ = \hat{x}_k^- + K_k\, y_k,
$$

The gain automatically approaches the "measurement" side when `R_k` is small
(confident VO) and the "prediction" side when `R_k` is large or `Q_k`
dominates.

### 25.3 Gating (the Mahalanobis test)

Before trusting a measurement, the filter checks whether the innovation is
statistically plausible:

$$
\large
y_k^{\top} S_k^{-1} y_k \;\le\; \chi^2_{\max},
$$

a **Mahalanobis-distance gate**.  If the (squared) normalized innovation
exceeds the threshold implied by a chi-square distribution with the right
number of degrees of freedom, the measurement is treated as an **outlier**
— e.g. a VO estimate that disagrees wildly with the inertial prediction
because it locked onto the wrong feature — and the *update is skipped*: the
filter dead-reckons on `F_k` alone rather than being dragged by a bad
measurement.

**Putting it together.**  The gating test is the *mathematical* form of the
confidence ladder's bottom rung: low-confidence measurements are either
down-weighted through `R_k` (§24) or vetoed outright by the gate (§25).  Both
mechanisms are the same philosophy — *know when to trust your sensor* — and
both are fed by the single, calibrated confidence number the front-end emits.

---

# Appendix A: Symbol table

Every symbol used in this document, with a plain-English meaning and the
section where it first appears.

| Symbol | Meaning | First appears |
|---|---|---|
| `x, x′` | a point in the image (and its transformed image) | §2 |
| `s`, `log s` | uniform scale; natural log of the scale | §2, §4 |
| `θ` | rotation angle, radians, wrapped to `(−π, π]` | §2, §4 |
| `t = (t_x, t_y)` | translation, pixels | §2 |
| `R_θ` | 2×2 rotation matrix by `θ` | §2 |
| `M` | 3×3 homogeneous similarity matrix | §2 |
| `A` | the 2×2 linear part of a similarity matrix | §5 |
| `u_i`, `q_i` | image corners; their destinations in the other frame | §6, §9 |
| `ρ`, `ρ̂` | true irreducible residual; the network's predicted residual (confidence) | §9, §16 |
| `φ`, `ψ` | camera pitch below horizontal, yaw about vertical | §7 |
| `c` | camera position (world coords) | §7 |
| `R_cw` | camera orientation matrix (axes as rows, world→camera) | §7 |
| `K` | intrinsic matrix (focal lengths, principal point) | §7 |
| `H`, `H_A`, `H_B` | ground-to-image homography; of camera A, of camera B | §8 |
| `I_a`, `I_b` | frames A and B (image arrays) | §11, §19 |
| `F_a`, `F_b` | feature maps of the two frames (shared encoder) | §12 |
| `r` | correlation radius (default 6, at 1/8 resolution) | §12 |
| `δy, δx` | displacement in feature-map pixels | §12 |
| `p_i`, `q_i` (Umeyama) | corresponding points in the two frames | §13 |
| `μ_p`, `μ_q` | centroids of the corresponding point sets | §13 |
| `H` (Umeyama) | cross-covariance matrix whose SVD drives the solve | §13 |
| `U, Σ, V` | SVD factors of the cross-covariance | §13 |
| `d` | reflection guard `det(U·Vᵀ) = ±1` | §13 |
| `σ₁, σ₂` | singular values of the cross-covariance | §13 |
| `Δ_i` | predicted offset of corner `i` | §14 |
| `λ_s, λ_θ, λ_c` | loss weights (scale, angle, confidence) | §16 |
| `L_mce` | mean corner error (primary supervised term) | §6, §16 |
| `L_photo` | photometric auxiliary loss `1 − zNCC(…)` | §17 |
| `zNCC` | zero-normalized cross-correlation | §17 |
| `(Δx, Δy)` | pure translation between frames | §19 |
| `F_a(ω), F_b(ω)` | 2-D Fourier transforms of the frames | §19 |
| `S(x)` | saliency map of a keypoint detector | §22 |
| `x_k` | EKF state | §23 |
| `F_k` | state-transition matrix | §23 |
| `Q_k`, `R_k` | process noise, measurement noise covariances | §23 |
| `z_k` | measurement vector | §23 |
| `H_k` | measurement Jacobian | §23 |
| `y_k`, `S_k`, `K_k` | innovation; its covariance; Kalman gain | §25 |

---

# Appendix B: Failure modes and shortcuts

Every design decision in this document exists because its *absence* causes a
specific failure.  This appendix is the field manual for those failures:
each entry follows the same shape
entry follows the same shape — **Symptom / Mechanism / Why the design
handles it**.  The "mechanism" is the mathematical reason; the "why handled"
is the defense already built into the system.

## B.1 The aperture problem

- **Symptom.**  A textureless region (flat ground, clear sky) contributes no
  usable alignment signal; the estimator drifts along the "flat" direction.
- **Mechanism.**  In Lucas–Kanade, a pixel with zero gradient supplies a
  *degenerate* equation: `∇I_b = 0 ⇒ 0·Δp = 0` — no constraint on motion.  The
  normal matrix `JᵀJ` becomes rank-deficient; motion perpendicular to the
  local edge direction is unobservable.  (A 1-D analogy: a pure horizontal
  edge in a 1-D image tells you nothing about vertical motion — there is no
  vertical information anywhere in the signal.)
- **Why the design handles it.**  Every pathway needs *texture*: (a) the
  learned corner head attaches to features that sharpen where gradients are
  informative (§14); (b) the confidence head, trained on the residual §9,
  grows exactly where texture is absent — a low-texture pair *predicts* a
  large `ρ̂`, which down-weights the measurement in the filter (§24); (c) the
  keypoint oracle only finds corners, so it reports *fewer matches* on
  textureless scenes and flags low confidence (§22).

## B.2 The 2π angle boundary

- **Symptom.**  Naive angle losses explode (or jump) when the prediction and
  target sit on opposite sides of ±π.
- **Mechanism.**  An angle is a *circle*, not a line: `359°` and `−1°` are 2°
  apart physically but 358° apart arithmetically.  An unwrapped loss
  `|θ̂ − θ|` is discontinuous at the boundary.
- **Why the design handles it.**  (a) All angle *differences* are wrapped
  before any loss or metric (§4.2, §16.3); (b) the network never regresses an
  angle at all — it regresses corner *pixels* (§14), so the discontinuity
  never enters the network's output space.

## B.3 Scale asymmetry and invalid scales

- **Symptom.**  A network regressing `s` directly can emit `s ≤ 0` (a
  mirror!), and its errors are asymmetric between "too big" and "too small".
- **Mechanism.**  Scale is multiplicative: "10% too big" and "10% too small"
  differ in linear units; and nothing in a raw linear regression prevents a
  negative output.
- **Why the design handles it.**  Scale is stored and regressed in *log*
  space: $`s = e^{log s} > 0`$ always, and log-space errors are symmetric
  relative errors (§4.1, §16.2).  The Umeyama solve additionally guarantees
  `s > 0` by construction (§13).

## B.4 Forward-warping holes

- **Symptom.**  Rendering a warp by iterating source pixels leaves holes
  (unwritten output pixels) and aliasing.
- **Mechanism.**  Forward mapping is not a function on the output grid:
  many-to-one collisions and empty destinations are inherent.
- **Why the design handles it.**  All warps use the **backward map** (§11.1):
  each output pixel samples its source independently, so every output pixel is
  defined and the warp is naturally differentiable (§15).

## B.5 Interpolation quantization of the ground truth

- **Symptom.**  Nearest-neighbor rendering quantizes sub-pixel shifts into
  whole-pixel steps; the ground truth becomes blocky, and the regressor
  learns to un-learn stair-step artifacts.
- **Mechanism.**  A kernel with no sub-pixel weighting cannot represent
  sub-pixel content.
- **Why the design handles it.**  The generator uses **bilinear** interpolation
  (§11.2), which is both sub-pixel-aware and what real resampling looks like;
  the convention is frozen in dataset metadata.

## B.6 The `align_corners` half-pixel bug

- **Symptom.**  An invisible 0.5-pixel translation error in every pair; the
  network asymptotes at MCE ≈ 0.5 px and cannot improve.
- **Mechanism.**  `align_corners=True` maps pixel *centers* to
  `±1`; `align_corners=False` maps pixel *edges*; mixing conventions between
  generator and warper shifts everything by half a pixel (§11.4).
- **Why the design handles it.**  One convention is fixed per dataset,
  recorded in metadata, and the ground-truth round-trip is unit-tested (the
  90° example of §3 is the canonical test, §11.4).

## B.7 Zero-padding shortcuts

- **Symptom.**  A network learns to "align" to the artificial zero border of
  the warped image instead of the scene.
- **Mechanism.**  Zero padding presents a constant, easy-to-detect boundary
  that is not real content, and it contaminates the photometric loss if the
  loss mask includes it.
- **Why the design handles it.**  The photometric zNCC is masked to the
  *valid* (non-padded) overlap (§17.3), and the padding policy is fixed and
  recorded (§11.3).

## B.8 Reflection fits

- **Symptom.**  A "similarity" fit that mirrors the image instead of rotating
  it; angles come back wrong by a sign.
- **Mechanism.**  The SVD's rotation is ambiguous up to a reflection; a naive
  `U·Σ·Vᵀ` can pick the mirrored branch when `det(UVᵀ) = −1`.
- **Why the design handles it.**  The Umeyama solve applies the reflection
  guard `d` (§13 Step 3), and `params_from_matrix` refuses to interpret a
  reflected matrix (§5).

## B.9 ECC / LK divergence on large motion

- **Symptom.**  The dense iteration diverges instead of refining.
- **Mechanism.**  LK/ECC linearize about the *current* estimate; outside a
  small basin (a few degrees, a few percent scale) the linearization is
  wrong and the update overshoots (§21.3).
- **Why the design handles it.**  A coarse-to-fine *init* (Fourier–Mellin for
  `(s, θ)`, phase correlation for translation) seeds the refinement inside
  the basin; the classical cascade is phase-correlation → Fourier–Mellin →
  ECC (§21.3).

## B.10 Wrong ground-truth labels (the meta-shortcut)

- **Symptom.**  The network "learns" the *generator's* artifacts instead of
  image alignment; it scores beautifully on synthetic pairs and fails on real
  ones.
- **Mechanism.**  Any systematic generator artifact (interpolation kernel,
  padding, border pattern, motion sampling envelope) is a *learnable
  shortcut* — a distribution shift between training and deployment.
- **Why the design handles it.**  (a) The generator is *geometrically exact*
  (homography, §8) rather than approximate; (b) rendering conventions are
  frozen in metadata (§11); (c) metrics are *sliced* per terrain/motion/AGL
  (§18) so a "great average" cannot hide a degenerate subtree; (d) the
  classical oracles of Part 5 provide an *independent* check that the learned
  front-end is doing real alignment.

---

# Appendix C: Reading list

The methods in this document come from a long and well-documented line of
work.  The following are the canonical references, each paired with where it
is used here.  Every entry is a public, checkable source.

**Image registration and alignment**

- C. Kuglin and D. Hines, *The Phase Correlation Image Alignment Method*,
  Proc. IEEE Int. Conf. on Cybernetics and Society, New York, 1975,
  pp. 163–165 — the original phase-correlation formulation of §19.
- B. S. Reddy and B. Chatterji, [*An FFT-Based Technique for Translation,
  Rotation, and Scale-Invariant Image Registration*](https://ieeexplore.ieee.org/document/506761),
  IEEE Transactions on Image Processing, 1996 — the Fourier–Mellin log-polar
  registration of §20.
- B. D. Lucas and T. Kanade, *An Iterative Image Registration Technique with
  an Application to Stereo Vision*, Proc. IJCAI, 1981 — the Lucas–Kanade
  iteration of §21.
- G. D. Evangelidis and E. Z. Psarakis, *Parametric Image Alignment Using
  Enhanced Correlation Coefficient Maximization*, IEEE Trans. PAMI, 2008 —
  the ECC objective of §21.

**Geometric fitting and estimation**

- S. Umeyama, [*Least-Squares Estimation of Transformation Parameters Between
  Two Point Patterns*](https://ieeexplore.ieee.org/document/1270), IEEE
  Transactions on Pattern Analysis and Machine Intelligence, 1991 — the
  closed-form similarity fit of §13 and §9.

**Keypoints, features, matching**

- C. Harris and M. Stephens, *A Combined Corner and Edge Detector*, Proc.
  Alvey Vision Conference, 1988 — the Harris corner detector of §22.
- D. G. Lowe, *Distinctive Image Features from Scale-Invariant Keypoints*,
  IJCV, 2004 — SIFT; the ratio test of §22.
- D. DeTone, T. Malisiewicz, A. Rabinovich, [*SuperPoint: Self-Supervised
  Interest Point Detection and Description*](https://arxiv.org/abs/1712.07602),
  CVPR 2018 — the learned keypoints of §22.
- P. Lindenberger, J. Sarlin, M. Pollefeys, et al., [*LightGlue: Local
  Feature Matching at Light Speed*](https://arxiv.org/abs/2304.12343), ICCV
  2023 — the learned matcher of §22.

**Textbooks and references**

- R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed.,
  Springer, 2022 — warping, interpolation, homographies, and Kalman basics
  (the best "go deeper" pointer for a student).
- R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
  2nd ed., Cambridge University Press, 2004 — the projective geometry
  (homographies, camera models) behind Part 2.
- R. Hastie et al. (for reference only), *The Elements of Statistical
  Learning*, Springer — background on the supervised-learning framing of
  Parts 3–4.

**Software documentation**

- OpenCV API reference — `cv2.phaseCorrelate`, `cv2.findTransformECC`,
  `cv2.estimateAffinePartial2D` (readily runnable implementations of
  §19–§21).
- PyTorch docs — `torch.nn.functional.grid_sample` (the `align_corners`
  semantics of §11.4 and the differentiable warp of §15).

---

# Appendix D: Math rendering notes

This document targets **GitHub's native Markdown math rendering** (MathJax
with KaTeX options).  The conventions used, and why:

- **Inline math** uses the `$ … $` delimiter where the expression is
  unambiguous, and the GitHub-supported `$ \`…\` $` (dollar-backtick) form
  where the expression contains characters that would collide with Markdown
  syntax (underscores, asterisks, backticks).  Both render identically to
  math.  This document mostly uses the second form for inline expressions
  (writing `$ \`θ\` $` for `θ`) because the mathematical notation is dense
  with underscores and slashes.
- **Display math** uses `$$ … $$` on their own lines.  Display blocks do
  *not* use the `\large` prefix that older versions of this document carried
  — GitHub's renderer treats `\large` as a control sequence and shows it
  verbatim (raw text), and it historically caused matrix rows to render all
  on one line.
- **Matrix row separators in display math** must be written as four
  backslashes in the raw Markdown file: `\\\\`.  GitHub's Markdown layer
  consumes one level of backslash escaping, so the four-character sequence
  in the file reaches KaTeX as the standard two-character LaTeX row
  separator `\\`.  Writing only two backslashes in the file causes KaTeX to
  see a single backslash — which is not a valid row separator — and the
  matrix collapses onto one line.  Every `bmatrix`/`matrix`/vector in this
  document follows this rule (§2, §7, §8, §13).
- GitHub strips the delimiter markers before handing the content to the
  renderer, and KaTeX accepts the standard `\\` row separator for `bmatrix`
  environments, so the matrix markup renders as intended.

The math rendering was verified by a simulation of GitHub's MathJax string
parsing on the document's delimiters before publication.
