# Visual Odometry Front-End — Algorithms & Math

The drone visual-odometry front-end consumes image pairs from an oblique
(~45°), body-mounted, unstabilized camera and outputs the 2-D similarity
transform between frames — image-plane rotation $`\theta`$, uniform scale
$`s`$, and translation $`(t_x, t_y)`$ — plus a confidence score.  This
document is the **single place where the math lives**: code docstrings
stay brief and point here.

- Source: integrated from the *ILOC Study — Parametric Image Alignment*
  (chapters 0–7), extended with drone-specific data-generation and EKF
  material (tagged *new* below).
- Numeric examples in this document are the same ones validated by the
  test suite (`tests/test_geometry_similarity.py`,
  `tests/test_vo_pairs.py`, `tests/test_vo_similar.py`,
  `tests/test_vo_training.py`), so doc and tests cannot drift.

## Contents

- [1. Problem, notation, parameterization](#1-problem-notation-parameterization)
- [2. Data-generation math](#2-data-generation-math)
  - [2.5 Warping mechanics: how B is rendered from A](#25-warping-mechanics-how-b-is-rendered-from-a)
- [3. Estimator math](#3-estimator-math)
- [4. Losses and metrics](#4-losses-and-metrics)
- [5. Classical baselines](#5-classical-baselines)
- [6. EKF consumption contract](#6-ekf-consumption-contract)
- [Appendix A. Symbol table](#appendix-a-symbol-table)
- [Appendix B. Failure modes and shortcuts](#appendix-b-failure-modes-and-shortcuts)

## 1. Problem, notation, parameterization

Source: ILOC Study §1.1–1.7.

### 1.1 The transform

A 2-D **similarity** transforms a point $`x`$ into

$$
\large
x' = s\,R_\theta\,x + t, \qquad
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\\\ \sin\theta & \cos\theta \end{bmatrix}
$$

where:

- $`s > 0`$ — the uniform scale (never negative; a negative scale is a
  mirror, a different transform family);
- $`R_\theta`$ — the rotation by $`\theta`$ radians, counter-clockwise in
  a standard image coordinate frame (x right, y **up**; image row order
  flips this visually — see the worked example below);
- $`t = (t_x, t_y)^{\top}`$ — the translation applied after rotation.

In homogeneous coordinates the same transform is a 3×3 matrix

$$
\large
M = \begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\\\ s\sin\theta & s\cos\theta & t_y \\\\ 0 & 0 & 1 \end{bmatrix}
$$

where:

- the top-left 2×2 block is the *linear part* $`sR_\theta`$;
- the last column is the translation;
- the bottom row $`(0, 0, 1)`$ makes the map affine (points map to
  points, lines to lines).

Homogeneous coordinates pay off immediately in two ways:

- **Composition.** Chaining transforms collapses to a single matrix
  product.  If $`M_1`$ is applied first and $`M_2`$ second, the combined
  map is $`M = M_2 M_1`$: the *last-applied* factor appears *leftmost*.
  Order matters — $`M_2 M_1 \ne M_1 M_2`$ in general (rotate-then-scale
  is not scale-then-rotate unless the scale is isotropic *and* centered).
  In the warp code this is the difference between $`M \circ M^{-1}`$
  and $`M^{-1} \circ M`$, so the composition order is pinned and
  unit-tested.
- **Center of rotation.** The formula above rotates about the **origin**
  $`(0, 0)`$ (the top-left corner of the image).  A center-based
  rotation about $`c = (W/2, H/2)`$ is the conjugation
  $`M_c = T_c\,M\,T_{-c}`$: translate the center to the origin, apply
  the similarity, translate back.  The pixel-grid convention (whether the
  center is $`W/2`$ or the exact half-pixel position `(W-1)/2`) is a
  source of the half-pixel bias discussed in §2.5, so it is fixed
  once per dataset and recorded in the metadata.

### 1.2 Worked example: the 90° rotation (pins every sign)

Apply $`\theta = 90^\circ`$, $`s = 1`$, $`t = 0`$ to the three canonical
points.  With $`\cos 90^\circ = 0`$ and $`\sin 90^\circ = 1`$:

$$
\large
\begin{bmatrix} 1 \\ 0 \end{bmatrix} \mapsto \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \qquad
\begin{bmatrix} 0 \\ 1 \end{bmatrix} \mapsto \begin{bmatrix} -1 \\ 0 \end{bmatrix}, \qquad
\begin{bmatrix} 1 \\ 1 \end{bmatrix} \mapsto \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

where:

- $`+x`$ maps to $`+y`$ — a **counter-clockwise** quarter turn in the
  standard frame;
- the sign convention is pinned by
  `tests/test_geometry_similarity.py::test_quarter_turn_sign_convention`;
- the mounting/sign calibration on the drone (plan §9.1) uses this exact
  case: a hand-computed 90° flight turn must produce $`\theta \approx
  +\pi/2`$ or $`\approx -\pi/2`$ depending on the camera's orientation,
  and that determination is done once, documented, and never re-derived.

### 1.3 Parameterization used everywhere in the code

$$
\large
(\log s,\ \theta,\ t) \quad \text{with} \quad s = e^{\log s}, \quad \theta \in (-\pi, \pi]
$$

where:

- $`\log s`$ — scale is stored and predicted in log space, so the network
  can additively combine scales and never predicts $`s \le 0`$;
- $`\theta`$ — radians in $`(-\pi, \pi]`$; angle *differences* are always
  wrapped with $`\mathrm{wrap}(\delta) = (\delta + \pi) \bmod 2\pi - \pi`$
  before use in losses or metrics;
- $`t`$ — pixels, normalized by image size at the network boundary when
  the model regresses it.

### 1.4 Reading parameters back from a matrix

Given a matrix with linear part $`A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}`$:

$$
\large
s = \sqrt{a^2 + c^2}, \qquad \theta = \mathrm{atan2}(c, a)
$$

where:

- $`\mathrm{atan2}`$ is **mandatory**: $`\arccos`$-based readout
  maps $`\theta`$ and $`-\theta`$ to the same value and silently folds
  negative rotations onto positive ones;
- $`s = \sqrt{a^2 + c^2}`$ (equivalently $`\tfrac12\log(a^2+c^2)`$ in log
  space) because $`a = s\cos\theta`$ and $`c = s\sin\theta`$.

This is implemented in
`genml_kit.geometry.similarity.params_from_matrix` and tested by
`test_params_matrix_round_trip`.

### 1.5 The metric: mean corner error (MCE)

$$
\large
\mathrm{MCE} = \frac{1}{N} \sum_{i=1}^{N} \left\lVert s R_\theta\, p_i + t - q_i \right\rVert_2
$$

where:

- $`\{p_i\}`$ — the reference points (by default the 4 image corners);
- $`\{q_i\}`$ — the target points (GT corners of the pair, or the
  network's own corner predictions during training);
- the MCE is the *only* headline accuracy metric: it is in pixels, it is
  directly interpretable, and it weights all four parameters together
  exactly as the downstream EKF consumes them.

Typical per-parameter magnitudes (ILOC §1.7): a 0.5 px corner error at
the image border corresponds to roughly 0.5% scale error, 0.3° rotation,
or 0.5 px translation — the MCE folds these into one number.

## 2. Data-generation math

Source: *new — review before merging*.  Extends ILOC §5 (synthetic
pairs) with the oblique-camera geometry the drone mission needs.

### 2.1 The oblique camera model

A pinhole camera at position $`c`$ with orientation $`R_{cw}`$ (camera
axes as **rows**, world→camera) projects a world point $`p`$ to

$$
\large
\tilde{p}_{cam} = R_{cw}(p - c), \qquad \tilde{u} = K\,\tilde{p}_{cam}
$$

where:

- $`K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}`$
  — the intrinsics ($`f_x, f_y`$ focal lengths in pixels, $`(c_x, c_y)`$
  the principal point);
- $`\tilde{\cdot}`$ — homogeneous coordinates;
- the camera frame is built from the optical axis
  $`f = (\cos\psi\cos\phi,\ \sin\psi\cos\phi,\ -\sin\phi)`$ (heading
  $`\psi`$, pitch $`\phi`$ below horizontal),
  $`r = \mathrm{norm}(f \times (0,0,1)^{\top})`$ (right) and
  $`d = f \times r`$ (down), stacked into $`R_{cw}`$.

### 2.2 The ground-to-image homography

Because the ground is a plane $`z = 0`$, the mapping from ground
coordinates $`(x, y)`$ to pixels is an exact **homography**:

$$
\large
H = K \begin{bmatrix} r_x & d_x & t_x \\\\ r_y & d_y & t_y \\\\ f_x & f_y & t_z \end{bmatrix},
\qquad t = -R_{cw}\,c
$$

where:

- the first two columns are $`K`$ times the camera-frame images of the
  world x and y axes;
- the third column is $`K t`$ with $`t = -R_{cw} c`$;
- frame B is rendered from frame A's tile through
  $`H_{b\leftarrow a} = H_b H_a^{-1}`$ — the composed ground-to-image
  map of the moved camera;
- implemented in
  `genml_kit.datasets.vo_pairs.look_at_ground_h`; the projection of the
  world origin equals the principal point for a camera aimed at it
  (`test_level_camera_optical_axis_hits_ground_origin`).

### 2.3 GT similarity = best similarity fit to the homography

The network predicts a similarity, but the true motion between oblique
frames is a homography (foreshortening).  The GT similarity is therefore
**defined** as the best similarity approximation of the homography:

$$
\large
(\hat{s}, \hat{\theta}, \hat{t}) = \arg\min_{s, \theta, t} \frac{1}{4}\sum_{i=1}^{4} \left\lVert s R_\theta\, u_i + t - H u_i \right\rVert_2^2
$$

where:

- $`\{u_i\}`$ — the four image corners of frame A, mapped through the
  homography $`H`$;
- the minimizer is the closed-form Umeyama fit of §3.2 applied to
  $`(u_i) \to (H u_i)`$;
- implemented in `homography_to_similarity`.

### 2.4 The GT residual is irreducible

The fit leaves a residual

$$
\large
\rho = \frac{1}{4}\sum_{i=1}^{4} \left\lVert \hat{s} R_{\hat{\theta}}\, u_i + \hat{t} - H u_i \right\rVert_2
$$

where:

- $`\rho`$ is the **irreducible** part of the task error: a similarity
  has 4 DOF while a homography has 8, and the 4 missing DOF are exactly
  the foreshortening (projective) part.  No network output, however
  good, can remove it — which is why it is the *target* for the
  confidence head (§4.3) rather than something to minimize away;
- $`\rho \to 0`$ as the view approaches nadir ($`\phi \to 90^\circ`$)
  or as the height change and off-axis motion go to zero — asserted by
  `test_residual_shrinks_toward_nadir` and
  `test_gt_residual_zero_for_pure_similarity_motion`;
- $`\rho`$ grows with off-nadir angle, height delta, and distance of the
  content from the image center.

Degeneracy: at exactly $`\phi = 0`$ (horizontal camera) the ground plane
passes through the optical axis, $`\det H = 0`$, and the residual is
undefined — the generator never samples this regime.

### 2.5 Warping mechanics: how B is rendered from A

The data-generator renders $`B = \mathrm{warp}(A, M)`$ with a chosen
interpolation kernel and padding policy.  These choices are
**recorded in the dataset metadata** and fixed across train and deploy,
because the generator defines both what the network sees and which
shortcuts exist to exploit it.

**Forward vs backward map.**  Forward mapping iterates source pixels
and writes to $`M p`$: it leaves holes where no source lands and
fights aliasing.  Backward mapping iterates *output* pixels and
samples $`M^{-1} p`$: every output pixel gets exactly one value and
the code never needs a rasterizer.  All warps in this project use the
backward map (see the differentiable variant in §3.4).

**Interpolation kernels.**  The choice quantizes the ground truth: a
sub-pixel shift rendered with nearest-neighbor produces visible
stair-stepping that a smooth regressor must then un-learn.

| Kernel | Behavior | Notes |
|---|---|---|
| Nearest | value of the rounded source pixel | blocky; quantizes GT — bad for training a smooth regressor; fastest |
| Bilinear | weighted average of the 4 surrounding pixels | the default; mild low-pass blur, which is also what a real resampled photo looks like |
| Bicubic | 4×4 = 16 neighbors with cubic weights ($`a=-0.5`$) | sharper; can overshoot/ring at edges (values may exceed the local range) |
| Lanczos | windowed sinc over a larger window | sharpest of the four; also rings; slower |

**Padding policies.**  Pixels that sample outside the source domain
must be filled, and the choice is a visible cue:

- *Constant fill* (e.g. zero) — reveals rotation/scale by the
  triangular "wedges" at the borders; a network can learn to read the
  wedge geometry instead of the content (the wedge shortcut, Appendix B).
- *Mirror / reflect* — the natural anti-aliasing choice for
  rotation/scale; no wedge cue, but it fabricates content that is not
  in the source image.

**Library conventions and the half-pixel trap.**  Every library gets
the math right and the signs/centers differently:

- OpenCV `getRotationMatrix2D(center, angle, scale)`:
  - `angle` is in **degrees**, and positive values rotate
    **counter-clockwise** in its y-down layout — the *opposite sign*
    of this project's convention, so callers pass `-deg(θ)`;
  - the returned matrix is the **forward map**; `warpAffine` (without
    `WARP_INVERSE_MAP`) internally inverts it and backward-samples.
- The rotation center is the pixel-grid center, and the exact
  half-pixel position (`(W-1)/2` vs `W/2`) is a 0.5 px bias source.
  PyTorch `grid_sample` uses normalized coordinates with
  `align_corners=True` mapping pixel centers to integer coordinates
  ($`x_n = 2x/(W-1) - 1`$), while some tooling assumes
  `align_corners=False` (pixel *edges*); mixing the two conventions is
  a silent 0.5 px translation error.
- Fix one convention per dataset, record it in the metadata, and
  unit-test the GT round-trip (the 90° worked example of §1.2 is the
  canonical test).

## 3. Estimator math

Source: ILOC §3.3 (cost volume), §4.5 (Umeyama), §2.4/§3.4 (warping).

### 3.1 Correlation cost volume

Given feature maps $`F_a, F_b \in \mathbb{R}^{C \times h \times w}`$
from the shared encoder, the correlation score at displacement
$`(\delta_y, \delta_x)`$ and pixel $`(y, x)`$ is

$$
\large
\mathrm{score}(y, x; \delta_y, \delta_x) = \sum_{c=1}^{C} F_a[c, y, x]\, F_b[c, y + \delta_y, x + \delta_x]
$$

where:

- displacements run over $`|\delta_y|, |\delta_x| \le r`$ with
  $`r`$ the configured radius (default 6 at 1/8 resolution);
- the stack of all scores is the cost volume
  $`\in \mathbb{R}^{(2r+1)^2 \times h \times w}`$, ordered dy-major:
  displacement $`(\delta_y, \delta_x)`$ sits at channel
  $`(\delta_y + r)(2r+1) + (\delta_x + r)`$;
- out-of-frame partners contribute zero (plain zero padding —
  `torch.roll` would wrap content around and fabricate border matches);
- the implementation uses only pad, slice, multiply and sum, so an int8
  exporter sees plain conv-shaped work (`test_correlate_matches_direct_semantics`).

### 3.2 Closed-form batched Umeyama fit

Given point sets $`P = \{p_i\}`$ and $`Q = \{q_i\}`$ with optional
weights $`w_i`$, the similarity minimizing
$`\sum_i w_i \lVert s R p_i + t - q_i \rVert^2`$ is computed in closed
form.

**Step 1 — weighted centroids.**

$$
\large
\mu_p = \frac{\sum_i w_i p_i}{\sum_i w_i}, \qquad \mu_q = \frac{\sum_i w_i q_i}{\sum_i w_i}
$$

**Step 2 — scatter matrix and SVD.**

$$
\large
\Sigma = \sum_i w_i (p_i - \mu_p)(q_i - \mu_q)^{\top} = U \,\mathrm{diag}(\sigma_1, \sigma_2)\, V^{\top}
$$

**Step 3 — rotation with reflection guard.**  Let
$`D = \mathrm{diag}(1, \det(UV^{\top}))`$; then

$$
\large
R = V D\, U^{\top}, \qquad \det R = +1 \ \text{always}
$$

where:

- the guard flips the singular value of the smallest singular direction
  whenever $`\det(UV^{\top}) = -1`$, so the fit can never emit a
  mirrored similarity (`test_umeyama_never_emits_reflection`);
- for $`H = \Sigma`$, the correct factor order is $`R = VDU^{\top}`$;
  the transpose $`UDV^{\top}`$ fits the *inverse* correspondence and is
  the classic silent-failure bug.

**Step 4 — scale and translation.**

$$
\large
s = \frac{\sigma_1 + d\,\sigma_2}{\sum_i w_i \lVert p_i - \mu_p \rVert^2}, \qquad
t = \mu_q - s R\, \mu_p
$$

where:

- $`d = \det(UV^{\top})`$ — the same guard enters the scale;
- $`t`$ is computed from the centroids directly, without ever forming
  the forward matrix;
- gradients flow through the SVD, so the fit sits inside the training
  graph (the plan's closed-form head);
- the whole derivation is batched over $`B`$ in
  `genml_kit.geometry.similarity.umeyama_similarity`, checked against an
  independent dense least-squares solve in
  `test_umeyama_matches_brute_force_least_squares`.

### 3.3 The corner head: why regression on corners

The network does **not** regress $`(\log s, \theta, t)`$ directly.  It
predicts offsets $`\Delta_i`$ for the four canonical corners, and the
Umeyama solve of §3.2 turns
$`\{(u_i) \to (u_i + \Delta_i)\}`$ into the parameters:

$$
\large
(\log s, \theta, t) = \mathrm{Umeyama}\big(u_i,\ u_i + \Delta_i\big)
$$

where:

- the output is *always a valid similarity*: any rotation angle and any
  positive scale can be produced, but no invalid $`(\theta, s)`$
  combination exists (`test_network_output_is_valid_similarity`);
- corners give free interpretability — plotting predicted vs GT corners
  is a direct sanity check (`utils/image_dump`);
- this is the HomographyNet template (ILOC §3.6) ported to 4 DOF.

### 3.4 Differentiable backward-map warp

To resample image $`I`$ by the similarity $`M`$, every **output** pixel
$`p`$ samples the input at $`M^{-1} p`$ (backward map):

$$
\large
I'_M(p) = I\big(M^{-1} p\big)
$$

where:

- the backward map avoids holes and aliasing that a forward map
  produces;
- `torch.nn.functional.grid_sample(align_corners=True)` implements it,
  differentiable with respect to both $`I`$ and $`M`$;
- the invariant used by the tests: sampling the warped image at
  $`M p`$ reproduces the source content at $`p`$, up to the
  interpolation of two bilinear stages —
  `test_warp_similarity_matches_point_map`;
- pixel centers sit at integer coordinates; normalized coordinates are
  $`x_n = 2x / (W - 1) - 1`$ (this is what `align_corners=True` means).

## 4. Losses and metrics

Source: ILOC §6.2; drone-specific staging from plan §6.

### 4.1 Supervised losses

$$
\large
\mathcal{L}_{\mathrm{sup}} = \mathcal{L}_{\mathrm{mce}} + \lambda_s \lVert \log \hat{s} - \log s \rVert_1 + \lambda_\theta \bigl\lvert \mathrm{wrap}(\hat{\theta} - \theta) \bigr\rvert + \lambda_c\, \mathrm{SmoothL1}(\hat{r}, \rho)
$$

where:

- $`\mathcal{L}_{\mathrm{mce}}`$ — the corner reprojection error of
  §1.5 applied to the network's corner offsets (in pixels);
- $`\lVert \log \hat{s} - \log s \rVert_1`$ — the scale loss in log
  space, symmetric in relative-error terms;
- the angle loss is computed on the **wrapped** difference, so training
  never sees a $`2\pi`$ discontinuity;
- $`\hat{r}`$ — the predicted fit residual, supervised against the GT
  residual $`\rho`$ of §2.4; this is what makes the confidence
  *calibrated* rather than decorative;
- $`\lambda_s, \lambda_\theta, \lambda_c`$ — loss weights
  (`vo_losses`).

### 4.2 Staged photometric auxiliary

Once the supervised loss converges, a photometric consistency term is
staged in:

$$
\large
\mathcal{L}_{\mathrm{photo}} = 1 - \mathrm{zNCC}\bigl(I_a,\ I_b \circ M^{-1}\bigr)
$$

where:

- $`\mathrm{zNCC}`$ — zero-normalized cross-correlation over the valid
  (non-zero-padded) overlap;
- $`I_b \circ M^{-1}`$ — frame B resampled by the backward map of the
  predicted similarity (§3.4);
- the term is a **gate and auxiliary**, never the primary output: it is
  minimized only inside the overlap, is blind to scale about the
  correlation peak, and degenerates on texture-poor terrain;
- at inference the same quantity is the residual confidence gate
  (plan §8.5).

### 4.3 Metrics

- **MCE** (§1.5) — headline accuracy in pixels.
- $`\Delta \log s = |\log \hat{s} - \log s|`$ — relative scale error.
- $`\Delta\theta = |\mathrm{wrap}(\hat{\theta} - \theta)|`$ — wrapped
  angular error.
- **Accuracy-vs-confidence** — MCE of the subset above each confidence
  threshold; the operating threshold is chosen here (plan §7).
- All metrics are always sliced per terrain class, motion-range bin,
  augmentation level and AGL config (`eval_vo.evaluate_sliced`).

## 5. Classical baselines

Source: condensed ILOC §4–5.  Enough math to reproduce each oracle.

### 5.1 Phase correlation (translation)

$$
\large
(\Delta x, \Delta y) = \arg\max \ \mathcal{F}^{-1}\!\left[\frac{F_a \overline{F_b}}{|F_a \overline{F_b}|}\right]
$$

where:

- $`F_a, F_b`$ — the 2-D DFTs of the frames; the normalized cross-power
  spectrum collapses to a single peak at the translation;
- rotation/scale must be removed first (see §5.2) — phase correlation
  alone handles only translation.

Practical caveats:

- **Cyclic (wrap-around) assumption.**  The DFT treats the image as
  toroidal: content leaving the right edge re-enters from the left.  A
  translation wraps the peak (a shift of $-5$ px appears at $`N-5`$;
  read peaks in $`[-N/2, N/2)`$), and borders/padding create artifacts.
  A **Hann window** (a smooth cosine edge taper) or edge-mirroring
  before the FFT is the standard mitigation.
- **Illumination.**  The normalized cross-power spectrum divides by the
  magnitude at every frequency, so per-pixel gain/bias differences are
  cancelled — the method is photometrically robust, which is why it
  is used as the initialization stage before the ECC refinement of
  §5.3.
- **Peak sharpness vs FFT size.**  Zero-padding to a larger FFT size
  sharpens the peak and improves sub-pixel interpolation, but enlarging
  the cyclic domain also wraps real content across the border — the
  Hann window belongs *before* the zero-pad.

### 5.2 Fourier–Mellin (rotation + scale)

Log-polar resampling of the log-magnitude spectra turns rotation and
scale into translations, which §5.1 then estimates:

$$
\large
\mathcal{F}\bigl[\log |F_a|\bigr] \to (\rho, \phi) \text{ space}, \qquad
\text{rotation} \to \phi\text{-shift}, \quad \text{scale} \to \rho\text{-shift}
$$

where:

- the magnitude spectra $`|F_a|, |F_b|`$ are rotation-invariant (the
  Fourier magnitude discards phase, and rotation of the image rotates
  the magnitude spectrum by the same angle);
- taking the **logarithm** $`\rho = \log r`$ turns the radial scale
  factor into an additive shift, so *both* rotation and log-scale are
  translations in $`(\rho, \phi)`$ space and are read off with a second
  phase correlation;
- because the magnitude spectrum is discarded, only the *modulus* of
  translation is recoverable per-stage — the pipeline is:
  log-polar magnitude → phase correlation → $`(\hat{\theta}, \hat{s})`$ →
  de-rotate/de-scale B → phase correlation → $`\hat{t}`$;
- log-polar resampling needs interpolation at high radii where the
  sampling density drops, which is the source of the "coarse θ, s"
  weakness — the estimate is refined by ECC (§5.3) after the log-polar
  init;
- this is the classic Fourier–Mellin oracle used as baseline 1.

### 5.3 Lucas–Kanade / ECC iteration

Both refine a warp $`M`$ by iterated linearization of the photometric
error:

$$
\large
\Delta p = \left( J^{\top} J \right)^{-1} J^{\top} \bigl(I_b(M(x; p)) - I_a(x)\bigr), \qquad
M \leftarrow M \circ M(\Delta p)^{-1}
$$

Derivation (Gauss–Newton):

1. **Brightness constancy.**  The true warp satisfies
   $`I_b(M(x; p^*)) \approx I_a(x)`$: the same physical surface radiates
   the same intensity into both frames.
2. **Linearize.**  For the current estimate $`p`$, a first-order Taylor
   expansion of the residual $`r(p) = I_b(M(x; p)) - I_a(x)`$ gives
   $`r(p + \Delta p) \approx r(p) + J\,\Delta p`$, with
   $`J = \partial r / \partial p`$ the Jacobian of the warped image
   w.r.t. the warp parameters (image gradient composed with the warp
   Jacobian).
3. **Normal equations.**  Minimizing $`\lVert r + J\,\Delta p \rVert^2`$
   w.r.t. $`\Delta p`$ gives $`J^{\top} J\,\Delta p = -J^{\top} r`$,
   whose solution is the update above; the warp is then composed with
   the *inverse* update so it stays a forward map.
4. **Iterate** to convergence; each step re-linearizes about the new
   $`p`$.

where:

- $`J`$ — the Jacobian of the warped image w.r.t. the parameterization
  $`p`$ of $`M`$ (step 2);
- $`M \leftarrow M \circ M(\Delta p)^{-1}`$ — compose with the inverse
  update so the map remains forward (step 3);
- ECC (Evangelidis & Psarakis 2008) *replaces the raw intensity
  residual* with a normalized-correlation objective (gradient
  preconditioning), which absorbs per-frame gain/bias and makes it
  robust to exposure/contrast differences;
- `cv2.findTransformECC` minimizes exactly this ECC objective and is the
  classical accuracy gold standard for the 4-DOF refine.

Practical properties (ILOC §5.3):

- **Small convergence basin.**  LK converges only for small
  $`\Delta p`$ (a few degrees, a few percent scale); outside it the
  linearization is wrong and it diverges.  Pyramid LK (coarse-to-fine)
  enlarges the basin, and seeding with the Fourier–Mellin estimate of
  §5.2 gives the same benefit with one pyramid level.
- **Aperture problem, quantified.**  Pixels with zero gradient
  contribute nothing ($`\nabla I_b = 0 \Rightarrow`$ no equation);
  texture-poor images make $`J^{\top}J`$ rank-deficient.  LK needs
  corners/edges — the same geometry keypoints (§5.4) and the
  learned confidence head rely on.
- **Photometric robustness** is why the scheme is phase correlation
  (photometrically robust, coarse) → ECC (photometrically robust,
  precise): both tolerate exposure differences, and ECC finishes the
  job the log-polar init starts.

### 5.4 Keypoint pipeline + RANSAC

The strongest classical-modern oracle: SuperPoint keypoints and
LightGlue matches, then ratio test, then RANSAC over similarity fits,
then a final Umeyama on inliers.

The pipeline components (ILOC Ch. 4):

- **Keypoint detector** — a saliency map $`\mathcal{S}(x)`$ over the
  image; local maxima above a threshold are keypoints.  Classical: Harris
  corners, SIFT.  Learned: SuperPoint's detector head (a CNN trained
  with a self-supervised homography-consistency objective).
- **Descriptor** — a fixed-length vector $`d \in \mathbb{R}^C`$
  summarizing the patch around each keypoint, built so the *same*
  physical point seen from slightly different poses gets nearly the
  same vector.  Classical: SIFT (gradient histograms, 128-D), ORB
  (binary strings).  Learned: SuperPoint's descriptor head.
- **Matcher** — pairs keypoints whose descriptors are closest
  (nearest-neighbor in cosine/L2), optionally with mutual-nearest
  checks and a ratio test to reject ambiguous matches.  Output:
  correspondences $`\{(p_i, q_i)\}`$.  LightGlue is a learned matcher
  that additionally prunes with geometric context (a candidate match
  inconsistent with its neighbors is suppressed) and stops attending
  once enough confident matches exist — compared to SuperGlue it is
  lighter, faster, and adaptive.

The RANSAC iteration count for inlier ratio $`w`$, model $`m = 2`$
point pairs, target probability $`p_{\mathrm{succ}}`$:

$$
\large
K = \frac{\log(1 - p_{\mathrm{succ}})}{\log(1 - w^{m})}
$$

where:

- $`m = 2`$ — two point pairs determine a 2-D similarity; the count
  grows logarithmically with $`1/(1-p_{\mathrm{succ}})`$ and
  polynomially with $`1/w`$;
- the final inlier refit is exactly the Umeyama of §3.2, which is why
  this baseline shares its algebra with the learned model.

## 6. EKF consumption contract

Source: *new — review before merging*.  Sensor-agnostic; the concrete
state instantiation (planar vs 6-DOF, IMU/altimeter wiring) is the
flight stack's business.  This section fixes only the measurement
contract.

### 6.1 Generic linearized KF

$$
\large
x_{k+1} = F_k x_k + w_k, \qquad w_k \sim \mathcal{N}(0, Q_k)
$$

$$
\large
z_k = H_k x_k + v_k, \qquad v_k \sim \mathcal{N}(0, R_k)
$$

where:

- $`x_k`$ — the filter state (instantiation is up to the flight stack);
- $`F_k`$ — the state-transition matrix integrated over the frame
  interval (gyro/accelerometer propagate the state between VO
  measurements);
- $`Q_k`$ — process noise, dominating when the platform maneuvers;
- $`z_k`$ — the measurement vector; the VO front-end contributes the
  inter-frame increment $`(\theta, s, t)`$ mapped into this vector;
- $`H_k`$ — the measurement Jacobian
  $`H_k = \partial h / \partial x \big|_{\hat{x}_k}`$;
- $`R_k`$ — the measurement noise **derived from the confidence**
  (§6.2), not a constant.

### 6.2 Confidence to measurement-noise mapping

$$
\large
R_k = R_{\min} + \alpha\, \hat{r}_k^2\, I
$$

where:

- $`\hat{r}_k`$ — the predicted fit residual (the confidence head's
  output, calibrated by §4.1 against the GT residual $`\rho`$);
- $`R_{\min}`$ — the noise floor for a perfect fit (quantization,
  interpolation);
- $`\alpha \hat{r}_k^2`$ — the quadratic penalty: twice the residual
  means four times the noise, so bad frames fade out smoothly instead
  of being hard-rejected;
- the mapping is monotone and linear in the *variance*, which is the
  statistically correct domain.

### 6.3 Update and gating

$$
\large
y_k = z_k - H_k \hat{x}_k, \qquad
S_k = H_k P_k H_k^{\top} + R_k, \qquad
K_k = P_k H_k^{\top} S_k^{-1}
$$

$$
\large
\hat{x}_k^{+} = \hat{x}_k + K_k y_k, \qquad
P_k^{+} = (I - K_k H_k) P_k
$$

where:

- $`y_k`$ — the innovation; $`S_k`$ — its predicted covariance;
- $`K_k`$ — the Kalman gain blending prediction and measurement;
- **gating**: if $`y_k^{\top} S_k^{-1} y_k > \chi^2_{\mathrm{max}}`$
  (a Mahalanobis gate on the innovation) the update is skipped and the
  filter dead-reckons on $`F_k`$ alone — this is the plan's "confidence
  low → ECC fallback → dead-reckon" ladder expressed in filter terms.

## Appendix A. Symbol table

| Symbol | Meaning | Defined in |
|---|---|---|
| $`s, \log s`$ | uniform scale, log-scale | §1.1, §1.3 |
| $`\theta`$ | rotation, radians, $`(-\pi, \pi]`$ | §1.1 |
| $`t`$ | translation (px) | §1.1 |
| $`M`$ | homogeneous similarity matrix | §1.1 |
| $`u_i, p_i, q_i`$ | image corners, source/target points | §1.5, §2.3 |
| $`K, R_{cw}, c`$ | intrinsics, camera rotation, center | §2.1 |
| $`H`$ | ground→image homography | §2.2 |
| $`\rho`$ | GT residual (irreducible); *also* log-polar radius in §5.2 | §2.4, §5.2 |
| $`F_a, F_b`$ | encoder feature maps | §3.1 |
| $`\Sigma, U, V, D`$ | scatter matrix, SVD factors, guard | §3.2 |
| $`J, \Delta p`$ | warped-image Jacobian, parameter update | §5.3 |
| $`w, p_{\mathrm{succ}}, K`$ | RANSAC inlier ratio, success prob., iteration count | §5.4 |
| $`\hat{r}`$ | predicted fit residual (confidence) | §4.1, §6.2 |
| $`x_k, P_k, Q_k, R_k`$ | filter state/covariances | §6.1 |

## Appendix B. Failure modes and shortcuts

Source: ILOC Ch. 7 glossary rows relevant to this front-end.

| Failure / shortcut | What it looks like | Countermeasure |
|---|---|---|
| Wedge shortcut | model reads constant-fill padding wedges instead of content | coverage checks in the generator (`test_frames_fully_covered`) |
| Zoom-to-fill / $`s_{\mathrm{cover}}`$ | scale that hides rotation-exposed corners | GT residual + overlap mask in metadata |
| Aperture problem | flat regions look identical under any small transform | confidence head trained on $`\rho`$; MCE slices per terrain |
| Mirror fit | negative-determinant linear part | reflection guard §3.2 (tested) |
| Angle folding | $`\theta`$ and $`-\theta`$ merged by arccos readout | atan2 readout §1.4 (tested) |
| Roll wrap in correlation | border content wraps and fabricates matches | zero-padding correlation §3.1 (tested) |

## References

- *ILOC Study — Parametric Image Alignment*, chapters 0–7 — the source
  study this document integrates (kept outside version control).
- DeTone et al., [Homography Estimation with Deep Learning
  (HomographyNet)](https://arxiv.org/abs/1606.03798) — the corner-offset
  + closed-form-solve head template.
- Umeyama, [Least-Squares Estimation of Transformation Parameters
  Between Two Point Patterns](https://ieeexplore.ieee.org/document/88573),
  PAMI 1991 — the closed-form fit of §3.2.
- Kabsch, [A Solution for the Best Rotation to Relate Two Sets of
  Vectors](https://onlinelibrary.wiley.com/doi/abs/10.1107/S0567739476001873),
  Acta Crystallographica A 1976 — the `s = 1` (rigid) special case of
  the same solve.
- DeTone et al., [SuperPoint: Self-Supervised Interest Point Detection
  and Description](https://arxiv.org/abs/1712.07629), CVPRW 2018 — the
  keypoint/descriptor head behind the §5.4 oracle.
- Lindenberger et al., [LightGlue: Local Feature Matching at Light
  Speed](https://arxiv.org/abs/2306.13643), ICCV 2023 — the matcher
  behind the §5.4 oracle.
- Evangelidis & Psarakis, [Parametric Image Alignment Using Enhanced
  Correlation Coefficient](https://ieeexplore.ieee.org/document/4359316),
  PAMI 2008 — the ECC objective behind the ECC oracle (§5.3).
- Lucas & Kanade, [An Iterative Image Registration Technique with an
  Application to Stereo Vision](https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf),
  IJCAI 1981 — the brightness-constancy Gauss–Newton derivation of
  §5.3.
- Kuglin & Hines, "The Phase Correlation Image Alignment Method", Proc.
  IEEE Int. Conf. on Cybernetics and Society, New York, 1975, pp.
  163–165 — the original phase-correlation formulation of §5.1.
- Reddy & Chatterji, [An FFT-Based Technique for Translation, Rotation,
  and Scale-Invariant Image Registration](https://ieeexplore.ieee.org/document/506761),
  IEEE Transactions on Image Processing 1996 — the Fourier–Mellin
  log-polar registration behind §5.2.
