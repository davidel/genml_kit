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
R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
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
M = \begin{bmatrix} s\cos\theta & -s\sin\theta & t_x \\ s\sin\theta & s\cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix}
$$

where:

- the top-left 2×2 block is the *linear part* $`sR_\theta`$;
- the last column is the translation;
- the bottom row $`(0, 0, 1)`$ makes the map affine (points map to
  points, lines to lines).

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
s = \sqrt{a^2 + c^2}, \qquad \theta = \operatorname{atan2}(c, a)
$$

where:

- $`\operatorname{atan2}`$ is **mandatory**: $`\arccos`$-based readout
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
H = K \begin{bmatrix} r_x & d_x & t_x \\ r_y & d_y & t_y \\ f_x & f_y & t_z \end{bmatrix},
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
\mathcal{L}_{\mathrm{sup}} = \mathcal{L}_{\mathrm{mce}} + \lambda_s \lVert \log \hat{s} - \log s \rVert_1 + \lambda_\theta \bigl\lvert \mathrm{wrap}(\hat{\theta} - \theta) \bigrigrvert + \lambda_c\, \mathrm{SmoothL1}(\hat{r}, \rho)
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

### 5.2 Fourier–Mellin (rotation + scale)

Log-polar resampling of the log-magnitude spectra turns rotation and
scale into translations, which §5.1 then estimates:

$$
\large
\mathcal{F}\bigl[\log |F_a|\bigr] \to (\rho, \phi) \text{ space}, \qquad
\text{rotation} \to \phi\text{-shift}, \quad \text{scale} \to \rho\text{-shift}
$$

where:

- the estimate is refined by ECC (§5.3) after the log-polar init;
- this is the classic Fourier–Mellin oracle used as baseline 1.

### 5.3 Lucas–Kanade / ECC iteration

Both refine a warp $`M`$ by iterated linearization of the photometric
error:

$$
\large
\Delta p = \left( J^{\top} J \right)^{-1} J^{\top} \bigl(I_b(M(x; p)) - I_a(x)\bigr), \qquad
M \leftarrow M \circ M(\Delta p)^{-1}
$$

where:

- $`J`$ — the Jacobian of the warped image w.r.t. the parameterization
  $`p`$ of $`M`$;
- ECC additionally normalizes both images (gradient preconditioning),
  which makes it robust to exposure/contrast differences;
- this is what `cv2.findTransformECC` minimizes and what the plan's ECC
  oracle runs to convergence.

### 5.4 Keypoint pipeline + RANSAC

The strongest classical-modern oracle: SuperPoint keypoints and
LightGlue matches, then ratio test, then RANSAC over similarity fits,
then a final Umeyama on inliers.  The RANSAC iteration count for
inlier ratio $`w`$, model $`m = 2`$ point pairs, target probability
$`p_{\mathrm{succ}}`$:

$$
\large
K = \frac{\log(1 - p_{\mathrm{succ}})}{\log(1 - w^{m})}
$$

where:

- $`m = 2`$ — two point pairs determine a 2-D similarity;
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
| $`\rho`$ | GT residual (irreducible) | §2.4 |
| $`F_a, F_b`$ | encoder feature maps | §3.1 |
| $`\Sigma, U, V, D`$ | scatter matrix, SVD factors, guard | §3.2 |
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
- Evangelidis & Psarakis, [Parametric Image Alignment Using Enhanced
  Correlation Coefficient](https://ieeexplore.ieee.org/document/4359316),
  PAMI 2008 — the ECC objective behind the ECC oracle (§5.3).
- Reddy & Anbarasu, the Fourier–Mellin log-polar registration line of
  work (§5.2).
