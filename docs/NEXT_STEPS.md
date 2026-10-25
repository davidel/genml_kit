# genml_kit — Roadmap / Next Steps

This document inventories the AI/ML topics that `genml_kit` does **not** yet
cover and ranks them by **Impact vs Complexity**. It is a planning artifact: it
describes *what* to build and *how it should slot into the existing
architecture*, not a commitment to implement everything.

> Status: proposal — nothing here is implemented. Review before acting.

## What genml_kit covers today

| Domain | Coverage |
|---|---|
| Image classification | Pluggable heads (`training/classifiers/`), XGBoost pipeline, TTA, metrics/eval |
| Semantic segmentation | UNet / FPN / linear decoders, Dice-CE loss, seg metrics |
| Visual odometry | Photometric + supervised, 2-D similarity algebra, differentiable Umeyama |
| Reinforcement learning | Model-free single-agent: DQN, PPO, SAC (gymnasium) |
| Self-supervised pre-training | BYOL, DINO, SimMIM, I-JEPA, SupCon / contrastive |
| PEFT / fine-tuning | LoRA via `peft`, freeze, LLRD, optim/scheduler factory |
| Infra | 4 registries (models/methods/pipelines/losses), unified `genml-kit-train` / `genml-kit-infer`, gradient monitor, dataset ensemble + samplers, checkpoint I/O (S3/GCS/R2), TensorBoard |

## Prioritization matrix

Scoring: **Impact** and **Complexity** are on a 1–5 scale. **Priority** =
`Impact − Complexity` (higher = do sooner). "Effort" is a rough engineering
estimate for a first, useful cut.

| Rank | Topic | Impact | Complexity | Priority | Effort |
|-----:|-------|:------:|:----------:|:--------:|:------:|
| 1 | Knowledge distillation | 5 | 2 | **3** | M |
| 2 | Object detection / instance segmentation | 5 | 3 | **2** | L |
| 3 | Learned depth / stereo / optical flow | 4 | 2 | **2** | M |
| 4 | Anomaly detection / OOD / calibration | 4 | 2 | **2** | S–M |
| 5 | Diffusion / VAE / GAN generative models | 5 | 4 | **1** | L |
| 6 | Experiment tracking + HPO (W&B / Optuna) | 3 | 2 | **1** | S |
| 7 | Interpretability + adversarial robustness | 3 | 2 | **1** | S–M |
| 8 | Vision-language / multimodal (CLIP-style) | 4 | 4 | **0** | L |
| 9 | Alignment / post-training (reward model, DPO) | 4 | 4 | **0** | M–L |
| 10 | Audio / video / time-series | 4 | 4 | **0** | L |
| 11 | Continual / meta / few-shot + domain adaptation | 3 | 3 | **0** | M |
| 12 | 3-D neural fields (NeRF / Gaussian splatting) + SLAM | 3 | 5 | **-2** | XL |

**Reading of the table**

- **Tier 1 — Quick wins (ranks 1–4):** high value, mostly reuse existing
  components (decoders, geometry, image pipelines). Good candidates for the next
  one or two milestones.
- **Tier 2 — Strategic bets (ranks 5–7):** large capability gains but either
  heavy compute or new dependency surface. Sequence after Tier 1.
- **Tier 3 — Long-term (ranks 8–12):** each opens a whole new modality or
  research axis; treat as its own project.

---

## Tier 1 — Quick wins

### 1. Knowledge distillation

**Why.** LoRA is the only "make it smaller/deployable" path today. Distillation
is what turns the research harness into a shippable one.

**Reuse.** `training/model_utils.py` already extracts features/freezes
encoders; `training/trainer.py` gives the loop; classifiers are pluggable.

**Proposed design**

- New method `genml_kit/methods/distillation.py` registered in `METHODS`, with a
  `DistillLoss` (logit KD + feature KD + optional relational KD) in
  `genml_kit/losses/distillation.py`.
- A teacher/student pairing driven by CLI flags
  (`--teacher_model`, `--kd_alpha`, `--kd_temperature`), mirroring the LLRD
  flag style in `training/optim_factory.py`.

**Deliverables.** `methods/distillation.py`, `losses/distillation.py`, tests,
README section.

**Risks.** None outstanding — the losses and the method shipped (see the Work
Log).

### 2. Object detection / instance segmentation

**Why.** Only *semantic* segmentation exists; the project has no boxes. This is
the most conspicuous missing dense-prediction task.

**Reuse.** `models/decoders/` (FPN already there), `seg_model_wrapper/`,
`pipelines/images.py`, `datasets/seg_folder.py` (extend to box targets).

**Proposed design**

- New pipeline `genml_kit/pipelines/detection.py` (image + boxes/labels blob),
  new method `methods/detection.py` with DETR-style set-prediction loss
  (Hungarian matching) and optional mask head for instance seg.
- New decoders registered in `DECODERS`: `detr_head`, `mask_head`.
- Dataset adapter for COCO-style JSON → reuse `datasets/ensemble.py` patterns.

**Deliverables.** detection pipeline + method + decoders + dataset adapter +
mAP metrics in `training/metrics.py`, tests, README section.

**Risks.** mAP metric implementation cost; DETR training is slow — start with
a frozen-backbone linear probe recipe.

### 3. Learned depth / stereo / optical flow

**Why.** `vo/` has strong geometric scaffolding but no *learned dense* methods.
Natural adjacency to existing visual-odometry work.

**Reuse.** `geometry/similarity.py` (warps, Umeyama), `vo_pairs` dataset, VO
pipelines, photometric loss machinery.

**Proposed design**

- New pipeline `pipelines/geometry_dense.py` producing stereo pairs /
  frame pairs; new method `methods/flow_depth.py` with photometric +
  smoothness + (optional) self-supervised losses.
- New decoders `disp_head` / `flow_head` registered in `DECODERS`.

**Deliverables.** pipeline + method + heads + eval metrics (end-point error,
abs-rel depth), tests, README + `vo/README.md` cross-link.

**Risks.** Requires stereo/sequence data; keep the synthetic `vo_pairs`
generator usable as a smoke-test corpus.

### 4. Anomaly detection / OOD / calibration

**Why.** The medical/industrial direction (`scdiag/`) needs "is this input
in-distribution / how confident am I?" — currently absent.

**Reuse.** Feature extractors in `model_utils.py`; scoring can wrap any method.

**Proposed design**

- New `genml_kit/eval/ood.py`: Mahalanobis / kNN / energy / MSP scores,
  plus calibration (temperature scaling, ECE).
- Optional one-class heads (`methods/anomaly.py`) — PatchCore-style memory bank.

**Deliverables.** `eval/ood.py`, `training/calibration.py`, CLI flags to emit
OOD + calibration reports in `train_reporting.py`, tests.

**Risks.** Low; mostly additive analysis code. Main cost is defining the
in-distribution reference split convention.

---

## Tier 2 — Strategic bets

### 5. Diffusion / VAE / GAN generative models

**Why.** The largest capability gap for a "general ML toolkit". No
DDPM/DDIM, latent diffusion, or controllable synthesis today.

**Reuse.** `models/decoders/unet.py` (adapt to a 2-D UNet with timestep
embedding), image pipelines, EMA utilities already used by BYOL/DINO.

**Proposed design**

- New package `genml_kit/models/diffusion/` (UNet2D, noise schedule, sampler)
  and `methods/diffusion.py` (denoising MSE + optional v-prediction).
- New pipeline `pipelines/diffusion.py` (image + optional conditioning).
- Sampler CLI `genml-kit-sample` with DDIM/DDPM, classifier-free guidance.

**Deliverables.** diffusion model + method + pipeline + sampler CLI + FID
scaffolding, tests, README section.

**Risks.** Compute-heavy; EMA/schedule correctness; FID needs a reference
statistics pipeline. Sequence after Tier 1.

### 6. Experiment tracking + HPO

**Why.** Only TensorBoard is wired. No W&B/MLflow, no Optuna/Hydra.

**Reuse.** `train_reporting.py` and the logging module are the right seams.

**Proposed design**

- Thin tracker abstraction in `genml_kit/io/tracking.py` with a TensorBoard
  backend (existing) + optional W&B/MLflow backends behind extras.
- `genml-kit-sweep` entry point wrapping Optuna with the existing parser so
  trials override `train.py` args.

**Deliverables.** `io/tracking.py`, optional extras, sweep CLI, tests.

**Risks.** Low. Keep backends optional to avoid growing core deps.

### 7. Interpretability + adversarial robustness

**Why.** No attribution/saliency and no robustness testing — both expected in
medical/industrial settings.

**Reuse.** `training/grad_monitor.py` already hooks gradients; reuse that
mechanism for grad-CAM.

**Proposed design**

- `genml_kit/eval/attribution.py`: grad-CAM, integrated gradients, attention
  rollout (the transformer utils already exist).
- `genml_kit/eval/robustness.py`: FGSM/PGD attack + accuracy-vs-epsilon curve,
  plus common-corruption robustness.

**Deliverables.** two eval modules + CLI flags + report rows in
`train_reporting.py`, tests.

**Risks.** Low.

---

## Tier 3 — Long-term / new modalities

### 8. Vision-language / multimodal (CLIP-style)

**Why.** No image-text contrastive, captioning, or VLM eval. A lone `caption`
key exists in `hf_proxy.py` but no NLP-side modelling.

**Reuse.** `losses/contrastive.py` + SupCon machinery generalize directly to
image-text pairs; `HFDatasetProxy` already reads HF multimodal datasets.

**Proposed design.** New `models/vl/` (dual encoder + projection), method
`methods/image_text.py` (InfoNCE over pairs), zero-shot classifier head, and a
captioning decoder head. New pipeline `pipelines/image_text.py`.

**Risks.** Tokenizer/processor handling; large data needs. Treat as its own
milestone.

### 9. Alignment / post-training (reward model, DPO)

**Why.** RL is environments/games only. No preference-based fine-tuning or
reward modelling despite a capable RL stack.

**Reuse.** `methods/rl_*` satisfy the same Method contract; loss registry can
hold DPO/IPO/BCO objectives; LoRA path is ready for PEFT-based alignment.

**Proposed design.** `methods/reward_model.py` (Bradley-Terry) and
`methods/dpo.py`; new pipeline `pipelines/preference_pairs.py`; reuse
existing checkpointing.

**Risks.** Assumes an LM backbone path — currently the harness is
vision-centric; would need a text head/processor.

### 10. Audio / video / time-series

**Why.** No speech (Whisper/wav2vec), no action recognition, no forecasting.

**Reuse.** `training/video_utils.py` (RL frames) is the only temporal code and
is a starting point for video pipelines.

**Proposed design.** Three separate pipelines (`audio.py`, `video.py`,
`timeseries.py`) + methods (CTC/ASR fine-tune, temporal action heads, sequence
forecasters). Each is effectively a new modality project.

**Risks.** High: new I/O, new metrics, new processors. Split into independent
efforts.

### 11. Continual / meta / few-shot + domain adaptation

**Why.** No EWC-style continual learning, MAML/prototypical few-shot, or
class-incremental / test-time adaptation (TTA today is inference-only).

**Reuse.** `training/tta.py` is the natural home to grow a test-time-training
(TTT) mode; samplers in `datasets/` support incremental task streams.

**Proposed design.** `methods/continual.py` (EWC/SI regularizers),
`methods/few_shot.py` (proto-net), and a TTT extension in `training/tta.py`.

**Risks.** Medium; mainly evaluation-protocol design (task boundaries, forgetting
metrics).

### 12. 3-D neural fields (NeRF / Gaussian splatting) + SLAM

**Why.** Real geometry + VO infrastructure exists, but no neural fields or SLAM
back-end — the furthest reach.

**Reuse.** `geometry/similarity.py` for camera poses/back-projection; VO
front-end could feed a SLAM back-end.

**Proposed design.** New `models/neural_fields/` (NeRF MLP / 3-D Gaussians),
`methods/nerf.py` (volume rendering loss), and an optimizer-backend SLAM module
reusing `geometry/`.

**Risks.** XL: rendering infra, ray sampling, pose optimization, GPU-heavy.
Recommend deferring until Tier 1–2 land.

---

## Suggested sequence

1. **Milestone A (Tier 1):** distillation (#1), detection (#2), depth/flow
   (#3), OOD/calibration (#4). Mostly additive; maximal reuse of existing
   decoders, geometry, and image pipelines.
2. **Milestone B (Tier 2):** tracking + HPO (#6) and interpretability/robustness
   (#7) as enablers, then diffusion (#5).
3. **Milestone C (Tier 3):** pick one modality/axis (VLM #8 or alignment #9 are
   the highest-leverage) and scope it as a standalone project.

## Cross-cutting concerns

- **Registry discipline.** Every new capability should register through the
  existing `MODELS` / `METHODS` / `PIPELINES` / `LOSSES` / `DECODERS` registries
  so the CLI picks it up with no new control flow in `train.py`.
- **Optional dependencies.** Follow the `pyproject.toml` extras pattern
  (`cv2`, `wandb`, `optuna`, …) — never harden new core deps.
- **Tests + style.** Add `tests/` coverage and keep 2-space indent via
  `yapf`/`.style.yapf`; validate with `ruff`.

## Progress Tracker

Live status of the work items. The matrix above is the original planning text
and is left untouched; this table (and the log below) record what has actually
shipped. The table mixes **ranked capability items** (rows 1–12, scored by the
matrix above) with unranked **cross-cutting chores** (row `L`). A chore being
flagged next does **not** outrank an item — the matrix remains the capability
ranking.

Legend: **✅ done** · **🚧 in progress** · **⬜ not started**.

| # | Item | Status | Notes |
|---|---|---|---|
| 1 | Knowledge distillation | ✅ | Losses + method **done**; ONNX export dropped (see Key decisions) |
| 1a | &nbsp;&nbsp;· KD losses (`losses/distillation.py`) | ✅ | 4 losses + registry entries |
| 1b | &nbsp;&nbsp;· `DistillationMethod` (EMA + external teacher) | ✅ | registered as `"distillation"` |
| L | Doc-Math Linter (`scripts/check_md_math.py`) | ⬜ | doc-hygiene chore (not a ranked capability) — do next; see "Continue Here Tomorrow" |
| — | Shared: `EMAModel` / `encoder_utils` contract / `layer_match` | ✅ | extracted first (no cut-and-paste) |
| 2–12 | Detection, depth/flow, OOD, diffusion, HPO, … | ⬜ | untouched (see the Tier sections above) |

---

## Work Log

Newest first. Each entry records the commits and the verification that backed
them.

### 2026-10 — KD method + losses (item 1a/1b) — **committed**

Shipped in `feat(distillation): add KD losses and DistillationMethod`
(the four new files plus five modified files):

- `genml_kit/losses/distillation.py` (new) — `kd_response_loss`,
  `kd_feature_loss`, `kd_attention_loss`, `kd_relation_loss`.
- `genml_kit/losses/builtin.py` (mod) — the four registered as `LOSSES` builders
  (`kd_response` / `kd_feature` / `kd_attention` / `kd_relation`).
- `genml_kit/losses/__init__.py` (mod) — export the four losses.
- `genml_kit/methods/distillation.py` (new) — `DistillationMethod`, both
  `--teacher_kind ema` and `--teacher_kind external`.
- `genml_kit/methods/__init__.py` (mod) — import for registration.
- `tests/test_distillation_losses.py`, `tests/test_distillation_method.py`
  (new) — 37 CPU-only tests.
- `knowledge_distillation/README.md` (mod) — implementation-status note.
- `docs/NEXT_STEPS.md` (mod) — this file.

Verified before handing over: `pytest -q` → **1477 passed, 4 deselected**;
`pytest -m slow -q` → **4 passed**; `ruff check .` clean; all new/changed
`.py` files pass `yapf` with `.style.yapf` (2-space).

**Divergence from the roadmap's sketch (intentional).** The roadmap proposed a
*class* named `DistillLoss` ("logit KD + feature KD + optional relational KD").
What shipped is instead **four plain loss functions** composed by the method in
`_kd_terms`, plus the FitNets projection handled by the shared `ProjectionHead`.
Rationale: (a) each family is one pure, independently testable function — the
repo's existing pattern (`byol_loss`, `supcon_loss`); (b) a single class would
have to own the teacher/student pairing and layer policy, which the method
already does via `match_layers`; (c) attention transfer was added as a fourth
loss (§10) beyond the sketch's three.

**Committed.** Item 1 (losses + method) is complete; the ONNX export (item 1c)
was dropped (see Key decisions).

> **Never commit without explicit approval.** The agreed process is: plan →
> review → approve → implement → review → **commit**. No `--amend`, no squash,
> no `Co-Authored-By`. Mark tasks here as they land.

### 2026-10 — Shared utilities (item 1 preamble, extract-before-write contract)

Committed, all local (not pushed):

| Commit | What |
|---|---|
| `66957f9` | `feat(models): add layer-matching helper for feature distillation` — `genml_kit/models/layer_match.py` (`iter_named_layers`, `match_layers`). |
| `60d4f78` | `refactor(models): document and guard the encoder feature-shape contract` — `require_matching_features` in `genml_kit/models/encoder_utils.py`. |
| `d4d835d` | `refactor(models): extract shared EMA/target-network utility` — `genml_kit/models/ema.py` (`EMAModel`, `linear_momentum`); BYOL / DINO / I-JEPA / DQN / SAC migrated, checkpoint keys byte-for-byte identical. |
| `779899f` | `docs: add knowledge-distillation design guide and AI-topic roadmap` — this file + `knowledge_distillation/README.md`. |

---

## Continue Here Tomorrow

**Resume point: the Doc-Math Linter (`scripts/check_md_math.py`).**
Item 1 (KD losses + method) is implemented and committed; the ONNX export
(item 1c) was dropped — see "Key decisions".

This is a small **doc-hygiene** chore, not a capability item: it clears the
known Markdown/LaTeX rendering regressions before more docs are written. The
top unstarted **capability** milestone is still **Object detection (#2, Tier 1)**
— see "Suggested sequence" / Milestone A.

### Step 0 — commit the reviewed KD work — ✅ done

The KD losses + `DistillationMethod` are committed (see the Work Log above).

### Step 1 — Doc-Math Linter (`scripts/check_md_math.py`) — ⬅ doc-hygiene, do next

The full task spec lives in the "Doc-Math Linter" section at the end of this
document. Summary: turn the truncated, copy-paste snippet in
`docs/MARKDOWN_LATEX.txt` §11 into a reusable, tested checker with a safe
`--fix`, wired into CI.

### Reference: how to run things

```bash
# full suite (slow deselected by default) + lint + format check
pytest -q
pytest -m slow -q
ruff check .

# training smoke for the new method (once labels are wired)
#   python -m genml_kit.training.train --method distillation --help
```

### Key decisions already locked (so we don't relitigate them)

- **No type annotations** — Google-style docstrings only (repo convention; the
  tree has ~3 annotated signatures total, all pre-existing outliers).
- **`namedtuple`s** over bare tuples for structured returns.
- **2-space indent** via `yapf`/`.style.yapf`; **`ruff` clean, no `noqa`**
  without explicit sign-off.
- **Extract-before-write:** the KD method reuses `EMAModel`,
  `encode_with_backbone` / `require_matching_features`, `match_layers`,
  `ProjectionHead` — no copy-paste.
- `by_name` layer matching is self-contained in `layer_match.py` (not
  `training/param_align.py`, whose exact-shape premise opposes cross-depth
  pairing and whose import would invert layering).
- Shape-changing operations are commented with the resulting shape on the line
  above, so tensor flow is trackable within a function.
- **ONNX export dropped (item 1c).** Investigated `torch.onnx.export` for the
  deployed students and dropped it: the VO head ends in `umeyama_similarity`,
  which calls `torch.linalg.svd`, and ONNX has no SVD operator at any opset —
  so `VOModelOutput.params` cannot be a graph output. Additionally the custom
  attention-pooling head emits a tensor→bool `TracerWarning` under the legacy
  tracer, and torch's new default exporter needs opset ≥ 18 plus `onnxscript`
  (and conflicts with the repo's `filterwarnings = error`). Not worth the
  dependency surface for a harness whose models stay in PyTorch.

### Working-tree hygiene

- `.chatty/models.json` is untracked tooling noise — **do not** commit it.
- Local commits are **not pushed**; `git log @{u}..HEAD` lists what is ahead.

---

## Doc-Math Linter

**Goal.** Turn the copy-paste, single-file, *truncated* snippet in
`docs/MARKDOWN_LATEX.txt` §11 into a reusable checker, so future
Markdown/KaTeX fixes are one command instead of a stream of one-off scripts.

**Deliverable.** `scripts/check_md_math.py` — stdlib only, Python 3.9+,
Google style, 2-space indent, `ruff` clean.

**CLI**

    python scripts/check_md_math.py [PATHS ...]   # default: all *.md in repo
        --check    (default) report findings, non-zero exit if any
        --fix      apply the SAFE, idempotent subset of fixes in place
        --json     machine-readable output
        --quiet    summary only

**Checks (each maps to a rule in `docs/MARKDOWN_LATEX.txt`).**
Output format: `file:line:col: [R<n>] message`.

| ID  | Rule (§) | What it detects |
|-----|----------|-----------------|
| R1  | §1  | bare `$...$`, `\(...\)`, `` ``$ `` extra backtick |
| R2  | §1  | inline `$`...`$` span crossing a newline (the KD bug) |
| R3  | §2  | `$$` not on its own line; `\large` missing/not first; blank line inside block; missing blank line before/after |
| R4  | §4  | `^*` or `^{*}` (must be `^{\ast}`) |
| R5  | §5  | line inside `$$` starting (0–3 spaces) with `+ - * \| # >` or a digit |
| R6  | §6  | malformed delimiters: `` ``$ ``, stray trailing `` `$ ``, odd `$`/`` ` `` counts |
| R7  | §7  | `*`/`**` straddling a math span; bold starting right after a math-close |
| R8  | §8  | display `\,` that must be `\\,`; inline `\\,` that must be `\,`; row-break `\\` sanity |
| R9  | §9  | `\operatorname`, raw unicode math symbols |
| R10 | §5/§8 | line-start `|...|` absolute value → suggest `\lvert...\rvert` |

**`--fix` (safe subset only):** R4 (`^*`→`^{\ast}`), R8 (backslash level,
display vs inline), R6 (drop the extra backtick). Everything else is
report-only; R2/R3/R5/R7 need human reflow.

**Tests.** `tests/test_check_md_math.py`: one case per rule, plus the two real
regressions — the cross-line inline span (KD) and `\,` vs `\\,` (rl/vo).

**Docs + CI.**
- `docs/MARKDOWN_LATEX.txt` §11: replace the snippet with a pointer to the script.
- `.github/workflows/ci.yml`: add a **blocking** step
  `python scripts/check_md_math.py` next to `ruff check .`.

**Exit codes.** `0` clean, `1` findings, `2` usage error.
