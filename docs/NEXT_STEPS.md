# genml_kit — Roadmap / Next Steps

This document inventories the AI/ML topics that `genml_kit` does **not** yet
cover and ranks them by **Impact vs Complexity**. It is a planning artifact: it
describes *what* to build and *how it should slot into the existing
architecture*, not a commitment to implement everything.

> Status: living document. Items 1 (distillation), 2a (detection, boxes-only)
> and chore `L` (doc-math linter) have shipped; everything else is still a
> proposal. See the Progress Tracker for live status.

## What genml_kit covers today

| Domain | Coverage |
|---|---|
| Image classification | Pluggable heads (`training/classifiers/`), XGBoost pipeline, TTA, metrics/eval |
| Semantic segmentation | UNet / FPN / linear decoders, Dice-CE loss, seg metrics |
| Visual odometry | Photometric + supervised, 2-D similarity algebra, differentiable Umeyama |
| Object detection | Anchor-free FCOS head, COCO-JSON `detfolder/`, mAP meter, NMS decode |
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
| L | Doc-Math Linter (`scripts/check_md_math.py`) | ✅ | doc-hygiene chore (not a ranked capability) — implemented; see Work Log |
| — | Shared: `EMAModel` / `encoder_utils` contract / `layer_match` | ✅ | extracted first (no cut-and-paste) |
| 2 | Object detection | ✅ | boxes-only cut shipped; 2b/2c deferred — see Work Log |
| 2a | &nbsp;&nbsp;· `detection` pipeline + `DetectionMethod` + `det_folder` | ✅ | COCO-JSON `detfolder/`, box-aware transforms, mAP meter |
| 2b | &nbsp;&nbsp;· instance segmentation (`mask_head`) | ⬜ | deferred; wrapper designed to accept a mask head later |
| 2c | &nbsp;&nbsp;· `detr_head` + Hungarian matcher | ⬜ | deferred; FCOS chosen for the first cut (fast, no matcher) |
| 3–12 | Depth/flow, OOD, diffusion, HPO, … | ⬜ | untouched (see the Tier sections above) |

---

## Work Log

Newest first. Each entry records the commits and the verification that backed
them.

### 2026-10 — Tiny-COCO generator promoted to `scripts/` — **committed** (`07b724e`)

Promoted the ad-hoc `.chatty/scratch/make_tiny_coco.py` helper into the repo
so smoke-test corpora can be produced without downloading COCO.

New files:

- `scripts/make_tiny_coco.py` — argparse CLI (`-n`, `--size`,
  `--num_classes`, `--boxes_per_image`, `--seed`) writing a COCO-JSON corpus
  (`images/` + `annotations.json`) for the `detfolder/` spec.  Per-class
  colours, boxes clamped to the frame, 5-digit file names.
- `tests/test_make_tiny_coco.py` — 8 tests via the `importlib` script-load
  pattern used by `test_check_md_math.py`: corpus structure, count flags,
  in-frame boxes with `area == w*h`, 1-based dense categories, non-blank
  images, default backward compatibility, and seed determinism.

Modified: one `README.md` pointer in the detection **Dataset Format** section.
Overlap with `tests/test_det_e2e.py::_write_dataset` is deliberate and left
in place (the test fixture stays minimal).

Verification: `pytest tests/test_make_tiny_coco.py -q` → **8 passed**;
`ruff check .` → clean; both files pass `yapf` with `.style.yapf` (2-space).

### 2026-10 — Object detection (item 2) — **committed** (`af622fa`, spec fix `c8bc24d`)

First cut of object detection, boxes-only per the agreed scope: an anchor-free
**FCOS** head over the existing encoder pyramid, reading **COCO-JSON**
corpora.  The roadmap's `detr_head` + `mask_head` are explicitly deferred
(rows 2b/2c above); FCOS was chosen because it reuses the `EncoderPyramid`
contract and `losses/focal.py` with **no new dependencies** and needs no
Hungarian matcher, so it trains fast enough for CI-friendly e2e tests.

New files:

- `genml_kit/datasets/det_folder.py` — `DetFolderDataset` / `load_detfolder`
  (COCO-JSON `images/` + `annotations.json`; `[x,y,w,h]`→`xyxy`; sparse
  category ids compacted to `[0, C)`; degenerate boxes dropped; empty-GT
  images kept as background).
- `genml_kit/datasets/det_transforms.py` — hand-written **letterbox** +
  box-aware horizontal flip.  Not reusable from the seg stack: `torchvision`
  remaps masks through geometric ops but leaves boxes alone, so each op moves
  the coordinates itself.
- `genml_kit/pipelines/detection.py` — `DetectionPipeline`; `DataBlob` carries
  a stacked `(B, 3, H, W)` image and **per-image** box/label lists (variable
  box counts), so the collate is custom rather than `torch.stack`.
- `genml_kit/models/det_model_wrapper/` — `det_wrapper` model loader;
  `DetModelWrapper` = encoder pyramid + head, reusing the seg wrapper's
  `TimmEncoder` / `probe_encoder` (a feature pyramid is a feature pyramid).
- `genml_kit/models/decoders/fcos_head.py` — `fcos_head`, registered in
  `DECODERS`; returns a `DetHeadOutput` (per-level class/box/centerness +
  strides) rather than one tensor, so locations keep their strides.
- `genml_kit/losses/detection.py` — `FCOSLoss` + `fcos` registry entry:
  focal + GIoU + centerness, with geometric centre-sampling assignment
  (smallest-area box wins on overlap).
- `genml_kit/training/det_metrics.py` — `MAPMeter`, COCO-style
  `mAP@[.5:.95]` + `map_50` / `map_75`; hand-rolled (torchvision has no mAP
  API) but reuses `torchvision.ops.box_iou` for the matching.
- `genml_kit/methods/detection.py` — `DetectionMethod`; `METRIC_KEY="map"`
  (maximize); infers the class count from the dataset; decodes + per-class NMS
  (`torchvision.ops.batched_nms`).

Modified (additive): the four registry `__init__`s, `datasets/factory.py`
(`detfolder/` spec), and a new **Object Detection** section in `README.md`.

Verification: `pytest -q` → **1583 passed, 5 deselected**; `ruff check .` →
clean; the two new `yapf`-formatted modules pass `.style.yapf` (2-space);
`python scripts/check_md_math.py` → 0 findings.  New tests:
`tests/test_det_{folder,transforms,head,loss,metrics,e2e}.py` (68 tests,
including a hand-computed mAP case and a slow overfit smoke test).

### 2026-10 — Doc-Math Linter (chore `L`) — **committed** (`8c5030a`, `26c559f`)

`scripts/check_md_math.py` (new) — stdlib-only, Python 3.9+, Google style,
2-space indent, `ruff` clean. Implements checks R1–R10 of
`docs/MARKDOWN_LATEX.txt` and prints `file:line:col: [R<n>] message`.
CLI: `[PATHS ...]` (default all git-tracked `*.md`), `--check` (default),
`--fix` (safe subset: R4/R6/R8), `--json`, `--quiet`; exit `0` clean /
`1` findings / `2` usage.

The scanner classifies fenced code blocks, `$$` display blocks, inline
``$`...`$`` spans and inline code spans before any rule runs, so a literal
``\\,`` inside `$$` is correctly *not* flagged while the same bytes inside
an inline span are. R9 (unicode / ASCII math) is scoped to math position
only, so prose such as `Python ≥ 3.9` stays clean.

Findings this surfaced in pre-existing docs, all fixed in the same change
(25 total): 15 cross-line inline spans in `rl/README.md`, bare `$...$` in
`rl/README.md` and `vo/README.md`, `\\,` inside inline spans in
`rl/README.md`, a missing `$` close in `README.md` and `rl/README.md`, and
`|...|` / emphasis hazards in `vo/README.md`.

`tests/test_check_md_math.py` (new) — 38 tests: one per rule, the two real
regressions (cross-line span; inline vs display `\\,`), `--fix`
idempotency, and a self-clean test asserting zero findings on this repo's
own docs.

Docs + CI: `docs/MARKDOWN_LATEX.txt` §11 now points at the script (the old
ad-hoc snippet is preserved in git history); `.github/workflows/ci.yml`
gained a blocking `python scripts/check_md_math.py` step after Ruff.

Verified before handing over: `pytest tests/test_check_md_math.py -q` →
**38 passed**; `python scripts/check_md_math.py` → **0 findings in 6
files**; `ruff check .` clean; both new files pass `yapf` with
`.style.yapf` (2-space).

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

**Resume point: learned depth / stereo / optical flow (item 3, Tier 1).**
Item 1 (KD losses + method), item 2 (object detection, boxes-only) and the
doc-hygiene chore `L` are implemented and committed; the ONNX export (item 1c)
was dropped — see "Key decisions". Deferred detection sub-items are 2b
(instance segmentation) and 2c (`detr_head`).

The top unstarted **capability** milestone is **Learned depth / stereo /
optical flow (#3, Tier 1)** — see "Suggested sequence" / Milestone A and the
Tier 1 section above. It reuses the `vo/` geometric scaffolding
(`geometry/similarity.py`, the `vo_pairs` synthetic generator, photometric
loss machinery) and needs a new `pipelines/geometry_dense.py`, a
`methods/flow_depth.py`, and `disp_head` / `flow_head` decoders.

### Step 0 — commit the reviewed KD work — ✅ done

The KD losses + `DistillationMethod` are committed (see the Work Log above).

### Step 1 — Doc-Math Linter (`scripts/check_md_math.py`) — ✅ done

Shipped as `docs: add a Markdown/KaTeX math linter and fix the findings it
surfaced`; the spec in the "Doc-Math Linter" section at the end of this
document records what was built (rules R1–R10, CLI, exit codes), not a TODO.

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

## Doc-Math Linter (delivered — reference spec)

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
