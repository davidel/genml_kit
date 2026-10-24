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
| 1 | Knowledge distillation + model export (ONNX) | 5 | 2 | **3** | M |
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

### 1. Knowledge distillation + model export (ONNX)

**Why.** LoRA is the only "make it smaller/deployable" path today. Distillation
and export are what turn the research harness into a shippable one.

**Reuse.** `training/model_utils.py` already extracts features/freezes
encoders; `training/trainer.py` gives the loop; classifiers are pluggable.

**Proposed design**

- New method `genml_kit/methods/distillation.py` registered in `METHODS`, with a
  `DistillLoss` (logit KD + feature KD + optional relational KD) in
  `genml_kit/losses/distillation.py`.
- A teacher/student pairing driven by CLI flags
  (`--teacher_model`, `--kd_alpha`, `--kd_temperature`), mirroring the LLRD
  flag style in `training/optim_factory.py`.
- Export CLI `genml-kit-export` (new entry point in `pyproject.toml`) wrapping
  `torch.onnx.export`, with an optional dynamic-axes + int8 path.

**Deliverables.** `methods/distillation.py`, `losses/distillation.py`,
`training/export.py`, console script, tests, README section.

**Risks.** ONNX export of the custom heads (attention pooling, ConvViT) needs
opset pinning; keep it behind the `dev`/new `export` extra.

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

1. **Milestone A (Tier 1):** distillation + export (#1), detection (#2),
   depth/flow (#3), OOD/calibration (#4). Mostly additive; maximal reuse of
   existing decoders, geometry, and image pipelines.
2. **Milestone B (Tier 2):** tracking + HPO (#6) and interpretability/robustness
   (#7) as enablers, then diffusion (#5).
3. **Milestone C (Tier 3):** pick one modality/axis (VLM #8 or alignment #9 are
   the highest-leverage) and scope it as a standalone project.

## Cross-cutting concerns

- **Registry discipline.** Every new capability should register through the
  existing `MODELS` / `METHODS` / `PIPELINES` / `LOSSES` / `DECODERS` registries
  so the CLI picks it up with no new control flow in `train.py`.
- **Optional dependencies.** Follow the `pyproject.toml` extras pattern
  (`cv2`, `onnx`, `wandb`, `optuna`, …) — never harden new core deps.
- **Tests + style.** Add `tests/` coverage and keep 2-space indent via
  `yapf`/`.style.yapf`; validate with `ruff`.
