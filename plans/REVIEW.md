# Deep Review & Fix Plan — Commit `ec19359` ("Introduce generic pipeline + method registry (v4.2)")

Status: **DRAFT for review** (implementation not started)
Scope: 46 files, +2937 / −3801. Replaces the classification / pre-training / VO
  harnesses with a single `genml-kit-train` CLI driven by a data-only
  `DataPipeline` registry and an objective-side `Method` registry.
Baseline: `pytest` → **914 passing**; `ruff check` → clean.

This document is the review of that commit plus the plan to fix what it found.
It is organized as:

- Part 1 — correctness regressions (behavioral)
- Part 2 — design / structure issues
- Part 3 — minor cleanups
- Part 4 — the fix plan (work items, sequencing, verification)
- Part 5 — open decisions still needed from the reviewer

No Python typing annotations are added anywhere (project convention). All new
code is 2-space indent, `yapf` (`.style.yapf`) formatted and `ruff`-clean.

---

## Part 1 — Correctness regressions

### 1.1 `_default_metric` sentinel is inverted — best checkpoint is never saved

Where: `genml_kit/training/train.py:424`

```python
def _default_metric(method):
  """Best-metric sentinel for the first validation of a run."""
  return float("inf") if not method.has_metric_improved(0.0, 1.0) else 0.0
```

The ternary is backwards. The sentinel is the value the *first* real metric must
beat, so it must be the worst possible value **in the method's metric direction**:

- **maximizer** (classification, `macro_f1`): `has_metric_improved(0.0, 1.0)`
  = `0.0 > 1.0` = `False` → returns `inf`. The sentinel for a maximizer must be
  `-inf` (old code used `default_metric=0.0`, `train.py:1301`).
- **minimizer** (vo_pair, `mce`): `has_metric_improved(0.0, 1.0)` = `0.0 < 1.0`
  = `True` → returns `0.0`. The sentinel for a minimizer must be `+inf`
  (old code used `float("inf")`, `train_vo.py`, with an explicit
  "an honest 'no error yet'" comment).

Verified empirically with the real method classes:

| method | direction | sentinel returned | first epoch improves? |
|---|---|---|---|
| classification | maximize | `inf` | **no** (`0.75 > inf` is False) |
| vo_pair | minimize | `0.0` | **no** (`5.0 < 0.0` is False) |
| simmim | minimize (loss) | `inf` | **no** |

Consequence: `saver.save_best(...)` is never called for the best epoch, and the
exit checkpoint stores `best_macro_f1 = inf`. This is a real functional
regression for every run, and it silently defeats the whole best-checkpoint
cycle.

Note on why tests missed it: `tests/test_trainer.py` constructs `BaseTrainer`
with an explicit `best_metric=float("inf")` and never routes through
`_default_metric`, so the inverted helper was never covered.

### 1.2 `method.needs_labels` is never wired into `args.needs_labels`

Where: `genml_kit/pipelines/images.py:241` reads
`getattr(args, \"needs_labels\", False)`; `genml_kit/training/train.py:345`
defaults `args.needs_labels` to `False`.

Design intent (plan \u00a76.1): the pipeline must read **`method.needs_labels`**;
the classification method is `True`, self-supervised methods are `False`. The
deleted `pretrain/cli.py` did exactly `needs_labels=method.needs_labels`.

Reality: nothing in `genml_kit/` ever sets `args.needs_labels` from the method.
Search confirms only tests set it. `_build_ensemble` *receives* `method` but uses
it only for `build_transform`, never for `needs_labels`.

**Verified**: `Method` base has no `needs_labels`. Only `SupConMethod`,
`DINOMethod`, `BYOLMethod`, `IJEPAMethod` define `needs_labels = False`.
`ClassificationMethod` and `VOPairMethod` do **not** define it.

Consequence: `--method supcon --datasets ...` applies the label-stripping branch
(`FieldSectorDataset` image-only items) and `SupConMethod.train_step` then raises
`ValueError(\"SupConMethod requires labels ...\")`. Only `supcon` currently
declares `needs_labels = True`, but the contract is broken for the whole family.

Fix in **A2**: add `needs_labels = False` to `Method` base, `True` to
`ClassificationMethod` / `VOPairMethod`, then propagate from method to
`args.needs_labels` in `train.py` before `pipeline.build_loader`.

### 1.3 Reconstruction visualization silently dropped; `log_validation_images` is dead code

Where: `genml_kit/pipelines/images.py:393` defines `log_validation_images`;
`genml_kit/training/train.py:24` imports it with `# noqa: F401`.

- The deleted `pretrain/cli.py` called `log_validation_images(...)` every
  `vis_every` steps, driven by `--vis_every`.
- The new `BaseTrainer` loop has no `vis_every` concept, and there is **no call
  site** for `log_validation_images` anywhere in `genml_kit/` (only tests).
- `README.md:570` still documents `--vis_every`.
- `SimMIM.validate()` (reconstruction path, `methods/simmim.py:143`) therefore
  has no live caller in a real run.

Net: the SimMIM reconstruction-to-TensorBoard feature was lost in the move,
leaving a public function plus an import as dead code and a stale doc row.

---

## Part 2 — Design / structure issues

### 2.1 Duplicated helpers \u2014 two sources of truth

Byte-identical definitions (verified by diffing the extracted functions) exist
in **both** `genml_kit/pipelines/images.py` and
`genml_kit/training/train_compat.py`:

- `compute_class_weights` (`images.py:471`, `train_compat.py:92`)
- `parse_class_multipliers` (`images.py:431`, `train_compat.py:109`)
- `fmt_weights` (`images.py:488`, `train_compat.py:87`)

`mixup_data` is **not** identical \u2014 `train_compat.py` uses `np.random.beta`
while `images.py` does not have it (defined in `classification.py:187`).  The
duplicated trio (`compute_class_weights`, `parse_class_multipliers`,
`fmt_weights`) creates two sources of truth for classification pipeline logic.
The pipeline is the canonical owner (data-side); `train_compat.py` re-exports
them for backward compatibility.  Recommended: keep them in `images.py` and
import from there in `train_compat.py` (or consolidate to a shared utils
module).  Current state is harmless but confusing.

`mixup_data` is additionally duplicated between `training/train_compat.py:149`
and `methods/classification.py:188` (near-identical; the method copy does an
inline `import numpy` and carries a longer docstring).

`train.py` re-exports from multiple of these modules, so it is ambiguous which
definition is canonical. Drift risk is real: a fix in one copy silently misses
the other.

### 2.2 `DataPipeline.build_transform` is dead and contradicts the stated contract

Where: `genml_kit/pipelines/base.py:46` declares
`build_transform(self, args, method)` ("pipeline MAY apply generic
preprocessing … the METHOD composes on top — two-phase contract").

Nothing ever calls it. The call site that exists is `Method.build_transform(args,
image_size)` (`pipelines/images.py:258`), a different signature. The pipeline
declares a hook it never invokes, and the plan's two-phase composition is not
implemented. Either wire it or delete it.

### 2.3 `Method.build_model` contract violated + `isinstance` type-sniffing in the driver

Where: `methods/classification.py:86` — `build_model` returns `None` with a
comment that the CLI stands in, even though `build_model` is
`@abc.abstractmethod` on `Method` (`methods/base.py:22`).

The driver then branches on `isinstance(method, get_method("classification"))`
in four places: `train.py:375`, `:384`, `:450`, `:512`. The stated goal of the
refactor was a branchless generic harness; classification is still hard-branched
into the CLI, and a core abstract method is implemented as a no-op.

### 2.4 Registration APIs are inconsistent between the two registries

- `register_method(cls)` — bare decorator, reads `cls.NAME`
  (`methods/registry.py`).
- `register_pipeline("name")` — takes the name as an argument
  (`pipelines/registry.py`), used as `@register_pipeline("images")`.

Same concept, two conventions. Worth unifying for symmetry (and with the
pre-existing `training/classifiers/__init__.py`).

### 2.5 Default `evaluate()` calls `train_step`, which has side effects

Where: `methods/base.py:56` \u2014 the default `evaluate` iterates the loader calling
`train_step` under `no_grad` to average metrics.

For DINO/BYOL, `train_step` calls `model.update_momentum(...)`; for SimMIM it
samples a fresh random mask. Any method relying on the default `evaluate` for
validation would mutate EMA state or be stochastic. Today this is masked because
the ensemble path only sets `train_loader` (`val_loader` stays `None` for
pre-training), but the base contract is a footgun for future methods.

### 2.6 `log_validation_images` imported but never called \u2014 SimMIM validation images lost

Where: `genml_kit/training/train.py:25` imports `log_validation_images` from
`pipelines.images`, but there is **no call site** in the training loop or
trainer. The function exists and is tested (`tests/test_pretrain_vis.py`), but
the integration was lost during the v4.2 refactor.

Consequence: `SimMIM.validate()` (`methods/simmim.py:143`) \u2014 which returns
reconstructed images for TensorBoard logging \u2014 has no live caller. The
`--vis_every` flag documented in `README.md:570` is also dead.

Recommendation: either (a) call `log_validation_images` from `BaseTrainer.run`
after validation when `args.vis_every > 0` and `epoch % args.vis_every == 0`,
or (b) remove the import, the function, the tests, and the stale doc row if
the feature is not needed.

### 2.7 `needs_labels` attribute missing from `Method` base class

Where: `Method` base class (`methods/base.py`) has no `needs_labels` attribute.
Only `SupConMethod`, `DINOMethod`, `BYOLMethod`, `IJEPAMethod` define
`needs_labels = False`. `ClassificationMethod` and `VOPairMethod` do not
define it (and thus default to... nothing, since it doesn't exist in base).

The CLI in `train.py` uses a hardcoded default `needs_labels=False` for all
methods (`normalize_args` in `utils/args.py:110`), which is incorrect for
supervised methods. This causes a regression where `--method supcon` on a
labeled ensemble silently strips labels (the old `pretrain/cli.py` used
`needs_labels=method.needs_labels`).

Fix tracked in **A2**: add `needs_labels = False` to `Method` base, set
`needs_labels = True` on `ClassificationMethod` and `VOPairMethod`, and
propagate from method to `args.needs_labels` in `train.py` before
`pipeline.build_loader`.

### 2.8 `pretrain/` folder is now a misnomer — move losses/augmentations up

Since the v4.2 refactor (commit `ec19359`), there is no longer a separate
"pretrain" CLI or bolted-in pretraining program. All methods (supervised +
self-supervised) are unified under the `Method` registry and the single
`genml-kit-train` binary. However, the shared utilities remain in
`genml_kit/pretrain/`:

```
genml_kit/pretrain/
├── augmentations/
│   ├── dual_view.py     → DualViewTransform (used by BYOL)
│   └── multicrop.py     → MultiCropTransform (used by DINO, IJEPA)
└── losses/
    ├── byol.py          → byol_loss (used by BYOL)
    ├── contrastive.py   → supcon_loss (used by SupCon)
    ├── dino.py          → DINOLoss (used by DINO)
    └── focal.py         → CombinedFocalLoss (used by Classification)
```

These are **general-purpose utilities** used by multiple methods, not
pretraining-specific. The `pretrain/` namespace is now misleading and
couples classification (via `CombinedFocalLoss`) to a "pretrain" module.

**Recommendation**: move to a neutral top-level namespace:
- `genml_kit/augmentations/dual_view.py`, `genml_kit/augmentations/multicrop.py`
- `genml_kit/losses/byol.py`, `genml_kit/losses/contrastive.py`, `genml_kit/losses/dino.py`, `genml_kit/losses/focal.py`

Then update imports in:
- `methods/byol.py`, `methods/dino.py`, `methods/supcon.py`, `methods/classification.py`
- `models/dino.py`, `models/byol.py`
- `training/train_compat.py`
- `pretrain/losses/__init__.py`, `pretrain/augmentations/__init__.py` (remove)

This is a mechanical rename with no behavioral change, but it cleans up the
legacy architecture and avoids the false implication that these are
"pretraining-only" utilities.

---

## Part 3 — Minor cleanups

- **C1.** No-op context block in `ClassificationMethod.train_step`
  (`classification.py:128-129`):
  ```python
  with model_mode(model, "eval"):
      pass  # no mode flip: reporting uses logits directly
  ```
  A context manager that flips mode and back for nothing — delete.
- **C2.** Unused attribute `DataPipeline.args` (`pipelines/base.py:19`):
  `build_pipeline(args.pipeline)` is always called without kwargs, so
  `self.args` is always `None`. Dead.
- **C3.** `del ckpt_extra` (`train.py:408`) \u2014 **NOT vestigial**. The variable
  `ckpt_extra` is a dict returned by `open_resume_context` containing the full
  checkpoint extras (optimizer state, scheduler state, scaler state, method
  state, etc.). After extracting the needed pieces (`ckpt_extra.get("method_state")`,
  `global_step`, and passing `ckpt_extra` to `build_optimization` for state
  restoration), the `del` explicitly drops the reference so the potentially
  large checkpoint dict can be garbage-collected before the trainer is
  constructed. This is intentional memory hygiene, not noise.  **Do not remove.**

  **Action**: Added explanatory comment in `train.py:405-407` to document the
  intent and prevent future reviewers from flagging it as vestigial.
- **C4.** Plan deviation: §6.1 says "if both `--dataset` and `--datasets` are
  present → `fatal`". Not implemented; `build_loader` silently prefers
  `--dataset`.
- **C5.** `_NoMonitor` is (re)defined inside `_init_grad_monitor` on every call
  (`trainer.py:175`); hoist to module scope.
- **C6.** `Method` base has no `needs_labels`; only `supcon/dino/byol/ijepa`
  define it \u2014 `classification/vo_pair` do not (verified). Adding the default to
  the base makes the contract explicit (folded into fix item A2).
- **C7.** Decorative section comments in `genml_kit/pipelines/images.py:335,428`
  (`# --- Moved verbatim from ... (v4.2 s ...) --------------------`) carry no
  semantic value and serve only as visual dividers from the legacy migration.
  Remove them.

---

## Part 4 — The fix plan

Work items are grouped; each lists files, the change, rationale and test impact.
Group A (correctness) should land regardless of the design decisions.

### Group A — Correctness (behavioral regressions)

#### A1. Fix the `_default_metric` sentinel (inverted direction)

Files: `genml_kit/training/train.py`

Make the direction explicit instead of hiding it in a negated predicate:

```python
def _default_metric(method):
  """Best-metric sentinel for the first validation of a run.

  The sentinel is the worst possible value in the method's metric direction,
  so the first real metric always wins.  Probe the direction with two
  constants rather than duplicating maximize/minimize knowledge.
  """
  return float("inf") if method.has_metric_improved(0.0,
                                                    1.0) else float("-inf")
```

Check: classification `0.0 > 1.0` = False → `-inf` ✅; vo_pair `0.0 < 1.0` =
True → `+inf` ✅.

Tests:
- `tests/test_pipeline_method_registry.py`: add `test_default_metric_direction`
  asserting `-inf` for `classification`/`simmim` and `+inf` for `vo_pair`.
- `tests/test_trainer.py`: add an end-to-end assertion that `save_best` runs on
  epoch 0 (drive the trainer through `main`'s sentinel, or assert that the
  first-epoch metric beats `_default_metric(method)`).

#### A2. Wire `method.needs_labels` into the pipeline

Files: `genml_kit/training/train.py`, `genml_kit/pipelines/images.py`,
  `genml_kit/methods/base.py`, `genml_kit/methods/classification.py`,
  `genml_kit/methods/vo_pair.py`

- Add `needs_labels = False` to `Method` base (makes the contract explicit;
  absorbs C6).
- Set `needs_labels = True` on `ClassificationMethod` and `VOPairMethod`.
- In `train.py:main`, after `method = build_method(args.method)` and before
  `pipeline.build_loader(...)`, propagate the flag from the method
  (mechanism chosen by open decision **Q1**).
- Keep `_post_process`'s default so the pipeline remains usable standalone in
  tests, but the CLI path must derive the value from the method.

Rationale: restores `pretrain/cli.py` semantics
(`needs_labels=method.needs_labels`) and fixes the SupCon-with-`--datasets`
label-stripping regression.

Tests: add an assertion that `--method supcon` on a labeled ensemble keeps
`meta["labels"]`; `tests/test_pipeline_method_registry.py:207/226` set
`args.needs_labels` manually — add a case that derives it from a method.

#### A3. Restore (or remove) the reconstruction-visualization path

Files: `genml_kit/training/trainer.py`, `genml_kit/training/train.py`,
  `genml_kit/pipelines/images.py`, `README.md`

Two options (open decision **Q2**):

- **(a) Restore:** add `--vis_every` to the generic logging args; in
  `BaseTrainer.train_epoch`, every `vis_every` optimizer steps call
  `log_validation_images(method, model, pipeline.train_loader, writer, step,
  device, image_column)`. Re-exercises `SimMIM.validate()`.
- **(b) Remove:** delete `log_validation_images`, the `train.py:24` import, the
  `--vis_every` row in `README.md`, and the two test modules that target it.

Recommendation: **(a)**, since `README.md` still documents the flag and
SimMIM's `validate()` otherwise has no live caller.

### Group B — Design / structure

#### B1. De-duplicate the moved helpers (single source of truth)

Files: `genml_kit/pipelines/images.py`, `genml_kit/training/train_compat.py`,
  `genml_kit/methods/classification.py`

Keep canonical definitions in **one** module. Because these are classification
helpers, put `compute_class_weights` / `parse_class_multipliers` / `fmt_weights`
/ `mixup_data` in `train_compat.py` (already the re-export hub), or in a small
`training/class_weights.py` if we want `pipelines` to stop importing from
`training`. The pipeline imports them instead of redefining; `train.py`
re-exports only from the canonical module.

Rationale: removes the two-sources-of-truth drift risk.

Tests: existing tests import from `training.train` / `train_mod`; keep those
re-exports so tests don't change (or update import paths if we prefer the new
module).

#### B2. Resolve the dead `DataPipeline.build_transform` hook

Files: `genml_kit/pipelines/base.py`

Options (open decision **Q3**):

- **(a) Delete** the pipeline hook, keep the single real
  `Method.build_transform` (matches what is implemented).
- **(b) Wire it** so the pipeline composes `pipeline.build_transform(args,
  method)` with `method.build_transform(...)`, implementing the plan's two-phase
  contract.

Recommendation: **(a)** unless the two-phase composition is genuinely needed;
(b) adds machinery with no current consumer.

#### B3. Remove classification `isinstance` special-casing from the driver

Files: `genml_kit/training/train.py`, `genml_kit/methods/classification.py`,
  `genml_kit/training/train_compat.py`, `genml_kit/methods/base.py`

Give `Method` an explicit, documented model/transform seam instead of
`isinstance`:

- Let the method own model construction via `build_model(args, device,
  pipeline)`; `ClassificationMethod.build_model` calls `load_model(...)` and
  applies LoRA / source-checkpoint logic (move the body of
  `_load_classification_model` into the method, reusing `train_compat`
  helpers).
- Make the remaining classification-only steps generic hooks with no-op
  defaults: e.g. `method.prepare(args, pipeline, device)` (transform resolution
  + criterion wiring) and `method.post_train(args, pipeline, device, result)`
  (XGBoost). This absorbs `train.py:375/:384/:450/:512`.

Rationale: fulfills the "branchless generic harness" goal.

Risk/effort: this is the largest item. `build_model` is `@abc.abstractmethod`,
so all methods + test fakes must adopt the new signature. Scope controlled by
open decision **Q4**.

#### B4. Unify the two registry APIs

Files: `genml_kit/methods/registry.py`, `genml_kit/pipelines/registry.py`

Same concept, two conventions. Make both class-decorators reading `cls.NAME`
(pipelines already have `NAME`), for symmetry with the method registry and with
`classifiers/__init__.py`.

Tests: `tests/test_pipeline_method_registry.py:100` uses
`@register_pipeline("custom_pipe")` — update.

### Group C \u2014 Minor cleanups

- **C1.** Delete the no-op `with model_mode(model, \"eval\"): pass`
  (`methods/classification.py:128-129`).
- **C2.** Remove the unused `DataPipeline.args` attribute
  (`pipelines/base.py:19`).
- **C3.** ~~Remove the vestigial `del ckpt_extra`~~ \u2014 **KEEP IT**.
  The `del ckpt_extra` at `train.py:405` is intentional memory hygiene
  (see Part 3).  No action needed.
- **C4.** Enforce the \"both `--dataset` and `--datasets` \u2192 `fatal`\" rule in
  `ImagesPipeline.build_loader` (`pipelines/images.py:133`).
- **C5.** Hoist `_NoMonitor` to module scope (`trainer.py:175`).
- **C6.** Folded into A2 (`Method.needs_labels` base default). Optionally
  document `Method.evaluate`'s side effects (scope per Q4).

### Sequencing

1. **A1** (isolated, highest priority) — fix + tests; run
   `pytest tests/test_trainer.py tests/test_pipeline_method_registry.py`.
2. **A2**, then **A3** (per Q2).
3. **B1**, **B2** (per Q3) — mechanical, low risk.
4. **B4**, **C1–C5** — mechanical.
5. **B3 / C6** (per Q4) — largest, done last so it never blocks the fixes.
6. After every step: `format_file` (yapf, `.style.yapf`) and
   `ruff check genml_kit/ tests/`; final `python -m pytest -q` must stay green.
   **No commit until the reviewer approves.**

### Test / verification matrix

| item | primary tests |
|---|---|
| A1 | `tests/test_pipeline_method_registry.py`, `tests/test_trainer.py`, `tests/test_checkpointing.py` |
| A2 | `tests/test_pipeline_method_registry.py`, `tests/test_supcon_method.py`, `tests/test_ensemble_collate.py` |
| A3 | `tests/test_pretrain_vis.py`, `tests/test_ensemble_collate.py` |
| B1 | `tests/test_pipeline.py`, `tests/test_pipeline_method_registry.py` |
| B3 | `tests/test_trainer.py`, `tests/test_headless_models.py`, `tests/test_supcon_method.py` |
| B4 | `tests/test_pipeline_method_registry.py` |

---

## Part 5 — Open decisions needed from the reviewer

- **Q1 (A2):** derive label need by setting `args.needs_labels =
  method.needs_labels` (minimal), **or** change the `build_loader` signature to
  take `needs_labels` explicitly (cleaner boundary)?
  _Recommendation: minimal attribute set._
- **Q2 (A3):** restore `--vis_every` / visualization, **or** delete the dead
  function + doc row?
  _Recommendation: restore._
- **Q3 (B2):** delete the unused `DataPipeline.build_transform`, **or** actually
  wire the two-phase composition?
  _Recommendation: delete._
- **Q4 (B3/C6):** how far to refactor the method/model seam now —
  (i) full consolidation (absorb transform / model / XGBoost hooks into
  `Method`); (ii) only move model construction into the method and keep the
  other branches; (iii) leave `isinstance` as-is for now?
  _Recommendation: (ii)._

Once these are confirmed, the plan is finalized and implementation begins on
explicit go-ahead. Nothing is committed to GIT until the reviewer approves the
changes.
