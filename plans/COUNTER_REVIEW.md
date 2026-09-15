# Counter-Review — `plans/REVIEW.md` vs `plans/GENERIC_PIPELINE.md` (v4 → v4.1)

Date: 2026-09-15 (same session as the v4.1 plan revision)
Scope: After re-verifying EVERY numbered claim in `plans/REVIEW.md` (22 points)
line-by-line against the source tree, this document records (a) where I
confirmed the review, (b) where I went further or chose a different resolution
than the one the review suggested, and (c) the one genuinely new defect the
review missed.  This is the companion to the v4.1 plan edits — everything in
category (a) has already been fixed in `plans/GENERIC_PIPELINE.md`.

**Bottom line: the review is rigorous and factually accurate.** Every cited
file:line, every method payload, and every "already-gone" claim was re-checked
and found true.  The disagreements below are about *resolution strategy*, not
about the existence of the defects.

---

## 1. Where the review is fully confirmed (no action beyond the plan fixes)

Verified against source, then fixed in the plan §4/§9/§11/§12:

| # | Claim | Evidence re-checked | Plan fix |
|---|---|---|---|
| 1 | `evaluate()` default contradiction | `plan:299-304` `raise NotImplementedError` vs `plan:635-637` "optional, default averages" — real contradiction | §3.1 now ships a concrete averaging default |
| 3 | Loss accumulation math wrong | `train.py:841` / `cli.py:257` multiply reported loss by `grad_accum_steps`; old §4 divided and then averaged → 1/G under-report | §4 notes: report RAW loss |
| 4 | AMP / scaler absent from generic loop | `train.py:778-810` verbatim autocast+scaler+unscale; old §4 had none | §4 loop restored from source |
| 5 | Scheduler placement | `train.py:1213-1214`, `cli.py:713-714` step per-epoch in `train_epoch`; old §4 never stepped | §4/#12.Q9: scheduler lives in `BaseTrainer.run()` |
| 6 | Checkpoint state hooks dropped | `byol.py`, `dino.py` (momentum+center), `simmim.py` (mask/decoder), `supcon.py`, `ijepa.py` all implement `get/load_checkpoint_state` | §3.1 restores both hooks; §4 wires `saver_extra` |
| 7 | epoch-end hook removed (IJEPA) | Verified the review's own correction: BYOL/DINO `on_epoch_end` bodies are `pass` (`byol.py:90`, `dino.py:159`); IJEPA's is real (`ijepa.py:212-215`) | §4 routes `epoch_end()` → `method.on_epoch_end()` |
| 8 | Transform ownership | Verified SimMIM has NO `build_transform`; `DualViewTransform`/`MultiCropTransform` are method-owned (byol.py:58 / dino.py:93) | §6.1 two-phase contract |
| 9 (mechanics) | `--vo_stage` string vs int | `methods/vo_pair.py:27-30` parses `type=int`; `train_vo.py:19` `STAGES = {supervised: 0, photometric: 1}`; plan's examples passed a string | §3.4/§5.2/§8 use int 0/1; §5.2 documents the type decision |
| 11 | Monitor sequencing lost | `train.py:804,811`, `cli.py:236,242` → `monitor.step(global_step)` after unscale, before clip; old §4 had none | §4 re-inserts `unscale → monitor.step → clip` |
| 12 | Loaders never wired | §2.3 `build_loader` (returns loader) vs §4 `self.pipeline.train_loader` vs §5.1 local vars — nothing populated `train_loader` | §2.3/§5.1: pipeline caches `self.train_loader/val_loader` |
| 14 | `loss_fn(out, blob)` drops model + global_step | BYOL `set_train_mode`+`_current_momentum(global_step, model)` (`byol.py:68-72`), DINO same (`dino.py:125-135`), IJEPA `set_train_mode` (`ijepa.py:194`) | §3.1: `train_step(model, blob, global_step, *, labels=None)` |
| 15 | train/eval mode missing | `train.py:758`, `cli.py:189` call `set_train_mode(model, 'train')`; old §4 never did | §4 calls `set_train_mode(self.model, "train")` |
| 17 | §3.4 stale binary | Old `:352-353` still invoked `genml-kit-pretrain`, contradicting §5.2/§12-Q6/Q8 | §3.4 rewritten to `genml-kit-train --vo_stage 0/1` |
| 18 | Risks table stale | `:574` v3 wording (`loss_fn` in `train_vo`); `:578` constructor signature mismatch with §4 | §10 refreshed |
| 19 | Doc inventory understated | `README.md:492,518,591,632,1061`; `scdiag/README.md:27,57,72`; `prepare_isic.py:201`, `prepare_derm1m.py:14,216`; `vo/README.md` has NO CLI refs | §12-Q7 corrected |
| 20 | File inventory misses losses/augmentations + helper homes | `train.py:26` imports `pretrain.losses.focal`; `base.py:79-80` imports `build_pretrain_transform` from deleted `pretrain.cli` | §9 keeps `pretrain/{losses,augmentations}`; §12-Q10 names new homes |
| 21 | Test inventory incomplete | All 12 files verified (see §11.2) including the private-symbol imports `_PatchEmbedder`/`_Predictor` | §11.2 expanded |
| 22 | Nits | `add_args` classmethod-vs-instance mix (byol.py:20-21, dino.py:50 classmethod vs rest instance); registry `NAME` vs plan `name`; `best_mce_negated` already gone (`test_trainer.py:262`) | §12-Q11 (instance methods), §3.1/3.2 (`NAME`), §7 rewritten |

---

## 2. Where I chose a different resolution than the review suggested

These are not disagreements about facts — the review's facts all hold.  They are
judgment calls about the FIX, which I made in the plan and want to flag for a
joint sign-off.

### 2.1 #2 — `ModelOutput`: I DROPPED it, and there is a reason the review could not have guessed

The review said "either the loop should wrap, or the namedtuple should be
dropped".  I verified a third fact that decides the question:

- **`genml_kit/models/registry.py:63` already defines a class `ModelOutput`**
  (a lightweight container holding `.logits`, used by every custom headless
  model to mimic HF output), and it is exported from `genml_kit/models`
  (`models/__init__.py:19,30`) and imported by **four production files**
  (`models/uvito/loader.py:15`, `models/timm/model.py:13`,
  `models/convvit/loader.py:16`, `models/cls_model_wrapper/model.py:9`) plus
  three tests (`test_timm_models.py:7`, `test_headless_models.py:14`,
  `test_custom_models.py:12`).

The plan's v4 declared a *second* `ModelOutput` — a namedtuple holding
`.predictions` — in `pipelines/contracts.py`.  Two different `ModelOutput`
types with different fields in one codebase is a name-crash waiting for an
implementer (or an `import *`), and the review's own #2 (declared but never
constructed) already showed it was dead weight.  So the resolution is:

- **Drop the contracts `ModelOutput`** (fixed in §2.2).
- The loop passes the raw `model(...)` return to `method.train_step`, and each
  method decodes its own shape (dict for VO, tensor for classification, tuple
  for SimMIM, multi-crop list handling inside DINO's train_step already).
- Consequence to track: `models.registry.ModelOutput` remains the ONLY
  `ModelOutput` in the codebase — no behavioural change there.

This also means §2.2's "Unit — contracts: namedtuple fields/defaults" test
(plan §11.1) must only assert `DataBlob` and `LossOutput`.

### 2.2 #13 — `to_device` meta movement: the review's fix is right, and the plan's own §11.1 unit-test bullet already said so

Review: "change §2.3's signature to `to_device(self, blob, device) -> DataBlob`
and §4 to `blob = self.pipeline.to_device(blob, self.device)`".  I agree, and
fixed exactly that.  A precision note on the review's framing: it credited
"§6's blob table" for already listing `meta` contents — true, but that table
describes the BLOB SHAPE, not the transfer rule.  The one place that *did*
state the transfer rule was the v4 §11.1 unit-test bullet for
`VOPairPipeline` ("`to_device` moves both tensors" — old `plan:593`), which
contradicted the §2.1 docstring and §4 code (both data-only).  So the plan
was self-contradictory rather than uninformed.

**Joint-resolution (2026-09-15):** `to_device` is a PIPELINE-SPECIFIC method
— there is no generic recursive utility imposed on the base class.  Each
pipeline implements the traversal for its own blob shape and meta nesting.
  - The VO pipeline must move `meta.gt` — a NESTED dict of tensors
    (`{log_s, theta, t}`, `vo_pairs.py:284-288`) — recursively; a
    top-level-only `hasattr(v, "to")` loop would silently leave GT on CPU
    and break the supervised VO loss.
  - The plan's §2.3 docstring and the §2.6 `VOPairPipeline.to_device` example
    now state this explicitly (small recursive `_move` helper shown for VO).
  - Interface also records the project-wide convention: **no PEP-484 type
    annotations anywhere** (matches `.style.yapf` / Ruff config, and the
    source's `NAME = ""` / `needs_labels = False` style).  The v4.1 draft
    had annotated `NAME: str = ""` / `metric_key: str = "loss"` — scrubbed.

### 2.3 #9 — `--vo_stage` type: I kept `int`, and did NOT rename `vo_pair`

The review left two options open: "One of the two must change, and the plan
never says which", and suggested distinct names (`vo_supervised` / `vo_ssl`) as
an alternative.  **Joint resolution (v4.2): keep ONE `vo_pair` method, but make
the CLI self-documenting with descriptive `--vo_stage` names while preserving
the int internals.**

- **Keep the single `vo_pair` method name.**  The supervised/SSL branching is a
  *loss-weight schedule* inside one objective (the model and the mce val metric
  are the same family; the photometric stage is literally a weighted extra loss
  term in `vo_losses`, `train_vo.py:51-55`).  Splitting into two registry names
  would duplicate the model-build, the pipeline, and the metric for no
  architectural gain.
- **CLI surface: descriptive names** `--vo_stage supervised|photometric`
  (argparse `type=` mapping to int, or `choices`), **default `supervised`**.
  - Self-documenting: a user typing `--method vo_pair --vo_stage photometric`
    understands the objective immediately (fixes the v4.1 ambiguity where
    `1` vs `0` was opaque).
  - Default `supervised` also fixes the DEFAULT foot-gun: a plain
    `genml-kit-train --method vo_pair` (no flag) now trains the supervised
    objective, which is the natural reading of a binary named *train*.  (Today
    `add_args` defaults `--vo_stage` to `1` photometric, `vo_pair.py:29`.)
- **Internal contract unchanged (int):** `STAGES = {"supervised": 0,
  "photometric": 1}` (`train_vo.py:19`) and the numeric
  `stage >= STAGES["photometric"]` comparison (`train_vo.py:51`) stay as-is;
  `tests/test_trainer.py:23` passes the internal int directly and keeps working.
- **BREAKING, documented in the plan (§5.2):** historic numeric
  `--vo_stage 1`/`--vo_stage 0` invocations must become `photometric` /
  `supervised`.  This is acceptable within v4's scope because
  `genml-kit-pretrain` is deleted anyway.

### 2.4 #10 — pipeline↔method pairing: I take the plan's side, with the review's logging suggestion

The plan resolved "no validation; any combination allowed" as a *user* decision;
the review called this a gap.  I verified the actual failure modes:

- `--pipeline images --method vo_pair` → `build_model` builds `VOSimilarityNet`
  but `train_step` receives a single image tensor where it unpacks
  `(image_a, image_b)` → crashes on the first forward.
- `--pipeline vo_pair --method simmim` → the images pipeline is bypassed;
  SimMIM gets `blob.data` = `(image_a, image_b)` and builds a mask over a
  tuple → crashes.

Both are deterministic-and-fast user errors, so a startup `fatal` is not needed
to "fail early" — the failure already happens on batch 0.  **Joint resolution
(2026-09-15, confirmed by user):** keep the no-validation decision but adopt the
review's own middle path — log the resolved `--pipeline` + `--method` pair at
`info` on startup (§5.1 `main()` now does `logging.info("Resolved
pipeline=%s method=%s", ...)`, and §12-Q3 documents the rationale), so a wrong
pairing is diagnosable without inventing a maintenance surface for a
compatibility matrix that the project deliberately does not want.

---

## 3. The one genuinely new defect the review missed

### 3.1 `ModelOutput` name collision (already covered in §2.1 — this is the find)

The only *new* defect I found that is absent from all 22 review points is the
pre-existing `genml_kit.models.registry.ModelOutput` (`.logits` semantics,
exported from `genml_kit.models`, imported by four production files and three
tests — full list in §2.1).  The review's #2 treated the plan's `ModelOutput`
as a gratuitous abstraction (true) but could not know the name was already
claimed.  This upgrades the fix from "pick one of the two options" to "you MUST
not introduce a second type with this name" — which is why the plan now
documents it in §2.2 and why `models/registry.ModelOutput` remains the only
canonical `ModelOutput`.

### 3.2 Secondary observation (retracted — verified FALSE)

v4.1 draft claimed the plan's §1.3 citation of `pretrain/cli.py:540-543` as the
"two-pass parse" was wrong ("line 559 is a plain parse_args; the two-pass lives
in main wiring").  **Retracted after re-verification:** the two-pass parse IS
at `pretrain/cli.py:540-543`, inside `parse_args()` itself:

```python
# Two-pass parse: first to get --method, then add its args.
known, _ = parser.parse_known_args(argv)
method_cls = get_method(known.method)
method_cls().add_args(parser)
```

It sits between the shared-flag block (ends :538) and the final
`args = parser.parse_args(argv)` (:559).  The plan's §1.3 citation is correct;
the counter-review's original observation was an error (a truncated read
skipped the 540-543 lines).  Kept here as a retraction record.

---

## 4. Recommendations for the joint review

1. Approve the **`Method.train_step(model, blob, global_step, *, labels=None)`**
   signature (§3.1) as the core contract fix — it is what allows BYOL/DINO
   momentum and IJEPA mode-switching to keep working; the earlier
   `loss_fn(model_output, blob)` could not.
2. Approve **dropping the contracts `ModelOutput`** in favour of the existing
   `models.registry.ModelOutput` (§2.2, §3.1 note).
3. Approve keeping **`--vo_stage` int (0/1)** and the single `vo_pair` name
   (§2.3) — or, if you prefer split names, treat it as a follow-up item.
4. Note that **`pretrain/losses/` and `pretrain/augmentations/` survive** (§9)
   — the review's #20.1 "never decided" is now decided: they stay, with only
   `pretrain/cli.py` and `pretrain/methods/` being deleted.
5. The **doc revision inventory (§12-Q7)** is now the verified full set; note
   also that `vo/README.md` needs NO changes (it is a pure-math tutorial with
   no CLI references) — the v4 plan's "`vo/README.md` references" clause was
   wrong and is dropped.

---

*Counter-review conducted 2026-09-15 against `plans/REVIEW.md` (revised
2026-09-15) and `plans/GENERIC_PIPELINE.md` v4.1.  Every claim re-verified
against the current source tree; no source files modified.*