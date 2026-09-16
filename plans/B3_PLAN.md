# B3 Implementation Plan — Method lifecycle contract (branchless driver)

Status: **DRAFT for review** (implementation not started)
Scope: `genml_kit/methods/base.py`, `genml_kit/methods/classification.py`, the
  six self-supervised methods, `genml_kit/training/train.py`,
  `genml_kit/training/train_compat.py`, `genml_kit/pipelines/base.py`,
  `genml_kit/pipelines/images.py`, `genml_kit/pipelines/vo_pair.py`,
  `README.md`, `tests/`.
Baseline: `pytest` → **914 passing**; `ruff check` → clean.
Depends on: nothing (self-contained on top of A1–A3, B1, B2, B4, C1–C7, 2.8).

---

## 1. Problem statement

Commit `ec19359` promised a *branchless generic harness*, but the driver
`genml_kit/training/train.py` still type-sniffs the method in three places:

```python
# train.py:381  (transform resolution)
is_classification = isinstance(method, get_method("classification"))
if is_classification:
  _resolve_classification_transforms(args, device)

# train.py:391  (label/criterion wiring)
if is_classification:
  _wire_classification(args, pipeline, method)

# train.py:462  (model construction)
if isinstance(method, get_method("classification")):
  model = _load_classification_model(args, device, pipeline, method)
  ...
  model = method.build_model(args, device)   # <- other methods

# train.py:526  (XGBoost post stage)
if isinstance(method, get_method("classification")):
  from genml_kit.training.train_compat import maybe_train_xgboost
  maybe_train_xgboost(args, pipeline, device)
```

Related defects this plan fixes:

- **Abstract-contract violation** (REVIEW §2.3): `build_model` is
  `@abc.abstractmethod` on `Method`, yet `ClassificationMethod.build_model`
  returns `None` with a comment that "the CLI stands in". The driver must
  dispatch around the broken implementation.
- **`apply_freeze_patterns` asymmetry** (train.py:462-472): the classification
  branch applies freeze patterns after model construction; the generic branch
  does **not**. A self-supervised run with `--freeze` silently ignores it.
- **Hidden ordering constraint**: the classification path requires transforms
  to be resolved *before* loaders are built, and the label space *before*
  model construction. Today this ordering is enforced only by comment.
- **Capabilities locked to one method**: LoRA and the XGBoost post-stage are
  reachable only through the classification branch, even though both are
  method-agnostic in principle.

## 2. Design — the `Method` lifecycle contract

The driver stops asking *what* the method is and starts asking the method to
*do its part* at well-defined points. `Method` gains four optional lifecycle
hooks with no-op base defaults; the driver invokes them unconditionally in a
fixed order.

### 2.1 The contract

| # | Hook | Base default | Driver call site | Absorbs |
|---|---|---|---|---|
| 1 | `prepare_transforms(args, device)` | no-op | before `build_loader` | `_resolve_classification_transforms` |
| 2 | `wire_data(args, pipeline)` | no-op | after both loaders built | `_wire_classification` |
| 3 | `build_model(args, device)` | **abstract (exists)** | replaces `_build_model` dispatch | `_load_classification_model` body |
| 4 | `post_train(args, pipeline, device, result)` | no-op | end of `main()` | XGBoost `isinstance` branch |

Lifecycle order (the one invariant the driver guarantees):

```
parse_args -> normalize_args -> seed -> device
  -> method.prepare_transforms(args, device)     # may set args.train/val/tta_transforms
  -> pipeline.build_loader(args, "train", method=method)
  -> pipeline.build_loader(args, "val",   method=method)
  -> method.wire_data(args, pipeline)            # label space, criterion, ...
  -> model = method.build_model(args, device)    # real model from every method
  -> method.load_checkpoint_state(model, {}, args)
  -> open_resume_context(...)                    # weights resume on top
  -> build_optimization(...)
  -> BaseTrainer(...).run()
  -> [interrupt guard] -> method.post_train(args, pipeline, device, result)
```

Docstring rule for every hook: **state the position in the lifecycle, the
caller, what may be mutated (`args`, pipeline attributes), and when overriding
is appropriate.** These hooks are part of the public extension surface; their
docs are the contract.

### 2.2 Capability-level hooks (LoRA for every method, XGBoost for any classifier)

Two capability helpers move to `Method` as *protected* reusable steps so every
method can opt in without inheriting behavior it did not ask for:

#### 2.2.1 `_apply_model_extras(args, model, device)` — post-construction polish

Exactly one place applies everything that wraps or adapts an already-built
model, **for every method**:

```python
# methods/base.py
def _apply_model_extras(self, args, model, device):
  """Apply grad checkpointing, freeze patterns and LoRA to a built model.

  Called by `build_model` implementations AFTER constructing the raw
  model and moving it to *device*.  Every method gets the same treatment;
  overrides are only needed when a wrapper model must NOT be adapted
  (e.g. pass the backbone instead of the wrapper).
  """
  if getattr(args, "grad_checkpoint", False):
    from genml_kit.training.model_utils import enable_grad_checkpointing
    enable_grad_checkpointing(model)
  if getattr(args, "lora", False):
    from genml_kit.training.model_utils import apply_lora
    model = apply_lora(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=(args.lora_target_modules.split(",")
                        if args.lora_target_modules else None),
    )
  elif getattr(args, "source_checkpoint", None):
    from genml_kit.io.checkpointing import load_checkpoint_weights
    load_checkpoint_weights(args.source_checkpoint,
                            model,
                            device=device,
                            param_rename=args.param_rename)
  from genml_kit.training.train_compat import apply_freeze_patterns
  apply_freeze_patterns(args, model)
  return model
```

Notes:

- **LoRA promotion, made generic**: today LoRA only fires on the
  classification branch. Moving the `apply_lora` call here makes `--lora`
  work for all seven methods. For wrapper models (SimMIM, DINO, BYOL, SupCon,
  IJEPA), the method decides *what* gets wrapped by passing the right module —
  see per-method notes in §3. PEFT requires `target_modules` to resolve; the
  default `None` lets PEFT auto-detect attention projections in the backbone,
  which is the correct default for ViT-style backbones; users can override
  with `--lora_target_modules`.
- **`source_checkpoint` promotion**: same reasoning; weights can be loaded
  into any method's backbone. `--lora` and `--source_checkpoint` stay
  mutually exclusive (`elif`), exactly as today.
- **`apply_freeze_patterns` for everyone**: fixes the asymmetry —
  `--freeze` currently only works on the classification branch. It also
  interacts correctly with LoRA: `apply_freeze_patterns` already folds LoRA
  adapter params into the trainable set (`extract_lora_params`).
- Grad-checkpointing moves here too (it is also model polish), and the
  driver's `enable_grad_checkpointing` call disappears with `_build_model`.

#### 2.2.2 `maybe_train_xgboost` — owned by the method, implemented once

XGBoost is a *label-space classifier head* concern, not a "classification
method" concern: any future method that produces a frozen embedding + label
space (e.g. a supervised-contrastive-then-probe method) can reuse it. The
`--xgboost_model` / `--xgb_*` flags stay on `ClassificationMethod.add_args`
today, but the *decision* moves into the method:

```python
# methods/classification.py
def post_train(self, args, pipeline, device, result):
  """Train the XGBoost head on frozen embeddings after a successful run."""
  from genml_kit.training.train_compat import maybe_train_xgboost
  maybe_train_xgboost(args, pipeline, device)
```

`maybe_train_xgboost` already no-ops when `--xgboost_model` is unset, so the
method's hook is safe to call unconditionally. A future second classifier
method re-uses the same two lines and the same flags; when that happens, the
`--xgb_*` arg block moves to a shared mixin (noted in §7, not built now).

The interrupt guard stays in the **driver** (`_post_train` keeps checking
`result.interrupt_signals`); *whether* post-training runs is orchestration,
*what* runs is the method's business.

### 2.3 What the driver looks like afterwards

```python
def main(argv=None):
  ...
  method.prepare_transforms(args, device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)
  method.wire_data(args, pipeline)
  model = method.build_model(args, device)
  method.load_checkpoint_state(model, {}, args)
  states_to_load = parse_state_flags(args.state_load)
  model, start_epoch, best_metric, ckpt_extra = open_resume_context(...)
  method.load_checkpoint_state(model, ckpt_extra.get("method_state", {}), args)
  optimization = build_optimization(args, model, device, ckpt_extra, states_to_load)
  global_step = ckpt_extra.get(...)
  del ckpt_extra  # GC: drop optimizer/scheduler/scaler state (see REVIEW C3)
  writer = open_writer(...)
  result = BaseTrainer(...).run()
  _post_train(args, pipeline, method, device, result)

def _post_train(args, pipeline, method, device, result):
  if result.interrupt_signals and result.interrupt_signals != ["SIGINT"]:
    logging.info("Interrupted by %s; checkpoint saved, exiting.",
                 result.interrupt_signals)
    return
  method.post_train(args, pipeline, device, result)
```

Deleted from `train.py`: `_wire_classification`,
`_resolve_classification_transforms`, `_build_model`, `_load_classification_model`,
`_resolve_transforms` (its body moves into `ClassificationMethod`), the
`is_classification` variable, and now-unused imports
(`apply_freeze_patterns`, `enable_grad_checkpointing`, `load_model`,
`load_processor`, `torch` if orphaned). The two `isinstance(method, ...)` in
`_build_model` / `_post_train` disappear with their functions.

`main()`'s docstring is updated: wire order becomes the lifecycle table, the
"transforms before loaders" constraint is stated as contract item 1, and the
`s 5.1` legacy-order comment is dropped (the code now *is* the order).

## 3. Per-method changes

### 3.1 `ClassificationMethod` (the broken contract, fixed)

- `build_model(args, device)`: real implementation. Verbatim move of
  `_load_classification_model` (registry `load_model` with
  `self._num_labels/_id2label/_label2id`, LoRA, source checkpoint), then
  `self._apply_model_extras(args, model, device)`. Preconditions (fail fast,
  clear message, instead of a confusing HF error):
  - `self._num_labels is None` → `fatal("--method classification requires "
    "wire_data(...) (label space) before build_model; check run order",
    RuntimeError)`. This converts the implicit "wire_data ran first" contract
    into an explicit, testable one.
- `prepare_transforms(args, device)`: processor + `resolve_augmentations`
  (verbatim move of `_resolve_classification_transforms` + `_resolve_transforms`;
  lazy imports, sets `args.train_transforms/val_transforms/tta_transform`).
- `wire_data(args, pipeline)`: label-space guard (moved from
  `_wire_classification`: `pipeline.num_labels is None` → `fatal`), then
  orchestrates the existing public steps `set_label_space` /
  `set_mixup_alpha` / `build_criterion` (they stay public; `wire_data` is the
  orchestrator, not a replacement).
- `post_train`: XGBoost (§2.2.2).
- Keeps `set_label_space` etc. — they are used by `evaluate` today and are
  the fine-grained API; `wire_data` composes them.

### 3.2 Self-supervised methods — LoRA enablement, model wrappers

All six already implement `build_model` via
`load_model(args.model, num_labels=0, ...)` + wrapper construction + `.to(device)`.
Changes:

1. **End every `build_model` with `return self._apply_model_extras(args, model, device)`** —
   giving all of them `--lora`, `--source_checkpoint`, `--freeze`,
   `--grad_checkpoint` for free.
2. **What gets wrapped matters.** `_apply_model_extras` must target the
   backbone when the outer object is not a plain `nn.Module` that PEFT can
   adapt:

| Method | Outer model | `_apply_model_extras` target | Rationale |
|---|---|---|---|
| SimMIM | `SimMIM(ConvViTMaskedImageEncoder(base))` | the `SimMIM` wrapper | `SimMIM.forward` drives the masked path; PEFT injects into the encoder's attention through the wrapper. |
| SupCon | `ContrastiveEncoder(encoder, ...)` | the `ContrastiveEncoder` wrapper | plain module chain; adapter injection reaches the backbone. |
| DINO | `DINO(encoder, ...)` (contains student+teacher) | **the student's encoder**, then rebuild the teacher | LoRA on the student only; the teacher is an EMA copy and must mirror structure, not adapters (wrapping the teacher would double the adapters and break EMA state-dict symmetry). Implementation: build student encoder → `encoder = self._apply_model_extras(args, encoder, device)` → construct `DINO(student, teacher=deepcopy, ...)`. |
| BYOL | `BYOL(online, target, predictor)` | the **online encoder** only | same EMA reasoning as DINO: target network is a momentum copy; adapters live on the online branch. |
| IJEPA | `_PatchEmbedder(encoder)` student + teacher copy | the **student** embedder's encoder | teacher is `copy.deepcopy(student)` *after* adapter injection, so teacher structurally mirrors the student and stays grad-free. |
| vo_pair | `VOSimilarityNet` (custom registry model) | the `VOSimilarityNet` | custom backend, PEFT handles `nn.Linear` targets inside; documented as "supported, untested" — the method docstring notes it. |

   For DINO/BYOL/IJEPA this means a **build order swap inside the method**:
   construct/annotate the student-side encoder first, apply extras to it, then
   build the composite (EMA copies happen after adaptation). The composite
   classes need no changes; `copy.deepcopy` after adapter injection yields a
   teacher with identical (frozen, never-trained) adapter weights — correct
   for EMA semantics and for checkpoint round-tripping (adapters are part of
   the state dict, `resume_checkpoint` already tolerates LoRA keys — see
   `checkpointing.py` lora filtering).
3. **`vo_pair` precondition**: `build_model` reads `args.vo_stage` /
   `vo_loss_cfg`; unchanged.

### 3.3 `Method` base — the three new hooks + `_apply_model_extras`

```python
# --- Lifecycle hooks (called by the genml-kit-train driver, in order) ------

def prepare_transforms(self, args, device):  # noqa: B027
  """Resolve any transforms this method needs BEFORE loaders are built.

  Called by the driver after parsing, before `pipeline.build_loader`.
  Override to set `args.train_transforms` / `args.val_transforms` /
  `args.tta_transform` (e.g. from a model processor).  Default: no-op.
  """

def wire_data(self, args, pipeline):  # noqa: B027
  """Consume pipeline-built data attributes (labels, weights).

  Called after both loaders are built, before `build_model`.  Override to
  read `pipeline.num_labels` / `pipeline.class_weights` etc. and construct
  criteria.  Default: no-op.
  """

def post_train(self, args, pipeline, device, result):  # noqa: B027
  """Optional post-training stage (e.g. shallow-head probing).

  Called by the driver after `BaseTrainer.run()` returns, unless the run
  was interrupted.  Default: no-op.
  """

def _apply_model_extras(self, args, model, device):
  ...  # see 2.2.1
```

(The `# --- ... ---` banner comments here are part of C8, §5 — added
deliberately to group the lifecycle surface, matching the file's existing
style, then normalized file-wide by C8.)

## 4. Driver + compat surgery (file by file)

### `genml_kit/training/train.py`

- Delete: `_wire_classification`, `_resolve_classification_transforms`,
  `_build_model`, `_load_classification_model`, `_resolve_transforms`,
  `is_classification`.
- `_post_train` reduces to the interrupt guard + `method.post_train(...)`.
- `main()` rewritten to the §2.3 sequence; docstring updated to the lifecycle
  contract; imports pruned (`apply_freeze_patterns`,
  `enable_grad_checkpointing`, `load_model`, `load_processor`; keep
  `build_optimization`, `open_resume_context`, ...).
- `_default_metric` untouched.

### `genml_kit/training/train_compat.py`

- `maybe_train_xgboost` and `apply_freeze_patterns` stay (implementation
  home). Their docstrings gain one line noting the caller (method hooks).
- No re-export changes: `train.py` still imports `apply_freeze_patterns`? No
  — after the move, `train.py` no longer imports it; `methods/base.py` does.
  Verify no other consumer breaks (`tests/test_pipeline.py` uses
  `build_transforms`/`resolve_augmentations` via `train_mod` — those remain).

### `genml_kit/pipelines/base.py`, `images.py`, `vo_pair.py`

- No signature changes (B2 already landed `needs_labels=`).
- Docstring touch-ups only: `DataPipeline` class docstring mentions the
  lifecycle position of `build_loader` relative to `wire_data`/`build_model`.

## 5. C8 — decorative comment cleanup (project-wide)

**Policy**: a comment must carry information the code cannot. Section banners
whose only content is an em-dash rule (`# --- Title ---`) are deleted unless
they group ≥3 members of one cohesive API *and* the title adds vocabulary not
in the members' names. Retro-fix the two banners that survive this test to a
consistent 72-char style, or drop them.

Audit result (17 occurrences found by `git grep -nE "# ---"`):

| File:line | Banner | Verdict |
|---|---|---|
| `pipelines/base.py:22,27,42,48` | `# --- Arg surface ---`, `Loading`, `Device transfer`, `Transforms` | **keep (normalized)** — they group the four-stage pipeline API; titles add the "surface/stage" vocabulary. |
| `pipelines/images.py:85,130,160,232,342` | `Arg surface`, `Loader construction`, `Classification path (...)`, `Ensemble path (...)`, `Collation` | **keep (normalized)** — `Classification path` / `Ensemble path` document *which legacy builder* the section mirrors (real information). |
| `pipelines/vo_pair.py:36` | `Collation` | **delete** — a single method under a banner; the method name says it. |
| `methods/classification.py:34,85,98,155` | `CLI surface`, `Model`, `Training step`, `Validation` | **keep (normalized)** — four cohesive stages of a 200-line class. |
| `methods/classification.py:87-90` (inline) | comment inside `build_model` explaining why it returns `None` | **delete** — the function is replaced by a real implementation (§3.1); the excuse comment goes with it. |
| `training/trainer.py:94,187` | `Epoch body (s 4)`, `Loop` | **delete** — `run()`/`train_one_epoch()` names already say it; `(s 4)` is a stale plan reference. |
| `training/train_compat.py` | (none) | — |
| `methods/*` other files | (none) | — |

Also in scope (same policy, different syntax — found by audit):

- `genml_kit/training/train.py:56`: module-level comment
  `load_dataset = _datasets.load_dataset  # patchable via ...` — **keep**
  (explains a non-obvious alias that tests monkeypatch).
- `tests/test_pipeline_method_registry.py:40,123,151,288` (`# --- Registries ---` etc.):
  **keep (normalized)** — test files use banners as visual sectioning; that is
  idiomatic for test layout and the titles match the test-class names.
- Any `# noqa: B027` comments: untouched (they are linter directives, not prose).

Normalization rule for kept banners: `# --- <Title> ---------------------` with
padding to a common width per file (existing style), 2-space context
consistent with surrounding code. The banner style lives next to the project's
formatting conventions (see §7.3).

## 6. Tests

### 6.1 New tests (`tests/test_method_lifecycle.py`)

1. `test_base_hooks_are_noops`: a bare `Method` subclass (implementing only
   abstract members) — `prepare_transforms`, `wire_data`, `post_train` are
   callable with sentinel args and return `None`.
2. `test_base_apply_model_extras_grad_checkpoint`: `args.grad_checkpoint=True`
   on a tiny model → `enable_grad_checkpointing` was applied (flag on model or
   `torch.utils.checkpoint` presence per `model_utils` implementation).
3. `test_base_apply_model_extras_lora`: `args.lora=True` with a tiny
   `nn.Linear` model and `--lora_target_modules=fc` → returns a `PeftModel`
   whose adapter params are exactly the trainable set (mirrors
   `tests/test_lora.py` fixtures; skipped if `peft` not installed).
4. `test_base_apply_model_extras_source_checkpoint`: writes a tiny checkpoint
   via `save_checkpoint`, `args.source_checkpoint=<path>` → weights loaded
   (mirrors `tests/test_checkpointing.py`).
5. `test_base_apply_model_extras_freeze`: `args.freeze="fc"` → `fc.weight`
   `requires_grad is False`.
6. `test_classification_build_model_requires_wire_data`:
   `ClassificationMethod.build_model` without a prior label space →
   `fatal(...)` (`RuntimeError`) with the wire_data message.
7. `test_classification_build_model_happy_path`: with `wire_data` done
   (tiny num_labels) and the `dummy/tiny-model` patch used by
   `test_train_smoke` → returns a module whose `forward` yields `.logits`,
   and LoRA wrapping works on it (skip if no `peft`).
8. `test_classification_post_train_runs_xgboost_when_flagged` /
   `test_classification_post_train_noop_without_flag`: patch
   `maybe_train_xgboost`; assert called/not called. (Driver interrupt-guard
   behavior is already covered by `tests/test_trainer.py` interrupt tests.)
9. `test_main_is_branchless`: source inspection (`inspect.getsource` on
   `train.main`) asserts neither `isinstance(method` nor
   `get_method("classification")` appears — the regression guard for B3's
   core promise.
10. `test_self_supervised_lora_path` (SimMIM or SupCon, smallest fixture):
    `args.lora=True` end-to-end through `build_model` → adapter modules
    present inside the wrapper's encoder; one `train_step` runs (skip if no
    `peft`).

### 6.2 Existing tests

- `tests/test_trainer.py`: untouched (FakeMethod is a plain class; the driver
  hooks are not `BaseTrainer`'s business).
- `tests/test_pretrain_smoke.py` / `test_train_smoke.py`: must pass
  **unchanged** — they are the end-to-end regression for the lifecycle move
  (classification path via `test_train_smoke`, self-supervised via
  `test_pretrain_smoke`).
- `tests/test_pipeline_method_registry.py`: the `_Custom(Method)` fake already
  implements `build_model`/`train_step` only — inherits no-op hooks, keeps
  passing.
- `tests/test_pipeline.py` (`build_transforms` / `resolve_augmentations`
  direct calls): unchanged; the functions stay in `train_compat`.

### 6.3 Expected counts

914 existing + ~10 new = **~924 passing**; ruff clean; yapf-formatted.

## 7. Sequencing, verification, process

### 7.1 Commit sequence (each commit = one reviewable step)

1. **C8** — decorative-comment cleanup (pure deletion/normalization; zero
   behavior; smallest possible review). Files: `pipelines/vo_pair.py`,
   `training/trainer.py`, `methods/classification.py` (one inline comment),
   banner normalization in the kept files. Includes the §5 policy text into
   `plans/REVIEW.md` as C8 so the policy is documented where the rest of the
   cleanups live.
2. **Base hooks + `_apply_model_extras`** — `methods/base.py` only +
   `tests/test_method_lifecycle.py` items 1–5. No consumer changes yet;
   everything still green because nothing calls the new hooks.
3. **Self-supervised methods adopt extras** — the six `build_model` bodies
   (order swaps for DINO/BYOL/IJEPA per §3.2) + test item 10. Still
   behavior-neutral for runs without `--lora`/`--freeze` (defaults are off).
4. **Classification method absorbs its wiring** — §3.1: `build_model` real
   implementation, `prepare_transforms`, `wire_data`, `post_train`; tests
   6–8. Driver untouched, so old `_build_model` still dispatches to
   `ClassificationMethod.build_model`... **no** — the old driver calls
   `_load_classification_model`, not the method. So step 4 lands the method
   implementation but keeps `_build_model` calling it internally:
   `_build_model` classification branch becomes
   `method.build_model(args, device)` (one line), the other branch unchanged.
   Transitional, still green.
5. **Driver branchless flip** — delete the four helpers, rewrite `main()` to
   the lifecycle, add test 9. This is the only step where driver behavior
   changes, and every piece it calls already exists and is tested.
6. **Docs** — README: the `genml-kit-train` section gains a "Method lifecycle"
   subsection (the contract table); `plans/REVIEW.md` B3 marked implemented
   with pointer to this plan.

### 7.2 Verification per step

- `format_file` (yapf, `.style.yapf`) on every touched file.
- `ruff check genml_kit/ tests/` clean.
- `python -m pytest -q` green (914 → 914 → 914 → 914 → 919 → 919).
- Diff shown hunk-by-hunk with per-hunk rationale before each commit; commit
  only on explicit reviewer approval; no amend/squash ever; no
  `Co-Authored-By:`.

### 7.3 Standing conventions honored

- 2-space indent everywhere (yapf + `.style.yapf`).
- Class-level constants UPPERCASE (`NEEDS_LABELS`, `METRIC_KEY` precedent).
- No type annotations (project convention).
- No `# noqa` unless pre-existing/ruff-required (B027 on no-op hooks kept).
- Ruff silence never added without asking.

## 8. Explicit non-goals

- No dissolution of `train_compat` (XGBoost helpers stay there).
- No new pipelines or methods.
- No changes to `DataPipeline` signatures.
- No checkpoint format changes (adapters already round-trip).
- No speculative `finalize_model` public hook — `_apply_model_extras` is
  protected; promotion to public happens when a second consumer exists.
- No behavior change for runs that do not pass `--lora` / `--source_checkpoint`
  / `--freeze` / `--grad_checkpoint` / `--xgboost_model`.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| PEFT auto-detect fails on custom backbones (`ConvViT*`, `VOSimilarityNet`) | `--lora_target_modules` escape hatch documented in README; test 10 pins the SimMIM path; vo_pair marked "supported, untested" in docstring. |
| EMA/teacher copies double adapters or break state-dict symmetry (DINO/BYOL/IJEPA) | build-order swap: adapt student encoder first, `deepcopy` after; checkpoint round-trip covered by existing resume tests + smoke tests. |
| Hidden ordering dependency (transforms before loaders) breaks for a future pipeline | the ordering becomes a documented lifecycle invariant; `ImagesPipeline._build_classification` already reads `args.train_transforms` lazily at loader build. |
| `main()` rewrite regresses UX (arg defaults, two-pass parse) | parse path untouched; smoke tests cover both `--pipeline` defaults end-to-end; test 9 is the structural guard. |
| LoRA changes param-group naming → optimizer groups shift | `build_optimization` matches by regex over `named_parameters()`; adapters add `lora_*` names which simply form their own group; `--lr_group` lets users pin them. Covered by existing `optim_factory` tests. |
