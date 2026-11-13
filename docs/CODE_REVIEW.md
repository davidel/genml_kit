# genml_kit — Code Review

Scope of this review:

- **In scope:** the `genml_kit/` package (the importable library) and its `tests/`.
- **Out of scope:** `experiments/` (local/temporary subrepo) and top-level
  documentation (`README.md`, `rl/`, `vo/`, `docs/`).
- **Out of scope:** `scripts/` and `scdiag/scripts/` are standalone scripts that
  must not import each other, so they may use plain `raise` and are not
  reviewed for cross-script duplication.

Project conventions applied as the yardstick:

1. Clear, professional APIs; **no cut-and-paste duplication** where a shared
   helper makes sense.
2. Shape-transformation comments must sit **on the line preceding** the
   transformation, not trailing it.
3. Python Google style, 2-space indent.
4. The **core `genml_kit` package must use `fatal(...)`**, never a bare
   `raise`; `scripts/` may raise directly.
5. No Ruff-silencing annotations without explicit sign-off.

Each issue below was verified against the source. Anything from the earlier
review that did **not** survive verification is listed in
[Appendix A](#appendix-a-retracted-claims) with the reason it was retracted.

## Resolution status

`scripts/` and `scdiag/scripts/` refactoring was intentionally **dropped**:
those files must stay standalone (no cross-file imports), so the duplication
there is accepted.

| # | Issue | Status |
|---|-------|--------|
| 1.1 | BYOL/DINO EMA momentum plumbing duplicated | **Fixed** — shared `compute_step_budget` + `linear_momentum` in `models/ema.py`; dead `contextlib` imports removed |
| 1.2 | RL model resolution triplicated + magic default-model string | **Fixed** — `RLMethod.resolve_model_name` / `model_kwargs`; new `DEFAULT_VISION_MODEL` constant |
| 2.1 | Dead `validate()` hooks in four methods | **Fixed** — removed; the stale test now asserts the real `log_validation` hook |
| 2.2 | Test scaffolding (`_ScriptedEnv`, `_make_scripted_env`) in production | **Fixed** — moved to `tests/scripted_env.py`; dead classmethod deleted |
| 2.3 | `QNetwork` missing `soft_update` | **Fixed** — added; DQN now calls it instead of hand-rolling the Polyak loop |
| 2.4 | `build_loader` signature drift | **Fixed** — RL/VO overrides now match the base contract |
| 2.5 | Inconsistent `fatal()` vs bare `raise` in core | **Fixed** — genuine-error sites converted; protocol/`__getattr__`/argparse/typed-search `raise`s deliberately kept |
| 3.1 | Inline shape comments | **Fixed** — core + test + README relocated to the preceding line |
| 3.2 | Inner-class `SACOptimization` | **Fixed** — hoisted to module scope |
| 3.3 | Classification-specific `resume_checkpoint` default | **Fixed** — `metric_key` is now required; callers updated |
| 3.4 | Object-key duplication in storage helpers | **Fixed** — single `_object_key` helper |
| 3.5 | Validation-transform duplication | **Fixed** — `train_compat` reuses `build_val_transform` |
| 3.6 | `raise fatal(...)` | **Fixed** — leading `raise` dropped |

Verification: `ruff check genml_kit tests` passes, and the full suite is
**2567 passed** (identical to the pre-change baseline, no regressions) with
`yapf` (`.style.yapf`) applied to all touched files.

---

## Severity 1 — High

### 1.1 Byte-identical EMA momentum plumbing duplicated in BYOL and DINO

**Files:** `genml_kit/methods/byol.py`, `genml_kit/methods/dino.py`
**Also relevant:** `genml_kit/models/ema.py` (already exposes `linear_momentum`)

`methods/ijepa.py` and `methods/distillation.py` already delegate to the shared
`linear_momentum`. BYOL and DINO do not: they carry three copy-pasted pieces.

**(a) `wire_data` — identical optimizer-step budget block** (`byol.py:116`,
`dino.py:171`), including the identical docstring:

```python
total = None
with contextlib.suppress(TypeError):
  # Iterable-only datasets: no len().
  total = len(pipeline.train_loader)
if total:
  total = total * args.epochs // max(args.grad_accum_steps, 1)
self._total_steps = total
```

**(b) `_current_momentum` — algebraically identical re-implementation**
(`byol.py:161`, `dino.py:234`):

```python
ratio = min(global_step / total, 1.0)
start = self._momentum_start()
end = self._momentum_end()
# Identical to ema.linear_momentum(global_step, total, start, end):
return end + (start - end) * (1.0 - ratio)
```

**(c) `_momentum_start()` / `_momentum_end()`** differ only in the attribute
name (`_byol_momentum` vs `_dino_momentum`).

**Impact:** three copies of the same schedule logic means three places to fix a
ramp bug; the two formulas *look* different (`end + (start-end)*(1-ratio)` vs
`start + (end-start)*ratio`) so a reviewer can easily miss that they agree.

**Suggested fix:** a small shared base/mixin (e.g. `EMAMomentumMixin` in
`models/ema.py`) holding `_total_steps`, a `wire_data` helper
`compute_step_budget(args, pipeline)`, and:

```python
# BYOL/DINO then only declare their flag names.
def _current_momentum(self, global_step, model):
  return linear_momentum(global_step, self._total_steps or 0,
                         self._momentum_start(), self._momentum_end())
```

---

### 1.2 RL model resolution triplicated + magic default-model string

**Files:** `genml_kit/methods/rl_ppo.py`, `genml_kit/methods/rl_sac.py`,
`genml_kit/methods/rl_dqn.py`

Every RL method repeats the same two-part prologue in `build_model`.

**(a) The default-model guard** (`rl_ppo.py:188`, `rl_sac.py:203`,
`rl_dqn.py:152`):

```python
model_name = args.model
if not model_name or model_name == "google/vit-base-patch16-224":
  model_name = "rl/actor_critic"   # or "rl/sac_critic" / "rl/qnet"
```

The literal `"google/vit-base-patch16-224"` is the default of `ARG_MODEL`
(`cli/train.py:86`). The dependency is implicit and unenforced: if that
default ever changes, all three RL methods silently start loading a vision
model into an RL run.

**(b) The `model_arg`-filter block**, each with its own hard-coded exclusion
set — 3 keys for PPO (`rl_ppo.py:191`), 7 for SAC (`rl_sac.py:196`), 9 for DQN
(`rl_dqn.py:155`):

```python
model_kwargs = {
    k: v for k, v in args.model_arg.items()
    if k not in {"log_std_init", "log_std_min", "log_std_max"}
}
```

**Impact:** cut-and-paste with three different exclusion sets and a hidden
coupling to the CLI default.

**Suggested fix:** in `methods/rl_utils.py` (`RLMethod`):

```python
def resolve_model_name(self, args, registered):
  """Fall back to the RL factory when --model is unset or is the vision default."""
  if not args.model or args.model == DEFAULT_VISION_MODEL:
    return registered
  return args.model

def filtered_model_arg(self, args):
  """forward --model_arg minus this method's own flags."""
  return {k: v for k, v in args.model_arg.items()
          if k not in self.MODEL_ARG_EXCLUDES}
```

with `DEFAULT_VISION_MODEL` defined once next to `ARG_MODEL`, and each method
declaring a class-level `MODEL_ARG_EXCLUDES` set.

---

## Severity 2 — Medium

### 2.1 Dead `validate()` hooks in four methods (unsupported by the base API)

**Files:** `genml_kit/methods/byol.py:187`, `genml_kit/methods/dino.py:276`,
`genml_kit/methods/supcon.py:108`, `genml_kit/methods/distillation.py:496`

Each defines:

```python
def validate(self, model, images, num_samples):
  # No pixel-space visualization for contrastive.
  pass
```

But `Method` (`methods/base.py`) has **no `validate` hook**. The real
visualization hook is `log_validation(...)` (`base.py:242`), which the trainer
calls (`training/trainer.py:520`). The only reason these `validate` methods
still "exist" is that `tests/test_distillation_method.py:236` asserts
`method.validate(None, None, 0) is None`; otherwise they are dead.

Only `methods/simmim.py` and `methods/segmentation.py` implement the real
`log_validation` hook. The four `validate` methods above are therefore pure
noise, kept alive solely by a test that locks in the wrong contract.

**Suggested fix:** delete the four `validate` methods and the asserting test
line. If a per-method no-op visualization is desired, rely on the **already
no-op** default `log_validation` in the base class.

---

### 2.2 Test scaffolding shipped inside production code

**File:** `genml_kit/pipelines/rl.py`

Two related problems:

1. `_ScriptedEnv` (`rl.py:136`) is a fake environment used only by tests
   (`tests/test_rl_trainer.py:16`, `tests/test_rl_pipeline.py`), yet it lives
   in the production module.
2. `_make_scripted_env(...)` (`rl.py:658`) is a `@classmethod` whose **only**
   grep hit is its own definition — it is dead code.

**Suggested fix:** move `_ScriptedEnv` into `tests/` (or a clearly-named
`genml_kit/pipelines/_testing.py`) and delete `_make_scripted_env`.

---

### 2.3 `QNetwork` is missing `soft_update`, so DQN hand-rolls the Polyak loop

**Files:** `genml_kit/models/rl/qnetwork.py`, `genml_kit/models/rl/sac_critic.py`,
`genml_kit/methods/rl_dqn.py:200`

`SACCritic` exposes both `hard_update()` and `soft_update(tau)` backed by the
shared `EMAModel` (`sac_critic.py:66-68`). `QNetwork` exposes only
`hard_update()` (`qnetwork.py:154`) — even though it *also* holds an
`EMAModel` (`qnetwork.py:121`). As a result DQN re-implements the Polyak blend
by hand in `update_target`:

```python
elif self._tau < 1.0:
  # Soft Polyak update: target = (1-tau)*target + tau*online.
  with torch.no_grad():
    for p, pt in zip(model.online.parameters(), model.target.parameters()):
      pt.data.mul_(1.0 - self._tau).add_(p.data, alpha=self._tau)
```

This duplicates `EMAModel.update()` (`ema.py:59`) — the very abstraction the
model already owns.

**Suggested fix:** add to `QNetwork`:

```python
def soft_update(self, tau):
  """Soft Polyak update: target = (1-tau)*target + tau*online."""
  self._ema.update(1.0 - tau)
```

and call `model.soft_update(self._tau)` in DQN, mirroring SAC.

---

### 2.4 `build_loader` override signature drift

**Files:** `genml_kit/pipelines/base.py:45` (contract),
`genml_kit/pipelines/rl.py:358`, `genml_kit/pipelines/vo_pair.py:91`

The base contract is:

```python
def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs): ...
```

`RLPipeline.build_loader(self, args, **kwargs)` and
`VOPairPipeline.build_loader(self, args, mode="train", **kwargs)` drop
`needs_labels` and (for RL) `mode`. Callers that pass these keywords
(`cli/train.py:380` passes `mode=`, and the base forwards `needs_labels=`) get
them silently absorbed into `**kwargs` rather than failing or being honoured.

**Suggested fix:** keep the full base signature on every override and simply
return `None` / ignore the unused arguments explicitly.

---

### 2.5 Inconsistent `fatal()` vs bare `raise` in the core package

**Scope:** all of `genml_kit/` (scripts excluded per convention)

The project rule is that the core package uses `fatal()`. In practice the
package mixes both. Confirmed bare-`raise` sites in core (non-`__getattr__`,
non-`NotImplementedError`):

| File | Line | Statement |
|------|------|-----------|
| `datasets/rollout_buffer.py` | 79 | `raise RuntimeError("RolloutBuffer overflow…")` |
| `datasets/rollout_buffer.py` | 105 | `raise ValueError("set_next_values no longer accepts scalars…")` |
| `datasets/rollout_buffer.py` | 241 | `raise ValueError("RolloutBuffer is empty")` |
| `datasets/ensemble.py` | 182, 470 | `raise TypeError("streaming … has no length")` |
| `datasets/hf_proxy.py` | 196, 296 | `raise TypeError(…)` |
| `datasets/text_proxy.py` | 94 | `raise TypeError("streaming text dataset has no length")` |
| `models/encoder_utils.py` | 64, 144, 151 | `raise ValueError("…backbone…")` |
| `models/layer_match.py` | 68 | `raise ValueError("No module name matched…")` |
| `models/rl/spaces.py` | 62 | `raise ValueError("Cannot infer obs_dim…")` |
| `models/uvito/model.py` | 80, 88, 92 | `raise ValueError("feature_tap…")` |
| `losses/distillation.py` | 48, 168 | `raise ValueError("kd_…")` |
| `methods/supcon.py` | 90 | `raise ValueError(…)` |
| `methods/classification.py` | 157 | `raise ValueError("…requires labels…")` |
| `training/vo/train_vo.py` | 63, 124 | `raise ValueError("…mce…")` |
| `io/checkpointing.py` | 106, 108 | `raise ValueError("Unsafe LoRA…")` |

Legitimate bare raises that should be **kept** as-is:

- `raise NotImplementedError` (`pipelines/base.py:56,60`,
  `datasets/iterable.py:59`, `datasets/stream_batching.py:113`,
  `search/{pruners,samplers}.py`) — abstract methods / unimplemented paths;
  `fatal()` would be inappropriate (no logging wanted, and subclasses raise it
  by design).
- `raise AttributeError(name)` inside `__getattr__`
  (`utils/distributed.py:467,806`, `datasets/capped.py:100`,
  `training/train.py:273`) — required protocol for attribute lookup; `fatal()`
  would break `hasattr`/`getattr` semantics.
- `raise argparse.ArgumentTypeError` (`utils/shared_args.py:46,48`) — argparse
  contract.
- `utils/signal.py` and `utils/logging.py` — `InterruptedException` and the
  `fatal` implementation itself.
- The custom `search/errors.py` hierarchy
  (`StudyNotFound`, `SearchSpaceExhausted`) — deliberately typed exceptions for
  programmatic handling; wrapping them in `fatal()` would lose the type. This is
  a defensible exception to the rule and should be documented as such.

**Suggested fix:** convert the first table's sites to `fatal(msg, Exc)`,
mirroring the many already-correct call sites (e.g. `classification.py:134`).
Where a typed exception must be preserved (search errors, `InterruptedException`),
add a one-line comment noting the deliberate exception to the convention.

---

## Severity 3 — Low

### 3.1 Shape comments written inline instead of on the preceding line

**Convention:** a comment describing a shape/dimension transformation MUST be
placed on the line **preceding** the transformation, never trailing on the same
line. (Project rule, reaffirmed.)

The great majority of the package already complies (30+ `# (B, …) -> (B, …)`
comments across `methods/`, `losses/`, `models/` all sit on the preceding line).
The trailing form below is the exception — and it is **not** a single outlier,
as an earlier draft claimed; it recurs across the package, the tests, and the
README.

**Core package (`genml_kit/`) — clear shape-transformation comments, must move
to the preceding line:**

| Location | Code |
|----------|------|
| `methods/segmentation.py:247` | `grid = np.concatenate(rows, axis=0)  # (n*H, 3*W, 3) uint8` |
| `methods/causal_lm.py:282` | `logits = out.logits  # [B, L, V]` |

**Borderline — axis annotation (judgment call):**

| Location | Code |
|----------|------|
| `losses/segmentation.py:56` | `dims = (0, 2, 3)  # sum over batch + spatial` |

This last one documents which axes a reduction runs over rather than a shape
transform; it is harmless but could move above the line for consistency.

`methods/causal_lm.py:282` is particularly illustrative: it sits directly below
a column of correctly-placed preceding-line comments, so the anomaly is easy to
overlook:

```python
# shift so token t predicts token t+1: (B, L) -> (B, L).
...
logits = out.logits  # [B, L, V]        # <-- should be on its own preceding line
```

**Tests (`tests/`) — should also conform:**

| Location | Code |
|----------|------|
| `tests/test_packed_attention.py:60` | `q = torch.randn(1, 2, 4, 5)  # [B, H, L, D]` |

**Documentation (`README.md`) — propagates the anti-pattern in examples:**

| Location | Code |
|----------|------|
| `README.md:1945` | `def forward(self, hidden_states):          # (B, N, D) tensor` |
| `README.md:1949` | `def extract_features(...):                # (B, N, D) → (B, D)` |

**Suggested fix** (example for `methods/causal_lm.py:282`):

```python
# LM head output: [B, L, V] logits over the vocabulary.
logits = out.logits
```

And analogously for each row above; the README examples should be reflowed so
the shape sits on its own line before the signature/statement.

---

### 3.2 Optimizer bundle class defined inside a method body

**File:** `genml_kit/methods/rl_sac.py:244` (`build_optimization`)

A class `SACOptimization` is defined *inside* the method and returned. Defining
a class per call is awkward for pickling/`isinstance` and hides a real type.

**Suggested fix:** hoist `SACOptimization` to module scope (or its own small
module next to `models/rl/sac_model.py`).

---

### 3.3 `resume_checkpoint` default `metric_key` is classification-specific

**File:** `genml_kit/io/checkpointing.py:482`

```python
def resume_checkpoint(ckpt_latest, ckpt_best, model, device,
                      metric_key="best_macro_f1", default_metric=0.0):
```

The sole production caller (`checkpointing.py:676`) always passes `metric_key`
explicitly; only tests lean on the default. The default leaks a
classification concern into generic IO code.

**Suggested fix:** make `metric_key` required, or default to `None` and resolve
explicitly (raising/`fatal` if a metric-bearing checkpoint is consulted without
a key).

---

### 3.4 Duplicated object-key construction in storage helpers

**File:** `genml_kit/io/storage_utils.py`

The idiom

```python
blob_name = os.path.basename(local_path)
if prefix:
  blob_name = f"{prefix}/{blob_name}"
```

is repeated across `_upload_s3`, `_upload_gcs`, `_upload_r2`, `_download_s3`,
`_download_gcs`, `_download_r2` (~6 sites).

**Suggested fix:** one helper, e.g.
`def _object_key(local_path, prefix): return f"{prefix}/{os.path.basename(local_path)}" if prefix else os.path.basename(local_path)`.

---

### 3.5 Validation-transform duplication

**Files:** `genml_kit/training/model_utils.py:311` (`build_val_transform`) vs
`genml_kit/training/train_compat.py:87` (inline `val_augmentations`)

Both build the same `Resize → CenterCrop → ToImage → ToDtype → Normalize` chain;
the only difference is the `*tail` list abstracting the processor step.

**Suggested fix:** have `train_compat.build_augmentations` call
`build_val_transform(processor, image_size)` (composing the tail), removing the
inline copy.

---

### 3.6 `raise fatal(...)` — dead, misleading `raise`

**File:** `genml_kit/datasets/ensemble.py:234`

```python
for member, value in next(iter(source)).items():
  if isinstance(value, Image.Image):
    return member
# Redundant keyword: fatal() already raises, so this outer `raise` is
# never reached.
raise fatal(
    "webdataset: the first sample holds no image member.  Pass "
    "--image_column naming the member that holds the image.", ValueError)
```

`fatal()` (`utils/logging.py:57`) logs and then `raise exc(msg)` internally, so
`fatal(...)` never returns to the caller and the trailing `raise` is **dead
code**. The emitted exception is still the correct `ValueError(msg)` (the inner
raise propagates first), so this is a readability hazard rather than a runtime
bug — but it is the only `raise fatal` in the package and reads as if `fatal`
were a factory returning an exception.

**Suggested fix:** drop the leading `raise`:

```python
fatal(
    "webdataset: the first sample holds no image member.  Pass "
    "--image_column naming the member that holds the image.", ValueError)
```

---

## Appendix A — Retracted claims

These appeared in earlier review drafts but did **not** survive verification.

| Claim | Verdict | Evidence |
|-------|---------|----------|
| `ARG_METHOD` / `ARG_PIPELINE` hard-code `choices` lists that must be kept in sync | **Bogus** | `shared_args.py:190` has no `choices`; `cli/train.py:76,83` already use `choices=PIPELINES.list_names()` / `METHODS.list_names()`. The suggested fix was already implemented. |
| Shape comments are "mostly inline" and need an audit | **Partially bogus (original direction), but the "only one offender" counter-claim was also wrong** | The bulk of shape comments ARE on the preceding line, so "mostly inline" is false. However a first correction claiming "exactly **one**" trailing comment was itself an undercount: **three** exist in core (`methods/segmentation.py:247`, `methods/causal_lm.py:282`, `losses/segmentation.py:56`), plus one in `tests/` and two in `README.md`. See issue 3.1. |
| `utils/logging.py` risks a circular import from `datasets` | **Bogus** | `logging.py` imports only stdlib + `torch.utils.tensorboard`; no `genml_kit` import. |
| Public functions are missing docstrings (e.g. `parse_dataset_specs`, `add_logging_args`) | **Unsubstantiated** | Both named functions have full Args/Returns docstrings; spot checks found thorough docstrings throughout. |
| I-JEPA, DINO and distillation all duplicate momentum scheduling | **Partially bogus** | `ijepa.py:347` and `distillation.py:378` already call the shared `linear_momentum`. Only BYOL and DINO duplicate it (issue 1.1). |
| `_select_image_field` is duplication that should move to `datasets/transforms.py` | **Not duplication** | Single definition (`images.py:128`) with a docstring, used twice *within its own module*. Cosmetic at most; not listed as an issue. |
| `fatal()` vs `raise` is "just an inconsistency" | **Refined, not bogus** | There *is* a real convention violation in core (issue 2.5) — but it must exclude the legitimate cases (`NotImplementedError`, `__getattr__`, argparse, typed search errors). |

---

## Appendix B — Verified strengths (keep as-is)

- `losses/rl.py` — every transformation documents its shape on the preceding
  line; no duplication.
- `methods/rl_utils.py` — `rl_evaluate` was correctly extracted from three
  near-identical `evaluate` implementations.
- `models/encoder_utils.py` — shared backbone-dim detection + feature
  extraction removes triplication across Contrastive/BYOL/DINO.
- `models/ema.py` — the `EMAModel` / `linear_momentum` seam; two of four
  callers already use it (see 1.1 to finish the job).
- `io/checkpointing.py`, `utils/distributed.py`, `search/` — cohesive, well
  documented, single-source-of-truth.
- CLI choices are already wired to the registries (`cli/train.py:76,83`).
- Surfaces are clean: `ruff check genml_kit tests` passes with no suppressions.
