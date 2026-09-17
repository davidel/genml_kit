# TODO — deferred items from plans/REVIEW_2.md

Second-round audit items that were **recorded but not implemented** in the
`72f5853` commit ("fix: implement REVIEW_2 findings #1-#6, #8-#13"). Each
section below states the open problem, the current code, the exact
reproduction, the decided/planned fix, and the validation expected.

Status anchor: working tree at commit `72f5853`.

---

## 1. `genml-kit-train --help` under-reports the CLI (REVIEW_2 #7)

**Status: IMPLEMENTED — single-pass always, `--help` renders every owner.**

### Problem (fixed)

`parse_args` was two-pass: `parser.parse_known_args(argv)` ran **before**
owner/shared flags were added, and argparse resolves `--help` inside
that first pass — so the tool printed only the ~18 generic flags.
`--dataset`, `--vo_length`, `--dino_local_num`, `--lr`, `--checkpoint`,
`--state_*` and every README CLI table flag were missing.

### What changed

`genml_kit/training/train.py`:
- New `register_all_owners(parser)` registers **every** pipeline and
  method via their `add_args` classmethod (`get_pipeline(name).add_args`
  / `get_method(name).add_args`), **owner args only** — no shared
  helpers inside it.
- `parse_args` is now single-pass-always: it builds the generic parser,
  calls `register_all_owners(parser)`, then the unchanged shared groups
  (`add_checkpoint_args`, `add_optimization_args`,
  `add_training_state_args`, `add_logging_args`, `add_source_checkpoint_args`,
  the inline `optimizer` group), then `parser.parse_args(argv)`. With
  everything registered before parsing, `--help` renders the complete
  CLI and `SystemExit(0)` fires naturally; no flag sniffing, no
  SystemExit catching, no second parser.

`add_args` became a **`@classmethod`** on every owner (both bases +
all concrete methods/pipelines). It no longer requires an instance, so
`register_all_owners` can call `ClassName.add_args(parser)` directly.
Classmethods remain callable on instances, so the existing
`method.add_args(parser)` call sites in tests are untouched.

### Collision policy (recorded from §7.1)

- Owner args are the owner's own declaration; the owner names its
  `add_argument_group`.
- Same-flag-same-intent: a flag is global only if semantically identical
  for every consumer and owned by a *concern* helper in `utils/args.py`,
  never a feature module. Different meaning ⇒ different name, never
  first-wins.
- **Enforcement**: `tests/test_cli_help.py::test_no_option_string_collisions`
  registers every owner on one parser and asserts no `ArgumentError`. If
  a future owner collides, `add_args` raises at registration time — the
  explicit signal to rename one of the flags (or promote it to a shared
  helper).

### Validation

- `tests/test_cli_help.py`:
  - `test_no_option_string_collisions` — all owners coexist on one
    parser (>80 actions registered).
  - `test_owner_groups_are_registered` — the group titles
    `"images pipeline"`, `"vo_pair pipeline"`, `"DINO"`, `"BYOL"`,
    `"SimMIM"`, `"I-JEPA"`, `"classification method"` all appear.
  - `test_help_prints_full_cli` — `--help` output contains `--dataset`,
    `--vo_length`, `--dino_local_num`, `--lr`, `--checkpoint`,
    `"images pipeline"`, `"DINO"`.
  - `test_help_short_flag` — `-h` behaves the same.
- Full suite green; `ruff check .` clean.

---

## 2. `--sampler balanced` divisibility: "warn but still run" (REVIEW_2 #10)

**Status: IMPLEMENTED — Option B with even-spread partial buckets
(default-on).**

### Decision

`BalancedBatchSampler` no longer hard-fails on indivisible batch sizes.
When `batch_size` is not a multiple of `samples_per_class`, the
remainder is spread **evenly (round-robin)** across the per-class
groups, so group sizes differ by at most one and every batch still
contains exactly `batch_size` samples:

- `batch_size=10, samples_per_class=3` → groups `[3, 3, 2, 2]`
- `batch_size=8,  samples_per_class=3` → groups `[3, 3, 2]`

`batch_size < samples_per_class` still fails fast (`ValueError`): a
batch cannot hold a full class group.

### What changed

`genml_kit/datasets/balanced_sampler.py`:
- The divisibility `fatal(...)` was replaced by a single
  `logging.warning(...)`; the hard error is kept only for
  `batch_size < samples_per_class`.
- `self._n_groups = ceil(batch_size / samples_per_class)` and a new
  `_group_sizes` attribute holds the even-spread distribution
  (`[q+1]*r + [q]*(spc-r)` where `r = batch_size % spc`).
- `__iter__` draws each group using its own size (not the nominal
  `samples_per_class`), preserving the per-group single-class invariant.

`genml_kit/pipelines/images.py`:
- Removed the `elif args.batch_size % args.samples_per_class:` branch
  that fell back to `shuffle=True`. `--sampler balanced` now always
  uses the sampler when a label column is present, even for
  non-divisible batch sizes.
- `--samples_per_class` help text updated.

Tests:
- `tests/test_balanced_sampler.py`: `test_raises_on_indivisible_batch`
  replaced by `test_indivisible_batch_spreads_remainder_evenly`,
  `test_indivisible_batch_group_sizes`, and
  `test_warns_on_indivisible_batch`; added
  `test_raises_when_batch_smaller_than_samples_per_class`.
- `tests/test_pipeline_method_registry.py`:
  `test_balanced_sampler_falls_back_on_indivisible_batch` became
  `test_balanced_sampler_used_on_indivisible_batch`, asserting the
  loader uses the sampler with group sizes `[2, 2]`.

Docs: README (`--sampler` table, `--samples_per_class` table + SupCon
tip) and `scdiag/README.md` updated to describe the even-spread
behavior; no "must be divisible" wording remains.

### Validation

- `tests/test_balanced_sampler.py` green (even-spread bucket sizes,
  single-class buckets, batch length, length semantics, warning).
- `tests/test_pipeline_method_registry.py` green (`--sampler balanced`
  used for non-divisible config; no RandomSampler fallback).
- Full suite green; `ruff check .` clean.

---

## 3. Missing-steps fallback for the EMA momentum ramp (REVIEW_2 #2)

**Status: IMPLEMENTED with the "start" fallback — the fallback VALUE is a
deliberate open question that was decided in review but not revisited
since.**

### The two parts of #2

1. **`_total_steps` was never set** → always `0` → `_current_momentum`
   returned the END momentum (1.0) → teacher permanently frozen.
   **FIXED in `72f5853`**: `wire_data` computes the budget
   (`epochs * len(train_loader) // grad_accum_steps`) and stores it on
   the method; `_current_momentum` reads it.
2. **Behavior when the budget is unknown** (e.g. `wire_data` not called
   — unit tests, custom pipelines, iterable-only datasets with no
   `__len__`). REVIEW_2 decided: fall back to the **START momentum**
   (`0.996`) so the teacher STILL MOVES rather than silently freezing.

### Current code (both DINO and BYOL)

`genml_kit/methods/dino.py:218-224` and
`genml_kit/methods/byol.py` (`_current_momentum`):

```python
total = getattr(self, "_total_steps", None)
if not total:
  return self._momentum_start()   # 0.996 for DINO and BYOL
ratio = min(global_step / total, 1.0)
start = self._momentum_start()
end = self._momentum_end()
return end + (start - end) * (1.0 - ratio)
```

`wire_data` (both files):

```python
total = None
with contextlib.suppress(TypeError):
  total = len(pipeline.train_loader)  # iterable-only datasets: no len()
if total:
  total = total * args.epochs // max(getattr(args, "grad_accum_steps", 1), 1)
self._total_steps = total
```

Regression tests in `tests/test_dino.py`
(`test_momentum_missing_budget_falls_back_to_start` etc.) assert the
start-fallback.

### Why this is still an open item

The `wire_data` budget computation has a subtle inconsistency with the
trainer's counter, and the fallback choice interacts with it:

1. **`grad_accum_steps` and partial last batch.** The trainer increments
   `global_step` only on grad-accum flushes (`trainer.py:117,132`), and
   `train.py:412` seeds the resumed counter with
   `start_epoch * (len(train_loader) // args.grad_accum_steps)` — note
   this uses **floor division**, while `wire_data` uses the same
   `// grad_accum_steps` on `epochs * len(train_loader)`. `epochs * L` is
   not always divisible by `grad_accum_steps`, and the last epoch's tail
   samples may never form a full optimizer step, so the ramp can end
   slightly before/after the true final step. Decide whether to compute
   the budget as `(epochs * L) // grad_accum_steps` (current, may
   undershoot a bit) or `sum over epochs of (L // grad_accum_steps)` /
   `epochs * (L // grad_accum_steps)` (matches the trainer's per-epoch
   flush semantics exactly). The plan's §6 clarification recommended the
   optimizer-step total; the exact formula should be re-verified against
   `trainer.py::train_epoch` when implementing.
2. **Fallback value re-check.** `_momentum_start()` was chosen over
   `_momentum_end()` because "a missing budget can no longer silently
   freeze the teacher". But for DINO, a start-momentum teacher at
   `0.996` moves very slowly — deliberate, but confirm it is the desired
   default when `wire_data` is absent (e.g. a user calling
   `train_step` directly, or a pipeline whose loader is iterable-only
   and `len()` raised `TypeError`, leaving `_total_steps = None`).
3. **BYOL parity.** Both files were updated identically; the tests only
   cover DINO (`tests/test_dino.py`). Add a BYOL-equivalent test for the
   budget fallback (`tests/test_byol.py`) to lock the parity.

### Intended follow-up

- Add a `grad_accum_steps>1` end-to-end or unit check that the ramp
  completes at the last optimizer step (not the last micro-batch),
  matching `trainer.py`'s counter semantics. Decide and document: is
  `global_step` the optimizer-step counter? (Yes, verify docstring at
  `trainer.py:78` and increment site `trainer.py:132`.)
- Add the BYOL fallback test.
- If the budget is intentionally not derivable (iterable-only datasets),
  keep `_momentum_start()` fallback and document it in the method
  docstring (`_current_momentum` already mentions "a missing budget
  never freezes the teacher" — extend to state the exact value and why).

### Validation expected

- New tests for `grad_accum_steps > 1` budget math (DINO + BYOL).
- BYOL `_current_momentum` fallback test.
- Full suite green; `ruff check .` clean.

---

## Reference

- Original audit: `plans/REVIEW_2.md` (removed after this TODO was
  created; the relevant §7 / §7.1 and §10 / §2 text is reproduced in
  this document).
- Implementation commit: `72f5853`.
- Plan-commit: `8777c59` (D-N legend); `24e016f` (original REVIEW_2).