# TODO — deferred items from plans/REVIEW_2.md

Second-round audit items that were **recorded but not implemented** in the
`72f5853` commit ("fix: implement REVIEW_2 findings #1-#6, #8-#13"). Each
section below states the open problem, the current code, the exact
reproduction, the decided/planned fix, and the validation expected.

Status anchor: working tree at commit `72f5853`.

---

## 1. `genml-kit-train --help` under-reports the CLI (REVIEW_2 #7)

**Status: DEFERRED (design recorded in REVIEW_2 §7 / §7.1, not implemented).**

### Problem

`parse_args` in `genml_kit/training/train.py` is two-pass: it calls
`parser.parse_known_args(argv)` on the 1st-pass parser **before** adding
pipeline/method/checkpoint/optimizer/logging args. argparse handles
`--help` *inside* `parse_known_args`, so the tool exits there and prints
only the flags that existed at that point — every owner-specific and
shared flag is missing.

### Current code

`genml_kit/training/train.py:246-252` (unchanged by `72f5853`):

```python
def parse_args(argv=None):
  """Two-pass parse: generic flags first, then pipeline + method flags."""
  parser = build_parser()
  known, _ = parser.parse_known_args(argv)
  pipeline_cls = get_pipeline(known.pipeline)
  method_cls = get_method(known.method)
  ...
```

### Reproduction (verified post-`72f5853`)

```python
import io
from contextlib import redirect_stdout
from genml_kit.training.train import parse_args

buf = io.StringIO()
try:
  with redirect_stdout(buf):
    parse_args(["--help"])
except SystemExit:
  pass
out = buf.getvalue()

print("help lines:", len(out.splitlines()))          # -> 62
print("--dataset:", "--dataset" in out)              # -> False
print("--lr:", "--lr" in out)                        # -> False
print("--checkpoint:", "--checkpoint" in out)        # -> False
print("--vo_length:", "--vo_length" in out)          # -> False
print("--dino_local_num:", "--dino_local_num" in out)  # -> False
print("images pipeline:", "images pipeline" in out)  # -> False
```

All of those flags **are** accepted at runtime (a full
`--dataset ... --lr ...` invocation reaches model download), so the
tool's own `--help` contradicts every README CLI table.

### Decided design (from REVIEW_2 §7, as revised in review round 2)

Keep the two-pass parser for real runs; intercept `--help` *before* the
probe pass; register ALL owners only on the help path:

```python
# genml_kit/training/train.py
_HELP_FLAGS = ("-h", "--help")   # argparse also accepts --help=<formatter>

def _wants_help(argv):
  return any(a in _HELP_FLAGS for a in (argv or []))

def _register_all(parser):
  for name in list_pipelines():
    get_pipeline(name)().add_args(parser)      # owner creates its own group
  for name in list_methods():
    get_method(name)().add_args(parser)        # "classification method", "DINO", ...

def parse_args(argv=None):
  if _wants_help(argv):
    parser = build_parser()
    _register_all(parser)                       # superset help, then exit
    parser.parse_args(["--help"])
  parser = build_parser()
  known, _ = parser.parse_known_args(argv)      # unchanged probe pass
  pipeline_cls = get_pipeline(known.pipeline)
  method_cls = get_method(known.method)
  pipeline_cls().add_args(parser)
  method_cls().add_args(parser)
  add_checkpoint_args(parser, ...)              # shared sections, unchanged
  ...
  args = parser.parse_args(argv)
  _post_process(parser, args)
  return args
```

**Important implementation notes (verified while prototyping, not yet
committed):**

1. **The shared arg sections are NOT all inside `build_parser()`.** The
   generic parser covers `--pipeline/--method/--model/--model_arg/...`,
   but `--lr`, `--weight_decay`, `--scheduler`, `--save_every`,
   `--device`, `--log_dir`, `--hf_token`, `--in_ch`, `--vis_every` and
   the `--state_save/--state_load` pair are added **inside `parse_args`**
   via `add_checkpoint_args` / `add_optimization_args` /
   `add_training_state_args` / `add_logging_args` /
   `add_source_checkpoint_args` plus an inline `opt = parser.add_argument_group("optimizer")`
   block (`train.py:254-365`). The help path MUST call those same helpers
   after `_register_all`, otherwise `--lr`/`--checkpoint`/`--state_*`
   still go missing.
2. **Option-string collision audit (current surface)**: registering all
   2 pipelines + 7 methods on one parser yields **no duplicate option
   strings** (verified with an audit script in review round 2; 63 owner
   flags + shared = 95 distinct, the only "--help" repeats are the
   per-parser `-h` auto-add artifact). Single-pass-always would work
   *today*, but is not the design (see §7.1 policy).
3. **Do NOT extract/move shared flags into a global section** to dodge a
   future collision: that orphans usage when an owner is retired. Shared
   helpers in `utils/args.py` are the single definition point for
   genuinely-global flags; per-owner flags stay per-owner.
4. **Collision diagnostic**: wrap each owner's `add_args` in
   `_register_all` and re-raise `argparse.ArgumentError` with an
   actionable message (which flag, which owners, intended fix: shared
   helper vs. rename). See §7.1 items 2-4.

### §7.1 policy to carry over (recorded, not yet enforced)

1. `add_args(parser)` is the owner's declaration of its own arg surface;
   the owner names its own `add_argument_group` (caller-as-knowledge).
2. No duplicate option strings anywhere (owner vs. owner, owner vs.
   shared helper).
3. Same-flag-same-intent rule: a flag is global only if semantically
   identical for every consumer (same default/type/help/dest) and owned
   by a *concern* helper, never a feature module. Different meaning ⇒
   different name (`--a_foo` vs `--b_foo`), never first-wins.
4. Enforcement: unit test registering every pipeline+method on one
   parser, asserting no `ArgumentError` (the audit script becomes the CI
   guard).

### Expected validation when fixed

- `parse_args(["--help"])` output contains `--dataset`, `--vo_length`,
  `--dino_local_num`, `--lr`, `--checkpoint` and the `"DINO"` /
  `"images pipeline"` group titles.
- A real run with no `--help` does NOT register unselected owners
  (probe via `parse_known_args`; selection-driven behavior unchanged).
- Full suite (currently 941 tests) stays green; `ruff check .` clean.

---

## 2. `--sampler balanced` divisibility: "warn but still run" (REVIEW_2 #10)

**Status: PARTIALLY IMPLEMENTED — the wiring landed, but the
divisibility semantics differ from the plan text and are recorded here
for a follow-up decision.**

### What was implemented in `72f5853`

`genml_kit/pipelines/images.py`:
- `--sampler` choices extended to `["none", "weighted", "balanced"]`
  (line ~121).
- `--samples_per_class` added (default 16).
- In `_build_classification` (lines 225-251):

```python
sampler = None
_sampler_balanced = args.sampler == "balanced"
if args.sampler == "weighted" and train_proxy.label_column:
  sampler = build_weighted_sampler(...)
elif _sampler_balanced:
  if not train_proxy.label_column:
    logging.warning("--sampler balanced requires a label column; "
                    "falling back to shuffle=True.")
    args.sampler = "none"
  elif args.batch_size % args.samples_per_class:
    logging.warning("--sampler balanced: batch_size %d is not divisible "
                    "by samples_per_class %d; falling back to shuffle=True.",
                    args.batch_size, args.samples_per_class)
    args.sampler = "none"
  else:
    from genml_kit.datasets.balanced_sampler import BalancedBatchSampler
    labels = train_proxy.dataset[train_proxy.label_column]
    sampler = BalancedBatchSampler(labels,
                                   batch_size=args.batch_size,
                                   samples_per_class=args.samples_per_class)
```

Regression tests added in `tests/test_pipeline_method_registry.py`
(`test_balanced_sampler_wired_into_train_loader`,
`test_balanced_sampler_falls_back_on_indivisible_batch`).

### The open discrepancy

REVIEW_2 §10 said:

> `batch_size % samples_per_class != 0` **warns but still runs**.

The implementation instead **falls back to `shuffle=True`** (still
"runs", but without balanced batching). Reason for the deviation:
`BalancedBatchSampler.__init__` **hard-fails** on indivisible sizes
(`genml_kit/datasets/balanced_sampler.py:37-40`):

```python
if batch_size % samples_per_class != 0:
  fatal(
      f"batch_size ({batch_size}) must be divisible by "
      f"samples_per_class ({samples_per_class})", ValueError)
```

So "warn and still use the balanced sampler" is impossible without
changing the sampler; "warn and still run the epoch" is what the
fallback provides.

### Open decision — pick ONE of the following

**Option A (current, minimal): keep the fallback-to-shuffle.**
Document the semantics in the README/help text as "not divisible ⇒
balanced sampler is unavailable, falls back to shuffling". No code
change beyond doc wording.

**Option B: make `BalancedBatchSampler` tolerant of indivisible sizes.**
Change `batch_size % samples_per_class != 0` from `fatal(...)` to a
warning, and adjust `self._n_groups = batch_size // samples_per_class`
so the last group is partial (or the last batch is smaller). This matches
REVIEW_2's literal wording but changes the sampler's contract for all
existing callers (unit tests in `tests/test_balanced_sampler.py` assert
the `ValueError`, e.g. `test_raises_on_indivisible_batch` — those tests
must be updated). Then revert the pipeline fallback so `--sampler
balanced` with a non-divisible batch uses the sampler anyway.

**Option C: reject at the CLI.** Keep the sampler strict and make the
pipeline `fatal(...)` instead of warning+fallback, so a misconfigured
`--sampler balanced --batch_size N --samples_per_class M` fails fast at
arg-parse/loader-build time.

Recommendation: **Option B** best matches the REVIEW_2 intent ("warns
but still runs" with the sampler) and is the most user-friendly, at the
cost of updating 1-2 unit tests. Confirm before implementing.

### Validation expected for whichever option

- `tests/test_balanced_sampler.py` updated to the new contract and
  green.
- Pipeline tests in `tests/test_pipeline_method_registry.py` updated to
  match (either the sampler is used with a partial group, or the CLI
  rejects).
- README `--sampler` / `--samples_per_class` wording matches the chosen
  behavior.
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