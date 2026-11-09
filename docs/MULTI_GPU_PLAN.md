# Multi-GPU Training Support

Status: **distributed plumbing landed (supervised and RL); §2.12 (epoch
advance), §2.13 (sampler preservation incl. the single-GPU `BalancedBatchSampler`
wiring), §2.14 (off-policy NaN reduction) and §2.15 (RL `--grad_accum_steps`
rejected) fixed. Remaining: §2.1--§2.11.**
Baseline: commit `629de39` (`test(rl): cover SAC reduction and the RL driver end
to end`), on top of `fbf3435` (`fix(rl): honour the rank gate in
RLTrainer.validate and gate eval videos`) and `a774d76` (`fix(rl): reduce
gradients explicitly and shard PPO rollouts under DDP`), on top of `67bfb87`
(`feat(dist): multi-GPU (DDP) training support`).

This document is the working record for multi-GPU (DDP) support in `genml_kit`.
It lists only what is **still open**, each entry stating the gap, why it matters,
where it lives, and the shape of the fix. Where a claim is backed by a test or a
real multi-process run it says so; where it is only reasoning from reading the
code it says that too.

> **Re-derivation note.** Auditing the code against this document on top of
> `629de39` surfaced four unrecorded items, added below as §2.12--§2.15. Three
> of them (§2.12--§2.14) are correctness bugs on the **default** DDP path, so
> "landed" applies to the *plumbing* (wrapping, reduction, sharding, checkpoint
> gating) and not yet to the sampler / NaN semantics built on top of it.

---

## 1. Goal (unchanged)

Train across multiple GPUs so that:

- throughput scales with the number of GPUs (data-parallel training);
- a run launched on N GPUs produces the *same kinds* of artifacts as a
  single-GPU run -- checkpoints, metrics, logs -- with no change to downstream
  consumers (`genml-kit-infer`, fine-tuning, `--source_checkpoint`);
- the single-GPU path is unchanged, and `torchrun` is never required for it.

---

## 2. What is missing or needs a better implementation

Ordered roughly by value. Each entry states the gap, why it matters, where it
lives, and the shape of the fix.

### 2.1 Per-method DDP graph knobs (mechanism shipped, payload undecided)

**Gap.** `find_unused_parameters` and `static_graph` are settable per method via
`Method.arg_defaults()` / `DataPipeline.arg_defaults()` (the general "declare
defaults for existing CLI flags, explicit CLI wins" seam). But **no method
declares a DDP default yet**: `arg_defaults()` returns `{}` everywhere, so
behaviour is unchanged from a plain `--dist_arg`.

**Why it matters.** The correct value is a property of the *method*: it depends
on whether that method's forward touches every grad-requiring parameter. A wrong
`False` looks like a training crash; a wrong `True` silently costs throughput.

**Fix.** Run 2 ranks x 2 steps under `torchrun` per method and declare
`find_unused_parameters=True` from the evidence, not from guesswork.

**Also relevant:** `_ddp_params_and_buffers_to_ignore` is a **module attribute
set before wrapping** (exact parameter names), *not* a `DistributedDataParallel`
constructor keyword. It is not an argparse default, so `arg_defaults()` cannot
express it; if any method needs it, the mechanism is a pre-wrap attribute set
and `wrap_model` would need a seam.

### 2.2 `validate_all` is redundant (either shard val, or drop it)

**Gap.** `--dist_arg validate_all` runs validation on every rank and averages the
result. But `val_loader` is **never sharded** (`shard_train_loader` touches only
`train_loader`), so every rank evaluates the *same* full validation set. The
averaged metric is therefore mathematically identical to rank 0's -- at N times
the cost.

**Why it matters.** A configuration knob that cannot change the result is a trap:
it invites someone to "enable correctness" and get nothing but a bill.

**Fix.** Two options:
- **(a) Shard `val_loader`** too and aggregate with the correct denominator
  (samples, not means-of-means, unless shards are equal).
- **(b) Remove `validate_all`** and hardcode rank-0-only, since it is exactly
  correct today. Simpler, and honest about the current behaviour.

Option (a) is the real fix if validation sets ever grow large enough that a
single-rank pass is the bottleneck. Option (b) is correct today. Doing neither
leaves a knob that lies.

### 2.3 Gradient-accumulation all-reduce cadence (not implemented)

**Gap.** With `grad_accum_steps > 1`, DDP all-reduces gradients on **every**
micro-step. `torch.nn.parallel.DistributedDataParallel`'s `no_sync()` context
manager exists precisely to skip those reductions on all but the final
micro-step.

**Why it matters.** Correct but wasteful: N-1 redundant all-reduces per
optimizer step, i.e. measurable bandwidth and latency cost scaling with
`grad_accum_steps`. Silent -- no error, just slower.

**Fix.** Wrap the non-final micro-steps in `with model.no_sync():`. The facade
must expose `no_sync` (or the trainer must reach `unwrap(model).no_sync()`), and
the accumulation boundary in `train_epoch` must be identified. Note the guard:
`no_sync()` must be active on *exactly* the micro-steps whose gradients are not
yet reduced, or gradients are silently dropped/double-counted.

### 2.4 Resume across a different rank count (not persisted)

**Gap.** Shard boundaries are a function of `world_size`. Resuming a
multi-GPU checkpoint with a different `--nproc_per_node` changes them, so the
epoch position is approximate whenever the dataset does not divide evenly.

**Why it matters.** Silent: the run resumes, the loss looks plausible, and the
data order is subtly wrong. It is also the one bug that only shows up in the
middle of an expensive resume.

**Fix.** Persist `world_size` and the sampler's epoch/offset in the checkpoint;
on a mismatch, warn and either re-shard from an epoch boundary or refuse to
resume mid-epoch.

### 2.5 SSL methods under DDP (blocked on an unrelated bug)

**Gap.** SimMIM / SupCon / DINO / BYOL / I-JEPA have never been run under DDP.

**Why it matters.** These are the methods most likely to have a parameter that
does not receive a gradient on a given forward, which is exactly the DDP failure
mode that surfaces on the second iteration.

**Fix.** Extend `tests/test_distributed_e2e.py` with a small per-method 2-rank
run (tiny model, `find_unused_parameters` at its default) for each SSL method.
I-JEPA is **blocked** by a pre-existing bug:

```
genml_kit/methods/ijepa.py:127-128
mask[r:r + block_size_h].unsqueeze(...)
RuntimeError: ...more than one element of the written-to tensor refers to a
single memory location`.
```

It is **unrelated to DDP** and must be fixed first (build the mask non-in-place,
e.g. via `mask.view(...)` assignment or `scatter`, rather than writing through
`expand`).

### 2.6 Decision F -- `ModelProtocol` (still unresolved)

**Gap.** The original doc listed "whether to add the untyped `ModelProtocol`" as
deferred by the reviewer. Decision C then shipped a concrete `DistributedModel`
facade, which changes the question: the facade now *is* the documented answer to
"what is 'the model'", so a separate protocol may be redundant.

**Fix.** Decide explicitly: either declare F moot (Decision C subsumes it) or add
the docstring-only protocol and make `DistributedModel` the reference
implementation. Either way, remove it from the open-decisions list so the
document does not carry a stale question.

### 2.7 FSDP-ready seams (design only)

**Gap.** Not designed beyond intent. The claim is that a future DDP -> FSDP swap
should be a single wrap-site change plus a checkpoint (un)shard layer.

**Why it matters.** Everything must route through the `wrap_model` / `unwrap`
seam for that claim to hold. It is cheap to keep true and expensive to retrofit.

**Fix.** Audit for direct `DistributedDataParallel` / model-type assumptions
outside `genml_kit/utils/distributed.py`, and confirm that FSDP would need only
a new branch at the wrap site plus sharded checkpoint I/O.

### 2.8 `--dist_arg` has no key validation (accepted risk)

**Gap.** A typo'd key (`--dist_arg find_unused_paramters`) is silently ignored,
matching how every other `*_arg` flag behaves (they are splatted into
constructors). This was a deliberate choice to stay consistent.

**Why it matters.** For DDP the miss is not benign: the run proceeds with the
wrong setting and fails later with an opaque DDP error rather than at parse time.
It is the one context where the permissive convention costs real debugging time.

**Fix.** Either accept and document (current state), or add a whitelist check at
the single consumption site. Note this is a conscious trade-off, not an
oversight; do not "fix" it without deciding, because it would make `--dist_arg`
behave differently from `--model_arg`.

### 2.9 `KVPairAction` treats a bare token as `key=True` (cross-cutting)

**Gap.** To make `--dist_arg shard` work, `KVPairAction.__call__` was changed from
erroring on a token without `=` to storing `{token: True}`. This affects **every**
`*_arg` flag: `--model_arg`, `--opt_arg`, `--sched_arg`, `--dataset_arg`,
`--tokenizer_arg`, `--proc_arg`, `--dist_arg`.

**Why it matters.** `--opt_arg weight_decay` (a forgotten `=0.05`) now silently
sets `weight_decay=1.0` -- and `opt_arg` is splatted into
`torch.optim.AdamW(**opt_arg)`, which accepts it without complaint. Typos shaped
like keys are also no longer caught for any of these flags. This was an
accepted, explicit decision, but it belongs in the record.

**Fix.** Document it in the README's argument sections (currently only the
`--dist_arg` help text mentions bare keys). If the `--opt_arg` case proves
painful, revisit scoping the permissive behaviour.

### 2.10 Multi-node and `nccl` are untested

**Gap.** Every test runs `gloo` on CPU in a single host. Multi-node runs
(`MASTER_ADDR` / `MASTER_PORT` spanning hosts) are unexercised -- no CUDA device
is available in the test environment.

**Why it matters.** `gloo` and `nccl` differ in ways that matter here: device
placement for collectives (`_reduce_device` in `distributed.py`), and the
`nccl` watchdog that turns a rank stall into a timeout.

**Fix.** A GPU/multi-node CI lane (or a documented manual verification
procedure), at minimum covering: `cuda:{LOCAL_RANK}` pinning, `all_reduce_dict`
dtype handling, and the L2 norm path under `nccl`.

### 2.11 Smaller robustness items

- **`resolve_device` and a stray `LOCAL_RANK`.** `resolve_device` prefers
  `cuda:{LOCAL_RANK}` whenever `LOCAL_RANK` is set (`gpu.py::_local_rank_env`),
  regardless of whether a process group exists. A user exporting `LOCAL_RANK`
  manually gets an unexpected device index with no error. Consider gating on
  `is_distributed()`.
- **`dist_arg` is read directly, like `model_arg`.** `KVPairAction` always
  installs a `dict` default, so a parsed namespace always carries the attribute,
  with no `getattr` fallback, matching the `args.model_arg` convention.
  Hand-built test namespaces must supply `dist_arg={}`.
- **Effective-batch documentation.** `loss/accum_steps` stays local and DDP
  averages gradients, so effective batch = `local_batch x accum x world_size`,
  and `--lr` may need linear scaling. This is documented in the README but is a
  silent-correctness trap worth stating in the doc too.
- **`grad_monitor` / `model_report`.** **[Resolved]** These are rank-0-owned by
  the same argument as the other sinks. `grad_monitor.py` contains no `open()`,
  `json.dump` or other write call -- it only emits `logging.*` records, which is
  safe under `torchrun` (one stderr stream per process, no shared file). Nothing
  left to do here; the "main-process-only writes" tests already pin the
  checkpoint and metrics sinks.

---

### 2.12 `set_epoch` never reaches `DistributedSampler` (every epoch is epoch 0)

**Gap.** Nothing advances the epoch on the sampler that actually drives the
default DDP path. Both call sites prefer `loader.batch_sampler`:

```python
# pipelines/base.py:106-112
sampler_epoch(
    getattr(loader, "batch_sampler", None) or getattr(loader, "sampler", None),
    epoch)

# training/trainer.py:333
maybe_call(self.pipeline.train_loader, "batch_sampler.set_epoch", epoch)
```

For an index-sharded loader, `shard_loader` rebuilds the `DataLoader` with
`sampler=DistributedSampler(...)`. `DataLoader.__init__` then wraps that sampler
in a `torch.utils.data.BatchSampler` whenever `batch_size` is set, so
`loader.batch_sampler` is a **plain `BatchSampler`**, which has no `set_epoch`.
Being truthy, it short-circuits the `or`, so the fallback to `loader.sampler`
never happens and both calls become silent no-ops.

Verified on CPU with
`DataLoader(ds, sampler=DistributedSampler(...), batch_size=8)`:

```
loader.sampler:              DistributedSampler
loader.batch_sampler:        BatchSampler (has set_epoch: False)
DistributedSampler.epoch after set_train_epoch(5):  0
```

**Why it matters.** This is the *default* configuration -- the images and text
pipelines request `shuffle=True`, `shard_loader` swaps the `RandomSampler` for a
`DistributedSampler`, and then every epoch draws the **identical** permutation
*and* the identical shard boundaries. The docstrings in both places ("Advance a
`DistributedSampler` ... so every epoch reshuffles") are therefore false. It is
silent: no error, no warning, just correlated data order for the whole run.

It also blocks §2.4: persisting "the sampler's epoch/offset" cannot make resume
correct while the epoch is pinned at 0.

**Status: FIXED.** Chosen shape: walk the wrapper chain and filter on capability.

- `sampler_epoch(root, epoch)` now accepts the *loader* (or a bare sampler, so
  the old call form keeps working) and delegates to a new `_epoch_targets`
  helper that yields `root`, `loader.batch_sampler`, `loader.sampler` and
  `loader.batch_sampler.sampler`; `sampler_epoch` then calls `set_epoch` on each
  one that actually defines it, de-duplicating by identity. `None` and
  `set_epoch`-less objects stay no-ops, so a plain `BatchSampler` and the
  balanced sampler are handled by construction rather than by special case.
- `pipelines/base.py::set_train_epoch` now passes the loader.
- `trainer.py` dropped its separate `maybe_call(loader, "batch_sampler.set_epoch")`
  call -- the duplicate walk that let the two call sites drift -- and keeps a
  fallback to `sampler_epoch(pipeline.train_loader, epoch)` when the pipeline
  does not define `set_train_epoch` (the trainer only requires a `train_loader`
  attribute, so without the fallback a minimal pipeline would silently stop
  receiving the epoch).

Tests: `tests/test_distributed.py::TestSamplerEpoch` (6 cases: epoch reaches the
`DistributedSampler` *under* a `BatchSampler`, three consecutive epochs produce
three distinct orders, a batch-stream sampler receives it, a bare sampler still
works, and `None` / `set_epoch`-less inputs are no-ops). Verified to fail
against the pre-fix code.

### 2.13 `shard_loader` silently discards the pipeline's chosen sampler

**Gap.** `shard_loader` picks its branch on `loader.batch_size is None`:

```python
# utils/distributed.py:641-646
if loader.batch_size is None and loader.batch_sampler is not None:
    kwargs["batch_sampler"] = shard_batch_sampler(loader.batch_sampler)
else:
    kwargs["sampler"] = make_sampler(
        loader.dataset, shuffle=_is_shuffling(loader), seed=seed)
```

The `else` branch **replaces** `loader.sampler` outright. But
`images.py:212-228` builds the training loader with
`sampler=<WeightedRandomSampler | BalancedBatchSampler>` *and*
`batch_size=args.batch_size`, so `batch_size is not None` and the `else` branch
runs.

Verified on CPU with `world_size` stubbed to 2:

```
--sampler weighted : WeightedRandomSampler      -> DistributedSampler
--sampler balanced : BalancedBatchSampler       -> DistributedSampler
text length-bucket : LengthBucketBatchSampler   -> _ShardedBatchSampler  (correct)
```

**Why it matters.** `--sampler weighted` (inverse-frequency / per-class
multipliers) and `--sampler balanced` (fixed samples-per-class batches) are
silently downgraded to uniform sampling under DDP -- the same run, on more GPUs,
optimises a *different objective*. No error, no warning. That breaks §1 goal #2
in the way that matters most: the same effective batch must mean the same thing.

**Prerequisite (unrelated single-GPU bug, found while verifying the above).**
`BalancedBatchSampler` is a **batch** sampler -- `__len__` returns a batch count
and each item is a `list[int]` of indices -- but `images.py:226-234` passes it as
`sampler=`, not `batch_sampler=`. `DataLoader` then treats each yielded *list* as
one **sample**, so a length-5 balanced sampler over 40 items produces
`len(loader) == 1` and yields a single batch whose samples are lists of index
lists. Measured on CPU with a 40-item, 4-class toy dataset:

```
len(BalancedBatchSampler)                     = 5   (batches)
as sampler=:       len(loader)=1  samples_seen=5   <-- wrong
as batch_sampler=: len(loader)=5  samples_seen=40  <-- correct
```

So `--sampler balanced` is broken on **one** GPU too, not just under DDP. (The
text pipeline gets this right -- `text.py:108-110` passes its
`LengthBucketBatchSampler` as `batch_sampler=`, which is why the length-bucket row
above takes the `_ShardedBatchSampler` path.) The existing regression tests
(`test_pipeline_method_registry.py:247`, `:270`) only assert
`isinstance(loader.sampler, BalancedBatchSampler)` and never iterate the loader,
so they pass while the loader is wrong. This must be fixed (pass the sampler as
`batch_sampler=`, and add an iteration/shape assertion to those tests) *before*
any distributed sharding of it is meaningful -- otherwise §2.13 would be
"correctly sharding a stream that is already malformed".

**Status: FIXED**, in three parts.

*C0 (prerequisite, single-GPU).* `images.py` now keeps the two sampler kinds in
separate slots: `build_weighted_sampler` (per-item indices) goes to `sampler=`,
`BalancedBatchSampler` (whole batches) goes to `batch_sampler=`, and
`batch_size`/`shuffle` are only set on the index-sampler path -- `DataLoader`
rejects them alongside `batch_sampler`. The two tests that had pinned the buggy
wiring now assert on `loader.batch_sampler` *and* inspect the index batches, so
they can no longer pass with a collapsed epoch; a new
`test_weighted_sampler_stays_an_index_sampler` pins the other side of the split.

*C1 (dispatch).* `shard_loader` now delegates the index-sampler case to a new
`shard_index_sampler`, which dispatches on the sampler's **type** rather than on
`batch_size`:

| sampler | result |
|---|---|
| `WeightedRandomSampler` | `_ShardedWeightedSampler` (weights preserved) |
| `RandomSampler` / `SequentialSampler` / `None` | `DistributedSampler` (as before) |
| anything else | `fatal(...)` naming the class and pointing at `--dist_arg shard=false` |

The `batch_sampler`-driven case is unchanged (`shard_batch_sampler`).

*C2a (sharded weighted sampler).* `_ShardedWeightedSampler` reproduces
`WeightedRandomSampler.__iter__` from a **rank-independent** generator seeded
`seed + epoch`, then yields `order[rank::world]` truncated to a multiple of the
world size. Both ranks therefore agree on one global draw and keep alternating
positions of it -- equal step counts, and a per-epoch reshuffle via the
`set_epoch` that §2.12 now actually delivers. This deliberately breaks the
`seed + rank()` convention: the *sampling* decision must be global, only the data
streams may differ. A side effect worth knowing: because sampling is with
replacement, an index can appear on both ranks -- the *positions* tile, the
values need not.

Tests: `tests/test_distributed.py::TestShardIndexSamplerPreservesSemantics`
(6 cases: weights preserved, global draw identical across ranks, slices equal
in length and tiling the draw, epoch reshuffles, plain shuffle still gets a
`DistributedSampler`, unknown sampler fails loudly). Verified to fail against
the pre-fix `shard_loader` (5 of 6) and pre-fix `images.py` (2 of 2).

### 2.14 Off-policy RL NaN/Inf skip is rank-local and can desync the reduction

**Gap.** The off-policy (DQN/SAC) flow skips a bad update without reducing the
decision across ranks:

```python
# training/rl_trainer.py:189-191
if torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any():
    logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
    continue
```

The on-policy (PPO) flow already does it correctly a few dozen lines later
(`rl_trainer.py:427-430`: `is_bad = ...; if any_rank_true(is_bad): ... continue`).

**Why it matters.** Skipping `_apply_grad` on one rank but not another means that
rank also skips `_reduce_rl_gradients` -> `all_reduce_gradients`, so the
collective runs on a subset of ranks. That is exactly what **invariant #7** below
forbids. Outcomes: a gloo/NCCL hang until the watchdog fires, or the
`all_reduce_gradients` count guard raising `fatal("inconsistent gradient sets
across ranks ...")` -- neither is a correct or diagnosable outcome for "one rank
had a NaN". RL runs are long, so this is an expensive way to learn a rank
diverged.

**Status: FIXED.** The off-policy flow now mirrors the PPO path:

```python
is_bad = bool(
    torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any())
if any_rank_true(is_bad):
    if is_bad:
        logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
    continue
```

`any_rank_true` was already imported in the module. Resulting semantics: one
rank's NaN discards the step on *all* ranks, which is the only choice that keeps
replicas in lockstep.

Tests: `tests/test_rl_trainer.py::TestOffPolicyNaNSkipIsReduced` (4 cases: a
finite rank skips when the peer is NaN, finite ranks step together, a local NaN
skips locally, and `inf` is treated as bad). Verified to fail against the
pre-fix code.

### 2.15 RL flows ignore `--grad_accum_steps` -- **RESOLVED**

**Resolution (rejected).** `RLTrainer.__init__` now calls `fatal()` when
`args.grad_accum_steps > 1`, pointing users at `--ppo_mini_batch_size` (PPO) or
`--batch_size` (off-policy). Covered by
`tests/test_rl_trainer.py::TestRLTrainer::test_rejects_grad_accum_steps`.

**Gap (historical).** Accumulation exists only in `BaseTrainer.train_epoch`
(`trainer.py:137-170`). `rl_trainer.py` has no reference to `grad_accum_steps`
at all; both the off-policy and PPO flows call `_apply_grad` once per mini-batch
unconditionally.

**Why it matters.** `--grad_accum_steps 4` is accepted globally and silently has
no effect on RL runs, so the "effective batch = local x accum x world_size"
relationship documented in the README (see §2.11) is false for RL. Low severity
when combined with DDP (world size already multiplies the batch), but it is a
silent no-op flag -- the same trap §2.2 complains about.

**Fix.** Either implement accumulation in `_apply_grad` (scale the loss, step the
optimizer only on the boundary, and for PPO align the boundary with the rollout's
mini-batch split), or reject the combination explicitly -- `fatal()` when
`grad_accum_steps > 1` and the method is RL -- and document that RL accumulation
is unsupported. The latter is one line and honest.

**Test.** Assert the flag is either honoured or rejected for an RL method.

---

## 3. Load-bearing invariants

Whatever is implemented next must preserve these:

1. Infra wrappers go **outermost**; semantic wrappers stay **inside**.
2. Checkpoint keys are the portable contract; infra never appears in them.
3. Introspection (`config`, report, param groups, freeze walk) always sees the
   semantic model, not infra.
4. Type checks (`isinstance`) see the semantic model too, so method branches keyed
   on the model's type take the same path with and without DDP.
5. Single-GPU behaviour is unchanged: every distributed path is a no-op when
   `world_size() == 1`.

RL-specific, because RL models are deliberately **not** DDP-wrapped
(`RLMethod.wrap_model` returns the model unchanged):

6. Every RL autograd path must be followed by an explicit gradient reduction
   before clipping and stepping -- the trainer's `_reduce_rl_gradients` for the
   shared flows, `SACMethod.apply_grad` for its own three-optimizer flow
   (including the bare `log_alpha` leaf tensor).
7. Any per-rank control-flow branch (early stop, NaN skip) must be reduced with
   `any_rank_true` before it changes whether a collective runs, or the ranks
   deadlock.

---

## 4. Non-goals

- FSDP / tensor / pipeline / sequence parallelism implementations.
- Multi-node orchestration beyond what `torchrun` provides. (Tested: 2.11.)
- `accelerate` / `deepspeed` integration.
- Changes to the single-GPU default path's behavior.

---

## 5. Rough risk/effort

| Item | Risk | Effort |
|---|---|---|
| 2.1 Per-method DDP knobs | Low (mechanism exists) | Small |
| 2.2 `validate_all` (shard val or remove) | Low, correctness-sensitive | Small-medium |
| 2.3 `no_sync()` accumulation | Medium (easy to get the boundary wrong) | Small |
| 2.4 Resume across rank counts | Medium (silent wrong-position risk) | Medium |
| 2.5 SSL methods under DDP | Medium; blocked on I-JEPA bug | Medium |
| 2.6 Decision F | Low | Trivial (a decision) |
| 2.7 FSDP seams | Low (design only) | Small |
| 2.8 / 2.9 / 2.10 | Low to medium | Small to medium |
| 2.12 `set_epoch` never reaches `DistributedSampler` | **High** (silent, default path) | ~~Small~~ **DONE** |
| 2.13 `shard_loader` drops weighted/balanced sampler | **High** (silent objective change) | ~~Small / Medium~~ **DONE** |
| 2.13 prerequisite: `BalancedBatchSampler` passed as `sampler=` | **High** (broken on 1 GPU too) | ~~Trivial~~ **DONE** |
| 2.14 Off-policy NaN skip not `any_rank_true` | **High** (hang or `fatal`) | ~~Trivial~~ **DONE** |
| 2.15 RL ignores `--grad_accum_steps` | Low (silent no-op flag) | Trivial (reject) / Medium (implement) |

### Suggested order

2.12, 2.13 (including its single-GPU prerequisite and the sharded weighted
sampler, option (a) -- one global draw, sliced per rank) and 2.14 are **done**.

Remaining correctness-adjacent item: 2.15 is a decision plus a one-liner. §2.2
and §2.3 remain the highest-value *design* items; §2.4 is the one that benefits
most from 2.12 landing, since the sampler epoch is now actually persisted-worthy.
