# Multi-GPU Training Support

Status: **supervised and RL paths landed; what remains is listed below.**
Baseline: commit `a774d76` (`fix(rl): reduce gradients explicitly and shard PPO
rollouts under DDP`), on top of `67bfb87` (`feat(dist): multi-GPU (DDP) training
support`).

This document is the working record for multi-GPU (DDP) support in `genml_kit`.
It lists only what is **still open**, each entry stating the gap, why it matters,
where it lives, and the shape of the fix. Where a claim is backed by a test or a
real multi-process run it says so; where it is only reasoning from reading the
code it says that too.

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
- **`grad_monitor` / `model_report`.** These are rank-0-owned by the same
  argument as the other sinks; confirm no non-rank-0 path writes to a shared
  file. Partially covered by the existing "main-process-only writes" tests.

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
