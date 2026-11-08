# Multi-GPU Training Support

Status: **supervised path landed and committed; RL path not started.**
Baseline: commit `67bfb87` (`feat(dist): multi-GPU (DDP) training support`,
15 files, +1576/-16), on top of `c301470`.

This document is the working record for multi-GPU (DDP) support in `genml_kit`.
It replaces the original proposal: the design is settled, the supervised path
works, and what remains is listed below with the implementation each item
actually needs. Where a claim is backed by a test or a real multi-process run it
says so; where it is only reasoning from reading the code it says that too.

---

## 1. Goal (unchanged)

Train across multiple GPUs so that:

- throughput scales with the number of GPUs (data-parallel training);
- a run launched on N GPUs produces the *same kinds* of artifacts as a
  single-GPU run -- checkpoints, metrics, logs -- with no change to downstream
  consumers (`genml-kit-infer`, fine-tuning, `--source_checkpoint`);
- the single-GPU path is unchanged, and `torchrun` is never required for it.

---

## 2. What is landed

Verified by `tests/test_distributed.py` (54 tests), `tests/test_distributed_e2e.py`
(2 tests, real `torchrun`), `ruff check .`, and the full suite
(**2030 passed, 38 deselected**).

### 2.1 Process group and helpers

`genml_kit/utils/distributed.py` provides `init_distributed()` (reads
`RANK`/`WORLD_SIZE`/`LOCAL_RANK`, `init_process_group(backend=...)`,
`torch.cuda.set_device(local_rank)`; returns `False` and creates nothing when
`WORLD_SIZE` is absent or 1), `destroy_distributed()`, `is_distributed()`,
`is_main_process()`, `rank()`, `world_size()`, `local_rank()`, `barrier()`, and
`all_reduce_sum` / `all_reduce_mean` / `all_reduce_dict`.

`main()` calls `init_distributed()` before seeding and tears the group down in a
`finally`, so teardown is unconditional and safe.

### 2.2 Model wrapping -- Decision C

`wrap_model(model, device, find_unused_parameters=, static_graph=)` returns a
`DistributedModel` facade, or **the model object itself** when `world_size < 2`
(so the single-device path has no facade and no added indirection).

The facade exists because DDP is the one non-conformer among the model wrappers:
it defines no `__getattr__` (so `ddp.config` / `ddp.student` raise
`AttributeError`) and `ddp.attr = value` lands on the DDP object rather than
`ddp.module` (so `model.mask_ratio = 0.5` would silently stop affecting the real
model). PEFT, by contrast, *does* proxy attribute reads.

`DistributedModel` delegates `__getattr__` / `__setattr__` / `state_dict` /
`load_state_dict` / `parameters` / `named_parameters` / `buffers` / `modules` /
`named_modules` / `children` / `train` / `eval` / `to` / `__str__` to the
semantic model, routes only the forward through DDP, and reports the semantic
model's `__class__` so `isinstance`-keyed method branches keep working.
`unwrap()` removes **only** allowlisted infrastructure wrappers
(`DistributedDataParallel` / `DataParallel`), never PEFT or semantic ones.

The wrap site is in `genml_kit/cli/train.py`, deliberately *after* checkpoint
restore and `build_optimization`, so both see the semantic model.

### 2.3 Checkpoint portability

`state_dict()` / `load_state_dict()` delegate to the semantic model, so no
`module.` prefix ever reaches a checkpoint key and no load-side stripping is
needed. `unwrap(self.model)` is applied at the one `isinstance`-sensitive
persistence site (`CheckpointSaver`, because `serialize_lora_state` tests
`isinstance(model, PeftModel)`). Proven end to end: a 2-rank `torchrun` run
writes keys `['fc.weight', 'fc.bias']`.

### 2.4 Data sharding

`make_sampler` / `shard_loader` / `shard_batch_sampler`, driven by
`DataPipeline.shard_train_loader` (once, from the driver) and
`DataPipeline.set_train_epoch` (per epoch). The loader is **rebuilt**, not
mutated, because torch forbids assigning a sampler after construction. A plain
index loader gets a `DistributedSampler`; a batch-sampler-driven loader gets a
rank-strided slice, truncated to a multiple of the world size so every rank runs
the same number of steps (DDP requires equal step counts). Shards are verified
disjoint, complete, and equal-length.

Sharding defaults to **on whenever `world_size() > 1`**; opt out with
`--dist_arg shard=false`. The default is derived from the rank count because the
opt-in form failed silently (every rank iterating the full dataset: N times the
compute, no error).

### 2.5 Validation policy

Rank 0 validates; other ranks skip and wait at a `barrier()` after validation
(so a slow validation cannot let a peer run ahead into the next epoch's
all-reduce). All-ranks validation with `all_reduce_dict(average=True)` is
available via `--dist_arg validate_all`.

This resolves the original Decision D. Note the load-bearing fact:
`shard_train_loader` shards **only** `train_loader`, never `val_loader`
(`pipelines/base.py:25`), so rank 0 already evaluates the entire validation set
-- rank-0-only is correct *and* N times cheaper. See item 3.3 for the
consequence.

### 2.6 Configuration surface

One repeatable `--dist_arg KEY=VALUE` flag (`KVPairAction`), matching
`--model_arg` / `--dataset_arg`, rather than one flag per knob. Registered as
`ARG_DIST_ARG` in `genml_kit/utils/shared_args.py`; addressable from the HPO
search space via the `dist.` prefix (`search/space.py`).

| Key | Default | Meaning |
|---|---|---|
| `backend` | auto (`nccl` on CUDA, else `gloo`) | Process-group backend. |
| `shard` | `true` when `world_size() > 1` | Shard the training set across ranks. |
| `find_unused_parameters` | `false` | Passed through to DDP. |
| `static_graph` | `false` | Passed through to DDP. |
| `validate_all` | `false` | Validate on every rank and average. |

### 2.7 Rank-dependent side effects

`resolve_device` prefers `cuda:{LOCAL_RANK}`; seeding is `base_seed + rank`;
checkpoint writes, metrics sinks, and file log targets are owned by rank 0.

### 2.8 Tests and docs

`tests/test_distributed.py` covers facade transparency, `isinstance`
transparency, single-device identity, fake-world-size wrapping, launcher-env
semantics, sharding disjointness/completeness/equal step counts, and a 2-rank
gloo all-reduce that asserts post-backward parameters are identical.
`tests/test_distributed_e2e.py` spawns real `torchrun` processes (2-rank and
1-rank) via `tests/dist_e2e_driver.py`, asserting both ranks joined, rank 0 alone
checkpointed, and keys are portable. `README.md` has a "Multi-GPU Training"
section.

No `pyproject.toml` change: `torchrun` is a console entry point of `torch`
(already a hard dependency), so an extra for it would install what is already
present -- and `torchrun` exists on PyPI as an unrelated third-party package, so
naming it in `dependencies` would pull the wrong thing.

---

## 3. What is missing or needs a better implementation

Ordered roughly by value. Each entry states the gap, why it matters, where it
lives, and the shape of the fix.

### 3.1 P4a -- Per-method DDP graph knobs (needs a better mechanism)

**Gap.** `find_unused_parameters` and `static_graph` are only settable globally
through `--dist_arg`. The correct value is a property of the *method*: it depends
on whether that method's forward touches every grad-requiring parameter.

**Why it matters.** DDP's `find_unused_parameters=False` raises on the **second**
iteration (not the first) for a parameter that genuinely never receives a
gradient. Today a user hitting that has to discover the global flag and set it
for the whole run. A per-method default plus a CLI override is the right shape.

**Where.** Method declaration surface (`genml_kit/methods/*`) plus the
`wrap_model` call in `genml_kit/cli/train.py`.

**Fix.** Add a `Method` hook (e.g. `ddp_config()` returning
`{"find_unused_parameters": ..., "static_graph": ...}`, defaulting to both
`False`), have the driver merge it with `--dist_arg`, with the explicit CLI value
winning. This is the item `wrap_model` already accommodates -- it accepts both
flags today, so only the declaration and the merge are missing.

**Also relevant:** `_ddp_params_and_buffers_to_ignore` is a **module attribute
set before wrapping** (exact parameter names), *not* a `DistributedDataParallel`
constructor keyword. If any method turns out to need it, the mechanism is a
pre-wrap attribute set, and `wrap_model` would need a seam for it.

### 3.2 P4b -- RL under DDP (largest remaining piece)

**Gap.** Not started. `RLTrainer` bypasses the `DataLoader` entirely: it drives
environments and replay buffers directly, so `shard_train_loader` never applies
and every rank would run an identical independent rollout.

**Why it matters.** RL is the one pipeline where "run the same command under
`torchrun`" produces *wrong* results rather than slow results -- N identical
rollouts with no data diversity, and gradient all-reduce over correlated
batches. It is not a wiring task; it is a design task.

**Fix (design, then implementation).**
- Per-rank environments and replay buffers (no sharing), with per-rank seeding so
  rollouts differ.
- Sharded on-policy rollouts for PPO; DDP then handles gradient all-reduce.
- A defined owner of replay-buffer / rollout-buffer checkpoint state on resume.
- Decide what "epoch" means when each rank has its own env.

This should be its own design pass with its own document, not folded into DDP
plumbing.

### 3.3 `validate_all` is currently redundant (either shard val, or drop it)

**Gap.** `--dist_arg validate_all` runs validation on every rank and averages the
result. But `val_loader` is **never sharded** (`shard_train_loader` touches only
`train_loader`, `pipelines/base.py:25`), so every rank evaluates the *same* full
validation set. The averaged metric is therefore mathematically identical to
rank 0's -- at N times the cost.

**Why it matters.** A configuration knob that cannot change the result is a trap:
it invites someone to "enable correctness" and pay N times for nothing.

**Fix -- pick one:**
- **(a) Shard `val_loader`** (per-rank slice) and reduce for the mean. Then
  `validate_all` becomes genuinely meaningful and cheaper per rank. Requires
  sharding the val loader, a `DistributedSampler` with `shuffle=False`, and a
  correct denominator (samples, not means-of-means, unless shards are equal).
- **(b) Remove `validate_all`** and hardcode rank-0-only, since it is exactly
  correct today. Simpler, and honest about the current behaviour.

Option (a) is the real fix if validation sets ever grow large enough that a
single-rank pass is the bottleneck. Option (b) is correct today. Doing neither
leaves a knob that lies.

### 3.4 Decision E -- gradient-accumulation all-reduce cadence (not implemented)

**Gap.** With `grad_accum_steps > 1` (`training/trainer.py:119`), DDP all-reduces
gradients on **every** micro-step. `torch.nn.parallel.DistributedDataParallel`'s
`no_sync()` context manager exists precisely to skip those reductions on all but
the final micro-step.

**Why it matters.** Correct but wasteful: N-1 redundant all-reduces per
optimizer step, i.e. measurable bandwidth and latency cost scaling with
`grad_accum_steps`. Silent -- no error, just slower.

**Fix.** Wrap the non-final micro-steps in `with model.no_sync():`. The facade
must expose `no_sync` (or the trainer must reach `unwrap(model).no_sync()`), and
the accumulation boundary in `train_epoch` must be identified. Note the guard:
`no_sync()` must be active on *exactly* the micro-steps whose gradients are not
yet reduced, or gradients are silently dropped/double-counted.

### 3.5 Resume across a different rank count (not persisted)

**Gap.** Shard boundaries are a function of `world_size`. Resuming a
multi-GPU checkpoint with a different `--nproc_per_node` changes them, so the
epoch position is approximate whenever the dataset size is not divisible by the
rank count.

**Why it matters.** Resume is a correctness feature; silently restarting at the
wrong position (or skipping/duplicating samples) is worse than refusing.

**Fix.** Persist the sampler epoch (and, if a `DistributedSampler` is in use, its
`set_epoch` value) in the checkpoint metadata, and either (a) restore it exactly,
or (b) detect a rank-count change and warn/refuse. The existing
`DataPipeline.set_train_epoch` hook is the natural place to consume it.

### 3.6 SSL methods under DDP are unvalidated (SimMIM / I-JEPA)

**Gap.** BYOL, DINO and distillation are expected to need no ignore-list because
`EMAModel.__init__` sets `requires_grad=False` on the target and `distillation`
freezes its teacher explicitly -- but **this is reasoning, not a test**. SimMIM
and I-JEPA are further out.

**Why it matters.** These are the methods most likely to have a parameter that
does not receive a gradient on a given forward, which is exactly the DDP failure
mode that surfaces on the second iteration.

**Fix.** Extend `tests/test_distributed_e2e.py` with a small per-method 2-rank
run (tiny model, `find_unused_parameters` at its default) for each SSL method.
I-JEPA is **blocked** by a pre-existing bug:

```
genml_kit/methods/ijepa.py:127-128
mask[r:r + block_size_h].unsqueeze(1).expand(-1, num_w)[:, c:c + block_size_w] = True
```

This writes in place through an `expand`ed view, which raises
`RuntimeError: ...more than one element of the written-to tensor refers to a
single memory location`. It is **unrelated to DDP** and must be fixed first
(build the mask non-in-place, e.g. via `mask.view(...)` assignment or
`scatter`, rather than writing through `expand`).

### 3.7 Decision F -- `ModelProtocol` (still unresolved)

**Gap.** The original doc listed "whether to add the untyped `ModelProtocol`" as
deferred by the reviewer. Decision C then shipped a concrete `DistributedModel`
facade, which changes the question: the facade now *is* the documented answer to
"what is 'the model'", so a separate protocol may be redundant.

**Fix.** Decide explicitly: either declare F moot (Decision C subsumes it) or add
the docstring-only protocol and make `DistributedModel` the reference
implementation. Either way, remove it from the open-decisions list so the
document does not carry a stale question.

### 3.8 P5 -- FSDP-ready seams (design only)

**Gap.** Not designed beyond intent. The claim is that a future DDP -> FSDP swap
should be a single wrap-site change plus a checkpoint (un)shard layer.

**Why it matters.** Everything must route through the `wrap_model` / `unwrap`
seam for that claim to hold. It is cheap to keep true and expensive to retrofit.

**Fix.** Audit for direct `DistributedDataParallel` / model-type assumptions
outside `genml_kit/utils/distributed.py`, and confirm that FSDP would need only
a new branch in `wrap_model` and a shard-aware `state_dict`. Record the seam
contract; write no FSDP code.

### 3.9 `--dist_arg` has no key validation (accepted risk)

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

### 3.10 `KVPairAction` now treats a bare token as `key=True` (cross-cutting)

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

### 3.11 Multi-node and `nccl` are untested

**Gap.** All verification is CPU/`gloo` with 2 ranks. `nccl` and multi-node
(`MASTER_ADDR` / `MASTER_PORT` spanning hosts) are unexercised -- no CUDA device
is available in the test environment.

**Why it matters.** `gloo` and `nccl` differ in ways that matter here: device
placement for collectives (`_reduce_device` in `distributed.py`), and the
`nccl` watchdog that turns a rank stall into a timeout.

**Fix.** A GPU/multi-node CI lane (or a documented manual verification
procedure), at minimum covering: `cuda:{LOCAL_RANK}` pinning, `all_reduce_dict`
device placement, and the watchdog safety of the post-validation `barrier()`.

### 3.12 Smaller robustness items

- **`resolve_device` and a stray `LOCAL_RANK`.** `resolve_device` prefers
  `cuda:{LOCAL_RANK}` whenever `LOCAL_RANK` is set (`gpu.py::_local_rank_env`),
  regardless of whether a process group exists. A user exporting `LOCAL_RANK`
  manually gets an unexpected device index with no error. Consider gating on
  `is_distributed()`.
- **`shard_train_loader` reads `dist_arg` defensively.** It uses
  `getattr(args, "dist_arg", {}).get(...)` because tests construct duck-typed
  arg namespaces. That is tolerated, not ideal; a shape contract for parsed args
  would remove the defensive read. (Related to Decision F.)
- **Effective-batch documentation.** `loss/accum_steps` stays local and DDP
  averages gradients, so effective batch = `local_batch x accum x world_size`,
  and `--lr` may need linear scaling. This is documented in the README but is a
  silent-correctness trap worth stating in the doc too.
- **`grad_monitor` / `model_report`.** These are rank-0-owned by the same
  argument as the other sinks; confirm no non-rank-0 path writes to a shared
  file. Partially covered by the existing "main-process-only writes" tests.

---

## 4. Load-bearing invariants

Whatever is implemented next must preserve these:

1. Infra wrappers go **outermost**; semantic wrappers stay **inside**.
2. Checkpoint keys are the portable contract; infra never appears in them.
3. Introspection (`config`, report, param groups, freeze walk) always sees the
   semantic model, not infra.
4. Type checks (`isinstance`) see the semantic model too, so method branches keyed
   on the model's type take the same path with and without DDP.
5. The single-GPU path is unchanged: `wrap_model` returns the same object when
   `world_size < 2`, and no distributed code runs without launcher env vars.

---

## 5. Non-goals

- FSDP / tensor / pipeline / sequence parallelism implementations.
- Multi-node orchestration beyond what `torchrun` provides. (Tested: 2.11.)
- `accelerate` / `deepspeed` integration.
- Changes to the single-GPU default path's behavior.

---

## 6. Rough risk/effort

| Item | Risk | Effort |
|---|---|---|
| 3.1 Per-method DDP knobs | Low (mechanism exists) | Small |
| 3.2 RL under DDP | **Highest** | **Largest**; own design doc |
| 3.3 `validate_all` (shard val or remove) | Low, correctness-sensitive | Small-medium |
| 3.4 `no_sync()` accumulation | Medium (easy to get the boundary wrong) | Small |
| 3.5 Resume across rank counts | Medium (silent wrong-position risk) | Medium |
| 3.6 SSL methods under DDP | Medium; blocked on I-JEPA bug | Medium |
| 3.7 Decision F | Low | Trivial (a decision) |
| 3.8 FSDP seams | Low (design only) | Small |
| 3.9 / 3.10 / 3.11 | Low to medium | Small to medium |
