# Multi-GPU Training Support

Status: **distributed plumbing landed (supervised and RL).** This document lists
only the work that is still open, ordered by urgency (highest first). See §2 for
the ranked list and §5 for the risk/effort summary.

Baseline: commit `f464b1c` (`feat(dist): shard validation and merge per-rank
metrics`), on top of `629de39` (`test(rl): cover SAC reduction and the RL driver
end to end`), `fbf3435` (`fix(rl): honour the rank gate in RLTrainer.validate and
gate eval videos`), `a774d76` (`fix(rl): reduce gradients explicitly and shard
PPO rollouts under DDP`) and `67bfb87` (`feat(dist): multi-GPU (DDP) training
support`).

This document is the working record for multi-GPU (DDP) support in `genml_kit`.
It lists only what is **still open**, each entry stating the gap, why it matters,
where it lives, and the shape of the fix. Where a claim is backed by a test or a
real multi-process run it says so; where it is only reasoning from reading the
code it says that too.

---

## 1. Goal

Train across multiple GPUs so that:

- throughput scales with the number of GPUs (data-parallel training);
- a run launched on N GPUs produces the *same kinds* of artifacts as a
  single-GPU run -- checkpoints, metrics, logs -- with no change to downstream
  consumers (`genml-kit-infer`, fine-tuning, `--source_checkpoint`);
- the single-GPU path is unchanged, and `torchrun` is never required for it.

---

## 2. What is missing or needs a better implementation

Ordered by urgency: silent-correctness risk on the **default** DDP path first,
then coverage gaps, then performance, then design-only items. Each entry states
the gap, why it matters, where it lives, and the shape of the fix.

### 2.1 Resume across a different rank count

**Urgency: high -- silent wrong data order, only on the mid-run resume path.**

**Gap.** Shard boundaries are a function of `world_size`. Resuming a
multi-GPU checkpoint with a different `--nproc_per_node` changes them, so the
epoch position is approximate whenever the dataset does not divide evenly.

**Why it matters.** Silent: the run resumes, the loss looks plausible, and the
data order is subtly wrong. It is also the one bug that only shows up in the
middle of an expensive resume.

**Fix.** Persist `world_size` and the sampler's epoch/offset in the checkpoint;
on a mismatch, warn and either re-shard from an epoch boundary or refuse to
resume mid-epoch.

### 2.2 SSL methods under DDP

**Urgency: high -- contains a real crash (I-JEPA) and zero 2-rank coverage of the
methods most likely to trip DDP's unused-parameter check.**

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

### 2.3 Per-method DDP graph knobs

**Urgency: high -- a wrong value is either a training crash or a silent
throughput loss, on the default path.**

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

### 2.5 Multi-node and `nccl` are untested

**Urgency: medium -- a coverage gap that could mask anything on the real
multi-node, GPU-backend path.**

**Gap.** Every test runs `gloo` on CPU in a single host. Multi-node runs
(`MASTER_ADDR` / `MASTER_PORT` spanning hosts) are unexercised -- no CUDA device
is available in the test environment.

**Why it matters.** `gloo` and `nccl` differ in ways that matter here: device
placement for collectives (`_reduce_device` in `distributed.py`), and the
`nccl` watchdog that turns a rank stall into a timeout.

**Fix.** A GPU/multi-node CI lane (or a documented manual verification
procedure), at minimum covering: `cuda:{LOCAL_RANK}` pinning, `all_reduce_dict`
dtype handling, and the L2 norm path under `nccl`.

### 2.8 Smaller robustness items

**Urgency: low -- mostly bookkeeping.**

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
### 2.9 FSDP-ready seams (design only)

**Urgency: low -- design only; cheap to keep true, expensive to retrofit.**

**Gap.** Not designed beyond intent. The claim is that a future DDP -> FSDP swap
should be a single wrap-site change plus a checkpoint (un)shard layer.

**Why it matters.** Everything must route through the `wrap_model` / `unwrap`
seam for that claim to hold. It is cheap to keep true and expensive to retrofit.

**Fix.** Audit for direct `DistributedDataParallel` / model-type assumptions
outside `genml_kit/utils/distributed.py`, and confirm that FSDP would need only
a new branch at the wrap site plus sharded checkpoint I/O.

### 2.10 Decision F -- `ModelProtocol`

**Urgency: low -- a decision, and likely moot now that Decision C shipped.**

**Gap.** The original doc listed "whether to add the untyped `ModelProtocol`" as
deferred by the reviewer. Decision C then shipped a concrete `DistributedModel`
facade, which changes the question: the facade now *is* the documented answer to
"what is 'the model'", so a separate protocol may be redundant.

**Fix.** Decide explicitly: either declare F moot (Decision C subsumes it) or add
the docstring-only protocol and make `DistributedModel` the reference
implementation. Either way, remove it from the open-decisions list so the
document does not carry a stale question.

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
- Multi-node orchestration beyond what `torchrun` provides. (Tested: 2.5.)
- `accelerate` / `deepspeed` integration.
- Changes to the single-GPU default path's behavior.

---

## 5. Rough risk/effort

Ordered by urgency, matching §2.

| Item | Risk | Effort |
|---|---|---|
| 2.1 Resume across rank counts | Medium (silent wrong-position risk) | Medium |
| 2.2 SSL methods under DDP | Medium; blocked on I-JEPA bug | Medium |
| 2.3 Per-method DDP knobs | Low (mechanism exists) | Small |
| 2.5 Multi-node / `nccl` untested | Medium (unverified backend) | Medium |
| 2.8 Smaller robustness items | Low | Small |
| 2.9 FSDP seams | Low (design only) | Small |
| 2.10 Decision F | Low | Trivial (a decision) |
