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

### 2.0 Recently landed (context for the list below)

- **Periodic + mid-epoch resume restored.** `--save_every` was silently inert
  since the v4.2 trainer rewrite (`ec19359`): the flag, the `CheckpointSaver`
  plumbing and `should_save()` all survived, but the periodic write site was
  dropped from the loop. It now fires at each grad-accum flush, writes
  `_latest.pt` with an `in_progress` marker (`{epoch, micro_step, phase}`) and
  `world_size`, and a resume **continues** the interrupted epoch (supervised:
  fast-forward by a grad-accum-aligned `islice`; PPO: skip the live-env
  collection, re-enter the optimize sweep). A `barrier()` brackets each
  periodic save so peers cannot run ahead into the next collective. Covered by
  `tests/test_midepoch_resume.py`, `tests/test_rl_trainer.py`
  (`TestPPOMidepochResume`) and the 2-rank `tests/test_distributed_e2e.py`.
  - Supervised resume is exact-as-to-order: the global per-epoch order is
    reconstructed from `(seed, epoch)`, so the fast-forward lands on the same
    sample the uninterrupted run would have drawn.
  - PPO resume rounds to a PPO-epoch boundary (no bit-exact RNG cursor); a
    checkpoint taken during COLLECT re-runs the epoch (the live env is not
    persisted).
  - Off-policy RL (DQN/SAC) mid-epoch resume is **out of scope**: the replay
    buffer is not checkpointed. See 2.11 below.

### 2.1 Resume across a different rank count

**Urgency: low -- the mismatch is now detected and the residual is bounded.**

**Correction (audited against the code).** An earlier draft of this entry
claimed a *silent, approximate epoch position*. That is not what the code does.
The per-epoch sample order is a pure function of `(seed, epoch)`
(`torch.randperm(..., generator=manual_seed(seed + epoch))` in
`DistributedSampler`, and the seeded custom samplers), so it is **independent of
`world_size`**. `world_size` only decides *which rank gets which slice* of that
fixed order. Resume is **epoch-aligned**: the checkpoint's `epoch` field is the
last fully completed epoch and the loop restarts the next one from its top (and
since mid-epoch resume landed, an interrupted epoch now continues from a
recorded micro-step rather than replaying -- see the landed note below).

**What actually differs.** Two bounded effects:
- `DistributedSampler` pads each rank to `ceil(N / world)` samples, so a few
  boundary samples are *repeated*, differently per `world` (at most `world - 1`
  across the whole epoch; measured: `N=101` re-draws 1 sample at `world=2`, 3 at
  `world=4`).
- The custom batch/weighted samplers truncate to `(len // world) * world`, so a
  different number of trailing batches is dropped.

Neither moves the epoch position; both are bounded by `world` and touch only the
epoch boundary.

**Landed.** `checkpoint_dict` now records `world_size`; `resume_checkpoint`
surfaces it; and the CLI calls `warn_world_size_mismatch` on resume
(`genml_kit/io/checkpointing.py`, `genml_kit/cli/train.py`). The mismatch is
**warn-only** (never aborts), covered by `tests/test_checkpointing.py`.

**Still open (low).** The padding/truncation residual is *warned*, not
eliminated. Eliminating it would mean persisting the sampler's epoch/offset and
re-slicing to the original `world` -- not worth it while resume is
epoch-aligned and the residual is a handful of boundary samples.

### 2.3 DDP ignore-list seam (`_ddp_params_and_buffers_to_ignore`)

**Urgency: low -- the existing knob covers today's methods; this is about
avoiding per-step reducer overhead and supporting per-name exclusion.**

`_ddp_params_and_buffers_to_ignore` is a **module attribute set before
wrapping** (exact parameter names), *not* a `DistributedDataParallel`
constructor keyword. It is not an argparse default, so `arg_defaults()` cannot
express it; if a method needs it (BYOL / DINO keep `find_unused_parameters=True`
as a defensive default even though the frozen EMA towers never trigger the
check, and that flag makes the reducer re-traverse every bucket each
iteration), the mechanism is a pre-wrap attribute set and `wrap_model` would
need a seam.

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

### 2.11 Off-policy RL mid-epoch resume (out of scope for now)

**Urgency: low -- off-policy RL is a narrower audience than LLM training.**

**Gap.** Supervised and PPO now resume mid-epoch (see 2.0). Off-policy RL
(DQN/SAC, `RLTrainer._train_epoch_offpolicy`) does not: `RLPipeline.get_checkpoint_state`
persists `obs_rms`, `ret_norm` and the rollout buffer, but **not the replay
buffer** (`self.replay_buffer`), and the environment is live, so neither the
experience nor the env state survives a restart. A mid-epoch interrupt therefore
restarts the warmup/learn epoch with an emptier buffer.

**Why it matters.** Semantically wrong for off-policy (the buffer *is* the
state), but the failure is not silent-corruption -- it is lost progress.

**Shape of the fix (not committed).** Either (a) persist the replay buffer
gated behind a flag, accepting the size cost (raw observations can be
gigabytes), or (b) declare "off-policy resume restarts the epoch" explicitly in
the README and keep the current behaviour. This is a storage-design decision,
deliberately kept out of the resume-semantics change.

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
| 2.1 Resume across rank counts | Low (detected + bounded; warn-only) | Small |
| 2.5 Multi-node / `nccl` untested | Medium (unverified backend) | Medium |
| 2.8 Smaller robustness items | Low | Small |
| 2.9 FSDP seams | Low (design only) | Small |
| 2.10 Decision F | Low | Trivial (a decision) |
| 2.11 Off-policy RL resume | Low (lost progress, not corruption) | Medium |
| 2.3 DDP ignore-list seam | Low | Small |
