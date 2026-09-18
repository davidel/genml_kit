# RL_PLAN: Training Reinforcement Learning Algorithms in genml_kit

Status: **Draft for review** — no code committed yet.

This document outlines how to extend the `genml_kit` project to train
reinforcement learning (RL) algorithms, reusing and extending the current
training infrastructure (methods, models, losses, training loop, checkpointing,
reporting, CLI) as much as possible.

> **Re-entry guide.** This plan is written to be the *complete reasoning record*:
> if we come back after days (or weeks) we should be able to (a) reconstruct
> every decision and why it was made, (b) re-verify against the code, and
> (c) resume implementation in the exact order described in the implementation
> phase checklist.  The mathematical foundations, derivations and proofs for
> every formula referenced here live in `rl/README.md` (the companion
> self-contained tutorial in the same style as `vo/README.md`).  Read
> `rl/README.md` first if the math needs re-derivation; this plan is about
> *wiring* that math into the existing harness.
>
> **Conventions used in both documents.** Inline math uses the GitHub
> dollar-backtick form `` $`...`$ ``; display math uses
> `$$\n\large\n...\n$$`; multi-letter operators are written
> `\mathop{\mathrm{...}}`; **all** math symbols are LaTeX (never raw unicode
> such as `\pi`, `\theta`, `\gamma`, `\to`, `\le` — see `rl/README.md`
> Appendix D and `vo/README.md` Appendix D).

---

## 0. Re-Entry Guide (read this first when returning to the work)

- Branch/base: work on top of `main` (currently clean, no RL code committed).
- Nothing under `genml_kit/` has been touched yet (all RL changes are new
  files; only registry re-export `__init__.py` files and a backward-compatible
  `Method` base extension are allowed).
- Companion document: `rl/README.md` (math, derivations, proofs, symbol
  table, reading list, failure modes).
- Order of implementation, strictly: Phase 1 tasks 1-9 (Section 8); each task
  ends with `ruff check .` + targeted `pytest` green.
- Quick self-check commands:
  ```bash
  pytest -q tests/test_rl_losses.py tests/test_rl_models.py tests/test_rl_method.py tests/test_rl_trainer.py tests/test_rl_cli.py
  ruff check genml_kit tests
  genml-kit-train --pipeline rl --method dqn --help   # must render rl/dqn groups
  ```
- The single most important architectural fact: **the driver
  (`genml_kit/training/train.py`) and `BaseTrainer.run()` stay branchless**;
  RL enters through one new pipeline (`rl`), one new method (`dqn`), one thin
  trainer subclass (`RLTrainer`), plus new model/loss/dataset modules.

---

## 1. Objective and Scope

Make `genml_kit` able to train RL agents through the *same* unified harness
that already trains classification, VO, and self-supervised objectives,
without forking the driver.

Phase 1 delivers **off-policy value-based DQN** (with Double DQN and Dueling
options). Phase 2 delivers **policy-gradient / actor-critic** (PPO) and
**maximum-entropy actor-critic** (SAC).  Phase 3 hardens (prioritized replay,
vector envs, n-step, observation normalization, remote checkpointing).

Why DQN first (deep analysis, see Decision Log D1):

- It is the *minimal* semantic delta over the existing
  `DataBlob -> LossOutput` contract: one `train_step` maps a batch of
  `(s, a, r, s', done)` transitions to one TD loss scalar.  No probability
  ratios, no log-probs, no episode-trajectory data layout.
- It exercises exactly the patterns the codebase already has strong support
  for: EMA/target networks (DINO/BYOL/IJEPA precedent), replay as a Dataset
  (Dataset/DataLoader precedent), epsilon schedule as method checkpoint
  state (momentum-schedule precedent), and best-checkpoint by a validation
  metric (all methods precedent).
- It forces us to build the *only genuinely new pieces (env stepping +
  replay + target update cadence) in isolation, before the policy-gradient
  machinery lands on top of a proven base.

The **responsibility split (the core design principle, see D2):

| Concern | Owning component |
|---|---|
| Environment + replay + data layout | `RLPipeline` (data side) |
| Stepping cadence (when to act vs learn) | `RLTrainer` (thin loop) |
| Learning math (loss, targets, schedules) | `DQNMethod` (objective side) |
| Target-network parameters + polyak updates | the method (model state dict) |
| Grad accumulation / AMP / monitor / checkpoints | reused unchanged from `BaseTrainer` |

---

## 2. Deep Look at What Exists Today (and What Maps to RL)

### 2.1 The unified driver and lifecycle

`genml_kit/training/train.py` is the single entry point (`genml-kit-train`).
It is *branchless* — it calls a fixed lifecycle and never inspects
`--method`/`--pipeline`:

```
parse -> seed -> device -> pipeline/method objects
  -> method.prepare_transforms   (processor normalization, if any)
  -> pipeline.build_loader x2    (data side)
  -> method.wire_data            (label space, criterion, ...)
  -> method.build_model          (model side, sized from wire_data)
  -> resume -> optimization -> BaseTrainer.run() -> post_train
```

Key contracts (`genml_kit/pipelines/contracts.py`):
- `DataBlob(data, meta)` — `data` is opaque to the loop; `meta` carries extras.
- `LossOutput(loss, metrics)` — `loss` is scalar (batch-averaged), UNSCALED;
  `metrics` is `dict[str, Tensor]` (logged).

`BaseTrainer.run()` (skeleton), `train_epoch` (grad accum + AMP + monitor),
`optim_factory.build_optimization`, `CheckpointSaver`, signal handling and the
save-on-exit `finally` — all reused by RL unchanged.

### 2.2 The Method contract (objective side)

`Method` (methods/base.py): `NAME`, `METRIC_KEY`, `METRIC_MINIMIZE`,
`NEEDS_LABELS`; abstract `build_model(args, device)` and
`train_step(model, blob, global_step, *, labels=None) -> LossOutput`; optional
`evaluate`, `has_metric_improved`, `add_args`, `build_transform`,
`prepare_transforms`, `wire_data`, `on_epoch_end`, `log_validation`,
`post_train`, `_apply_model_extras`.

EMA/target patterns already present: BYOL (online+target in one module,
`copy.deepcopy` + `requires_grad_(False)`, momentum schedule in
`get_checkpoint_state`/`load_checkpoint_state`), DINO (same + center EMA),
IJEPA (teacher momentum ramp).  DQN/SAC target nets reuse this exactly.

### 2.3 The Pipeline contract (data side)

`DataPipeline` (pipelines/base.py): loaders + blob contract + device transfer.
`images` (HF/ImageFolder/ensemble; samplers), `vo_pair` (generative synthetic
pairs — the closest existing analog to an RL env: data produced by a
stochastic process at `__getitem__` time).  `vo_pair` also shows non-tensor
`meta` handling (ints/strings stay on CPU in `to_device`) and staged
objectives — good references for `RLPipeline` and `DQNMethod`.

### 2.4 Optimizers/schedulers/AMP/checkpointing — reused unchanged
2.5 Models and losses — registries; custom loaders; standalone `losses/` modules
2.6 Testing/style — fake-method/pipeline unit tests; yapf 2-space Google; Ruff;
pytest warnings-as-errors.

(All detailed in the original review, kept verbatim in spirit; see the
"Reviewed against" appendix.)

---

## 3. Design Principles (extended)

1. **No branch in the driver. 2. Reuse skeleton, override only env-specific
   behavior. 3. RL is about the data contract. 4. No touching
   classification/SSL code. 5. Determinism where it matters (`utils.seed`).
6. Everything CLI-driven, AMP/grad-monitor compatible.**
7. **Math in one place.** All RL mathematics lives in `rl/README.md`
   (self-contained tutorial with proofs); the code implements formulas that
   are *named* there, and this plan references them by section.
8. **Learnable state == serializable state.** Everything the method needs to
   resume (epsilon, polyak tau, target net, replay stats) is either in the
   model state dict (target `.*` parameters) or in
   `method.get_checkpoint_state()` — nothing in closure state.
9. **`train_step` purity.** Env stepping never happens inside `train_step
   (the way `simmiim`/`dino` never step datasets inside `train_step`).
   The loop's microbatch "backward/step" primitives stay shared.

---

## 4. Algorithm Selection and Rationale

| Criterion | DQN (Phase 1) | PPO (Phase 2) | SAC (Phase 2) |
|---|---|---|---|
| Data regime | off-policy (replay) | on-policy (rollouts) | off-policy (replay) |
| Loop delta | minimal (replay TrainLoader) | rollout IterableDataset | replay + target critics + alpha |
| New math | TD target + replay + target net | GAE + clipped surrogate + entropy | soft Bellman + reparam + auto-alpha |
| EMA/target net needed | yes (hard/polyak) | no | yes (2 target critics |
| Best-checkpoint metric | `eval_return` | `eval_return` | `eval_return` |
| Risk to existing loop | lowest | medium (on-policy data flow) | medium |

Selection: Phase 1 DQN; Phase 2 PPO then SAC.  Rationale: DQN validates the
entire pipeline choice end-to-end with the *smallest* surface; PPO is the
most deployable on-policy method and stresses on-policy data flow; SAC adds
max-entropy machinery on top of the DQN replay + target twin critic
infrastructure.

---

## 5. Reference Mapping RL Concepts -> Existing Pieces

| RL concept | Existing piece | Reuse / extend |
|---|---|---|
| Env as data source | `DataPipeline` + `DataBlob(data, meta)` | New `RLPipeline` (`pipelines/rl.py`) |
| Replay buffer | `Dataset`/`DataLoader`, `seed_worker`, `getitem_retry` | New `ReplayBufferDataset` (`datasets/replay_buffer.py`) |
| Q-network / actor-critic | `models/registry` custom loaders (`vo/`) | New `models/rl/{qnetwork,actor_critic}.py` |
| TD loss / policy losses | `losses/` standalone modules | New `losses/rl.py` |
| Target-network EMA | BYOL/DINO/IJEPA EMA pattern | `models/rl/qnetwork.py` (target = deepcopy, no-grad) |
| Exploration schedule + tau | BYOL/DINO momentum ramp in method state | method fields + `get/load_checkpoint_state` |
| Best-checkpoint metric | `METRIC_KEY="eval_return"`, `METRIC_MINIMIZE=False` | method attrs |
| Grad accum/AMP/monitor | `BaseTrainer` | reused |
| Checkpoint extra state | `method.get_checkpoint_state`/`CheckpointSaver` | epsilon, tau, replay stats |
| Post-training export/eval | `Method.post_train` | policy export / final eval |
| CLI surface | `add_args` per pipeline/method | auto via `register_all_owners` |
| Custom env scripts | `utils/script.load_extern` | env factory / exploration schedule |
| Reproducibility | `utils.seed` | env + replay + torch generators |

---

## 6. Proposed Architecture (full component specs)

All new files.  Only registry re-export `__init__.py` files change, plus an
optional backward-compatible `Method` base addition (default-implemented
hook `update_target(model, global_step)` that no-ops, D9).

```
genml_kit/
  datasets/replay_buffer.py     # ReplayBufferDataset (Section 6.2)
  pipelines/rl.py               # RLPipeline (Section 6.3)
  methods/rl_dqn.py             # DQNMethod (Section 6.6)
  methods/rl_ppo.py             # (Phase 2) PPOMethod
  methods/rl_sac.py             # (Phase 2) SACMethod
  models/rl/__init__.py
  models/rl/qnetwork.py         # QNetwork, DuelingQNetwork (Section 6.4)
  models/rl/actor_critic.py     # (Phase 2) ActorCritic
  losses/rl.py                  # Section 6.5 (Section 6.5 formulas, math)
  training/rl_trainer.py         # RLTrainer (Section 6.7)
plans/RL_PLAN.md                  # this document
rl/README.md                     # math tutorial (companion)
tests/test_rl_*.py
```

### 6.1 Data layouts (blob contract)

Off-policy (Phase 1): one `DataBlob` is a batch **sampled from the replay
buffer**:

```
data = TransitionBatch(obs, actions, rewards, next_obs, dones)   # tensors
meta = {"step_type": "replay", "indices": ..., "weights": ...}    # prioritized later
```

On-policy (Phase 2): `data = obs`, `meta = {"actions_with_logp", "advantages",
"returns", ...}` from rollouts.

`to_device`: move tensors, keep ints/floats as given; dones float; no strings.

### 6.2 ReplayBufferDataset (`datasets/replay_buffer.py`)

- Fixed-capacity circular buffers (arrays) for `obs, action, reward, next_obs,
  done`; `__len__` = min(filled, capacity); `__getitem__(idx) -> dict (for
  `default_collate`-compatible, like ensemble datasets) with keys
  `("obs", "action", "reward", "next_obs", "done")`.
- API surface:
  ```
  push(obs, action, reward, next_obs, done)     # single transition
  sample(batch_size, generator) -> dict batch   # uniform (Phase 1)
  set_priorities / sample_prioritized           # Phase 3 (via WeightedRandomSampler reuse)
  stats() -> dict                               # filled/capacity, mean reward, episode count
  ```
- No labels: method `NEEDS_LABELS=False`; images-style label stripping not
  used (RL pipeline never exposes labels).
- Seeding: buffer owns a `torch.Generator` seeded from `args.seed`.

### 6.3 RLPipeline (`pipelines/rl.py`)

`@register_pipeline` `class RLPipeline(DataPipeline)`:

- `add_args`: `--env` (gym id or `utils.script.load_extern` env-factory
  script path/URL), `--env_seed`, `--max_episode_steps`, `--num_envs`
  (Phase 2/3), `--replay_capacity`, `--warmup_steps`, `--batch_size`,
  `--train_freq`, `--target_update_freq`, `--eval_episodes`, `--obs_dim`,
  `--n_actions`, `--obs_transform`.
- Loaders: `train_loader` = `DataLoader(ReplayBufferDataset, batch_size,
  collate_fn=_transitions_collate, generator=..., worker_init_fn=seed_worker)`;
  `val_loader = None` (BaseTrainer.validate no-ops; RLTrainer.validate runs
  policy rollouts instead).
- Env interface (protocol, not a class):
  ```
  env.reset(seed) -> obs
  env.step(action) -> (obs, reward, done, info)     # gymnasium-style
  env.close()
  ```
  or a `load_extern(env, "create_env")` factory script (project pattern for
  custom optimizers/processors/TTA).
- API to the trainer:
  ```
  self.act(obs, epsilon_or_greedy) -> action     # epsilon-greedy via method policy? No:
  self.step_env(action) -> transition             # env stepping only here
  self.push(transition)
  self.reset_env()
  self.eval_rollout(policy_fn, episodes) -> {"eval_return": float, ...}
  ```
  Decision D2: the pipeline owns *data* (env + replay); the trainer owns
  *cadence*; the method owns *learning* (the policy/epsilon live in the
  method and are passed in as callables where needed).

### 6.4 Models: `models/rl/qnetwork.py`

`QNetwork(nn.Module)`:
- Backbone: `load_model(args.model, num_labels=0, ...)` for image obs OR a
  plain MLP for vector obs (`--obs_dim`), mirroring `ProjectionHead` style.
- Heads: `QHead` (linear to `n_actions`) and `DuelingQHead`
  (advantage stream + value stream; `Q = V + A - mean(A)`), see
  `rl/README.md` Part 3 for the identifiability argument.
- Registered via `@register_model("rl/qnet")` and
  `@register_model("rl/qnet_dueling")` so `--model rl/qnet` works and
  inherits freeze/LoRA/source-ckpt/grad-checkpoint extras through
  `_apply_model_extras`.
- Target net: same class `deepcopy` with `requires_grad_(False)` created in
  `build_model` (BYOL pattern); parameters live in the model state dict under
  `target.*` so CheckpointSaver covers them with zero changes.

Phase 2: `ActorCritic(nn.Module)` — policy head (softmax/gaussian) + value
head(s) + log-prob helper.

### 6.5 Losses: `losses/rl.py` (formulas; math in `rl/README.md`)

All pure-torch, batched, reduction-aware (mirrors `losses/focal.py`):

- `td_loss(pred_q, td_target, reduction="mean")` — Huber (or MSE).
- `td_target(
    rewards, next_obs, dones, policy_net, target_net, gamma, double_q=True,
    n_step=1)` -> Tensor, implementing (see `rl/README.md` §3.3):
  $`y^{(n)} = r^{(n)} + \gamma^n (1 - d)\, Q_{\mathrm{target}}(s', \arg\max_a Q_{\mathrm{policy}}(s', a))`$   # Double DQN
  $`y^{(n)} = r^{(n)} + \gamma^n (1 - d)\, \max_a Q_{\mathrm{target}}(s', a)`$                             # plain DQN
- Phase 2:
  - `clipped_surrogate(ratio, advantages, clip_eps) -> Tensor` (see
    `rl/README.md` §4.4):
  $`L^{\mathrm{CLIP}} = \min\left( r_t(\theta)\, A_t,\; \mathrm{clip}(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon)\, A_t \right)`$
  - `entropy_bonus(logits)` (SAC uses $`-\log \pi`$; PPO uses $`+\beta H(\pi)`$).
  - `value_loss(pred_v, returns)` — clipped VF loss for PPO.
  - soft-Q twin-critic losses + soft target (see `rl/README.md` §5.3):
  $`y = r + \gamma(1-d)\left(\min_j Q_{\mathrm{target},j}(s', a') - \alpha \log \pi(a' \mid s')\right)`$
  for SAC (math in `rl/README.md` §5).

### 6.6 DQNMethod (`methods/rl_dqn.py`)

`@register_method` `class DQNMethod(Method)`:
- `NAME="dqn"`, `METRIC_KEY="eval_return"`, `METRIC_MINIMIZE=False`,
  `NEEDS_LABELS=False`; method stores `_epsilon` schedule params.
- `add_args` (all group "dqn method"):
  `--gamma`, `--ddqn`, `--dueling`, `--epsilon_start`, `--epsilon_end`,
  `--epsilon_decay_steps`, `--tau` (polyak, optional soft), `--target_update_freq`
  (hard sync step count; if 0, soft polyak with `--tau`),
  `--replay_alpha` (Phase 3), `--n_step` (>=1, Phase 3), `--eval_episodes`,
  `--post_train_eval` (bool; export policy via post_train).
- `build_model(args, device)`:
  1. `self._epsilon = args.epsilon_start`; store start/end/decay.
  2. `model = load_model("rl/qnet", device=device, obs_dim=..., n_actions=..., dueling=args.dueling)`.
  3. `model = self._apply_model_extras(args, model, device)`.
  4. Create target: `model.target = copy.deepcopy(model.online); requires_grad_(False)`
     (or a composite `QNetwork` holding online+target; choose composite so
     state dict includes both, D3).
- `train_step(model, blob, blob_meta, global_step, *, labels=None)`:
  1. Unpack `(obs, action, reward, next_obs, done)` from `blob.data`.
  2. `q = model.online(obs).gather(1, action.unsqueeze(1))` (B,1).
  3. `target = td_target(...)` with online/target nets; `loss = td_loss(q, target)`.
  3b. Optional entropy/metric extras; `q_mean`, `q_max`, `epsilon`.
  3c. `self._epsilon = max(eps_end, eps_start * decay ** global_step)`
      (or linear schedule).
  4. `return LossOutput(loss=loss, metrics={"td_loss": loss.detach(),
     "q_mean": q.detach().mean(), "epsilon": self._epsilon})`
- `get_checkpoint_state(model, args)`:
  `{"method": "dqn", "epsilon": self._epsilon, "eps_start": ...,
   "eps_end": ..., "eps_decay_steps": ...}`; target-net params already in
  state dict under `model.online.*`/`model.target.*` (whatever the composite
  layout — ensure online+target under prefixes; D3 documents the exact keys).
- `load_checkpoint_state(model, state, args)`: restore epsilon + schedule;
  nothing else (target in state dict).
- `evaluate(model, env_pipeline, num_episodes)` (RLTrainer passes an env
  runner or an "eval loader" wrapper): greedy rollouts, returns `{"eval_return", "eval_steps"}`.
- `has_metric_improved` default from `METRIC_MINIMIZE=False` (max).
- `post_train(args, pipeline, device, result)`: optional final
  `evaluate(...)`; export trained policy (`torch.save(model.online)`) —
  mirrors classification's XGBoost post_train stage.
- `on_epoch_end(model, epoch, writer)` optional (used for hard target sync if
  cadence by epoch rather than step; default owner: trainer calls
  `method.update_target(model, global_step)` — see 6.7/`update_target`).

Optional base addition (D9): `Method.update_target(model, global_step) # noqa: B027 -> None`
(the no-op default).  DQNMethod overrides per its cadence arg.  RLTrainer
calls it after each learn microbatch; default no-op keeps existing methods
unchanged.

### 6.7 RLTrainer (`training/rl_trainer.py`)

`class RLTrainer(BaseTrainer)` — thin override, ~80-120 lines:

```
def train_epoch(self, epoch, saver, step, monitor):
  set_train_mode(self.model, "train")
  # 1. WARMUP: env steps until replay filled >= warmup_steps
  # 2. LEARN LOOP: for learn_step_in_epoch in range(self.args.learn_steps_per_epoch):
  #      every args.train_freq: env.step and push transition
  #      sample batch -> blob -> to_device -> train_step (same as base's
  #        microbatch path: LossOutput -> grad = loss/accum -> backward
  #        -> scaler.step/optimizer.step -> zero_grad -> monitor.step) (shared helper D10)
  #      step += 1
  #      method.update_target(self.model, global_step=step)
  # 3. return avg_loss, new_step
def validate(self):
  return self.method.evaluate(self.model, self.pipeline, self.args.eval_episodes)   # policy eval on env
```

Reuses from base: `run()` (signals, saver, epoch loop, `_init_grad_monitor`,
`best_metric_key`, `saver_extra`/`ckpt_extra`, `finally` save-on-exit),
`optimization`, AMP, grad accum.  D10: extract `_apply_grad(loss, scaler,
amp_dtype)` protected helper from `BaseTrainer` (a safe, behavior-preserving
refactor) so both loops share the microbatch backward/step primitives.

Cadence summary (all configurable, defaults typical DQN values):
`warmup_steps=1000`, `train_freq=4`, `target_update_freq=500` (hard sync) or
`tau=0.005` (soft polyak every microbatch), `learn_steps_per_epoch` derived
(Phase 1 default: 1 learn epoch corresponds to `total_learn_steps =
max(1, round(train_freq * train_loader_len_per_epoch))` or a simple
`--learn_steps_per_epoch`; document exact default in CLI spec below).

### 6.8 CLI spec (Phase 1)

```bash
genml-kit-train \
  --pipeline rl --method dqn \
  --env CartPole-v1 --env_seed 42 \
  --replay_capacity 100000 --warmup_steps 1000 --batch_size 128 \
  --train_freq 4 --target_update_freq 500 --tau 0.005 \
  --gamma 0.99 --epsilon_start 1.0 --epsilon_end 0.05 --epsilon_decay_steps 50000 \
  --model rl/qnet --obs_dim 4 --n_actions 2 \
  --learn_steps_per_epoch 1000 --epochs 100 --lr 1e-3 --scheduler CosineAnnealingLR \
  --checkpoint rl_cartpole --remote_checkpoint gs://bucket/rl_cartpole \
  --grad_monitor 100 --seed 42
```

(`--help` auto-renders rl pipeline + dqn method groups; regression test like
`test_cli_help.py` asserts no collisions and presence of `rl pipeline` /
`dqn method` group names.)

---

## 7. Checkpoint and Reproducibility Spec

Checkpoint dict (additive, all existing keys untouched):
- `model.state_dict()` includes online+target (see D3 exact prefix);
  restored by resume path unchanged.
- `best_eval_return` under `best_eval_return` (metric-key cycle via
  `METRIC_KEY`); the loop never negates (`has_metric_improved`).
- `method.get_checkpoint_state()` -> `method`, `epsilon`, schedule fields,
  optional replay stats (`filled`, mean reward etc.) Phase 3, and `tau`.
- State flags (`--state_load`) reuse `restore_training_state` unchanged for
  opt/sched/amp.

Reproducibility: `--seed 42` seeds python/random/numpy/torch (existing
`utils.seed`), `env_seed` seeds the env, the replay sampler generator, and
torch RNG; document that env determinism requires gym's own seed support.

---

## 8. Implementation Phases (with task checklist and exit criteria)

## (full phase breakdown — each task: files, tests run first, exit gate)

### Phase 1 — Off-policy DQN vertical slice

T1. `losses/rl.py` (+ `tests/test_rl_losses.py`).
T2. `models/rl/qnetwork.py` (+ registry wiring, `tests/test_rl_models.py`).
T3. `datasets/replay_buffer.py` (`tests/test_replay_buffer.py`).
T4. `pipelines/rl.py` (`tests/test_rl_pipeline.py` with scripted FakeEnv).
T5. `methods/rl_dqn.py` (`tests/test_rl_method.py`).
T6. `training/rl_trainer.py` (`tests/test_rl_trainer.py` FakeGymPipeline +
   FakeRLMethod mirroring test_trainer.py Fakes).
T7. Re-exports (`pipelines/__init__.py`, `methods/__init__.py`, `losses/__init__.py`),
   CLI wiring; `tests/test_rl_cli.py` (help render + no collisions).
T8. `rl/README.md` companion doc (written with this plan; verified math renders).
T9. Docs + README additions; full regression suite.

Exit: scripted FakeEnv toy problem (e.g. a 4-state chain or CartPole
equivalent) trains near-optimal policy (`eval_return` improves monotonically);
checkpoint round-trip (latest/best) preserves epsilon schedule + target net +
`best_eval_return`; `--seed 42` reproduces a run; existing tests green.

### Phase 2 — PPO (on-policy) and SAC (max-entropy off-policy) on top of Phase 1

- `models/rl/actor_critic.py`, `losses/rl.py` extensions (GAE + clipped surrogate
  PPO objective; SAC soft twin critic + target entropy alpha).
- `pipelines/rl.py` on-policy mode (rollout IterableDataset yielding DataBlobs;
  `--pipeline rl` with `--method ppo` selects on-policy data flow, still
  branchless); SAC reuses off-policy replay + target twin critics, adds alpha
  learner; RLTrainer reused verbatim if possible (cadence expressed via
  pipeline loaders/method hooks).
- math already fully covered in `rl/README.md` (Part 4 PPO, Part 5 SAC).

### Phase 3 — Hardening

Prioritized replay (reuse `WeightedRandomSampler` or priority-aware sampler),
n-step + frame stack, observation normalization via model-processor path,
vector envs (`gymnasium.vector`), distributed checkpointing (already
`--remote_checkpoint` GCS/R2/S3), CI `[rl]` extra, `--help` regression.

---

## 9. Decision Log (complete rationale record)

| # | Decision | Status | Rationale / alternative |
|---|---|---|---|
| D1 | DQN first, then PPO, then SAC | agreed | smallest loop delta; pattern reuse; on-policy after off-policy plumbing proven |
| D2 | pipeline owns data; trainer owns cadence; method owns learning | agreed | keeps `train_step` pure; testable; mirrors existing separation |
| D3 | online+target in one module (state dict covers both; targets under `target.*` prefix) | agreed | BYOL/DINO/IJEPA precedent; CheckpointSaver zero-change; state-serialization principle 8 |
| D4 | `METRIC_KEY="eval_return"` maximize; `has_metric_improved` default | agreed | BaseTrainer skips negating; RL noise handled by `--eval_episodes` smoothing |
| D5 | RLTrainer thin override (+ optional `_apply_grad` refactor D10) | agreed | run() skeleton/signals/saver/AMP reused verbatim |
| D6 | gymnasium as optional `[rl]` extra | agreed | core stays light; custom envs via `utils.script.load_extern` need no gym dep; tests avoid hard dep (scripted FakeEnv) |
| D7 | `--help` regression test + no-option-collision (register_all_owners) | agreed | mirrors `test_cli_help.py`; driver stays branchless |
| D8 | replay as Dataset; prioritized later via WeightedRandomSampler reuse | agreed | Dataset collate-compatible dicts + DataLoader/samplers existing |
| D9 | add optional default `Method.update_target(model, global_step)` no-op hook | proposed | lets RLTrainer call cadence hook; existing methods unaffected (B027 default) |
| D10 | extract shared microbatch backward/step helper `_apply_grad` (behavior-preserving) | proposed | avoids duplicating AMP/grad-accum logic between BaseTrainer loop and RLTrainer — do not implement with behavior change |

---

## 10. Testing Strategy (matrix)

| File | Covers | Mirrors |
|---|---|---|
| `test_rl_losses.py` | TD target incl. done mask, DDQN argmax, Huber/MSE, batch/reduction shapes | `test_contrastive.py-style` |
| `test_rl_models.py` | QNet forward, dueling invariant `Q = V+A-mean(A)`, target copy = deepcopy + no-grad, freeze/LoRA extras work via `_apply_model_extras | `test_byol.py` patterns |
| `test_replay_buffer.py` | capacity overwrite, sampling shape, generator seed, stats() | `test_balanced_sampler.py` |
| `test_rl_method.py` | FakeEnv transitions through `DQNMethod.train_step` toy problem; LossOutput fields; schedule math; checkpoint keys round-trip | `test_trainer.py` FakeMethod/FakePipeline |
| `test_rl_trainer.py` | RLTrainer.run() with FakeGymPipeline + FakeRLMethod; global_step advances; best_eval_return cycle; save-on-exit `finally`; resume preserves epsilon/target-net | `test_checkpointing.py` |
| `test_rl_cli.py` | single-pass parser no collisions; `--pipeline rl --method dqn --help` renders groups | `test_cli_help.py` |
| `test_rl_pipeline.py` (Phase 1) | env protocol, push/step/reset/eval_rollout, seeding | `test_vo_pairs.py` |

---

## 11. Risks and Mitigations (kept from original review, updated)

| Risk | Mitigation |
|---|---|
Loop semantics vs epochs | thin RLTrainer override; epochs preserved (each epoch = warmup + N learn steps + optional eval) |
Replay buffer memory | array-backed circular buffer, capacity config, no Python objects hot path |
| `train_step` purity | env stepping in pipeline/trainer only; never inside `train_step` |
| gym dep pollution | `[rl]` extra; `utils.script.load_extern` custom env factories; tests avoid importing gym at module import |
| checkpoint backward-compat | additive keys only (method, epsilon, schedule, target under model state dict prefix); old checkpoints load fine |
| RL metric noise (stochastic env) | `--eval_episodes` mean over episodes + `has_metric_improved` max; optional `--metric_smoothing` later |
| reproducibility | seed env, replay sampler, torch RNGs; document `--seed` semantics (env seed separate `--env_seed`) |
| RL math correctness drift | `rl/README.md` = single source of math truth; code and docs co-reviewed; formulas named by `rl/README.md` section |

12. Out Of Scope (kept): multi-agent RL, world models/model-based, distributed rollout
workers, reward design DSLs (env provides), hierarchical RL.

13. Acceptance Criteria (Phase 1): keep list from original review (1-5) + add:
6. `rl/README.md` renders with all math (checklist: symbols LaTeX, display blocks,
   proofs present for TD/DQN/DDQN/GAE/PPO/SAC/Dueling/soft-policy-improvement/
   reparametrization, Appendix A-D).
7. Plan revisitable after days via Section 0 Re-Entry Guide.

---

## Appendix: Reviewed-Against (initial deep dive)

`train.py`, `trainer.py` (incl. full `train_epoch`/`run`), `optim_factory.py`,
`checkpointing.py`, `storage_utils.py`, `pipelines/{base,images,vo_pair,registry}.py`,
`contracts.py`, `methods/{base,classification,vo_pair,dino,byol,ijepa,simmim,supcon}.py`,
`losses/*.py`, `models/{registry,byol,dino,encoder_utils,contrastive}.py`,
`datasets/{ensemble,hf_proxy,vo_pairs,balanced_sampler,weighted_sampler,field_dataset,
image_folder,retry,transforms,__init__}.py`,
`utils/{args,cli,attr,script,seed,signal,logging,gpu,table,label,image_dump,transformer}.py`,
`tests/{test_trainer,test_checkpointing,test_cli_help,test_byol,test_vo_pairs}.py`,
`README.md`, `vo/README.md`, `pyproject.toml`, `.github/workflows/ci.yml`, `vo/README.md` LaTeX
conventions (Appendix D).

`rl/README.md` companion doc and this plan reference each other; both
maintain 2-space prose style, no tabs, column ~88.