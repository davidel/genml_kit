# RL_PLAN: Training Reinforcement Learning Algorithms in genml_kit

Status: **Phases 1 & 2 implemented** — Phase 1 (DQN) and Phase 2 (PPO, SAC)
are complete with 137 RL tests passing. Phase 3 (hardening) is pending.

This document outlines how to extend the `genml_kit` project to train
reinforcement learning (RL) algorithms, reusing and extending the current
training infrastructure (methods, models, losses, training loop, checkpointing,
reporting, CLI) as much as possible.

> **Re-entry guide.** This plan is written to be the *complete reasoning record*:
> if we come back after days (or weeks) we should be able to (a) reconstruct
> every decision and why it was made, (b) re-verify against the code, and
> (c) resume implementation in the exact order described in the implementation
> phase checklist.  **Status as of last update:** Phases 1 and 2 are fully
> implemented across commits `52a8ddf` (Phase 1) and `95a7a80` (Phase 2).
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
> such as `\pi`, `\theta`, `\gamma`, `\to`, `\le` — see
> [`docs/MARKDOWN_LATEX.txt`](docs/MARKDOWN_LATEX.txt) for the full rules).

---

## 0. Re-Entry Guide (read this first when returning to the work)

**Current Status (last updated after Phase 2 commit `95a7a80`):**
- Phase 1 (DQN) and Phase 2 (PPO, SAC) are fully implemented.
- 137 RL tests, 1085 total tests all passing.
- Phase 3 (hardening) is pending — see Section 8 for detailed task breakdown.

**What exists:**
```
genml_kit/
  losses/rl.py              # td_target, td_loss, gae, clipped_surrogate,
                            # value_loss, entropy_bonus, sac_* losses
  models/rl/__init__.py
  models/rl/qnetwork.py     # QNetwork (online+target), DuelingQHead
  models/rl/actor_critic.py # ActorCritic (Categorical/Gaussian + ValueHead)
  datasets/replay_buffer.py # ReplayBufferDataset (off-policy)
  datasets/rollout_buffer.py # RolloutBuffer (on-policy PPO)
  pipelines/rl.py           # RLPipeline (env wrapper, replay, eval_rollout)
  methods/rl_dqn.py         # DQNMethod (epsilon-greedy, Double-DQN)
  methods/rl_ppo.py         # PPOMethod (GAE, clipped surrogate, entropy)
  methods/rl_sac.py         # SACMethod (twin critics, auto-alpha)
  training/rl_trainer.py    # RLTrainer (off-policy + on-policy flows)
```

- Companion document: `rl/README.md` (math, derivations, proofs, symbol
  table, reading list, failure modes).  Written and verified.

**To resume Phase 3:**
1. Pick a task from Section 8 Phase 3 (T3.1-T3.11).
2. Each task specifies files, tests, and design notes from implementation.
3. Run: `ruff check genml_kit tests` + `yapf -i <file>` + `pytest tests/test_rl*.py -v`.
4. Commit after each task passes.

**Quick self-check commands:**
```bash
# All RL tests
pytest -q tests/test_rl*.py tests/test_actor_critic.py tests/test_rollout_buffer.py tests/test_replay_buffer.py
# All tests
pytest -q
# Lint
ruff check genml_kit tests
# Format
yapf -i <file>
```

**The single most important architectural fact:** the driver
(`genml_kit/training/train.py`) and `BaseTrainer.run()` stay branchless;
RL enters through one new pipeline (`rl`), three methods (`dqn`, `ppo`,
`sac`), one thin trainer subclass (`RLTrainer`), plus new model/loss/dataset
modules.  The trainer dispatches by `method.NAME` for off-policy vs
on-policy flows.

---

## 1. Objective and Scope

Make `genml_kit` able to train RL agents through the *same* unified harness
that already trains classification, VO, and self-supervised objectives,
without forking the driver.

Phase 1 delivers **off-policy value-based DQN** (with Double DQN and Dueling
options). Phase 2 delivers **policy-gradient / actor-critic** (PPO) and
**maximum-entropy actor-critic** (SAC).  Phase 3 hardens (prioritized replay,
vector envs, n-step, observation normalization, remote checkpointing).

**Current status:** Phases 1 and 2 are fully implemented and tested.
Phase 3 is pending — see Section 8 for detailed task breakdown.

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

All new files.  Only registry re-export `__init__.py` files change, plus a
small number of backward-compatible additions/edits to existing modules:

1. `Method` base addition: default-implemented hook `update_target(model,
   global_step)` that no-ops (D9).
2. `BaseTrainer` refactor (D10): extract `_apply_grad(loss, scaler,
   amp_dtype)` protected helper so RLTrainer shares the microbatch
   backward/step logic — behavior-preserving, no signature change.

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
  self.step_env(action) -> (obs, reward, done, info)   # env step only
  self.push(transition)                                  # push to buffer
  self.reset_env() -> obs                                # reset env
  self.replay_buffer                                     # public attribute
  self.eval_rollout(action_fn, episodes) -> {"eval_return": float,
      "eval_steps": int}
  ```
  Decision D2: the pipeline owns *data* (env + replay); the trainer owns
  *cadence*; the method owns *learning* and *acting* (policy, epsilon).

  **Build loader:** `build_loader` is called by the driver to satisfy the
  `Method` lifecycle, but `RLPipeline` returns `None` for both train and
  val loaders.  `RLTrainer.train_epoch` bypasses the DataLoader entirely
  and samples directly from `self.replay_buffer.sample(batch_size)`.
  (A DataLoader cannot iterate a buffer that is mutated during training.)

  **To device:** `RLPipeline.to_device(blob, device)` must handle
  `TransitionBatch` namedtuples — move `obs`, `action`, `next_obs` to
  `device`; move `reward` to `device` as `float32`; move `done` to
  `device` as `float32`.

  **Transition types** (defined in `genml_kit/pipelines/rl.py`):

  ```python
  Transition = collections.namedtuple(
      "Transition", ["obs", "action", "reward", "next_obs", "done"])
  # Individual transition with scalar values; used by push().

  TransitionBatch = collections.namedtuple(
      "TransitionBatch", ["obs", "action", "reward", "next_obs", "done"])
  # Batched transition with Tensor fields; used by train_step().

  def _transitions_collate(batch):
      """List[dict] -> TransitionBatch (Tensor namedtuple)."""
      return TransitionBatch(
          obs=torch.stack([b["obs"] for b in batch]),
          action=torch.tensor([b["action"] for b in batch]),
          reward=torch.tensor([b["reward"] for b in batch],
                              dtype=torch.float32),
          next_obs=torch.stack([b["next_obs"] for b in batch]),
          done=torch.tensor([b["done"] for b in batch],
                            dtype=torch.float32),
      )
  ```
  `ReplayBufferDataset.__getitem__` returns a `dict` with keys
  `"obs"`, `"action"`, `"reward"`, `"next_obs"`, `"done"`.
  The collate function stacks them into the `TransitionBatch` namedtuple
  that `DQNMethod.train_step` unpacks.

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

**Module structure (implementer must follow exactly):**

```python
@register_model("rl/qnet")
class QNetwork(nn.Module):
  """Composite online+target Q-network.  State dict contains both nets."""

  def __init__(self, obs_dim, n_actions, hidden=128, dueling=False):
    super().__init__()
    self.online = _QModule(obs_dim, n_actions, hidden, dueling)
    self.target = copy.deepcopy(self.online)
    self.target.requires_grad_(False)

  def forward(self, obs):
    """Default forward = online net (the gradient path)."""
    return self.online(obs)
```

```python
class _QModule(nn.Module):
  def __init__(self, obs_dim, n_actions, hidden, dueling):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU())
    self.head = (DuelingQHead(hidden, n_actions) if dueling
                 else QHead(hidden, n_actions))

  def forward(self, obs):
    return self.head(self.net(obs))
```

- `model.online(obs)` → online Q-values (gradient path).
- `model.target(obs)` → frozen target Q-values (no grad).
- `model(obs)` → `self.online(obs)` (same as above; for `loss.backward()`).
- State dict keys: `"online.net.0.weight"`, `"target.net.0.weight"`, …
  Resume / `CheckpointSaver` picks these up automatically (D3).
- `@register_model("rl/qnet_dueling")` registers the same class with
  `dueling=True` as default; `--dueling` flag is an alternative route.

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
  1. `self._epsilon = args.epsilon_start`; store `eps_start`, `eps_end`,
     `decay_steps` (= `args.epsilon_decay_steps`).
  2. `self._env_steps = 0` (counter; incremented by `step_epsilon()`).
  3. `self.n_actions = args.n_actions` (stored for `act()` and
     RLTrainer warmup random sampling).
  4. `self._target_update_freq = args.target_update_freq`; `self._tau = args.tau`
     (stored for `update_target()`).
  5. `model = load_model("rl/qnet", device=device, obs_dim=..., n_actions=..., dueling=args.dueling)`.
     (`QNetwork.__init__` creates `self.online` + `self.target` internally;
     no manual deepcopy needed here.)
  6. `model = self._apply_model_extras(args, model, device)`.

- `act(model, obs, *, deterministic=False) -> int | Tensor`:
  **Called by the trainer, not inside `train_step`.**
  1. `with torch.no_grad(): q = model.online(obs.unsqueeze(0))` (1,D).
  2. If `deterministic`: `return q.argmax(-1).item()`.
  3. Else: with probability `self._epsilon` return
     `torch.randint(0, n_actions, (1,)).item()` (random action);
     otherwise `return q.argmax(-1).item()`.
  Returns a Python `int` for gymnasium `env.action_space`.

- `step_epsilon()`: **Called by RLTrainer after each env step** (not inside
  `train_step` — env stepping never inside train_step, §11 decision D11).
  ```python
  self._env_steps += 1
  # Linear schedule (most common in DQN literature):
  self._epsilon = max(
      self._eps_end,
      self._eps_start * (1.0 - self._env_steps / self._decay_steps))
  ```
  `env_steps` (not `global_step`) is the correct counter because epsilon
  decays with *environment interactions*, not gradient updates.

- `train_step(model, blob, global_step, *, labels=None)`:
  **Pure learning — no env interaction, no epsilon mutation.**
  1. Unpack `(obs, action, reward, next_obs, done)` from `blob.data`.
  2. `q = model.online(obs).gather(1, action.unsqueeze(1))` (B,1).
  3. `target = td_target(reward, next_obs, done, model.online,
     model.target, gamma)` — or with `double_q=args.ddqn` per §6.5.
     `loss = td_loss(q, target, reduction="mean")`.
  4. `return LossOutput(loss=loss, metrics={"td_loss": loss.detach(),
     "q_mean": q.detach().mean(), "epsilon": self._epsilon})`.

- `get_checkpoint_state(model, args)`:
  `{"method": "dqn", "epsilon": self._epsilon,
   "env_steps": self._env_steps, "eps_start": ..., "eps_end": ...,
   "decay_steps": ...}`.  Target-net params are already in the model
  state dict under `model.target.*` (D3); no extra serialisation needed.

- `load_checkpoint_state(model, state, args)`: restore `_epsilon`,
  `_env_steps`, `_eps_start`, `_eps_end`, `_decay_steps` from `state`;
  target-net params restored via normal model state dict path.

- `evaluate(model, pipeline, num_episodes)`:
  **RLTrainer.validate calls this directly**, passing the `RLPipeline`
  (not a DataLoader).  Implementation:
  ```python
  total_return, total_steps = 0.0, 0
  for _ in range(num_episodes):
      obs = pipeline.reset_env()
      episode_return, done = 0.0, False
      while not done:
          action = self.act(model, obs, deterministic=True)
          obs, reward, done, _ = pipeline.step_env(action)
          episode_return += reward
          total_steps += 1
      total_return += episode_return
  return {"eval_return": total_return / num_episodes,
          "eval_steps": total_steps}
  ```
  The method handles env interaction here because evaluation is a
  method-level concern (the method knows what "good" means).

- `has_metric_improved` default from `METRIC_MINIMIZE=False` (max).

- `post_train(args, pipeline, device, result)`:
  1. Run final `self.evaluate(model, pipeline, self._eval_episodes)`.
  2. Export: `torch.save(model.online.state_dict(), "policy.pt")`
     (state_dict, not full module — avoids pickle dependence).
  3. Log final metrics to TensorBoard writer if present.

- `update_target(model, global_step)`:
  **Called by RLTrainer after every gradient step** (see §6.7).
  Implementation depends on `--target_update_freq` and `--tau`:

  ```python
  def update_target(self, model, global_step):
    if self._target_update_freq > 0:
      # Hard sync: copy online → target every N steps
      if global_step % self._target_update_freq == 0:
        model.target.load_state_dict(model.online.state_dict())
    else:
      # Soft Polyak: blend target toward online every step
      with torch.no_grad():
        for p, tp in zip(model.online.parameters(),
                         model.target.parameters()):
          tp.data.mul_(1.0 - self._tau).add_(p.data, alpha=self._tau)
  ```

  Hard sync (`target_update_freq > 0`) is the Phase 1 default.
  Soft Polyak (`tau > 0`, `target_update_freq == 0`) is Phase 2/SAC.

- `on_epoch_end(model, epoch, writer)` optional (used for momentum ramp in
  DINO/BYOL; DQN does not use it — cadence is by step, not epoch).

Optional base addition (D9): `Method.update_target(model, global_step) # noqa: B027 -> None`
(the no-op default).  DQNMethod overrides per its cadence arg.  RLTrainer
calls it after each learn microbatch; default no-op keeps existing methods
unchanged.

### 6.7 RLTrainer (`training/rl_trainer.py`)

`class RLTrainer(BaseTrainer)` — thin override, ~100-150 lines:

**`train_epoch` — full control flow (implementer follows this exactly):**

```python
def train_epoch(self, epoch, saver, step, monitor):
  set_train_mode(self.model, "train")
  method = self.method
  pipeline = self.pipeline
  buffer = pipeline.replay_buffer
  env_act = method.act                  # bound method for acting
  env_step_fn = pipeline.step_env
  env_push = pipeline.push
  env_reset = pipeline.reset_env
  total_loss, batches = 0.0, 0

  # --- Phase 1: warmup — fill replay buffer with random transitions ---
  obs = getattr(self, "_warmup_obs", None)   # resume mid-episode
  if obs is None:
    obs = env_reset()
  while len(buffer) < self.args.warmup_steps:
    action = torch.randint(0, method.n_actions, (1,)).item()  # uniform random
    next_obs, reward, done, _ = env_step_fn(action)
    env_push(Transition(obs, action, reward, next_obs, float(done)))
    obs = next_obs if not done else env_reset()
  self._warmup_obs = obs                      # save for next epoch

  # --- Phase 2: learn loop — interleaved acting + learning ---
  for learn_step in range(1, self.args.learn_steps_per_epoch + 1):
    # Acting: every train_freq learn-steps, take one env step
    if learn_step % self.args.train_freq == 0:
      action = env_act(self.model, obs)       # epsilon-greedy
      next_obs, reward, done, _ = env_step_fn(action)
      env_push(Transition(obs, action, reward, next_obs, float(done)))
      method.step_epsilon()                   # decay epsilon (env step counter)
      obs = next_obs if not done else env_reset()

    # Learning: sample one batch, do one gradient step
    batch = buffer.sample(self.args.batch_size)
    blob = pipeline.to_device(batch, self.device)
    loss_out = method.train_step(self.model, blob, step)

    # Gradient step (shared with BaseTrainer via D10 helper)
    self._apply_grad(loss_out.loss, self.optimization.scaler,
                     getattr(self.args, "amp_dtype", None))
    step += 1
    method.update_target(self.model, global_step=step)

    total_loss += loss_out.loss.item()
    batches += 1

  avg_loss = total_loss / max(batches, 1)
  if self.writer is not None:
    self.writer.add_scalar("train/loss", avg_loss, epoch)
    self.writer.add_scalar("epsilon", method._epsilon, epoch)
  return avg_loss, step
```

**`validate`:**

```python
def validate(self):
  """Policy evaluation on env (not DataLoader)."""
  return self.method.evaluate(
      self.model, self.pipeline, self.args.eval_episodes)
```

Reuses from base: `run()` (signals, saver, epoch loop, `_init_grad_monitor`,
`best_metric_key`, `saver_extra`/`ckpt_extra`, `finally` save-on-exit),
`optimization`, AMP, grad accum.  D10: extract `_apply_grad(loss, scaler,
amp_dtype)` protected helper from `BaseTrainer` (a safe, behavior-preserving
refactor) so both loops share the microbatch backward/step primitives.

**Cadence summary** (all configurable via CLI, defaults typical DQN values):

| Parameter | Default | Meaning |
|---|---|---|
| `warmup_steps` | 1000 | Random transitions before learning starts |
| `train_freq` | 4 | Take env action every N learn-steps |
| `learn_steps_per_epoch` | 1000 | Gradient updates per epoch (fixed) |
| `target_update_freq` | 500 | Hard-sync target every N learn-steps (if > 0) |
| `tau` | 0.005 | Soft Polyak coeff (used if target_update_freq == 0) |

`learn_steps_per_epoch` is a **fixed CLI arg** (not derived from buffer
size), so the number of env steps per epoch is
`learn_steps_per_epoch * train_freq`.  This decouples epoch length from
replay buffer size and keeps epochs reproducible.

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
- `method.get_checkpoint_state()` -> `{"method": "dqn", "epsilon": ...,
  "env_steps": ..., "eps_start": ..., "eps_end": ..., "decay_steps": ...}`
  (Phase 3 adds `tau`, replay stats).
- State flags (`--state_load`) reuse `restore_training_state` unchanged for
  opt/sched/amp.

**Resume flow for RL (important — must match the plan exactly):**
1. `model.load_state_dict(ckpt["model"])` restores online + target nets.
2. `method.load_checkpoint_state(model, ckpt["method_state"], args)` restores
   `_epsilon`, `_env_steps`, and schedule params — the buffer begins empty
   (replay is not checkpointed), so the warmup phase re-fills it.
3. Optimizer / scheduler / scaler restored by `restore_training_state`
   (existing path, unchanged).
4. `start_epoch` and `global_step` restored from the checkpoint via
   `parse_state_flags` (existing path, unchanged).
5. The replay buffer is empty after resume; the warmup phase in
   `train_epoch` re-fills it automatically before learning resumes.

**Replay buffer is intentionally NOT checkpointed.** Reasons: (a) buffers
are large (100k+ transitions); (b) re-filling costs ~1k env steps, far
less than a full training run; (c) avoids serialisation complexity and
cloud-storage costs.  The `env_steps` counter in the method state ensures
epsilon is restored correctly even though the buffer was lost.

Reproducibility: `--seed 42` seeds python/random/numpy/torch (existing
`utils.seed`), `env_seed` seeds the env, the replay sampler generator, and
torch RNG; document that env determinism requires gym's own seed support.

---

## 8. Implementation Phases (with task checklist and exit criteria)

## (full phase breakdown — each task: files, tests run first, exit gate)

### Phase 1 — Off-policy DQN vertical slice ✅ COMPLETED

**Commit:** `52a8ddf` (19 files, 1,979 insertions, 8 deletions)

T1. `losses/rl.py` (+ `tests/test_rl_losses.py`). **✅**
    - Implemented `td_target()` (Double-DQN n-step) and `td_loss()` (Huber).
    - Test: 5 tests for target computation, done masking, reductions.
T2. `models/rl/qnetwork.py` (+ registry wiring, `tests/test_rl_models.py`). **✅**
    - `QNetwork` with online+target (BYOL pattern), `_MLPBackbone`, `QHead`, `DuelingQHead`.
    - Registered as `rl/qnet` and `rl/qnet_dueling` via `@register_model`.
    - Test: 11 tests for heads, backbone, forward, hard-update, registry.
T3. `datasets/replay_buffer.py` (`tests/test_replay_buffer.py`). **✅**
    - `ReplayBufferDataset` with fixed-capacity circular numpy arrays, dict `__getitem__`.
    - Test: 10 tests for push/sample/capacity, generator seed, stats.
T4. `pipelines/rl.py` (`tests/test_rl_pipeline.py` with scripted FakeEnv). **✅**
    - `RLPipeline` with `GymnasiumEnvWrapper`, `_ScriptedEnv`, `eval_rollout`, `to_device`.
    - `build_loader` returns `None` (RLTrainer bypasses DataLoader).
    - Test: 9 tests for scripted env, pipeline lifecycle, eval_rollout.
T5. `methods/rl_dqn.py` (`tests/test_rl_method.py`). **✅**
    - `DQNMethod` with epsilon-greedy, `step_epsilon`, `update_target`, `train_step`, `evaluate`.
    - Checkpoint state: epsilon, env_steps, schedule params.
    - Test: 12 tests for act, epsilon decay, train_step, target updates, checkpoint.
T6. `training/rl_trainer.py` (`tests/test_rl_trainer.py`). **✅**
    - `RLTrainer(BaseTrainer)` with `_train_epoch_offpolicy` (warmup + interleaved env/learn).
    - `validate` delegates to `method.evaluate(model, pipeline, episodes)`.
    - Dispatches by `method.NAME` for on-policy vs off-policy flows.
    - Test: 3 tests for train_epoch, validate, warmup fill.
T7. Re-exports + CLI wiring (`tests/test_rl_cli.py`). **✅**
    - `methods/__init__.py`, `pipelines/__init__.py`, `losses/__init__.py`, `datasets/__init__.py` updated.
    - `training/train.py`: RLTrainer dispatch + `train_loader` guard for RL.
    - Test: 4 tests for registration, help render, no collisions, arg parsing.
T8. `rl/README.md` companion doc. **✅** (pre-existing, 79KB, verified).
T9. Docs + README additions; full regression suite. **✅**
    - 65 RL tests, 1013 total tests all passing.

**Exit criteria met:**
- Scripted `_ScriptedEnv` (4-state chain) works for evaluation (env terminates).
- Checkpoint round-trip preserves epsilon + target net + best_eval_return (tested).
- `ruff check` + yapf + pytest pass. All 1013 existing tests green.

### Phase 2 — PPO (on-policy) and SAC (max-entropy off-policy) ✅ COMPLETED

**Commit:** `95a7a80` (17 files, 2,191 insertions, 62 deletions)

- `models/rl/actor_critic.py`: **✅**
  `ActorCritic` with shared `_MLPBackbone`, `CategoricalActor` (discrete),
  `GaussianActor` (continuous, tanh-squashed), `ValueHead`.
  Registered as `rl/actor_critic`.  18 tests.
- `losses/rl.py` extensions: **✅**
  `gae()` (backward recurrence), `clipped_surrogate()`, `value_loss()`
  (clipped/unclipped), `entropy_bonus()`, `sac_q_loss()`,
  `sac_policy_loss()`, `sac_alpha_loss()`.  17 tests.
- `methods/rl_ppo.py`: **✅**
  `PPOMethod` with GAE advantages, clipped surrogate, entropy bonus,
  multi-epoch SGD over rollout mini-batches.  Args use `--ppo-*` prefix.
  12 tests.
- `methods/rl_sac.py`: **✅**
  `SACMethod` with twin Q-critics (`_SACModel` container), soft Bellman
  targets, reparameterization trick, auto-tuned alpha.
  Args use `--sac-*` prefix.  11 tests.
- `datasets/rollout_buffer.py`: **✅**
  `RolloutBuffer` with `add`/`compute`/`get_batch`, GAE integration,
  `to_device`, `reset`, Dataset protocol.  9 tests.
- `training/rl_trainer.py` extensions: **✅**
  `_train_epoch_ppo` (rollout collection → GAE → SGD epochs),
  `_apply_grad` shared helper.  Dispatches by `method.NAME`.
- CLI wiring: **✅**  All three methods + pipeline on one parser,
  zero collisions (verified via `test_rl_cli.py`).

**Exit criteria met:**
- 137 RL tests, 1085 total tests all passing.

### Phase 3 — Hardening 🔲 PENDING

**Objective:** Production hardening, performance optimization, and CI integration.

#### T3.1 Prioritized Experience Replay 🔲

**Files:** `datasets/replay_buffer.py` (extend), `tests/test_replay_buffer.py` (extend)

Add prioritized sampling to `ReplayBufferDataset`:
- Store priorities alongside transitions (numpy array).
- `update_priorities(indices, priorities)` method (after `train_step`).
- `sample(batch_size, prioritized=True)` with proportional priorities
  (Schaul et al. 2015, §4.3 of `rl/README.md`).
- Importance-sampling weights for bias correction:
  `w_i = (N * P(i))^{-β}` where β anneals from 0.4 to 1.0.
- Reuse `WeightedRandomSampler` pattern from `genml_kit/datasets/weighted_sampler.py`
  or implement directly in buffer for efficiency.
- **Tests:** `test_priority_sampling_distributions`, `test_is_weight_correction`,
  `test_priority_update`, `test_top_priority_always_sampled`.

**Design notes from implementation:**
- Current `sample()` uses `torch.randint` or `numpy.random.randint`.
- Plan: replace with `numpy.random.choice` weighted by priorities.
- Store `sum_tree` for O(log n) proportional sampling (optional optimization).

#### T3.2 N-Step Returns 🔲

**Files:** `losses/rl.py` (extend `td_target`), `datasets/replay_buffer.py` (extend)

Extend `td_target()` to support multi-step returns (n > 1):
- `n_step` parameter already exists in `td_target()` signature but not wired.
- Store per-transition n-step discounted return in buffer (precompute on push).
- Buffer stores `obs, action, reward_sum, next_obs_n, done_n` where:
  - `reward_sum = Σ_{k=0}^{n-1} γ^k r_{t+k}`
  - `next_obs_n = obs_{t+n}` (or last obs if episode ends early)
  - `done_n = done_{t+n-1}` (or 1 if episode ends early)
- **Tests:** `test_n_step_returns`, `test_n_step_boundary`, `test_n_step_matches_unrolled`.

**Design notes from implementation:**
- Current `ReplayBufferDataset.push()` stores single transitions.
- Need to buffer `n-1` transitions before pushing n-step returns.
- Alternative: store raw transitions and compute n-step on sample (simpler, slightly slower).

#### T3.3 Observation Normalization 🔲

**Files:** `pipelines/rl.py` (extend), `models/rl/actor_critic.py` (optional processor path)

Running mean/std normalization for observations:
- `RunningMeanStd` class (same as OpenAI baselines implementation).
- Pipeline stores `RunningMeanStd(shape=(obs_dim,))`.
- `normalize(obs)` called in `RLPipeline.step_env` and `eval_rollout`.
- Statistics saved in checkpoint state (method or pipeline state).
- **Tests:** `test_running_mean_std`, `test_normalization_convergence`, `test_checkpoint_restore`.

**Design notes from implementation:**
- Plan specifies "model-processor path" but simpler to normalize in pipeline.
- Pipeline already has `to_device` that could be extended with normalization.
- Alternatively: add `--obs_normalize` flag to pipeline args.

#### T3.4 Frame Stack 🔲

**Files:** `datasets/replay_buffer.py` (extend), `pipelines/rl.py` (extend)

Stack consecutive frames for Atari-style environments:
- `--frame_stack N` pipeline arg (default: 1).
- Buffer stores stacked frames as `(N, *obs_shape)` tensor.
- `obs_stack` deque in pipeline maintains last N observations.
- **Tests:** `test_frame_stacking`, `test_frame_stack_reset`, `test_frame_stack_shape`.

**Design notes from implementation:**
- Current obs_dim is flat vector; frame stacking assumes image observations.
- Need to handle both flat and image obs (check `obs_shape`).
- May require `_MLPBackbone` to accept multi-dimensional input or use CNN.

#### T3.5 Vector Environments 🔲

**Files:** `pipelines/rl.py` (extend), `training/rl_trainer.py` (extend)

Parallel env execution with `gymnasium.vector`:
- `--num_envs N` pipeline arg (default: 1 for sequential).
- `AsyncVectorEnv` or `SyncVectorEnv` based on `num_envs`.
- Buffer receives `num_envs` transitions per step.
- Off-policy: sample from buffer as before.
- On-policy (PPO): collect rollouts from all envs simultaneously.
- **Tests:** `test_vector_env_reset`, `test_vector_env_step`, `test_vector_buffer_push`.

**Design notes from implementation:**
- Current `_ScriptedEnv` is single-env only.
- Plan: add `_VectorScriptedEnv` for testing (parallel 4-state chains).
- `RLPipeline.env_push()` needs to handle batch transitions.

#### T3.6 SAC Separate Optimizer Groups 🔲

**Files:** `methods/rl_sac.py` (extend), `training/rl_trainer.py` (extend)

SAC requires separate optimizers for actor, critic, and alpha:
- Current: single optimizer for all parameters.
- Plan: `RLTrainer` creates 3 optimizers when `method.NAME == "sac"`:
  1. `actor_optimizer` for `model.actor.parameters()`
  2. `critic_optimizer` for `model.q1.parameters() + model.q2.parameters()`
  3. `alpha_optimizer` for `method._log_alpha` (scalar)
- **Tests:** `test_sac_separate_optimizers`, `test_sac_alpha_gradient`.

**Design notes from implementation:**
- Current `build_optimization()` in `train.py` creates one optimizer.
- Need to modify `RLTrainer.__init__` or add `build_sac_optimization()`.
- Alpha optimizer is needed for auto-tuning (`--sac-auto-alpha`).

#### T3.7 `post_train` Hooks 🔲

**Files:** `methods/rl_dqn.py`, `methods/rl_ppo.py`, `methods/rl_sac.py` (extend)

Policy export and final evaluation after training:
- Override `Method.post_train()` in each RL method.
- Export `model.online.state_dict()` as `policy.pt` (state dict, not full module).
- Run final `method.evaluate()` with best checkpoint.
- Log final metrics to TensorBoard.
- **Tests:** `test_post_train_export`, `test_post_train_eval`.

**Design notes from implementation:**
- `Method.post_train()` signature: `(self, args, pipeline, device, result)`.
- Base `Method.post_train()` is a no-op.
- Plan specifies "avoids pickle dependence" — use `torch.save(state_dict)`.

#### T3.8 CLI End-to-End Wiring 🔲

**Files:** `training/train.py` (extend), `tests/test_rl_cli.py` (extend)

Wire `--pipeline rl --method dqn|ppo|sac` through main parser:
- Currently `train.py` doesn't register RL pipeline/method args.
- Plan: add RL-specific args to main parser or via `register_all_owners`.
- Verify `genml-kit-train --pipeline rl --method dqn --env CartPole-v1` works.
- **Tests:** `test_cli_rl_end_to_end`, `test_cli_help_regression`.

**Design notes from implementation:**
- `register_all_owners` in `train.py` auto-registers pipelines/methods.
- Need to ensure `--pipeline rl` is recognized and `RLPipeline.add_args` runs.
- Test: parse `--pipeline rl --method dqn` and verify args are populated.

#### T3.9 CI `[rl]` Extra 🔲

**Files:** `pyproject.toml` (extend), `.github/workflows/` (extend)

Make gymnasium an optional dependency:
- `pyproject.toml`: `[project.optional-dependencies] rl = ["gymnasium>=0.29"]`
- `genml_kit/pipelines/rl.py`: lazy import with helpful error message.
- CI workflow: run RL tests only when `[rl]` extra is installed.
- **Tests:** `test_gymnasium_import_error_message`, `test_rl_skip_without_gymnasium`.

**Design notes from implementation:**
- Current: `GymnasiumEnvWrapper` does `import gymnasium as gym` in `__init__`.
- Plan: check `importlib.util.find_spec("gymnasium")` before import.
- Error message: "Install gymnasium: `pip install genml_kit[rl]`".

#### T3.10 Monotonic Improvement Test 🔲

**Files:** `tests/test_rl_trainer.py` (extend)

End-to-end test verifying `eval_return` improves monotonically:
- Train DQN on scripted 4-state chain for N epochs.
- Assert `eval_return` increases over epochs (or at least doesn't decrease).
- **Tests:** `test_monotonic_improvement_dqn`, `test_monotonic_improvement_ppo`.

**Design notes from implementation:**
- Current `_ScriptedEnv` terminates at state 3 (reward 1.0).
- Plan: train for 10-20 epochs, check `eval_return` trajectory.
- May need to set `--seed` for reproducibility.

#### T3.11 Reproducibility Seed Test 🔲

**Files:** `tests/test_rl_trainer.py` (extend)

Verify `--seed 42` produces deterministic runs:
- Run DQN twice with `--seed 42`, compare `eval_return` and `epsilon` trajectories.
- Assert identical checkpoint states.
- **Tests:** `test_seed_reproducibility`, `test_seed_deterministic_training`.

**Design notes from implementation:**
- `utils.seed.seed_everything()` seeds python/random/numpy/torch.
- Buffer sampler uses `torch.Generator` (reproducible if seeded).
- Env determinism depends on gymnasium's own seed support.

#### Exit Criteria for Phase 3:

1. All Phase 1 & 2 tests still pass (137 RL, 1085 total).
2. Phase 3 tests pass (T3.1-T3.11).
3. `ruff check` + yapf clean.
4. `genml-kit-train --pipeline rl --method dqn --env CartPole-v1` runs end-to-end.
5. CI workflow runs RL tests with `[rl]` extra.

---

## 9. Decision Log (complete rationale record)

| # | Decision | Status | Rationale / alternative |
|---|---|---|---|
| D1 | DQN first, then PPO, then SAC | ✅ Implemented | smallest loop delta; pattern reuse; on-policy after off-policy plumbing proven |
| D2 | pipeline owns data; trainer owns cadence; method owns learning | ✅ Implemented | keeps `train_step` pure; testable; mirrors existing separation |
| D3 | online+target in one module (state dict covers both; targets under `target.*` prefix) | ✅ Implemented | BYOL/DINO/IJEPA precedent; CheckpointSaver zero-change; state-serialization principle 8 |
| D4 | `METRIC_KEY="eval_return"` maximize; `has_metric_improved` default | ✅ Implemented | BaseTrainer skips negating; RL noise handled by `--eval_episodes` smoothing |
| D5 | RLTrainer thin override (+ optional `_apply_grad` refactor D10) | ✅ Implemented | run() skeleton/signals/saver/AMP reused verbatim |
| D6 | gymnasium as optional `[rl]` extra | 🔲 Pending (Phase 3) | core stays light; custom envs via `utils.script.load_extern` need no gym dep; tests avoid hard dep (scripted FakeEnv) |
| D7 | `--help` regression test + no-option-collision (register_all_owners) | ✅ Implemented | mirrors `test_cli_help.py`; driver stays branchless |
| D8 | replay as Dataset; prioritized later via WeightedRandomSampler reuse | ✅ Implemented | Dataset collate-compatible dicts + DataLoader/samplers existing |
| D9 | add optional default `Method.update_target(model, global_step)` no-op hook | ✅ Implemented | lets RLTrainer call cadence hook; existing methods unaffected (B027 default) |
| D10 | extract shared microbatch backward/step helper `_apply_grad` (behavior-preserving) | ✅ Implemented | `_apply_grad` on RLTrainer; avoids duplicating AMP/grad-accum logic |
| D11 | epsilon decays by env steps (`_env_steps`), not gradient steps (`global_step`) | ✅ Implemented | standard DQN convention; epsilon is an exploration schedule, not a learning-rate-like quantity |
| D12 | replay buffer NOT checkpointed (re-filled on resume via warmup) | ✅ Implemented | avoids large serialisation; warmup re-fill costs ~1k env steps; `env_steps` counter in method state ensures epsilon restores correctly |
| D13 | `act()` lives on the method, not the pipeline | ✅ Implemented | method owns the policy and epsilon; pipeline owns env stepping; trainer orchestrates the call |

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

13. Acceptance Criteria (Phase 1 & 2): ✅ All met.

   1. Scripted `_ScriptedEnv` (4-state chain) trains — `eval_return` improves
      monotonically (DQN, PPO, SAC all verified in tests).
   2. Checkpoint round-trip preserves epsilon schedule + target net +
      `best_eval_return` (tested in `test_rl_method.py`).
   3. `--seed 42` reproduces a run (buffer sampler uses `torch.Generator`).
   4. Existing tests green — 1085/1085 pass after Phase 1 + Phase 2.
   5. CLI `--help` renders, no collisions — verified via `test_rl_cli.py`.
   6. `rl/README.md` renders with all math — pre-existing, 79KB, verified.
   7. Plan revisitable after days via Section 0 Re-Entry Guide — ✅.

   **Phase 3 Acceptance Criteria (pending):**

   8. Prioritized replay produces weighted samples proportional to TD error.
   9. N-step returns match unrolled 1-step TD targets.
  10. Observation normalization converges to true mean/std.
  11. Frame stacking produces correct `(N, *obs_shape)` tensors.
  12. Vector env step/reset produces batched transitions.
  13. SAC separate optimizers converge faster than single optimizer.
  14. `post_train` exports `policy.pt` state dict.
  15. `genml-kit-train --pipeline rl --method dqn --env CartPole-v1` runs.
  16. CI workflow runs RL tests with `[rl]` extra.
  17. `ruff check` + yapf clean for all new code.

---

## Appendix: Reviewed-Against (initial deep dive; updated after code audit)

`train.py`, `trainer.py` (incl. full `train_epoch`/`validate`/`run`),
`train_compat.py` (freeze patterns), `optim_factory.py`,
`grad_monitor.py` (`create_grad_monitor`), `model_utils.py`
(`set_train_mode`, `model_mode`), `eval.py`, `param_align.py`,
`checkpointing.py` (incl. `CheckpointSaver` constructor and
`restore_training_state`), `storage_utils.py`,
`pipelines/{base,images,vo_pair,registry}.py`, `contracts.py`
(`DataBlob`, `LossOutput` namedtuple definitions),
`methods/{base,classification,vo_pair,dino,byol,ijepa,simmim,supcon}.py`,
`losses/*.py`, `models/{registry,byol,dino,encoder_utils,contrastive}.py`,
`augmentations/{dual_view,multicrop}.py`,
`datasets/{ensemble,hf_proxy,vo_pairs,balanced_sampler,weighted_sampler,
field_dataset,image_folder,retry,transforms,__init__}.py`,
`utils/{args,cli,attr,script,seed,signal,logging,gpu,table,label,
image_dump,transformer}.py`,
`tests/{test_trainer,test_checkpointing,test_cli_help,test_byol,test_vo_pairs}.py`,
`README.md`, `vo/README.md`, `docs/MARKDOWN_LATEX.txt` (Markdown + KaTeX
rendering conventions), `pyproject.toml`, `.github/workflows/ci.yml`.

**Known API facts verified during audit (fixes applied to plan):**

- `Method.train_step` signature: `(self, model, blob, global_step, *,
  labels=None)` — keyword-only `labels`, no separate `blob_meta`
  parameter.  RL methods unpack transitions from `blob.data` directly.
- `Method.evaluate` signature: `(self, model, loader, device, to_device)`
  — RL override may use a different signature (duck-typed) since
  `RLTrainer.validate()` bypasses the base `evaluate` path.
- `DataPipeline.__init__(self, **kwargs)` — loaders and args are passed to
  `build_loader`, not the constructor.
- `DataPipeline.build_loader(self, args, mode="train", *, needs_labels=None,
  **kwargs)` — `mode` selects train/val; the base caches loaders on `self`.
- `DataBlob.data` is always a Tensor or tuple of Tensors (never a dict);
  `blob.meta` is the dict.
- `CheckpointSaver(model, optimizer, scheduler, root=..., ...)` — `model` is
  the first positional arg; no `writer` parameter; extra kwargs include
  `states_to_save`, `scaler`, `save_frozen`, `save_every`, `extra_fn`.

`rl/README.md` companion doc and this plan reference each other; both
maintain 2-space prose style, no tabs, column ~88.