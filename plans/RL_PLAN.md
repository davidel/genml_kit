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
> such as `\pi`, `\theta`, `\gamma`, `\to`, `\le` — see
> [`docs/MARKDOWN_LATEX.txt`](docs/MARKDOWN_LATEX.txt) for the full rules).

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
| D11 | epsilon decays by env steps (`_env_steps`), not gradient steps (`global_step`) | agreed | standard DQN convention; epsilon is an exploration schedule, not a learning-rate-like quantity; `_env_steps` counter lives on the method, incremented by `step_epsilon()` called from RLTrainer after each env push |
| D12 | replay buffer NOT checkpointed (re-filled on resume via warmup) | agreed | avoids large serialisation (~400 MB for 100k transitions × 4 obs floats × 8 bytes); warmup re-fill costs ~1k env steps, far cheaper than a full training run; `env_steps` counter in method state ensures epsilon restores correctly |
| D13 | `act()` lives on the method, not the pipeline | agreed | method owns the policy and epsilon; pipeline owns env stepping; trainer orchestrates the call — consistent with D2 (policy = method concern) |

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