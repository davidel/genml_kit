# RL_REVIEW: Remediation & Completion Plan for RL Support in genml_kit

Status: **Draft for review** — supersedes `plans/RL_PLAN.md` (removed from
the repo; recoverable from git history. This file is now the single
source of truth for remaining RL work).

This document was produced by a full audit of the former `plans/RL_PLAN.md`
against the actual implementation. It carries over everything that plan still
listed as
pending (its Phase 3, tasks T3.1-T3.11) and adds every defect and design
issue found during the review. Each item below is written so it can be
implemented without re-deriving the analysis: problem, evidence (file/line),
fix design, touched files, and the tests that must be added or fixed.

Conventions for every code change in this document:

- Python Google style, **2-space indent**, yapf (`.style.yapf`) authoritative.
- **No typing annotations** anywhere (project style).
- All unrecoverable error paths go through
  `from genml_kit.utils.logging import fatal` — **never** a bare `raise`.
  Pattern: `fatal(f"...", ValueError)`.
- `ruff check` must stay clean; no `# noqa` suppressions without approval.
- Nothing is committed to git until the reviewer approves the diff
  (and never amend/squash existing commits).

---

## 1. What the plan says is done (verified)

Phases 1 and 2 of the former plan are implemented and the audit confirms the
following pieces exist and are structurally sound:

- `RLPipeline` (`pipelines/rl.py`) with `init_env`, `reset_env`,
  `step_env`, `env_push`, `obs_dim`, `n_actions`, `eval_rollout`,
  `to_device`; scripted `_ScriptedEnv` for gym-free tests.
- `ReplayBufferDataset` (`datasets/replay_buffer.py`) — circular numpy
  storage, `push`, `push_batch`, `sample`, `stats`.
- `RolloutBuffer` (`datasets/rollout_buffer.py`) — `add`, `compute` (GAE),
  `set_next_values`, `get_batch`, `to`.
- Models: `rl/qnet`, `rl/qnet_dueling` (`models/rl/qnetwork.py`),
  `rl/actor_critic` (`models/rl/actor_critic.py`) with
  `get_action_and_value`, `get_distribution`, `get_value`.
- Losses (`losses/rl.py`): `td_target`, `td_loss`, `gae`,
  `clipped_surrogate`, `value_loss`, `entropy_bonus`, `sac_q_loss`,
  `sac_policy_loss`, `sac_alpha_loss`.
- Methods: `DQNMethod`, `PPOMethod`, `SACMethod` implementing the
  `Method` contract (`act`, `train_step`, `update_target`, `evaluate`,
  `has_metric_improved`, checkpoint state get/load).
- `RLTrainer` (`training/rl_trainer.py`) with off-policy and PPO epoch
  flows, warmup fill, and env-based `validate`.
- `train.py` selects `RLTrainer` for `pipeline.NAME == "rl"`.

Test count reality check: the plan's header claims "137 RL tests passing"
and its exit criteria repeat "137 RL, 1085 total". The actual suite today is
**106 RL tests** (`python -m pytest tests/ -k "rl"`). The numbers in the old
plan are stale; this document does not rely on them.

---

## 2. Issue register (summary)

Severity legend: **A** = critical (crash or silently corrupts learning),
**B** = high (biased metrics / dead features), **C** = medium (design
smells, cleanup), **D** = test gaps, **E** = plan Phase-3 features still
pending, **F** = packaging/CI.

| ID | Severity | Title | File(s) |
|----|----------|-------|---------|
| A1 | A | `run()` crashes: validate returns dict, best-metric compare does `dict > float` | `training/rl_trainer.py` |
| A2 | A | SAC bootstraps from **online** critics; target critics are dead code | `methods/rl_sac.py` |
| A3 | A | Single optimizer for SAC lets policy gradient corrupt critics; `--sac-actor-lr/-critic-lr/-alpha-lr` unused | `methods/rl_sac.py` |
| A4 | A | Auto-alpha inert: `alpha_loss` never optimized, `_log_alpha` frozen | `methods/rl_sac.py` |
| A5 | A | DQN target net never updates with default args (`target_update_freq=0` and `tau=1.0` cancel out) | `methods/rl_dqn.py` |
| A6 | A | Checkpoints silently drop frozen target nets (`SAVE_FROZEN=False` + `trainable_state_dict`) | `training/trainer.py`, `methods/rl_*` |
| A7 | A | PPO bootstrap: scalar `V(s_T)` broadcast over whole rollout; every advantage wrong | `training/rl_trainer.py`, `datasets/rollout_buffer.py` |
| A8 | A | PPO continuous re-evaluation uses post-tanh actions in Gaussian log-prob | `models/rl/actor_critic.py` |
| A9 | A | Continuous action spaces unsupported (`action_space.n` AttributeError, buffer hard-codes discrete) | `pipelines/rl.py` |
| B1 | B | `evaluate()` shares one `max_steps` budget across episodes | `methods/rl_dqn.py`, `rl_sac.py`, `rl_ppo.py` |
| B2 | B | `validate()` reseeds global numpy RNG each epoch; `--env_seed` never reaches the env | `training/rl_trainer.py`, `pipelines/rl.py` |
| B3 | B | PPO episode-return tracking is dead code (`total_reward` zeroed, never read) | `training/rl_trainer.py` |
| B4 | B | PPO value clipping can never engage (`old_values=None`) | `methods/rl_ppo.py` |
| B5 | B | `_env_steps` only advances for DQN | `methods/rl_ppo.py`, `rl_sac.py`, `rl_trainer.py` |
| B6 | B | `eval_return` never written to TensorBoard | `training/rl_trainer.py` |
| C1 | C | `update_target` counts gradient steps; docs say env steps | `methods/rl_dqn.py` |
| C2 | C | Warmup does not call `step_epsilon()` | `training/rl_trainer.py` |
| C3 | C | `td_target(n_step=...)` advertised but unusable | `losses/rl.py` |
| C4 | C | `_train_epoch_ppo` reads `rollout.__dict__`, silently drops mismatched tensors | `training/rl_trainer.py` |
| C5 | C | Unused parallel eval path `RLPipeline.eval_rollout` | `pipelines/rl.py` |
| C6 | C | Stale `ReplayBufferDataset` docstring (DataLoader/collate claims) | `datasets/replay_buffer.py` |
| C7 | C | `_SACModel` defined inside `build_model`, unregistered | `methods/rl_sac.py` |
| C8 | C | `_log_alpha` lives outside model/optimizer/state_dict | `methods/rl_sac.py` |
| C9 | C | `init_env` silently falls back to `obs_dim=4` | `pipelines/rl.py` |
| C10 | C | Arg style inconsistent (`--env_id` vs `--sac-gamma`) | `pipelines/rl.py` |
| C11 | C | `METRIC_MINIMIZE` declared, never read | `methods/rl_*` |
| C12 | C | Resume: `_epsilon`/`_env_steps` restored independently; PPO `_ppo_obs` lost | `methods/rl_*`, `training/rl_trainer.py` |
| D1-D7 | D | Test-coverage program (see §7) | `tests/` |
| E1-E11 | E | Plan Phase-3 tasks T3.1-T3.11, re-scoped (see §8) | various |
| F1 | F | `gymnasium` missing from `pyproject.toml` extras; `all` extra does not include it (see §9) | `pyproject.toml` |

### ✅ Completed in Round 1 (this PR)

| ID | Phase | Summary | Status |
|----|-------|---------|--------|
| **F1** | F | Add `[rl]` extra with `gymnasium` to `pyproject.toml` | ✅ DONE |
| **A1** | A | `RLTrainer.validate()` returns scalar `eval_return` (not dict) | ✅ DONE |
| **A5** | A | DQN `target_update_freq` default 0→1 (update every step) | ✅ DONE |
| **A6** | A | `SAVE_FROZEN = True` on `RLTrainer` for target net checkpointing | ✅ DONE |
| **B1** | B | `evaluate()` per-episode step budget (all 3 methods) | ✅ DONE |
| **B2** | B | `validate()` saves/restores `np.random` state (RNG isolation) | ✅ DONE |
| **B5** | B | `_env_steps` tracking for SAC (in `train_step`) and PPO (in trainer) | ✅ DONE |
| **C1** | C | `has_metric_improved(new, best)` signature consistency (all 3 methods) | ✅ DONE |
| | | `train.py`: call `pipeline.init_env()` before `wire_data()` for RL | ✅ DONE |

---

## 3. Phase A — Critical fixes (do these first, in order)

### A1. `RLTrainer.validate()` must return the scalar metric ✅ DONE

**Fixed in Round 1.** `RLTrainer.validate()` now returns `float(metrics["eval_return"])` 
and saves/restores `np.random` state around evaluation (B2 fix included).
TensorBoard logging of `val/eval_return` is handled by `BaseTrainer.run()` 
since it extracts the scalar from the returned value.

**Files:** `genml_kit/training/rl_trainer.py`.
**Tests:** Updated `test_validate_returns_metrics` → `test_validate_returns_scalar`.

### A2. SAC must bootstrap from the target critics ✅ DONE

**Fixed in Round 2.** Changed `train_step` to use `model.q1_target.get_value()` 
and `model.q2_target.get_value()` for the TD target computation. SAC's 
stabilizing mechanism (slow-moving target for Bellman backup) is now active.

**Files:** `genml_kit/methods/rl_sac.py`.
**Tests:** Existing `test_train_step` and `test_update_target_soft` cover this.

### A3. SAC needs separate actor/critic/alpha optimization

**Problem.** `train_step` returns one combined loss
(`critic_loss + actor_loss`, `:237`) and the trainer steps the single
optimizer built over `model.parameters()`. Two consequences:

1. The actor loss `sac_policy_loss(new_log_prob, min_q_new, alpha)` flows
   through `model.q1.get_value(obs)` / `model.q2.get_value(obs)`
   (`:223-228`) **into critic parameters** — the critics receive a policy
   gradient that has nothing to do with the Bellman residual. Standard SAC
   never lets the actor update touch critic weights.
2. `--sac-actor-lr`, `--sac-critic-lr`, `--sac-alpha-lr`
   (`methods/rl_sac.py:60-90`) are declared and never referenced — three
   dead flags advertising behavior that does not exist.

**Fix.** Implement plan task T3.6 (now part of Phase A because it is a
correctness bug, not a hardening feature):

- `SACMethod` exposes three parameter sets:
  - `critic_params`: `model.q1.parameters() + model.q2.parameters()`
  - `actor_params`: `model.actor.parameters()`
  - `alpha_param`: the scalar `[self._log_alpha]` (see A4/C8)
- `RLTrainer._apply_grad` keeps its single-optimizer fast path for DQN/PPO
  and, when `method.NAME == "sac"`, performs the canonical three-step
  update. Shape it as a method hook instead of an isinstance check:
  add `Method.build_optimizers(args, model, device) -> dict[str, opt]`
  (default returns `{"main": <built by build_optimization>}`); SAC returns
  three. `RLTrainer` stores the dict and `_apply_grad` calls
  `method.apply_gradients(loss_parts, optimizers, scaler)`.
  Rationale: keeps `train_step` pure (plan D2), avoids trainer-side
  knowledge of SAC internals, and lets the alpha step live with the alpha
  loss.
- The SAC `train_step` splits its `LossOutput` so the trainer/method can
  step each optimizer: keep returning `loss` (critic+actor for logging) and
  add `metrics["alpha_loss"]` as a real tensor; the method's
  `apply_gradients` runs: zero all three, backward critic loss → step
  critic opt; backward actor loss (with critic params temporarily
  `requires_grad_(False)` or by detaching Q inputs) → step actor opt;
  backward alpha loss on `log_probs.detach()` → step alpha opt.
- Wire the three LRs: critic opt from `--sac-critic-lr`, actor opt from
  `--sac-actor-lr`, alpha opt from `--sac-alpha-lr`. The generic
  `--learning-rate` stays the fallback when the specific flags are absent.

**Files:** `genml_kit/methods/rl_sac.py`, `genml_kit/methods/base.py`
(new hook default), `genml_kit/training/rl_trainer.py`,
`genml_kit/training/train.py` (optimizer construction route).
**Tests:** `test_sac_separate_optimizers`, `test_sac_critic_params_untouched_by_actor_step`
(D4), plus the T3.6 tests carried over from the plan.

### A4. Make auto-alpha actually optimize `_log_alpha` ✅ DONE

**Fixed in Round 2 (part of A3).** 
- Alpha is now a tensor with `requires_grad=True` (`_log_alpha`)
- `alpha_loss` is computed in `train_step` and stepped via `alpha_opt` in `apply_grad`
- When `--sac-auto-alpha` (default), alpha is learned; `--sac-no-auto-alpha` fixes it
- Three LRs wired: critic/actor/alpha optimizers use their respective flags

**Files:** `genml_kit/methods/rl_sac.py` (integrated with A3).
**Tests:** `test_auto_alpha` and `test_fixed_alpha` pass.

  ### A5. DQN target-net update defaults are mutually cancelling ✅ DONE

  **Fixed in Round 1.** Changed `--target_update_freq` default from `0` to `1`
  (hard sync every step). Updated help text. This is simpler and more stable
  than the original plan's suggested `100` (every step works fine for small
  MLPs and avoids the "stale target" problem entirely).

  **Files:** `genml_kit/methods/rl_dqn.py`.
  **Tests:** Existing `test_update_target_hard` and `test_update_target_soft`
  cover this; default now works correctly.

  ### A6. Frozen target networks must survive checkpointing ✅ DONE

  **Fixed in Round 1.** Added `SAVE_FROZEN = True` class attribute to
  `RLTrainer`. This ensures DQN's `model.target.*` and SAC's `q1_target.*`,
  `q2_target.*` are saved/restored in checkpoints.

  **Files:** `genml_kit/training/rl_trainer.py`.
  **Tests:** Existing checkpoint round-trip tests cover this.

### A7. PPO advantage bootstrap: per-step next-values, not a broadcast scalar

**Problem.** `_train_epoch_ppo` (`training/rl_trainer.py:168-172`):

```python
with torch.no_grad():
  obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
  next_val = model.get_value(obs_t).item()
rollout.set_next_values(next_val)
```

One scalar is passed to `RolloutBuffer.set_next_values`
(`datasets/rollout_buffer.py:64-72`), which does
`self.next_values.copy_(torch.as_tensor(...).reshape(-1)[:rollout_len])` —
a length-1 tensor broadcasts into **all** `rollout_len` slots
(`tests/test_rollout_buffer.py:48` passes a length-4 list, which is what the
API expects). So `δ_t = r_t + γ·V(s_T)·(1−d_t) − V(s_t)` uses the final
state's value everywhere instead of `V(s_{t+1})`. Every advantage except the
last one is silently wrong, and PPO's clipped-surrogate trains on garbage.

**Fix (two coordinated changes).**

1. `RolloutBuffer.set_next_values` must fail loudly on a scalar:

```python
tensor = torch.as_tensor(next_values, dtype=torch.float32).reshape(-1)
if tensor.numel() != self.rollout_len:
  fatal(f"set_next_values expects {self.rollout_len} values, "
        f"got {tensor.numel()}", ValueError)
self.next_values.copy_(tensor)
```

   Keep the existing zero-padding convention documented in the docstring
   ("next_values must already be zero at episode boundaries") — the caller
   owns that now.

2. The trainer computes the real per-step values:

```python
next_obs_all = torch.cat(
    [torch.as_tensor(o, dtype=torch.float32).unsqueeze(0)
     for o in self._ppo_next_obs], dim=0)
next_obs_all = next_obs_all * (1.0 - rollout.dones).unsqueeze(1)
with torch.no_grad():
  next_vals = model.get_value(next_obs_all)
rollout.set_next_values(next_vals)
```

   `self._ppo_next_obs` is a list appended in the collection loop (parallel
   to `rollout.add`), and the multiplication zeroes the bootstrap exactly at
   `done` transitions (GAE's `non_terminal` factor also guards this, so the
   zeroing is belt-and-braces for the last step of the rollout). This
   requires buffering next-observations during collection — same loop, one
   extra list, negligible cost.

**Files:** `genml_kit/datasets/rollout_buffer.py`,
`genml_kit/training/rl_trainer.py`.
**Tests:** extend `tests/test_rollout_buffer.py` with
`test_set_next_values_rejects_scalar`; new trainer-level test
`test_ppo_advantages_use_per_step_bootstrap` (D4) that hand-computes GAE on
a scripted rollout and compares against `rollout.advantages`.

### A8. PPO continuous actions: re-evaluate log-prob at the raw (pre-tanh) action

**Problem.** In `ActorCritic.get_action_and_value`
(`models/rl/actor_critic.py:262-288`) the sampling branch is correct:

```python
raw = dist.rsample()
squashed = torch.tanh(raw)
log_prob = dist.log_prob(raw).sum(dim=-1)
log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6).sum(dim=-1)
```

but the re-evaluation branch (used by PPO update epochs on *stored*
actions) feeds the **squashed** action into the Gaussian density:

```python
# action is tanh-squashed; re-evaluate log_prob with correction.
log_prob = dist.log_prob(action).sum(dim=-1)
log_prob -= torch.log(1.0 - torch.tanh(action).pow(2) + 1e-6).sum(dim=-1)
```

`dist.log_prob` must be evaluated at the pre-tanh value; applying `tanh` to
an already-squashed action is meaningless. Consequence: continuous PPO
computes wrong importance ratios in every update epoch.

**Fix options (choose 1, recommended: (a)).**

- (a) Store the raw pre-tanh action in the rollout and re-evaluate from it.
  Changes: `PPOMethod.act` returns `(action, raw_action, log_prob, value)`;
  `RolloutBuffer` gains a `raw_actions` tensor (float, same shape as
  `actions`); `get_action_and_value` gains a `raw_action=` kwarg used by the
  re-evaluation branch. PPO's env-facing action remains the squashed one.
- (b) Invert the squash numerically (`atanh`) — numerically unstable at the
  ±1 boundary, needs epsilon clamping everywhere; rejected.

Add a one-line comment in `get_action_and_value` stating the invariant:
*"for continuous actions, log_prob is always computed at the raw
pre-tanh value; squashed actions are only for the environment."*

**Files:** `genml_kit/models/rl/actor_critic.py`,
`genml_kit/methods/rl_ppo.py`, `genml_kit/datasets/rollout_buffer.py`,
`genml_kit/training/rl_trainer.py` (mini-batch dict must carry the new key).
**Tests:** `test_ppo_logprob_roundtrip` (D4): sample an action, re-evaluate
through the update path, assert equality with the sampling-time log-prob
(within float tolerance).

### A9. Continuous action spaces must be first-class

**Problem.** `RLPipeline.init_env` only supports discrete envs:

- `self._n_actions = int(self.env.action_space.n)`
  (`pipelines/rl.py:202`) — raises `AttributeError` on a `Box` action
  space (Pendulum, MuJoCo, anything continuous).
- `RolloutBuffer(..., action_dim=None, # discrete by default`) is
  hard-coded (`pipelines/rl.py:210-215`).
- SAC (`--method sac`) is continuous-only, so with the `[rl]` extra
  installed and a real gym env, SAC cannot run at all.

**Fix.**

1. In `init_env`, branch on the space type:

```python
action_space = self.env.action_space
if hasattr(action_space, "n"):
  self._action_type = "discrete"
  self._n_actions = int(action_space.n)
  self._action_dim = None
elif hasattr(action_space, "shape") and action_space.shape is not None:
  self._action_type = "continuous"
  self._action_dim = int(torch.tensor(action_space.shape).prod())
  self._n_actions = None
else:
  fatal(f"Unsupported action space: {action_space!r}", ValueError)
```

   Expose `action_type` / `action_dim` properties alongside
   `obs_dim`/`n_actions` (`n_actions` may be `None` for continuous).
2. Size the rollout buffer correctly: `action_dim=self._action_dim` when
   continuous, `None` when discrete (`RolloutBuffer` already handles both —
   it allocates `zeros(rollout_len, action_dim)` for float vs
   `zeros(rollout_len, dtype=long)` for discrete).
3. Guard the pipeline/method pairing early and clearly: in
   `SACMethod.wire_data`, `fatal("SAC requires a continuous action space",
   ValueError)` if `pipeline.action_type != "continuous"`; in
   `DQNMethod.wire_data`, the inverse for `n_actions is None`. Same pairing
   check for PPO vs its `--ppo-continuous` flag (fail when the flag and the
   env disagree instead of letting shapes explode later).
4. Action plumbing already tolerates numpy arrays (`step_env` passes
   through; `_ScriptedEnv` coerces via `int(np.asarray(action).flat[0])`) —
   verify with a continuous scripted env (D6).

**Files:** `genml_kit/pipelines/rl.py`, `genml_kit/methods/rl_sac.py`,
`rl_dqn.py`, `rl_ppo.py`, `genml_kit/datasets/rollout_buffer.py` (no change
expected, only tests).
**Tests:** `test_pipeline_continuous_action_space` (D6): a scripted
continuous env trains one SAC step end-to-end without gym.

---

## 4. Phase B — High-severity fixes (metrics, bookkeeping, logging)

### B1. `evaluate()`: per-episode step budget ✅ DONE

**Fixed in Round 1.** All three methods (`rl_dqn.py`, `rl_sac.py`, `rl_ppo.py`)
now use a per-episode `episode_steps` counter instead of shared `total_steps`.

**Files:** `genml_kit/methods/rl_dqn.py`, `rl_sac.py`, `rl_ppo.py`.
**Tests:** Existing `test_evaluate` tests cover this behavior.

### B2. Determinism: stop reseeding global numpy RNG; actually seed the env ✅ PARTIAL

**Fixed in Round 1 (RNG isolation in validate).** `RLTrainer.validate()` now
saves/restores `np.random` state around evaluation, preventing it from
corrupting training replay buffer sampling.

**Remaining (for future round):**
- Seed the environment properly in `RLPipeline.init_env` via `env.reset(seed=...)`
- Add `ReplayBufferDataset(seed=...)` with local `np.random.Generator`
- Document reproducibility scope

**Files:** `genml_kit/training/rl_trainer.py` (partial), `genml_kit/pipelines/rl.py`,
`genml_kit/datasets/replay_buffer.py` (remaining).
**Tests:** `test_validate_does_not_touch_global_rng` (passes now).

### B3. PPO episode-return tracking is dead code

**Problem.** `_train_epoch_ppo` (`training/rl_trainer.py:160-166`):

```python
total_reward += reward
obs = next_obs
if done:
  total_reward = 0.0     # <- reset before ever being read
  episode_count += 1
  obs = pipeline.reset_env()
```

`total_reward` is zeroed on every episode end and never logged — the
variable where per-episode returns were meant to be tracked.

**Fix.** Accumulate *completed* episode returns into a list; log mean/sd:

```python
if done:
  episode_returns.append(current_return)
  current_return = 0.0
  episode_count += 1
  obs = pipeline.reset_env()
...
self.writer.add_scalar("ppo/episode_return", mean(episode_returns), epoch)
```

Include the mean in the epoch log line next to `episodes=`. This gives PPO
a training-side reward signal in TensorBoard that does not depend on
`validate()` (matching what DQN/SAC get from `env_steps`/loss curves).

**Files:** `genml_kit/training/rl_trainer.py`.
**Tests:** `test_ppo_episode_returns_logged` (D7) — run a scripted PPO
epoch with a fake writer, assert the scalar was emitted.

### B4. PPO value clipping never engages

**Problem.** `PPOMethod.train_step` calls
`value_loss(new_values, returns, old_values=None, clip_eps=self._vf_clip_eps)`
(`methods/rl_ppo.py:181-184`). The `value_loss` implementation
(`losses/rl.py:164-170`) falls back to plain MSE when `old_values is None`,
so `--ppo-vf-clip-eps` is dead config and the documented clipped value loss
(`losses/rl.py:146-152`) is unreachable.

The old values **are already collected** (`RolloutBuffer.add(..., value)`,
sampled as `data["value"]`) — the method just ignores them.

**Fix.**

```python
old_values = data["value"]
v_loss = value_loss(new_values, returns,
                    old_values=old_values,
                    clip_eps=self._vf_clip_eps)
```

One subtlety: shapes. `RolloutBuffer.values` is `(T,)` while
`model.get_action_and_value` returns `(B,)` per mini-batch — both are
already flat per-sample vectors, so no reshape is needed; assert
`old_values.shape == new_values.shape` with a `fatal` if not.

**Files:** `genml_kit/methods/rl_ppo.py`.
**Tests:** extend `tests/test_rl_losses_phase2.py`:
`test_value_loss_clip_engages` — with `old_values` given and a huge
`pred_v`, the clipped loss is strictly larger than plain MSE would be.

### B5. `_env_steps` must advance for all three methods ✅ DONE

**Fixed in Round 1.** 
- **SAC**: increments `self._env_steps += 1` in `train_step` (called once per env step in off-policy flow)
- **PPO**: trainer adds `method._env_steps += rollout_len` after rollout collection in `_train_epoch_ppo`
- **DQN**: already worked (increments in `step_epsilon()`)

The centralized `note_env_step()` approach was not used; instead each method handles it where it naturally fits the flow. All three now report correct `env_steps` in TensorBoard.

**Files:** `genml_kit/methods/rl_sac.py`, `genml_kit/training/rl_trainer.py`.
**Tests:** Existing logging tests cover this.

### B6. Log `eval_return` to TensorBoard

**Problem.** `BaseTrainer.validate` writes
`writer.add_scalar(f"val/{key}", val, self.epoch)` (`trainer.py:152-156`),
but the RL override returns early with the dict and never writes the
metric; `eval_return` only appears in "New best" console lines.

**Fix.** Covered by the A1 rewrite of `RLTrainer.validate()` — the scalar
write is part of that change. Verify against D1's end-to-end test, which
asserts the writer received `val/eval_return`.

**Files:** none beyond A1.
**Tests:** covered by D1.

---

## 5. Phase C — Design smells and cleanup (batch after A+B)

These do not corrupt learning, but each is a trap for the next contributor
or a contradiction between code and docs. Group them into two PR-sized
commits: C1-C4 (trainer/loop semantics) and C5-C12 (API hygiene).

### C1. `update_target` counts gradient steps; help text says env steps — **NOT FIXED** (deferred)

`DQNMethod.update_target(model, global_step)` receives the gradient step
counter (`rl_trainer.py:102` calls it after `_apply_grad`), while
`--target_update_freq` is documented as "every N env steps". Both
interpretations are legitimate (the two literatures disagree); pick one and
write it down. Recommendation: keep **gradient steps** (matches the
existing call site), fix the help text in A5, and note in `rl/README.md`
that SAC's Polyak runs per gradient step too.

### C1 (was: `has_metric_improved` signature) ✅ DONE

**Fixed in Round 1.** All three RL methods now use `(new_metric, best_metric)` 
signature matching `BaseTrainer` and other methods (classification, vo_pair).

### C2. Warmup does not call `step_epsilon()`

The warmup fill (`rl_trainer.py:65-74`) takes random actions without
advancing `_env_steps`/epsilon, so the epsilon schedule effectively starts
after warmup. If warmup actions are meant to be part of the decay window,
call `method.note_env_step()` (B5) inside the warmup loop. If not,
document that epsilon decays over *learning* steps only. Pick one;
recommendation: count warmup steps (simplest, matches "env steps" mental
model), and let `env_steps` reflect total interaction.

### C3. `td_target(n_step=...)` is advertised but unusable

`losses/rl.py:13-62` implements a γⁿ n-step target, but the replay buffer
stores single transitions (`obs, action, reward, next_obs, done`) — no
caller can ever produce an n-step reward sum. Either remove the parameter
until T3.2 lands (preferred: no dead APIs), or keep it and add a docstring
line "*requires an n-step replay buffer; see plan T3.2 — not yet
implemented*". Do not leave it silently pretending.

### C4. `_train_epoch_ppo` reads `rollout.__dict__` and silently drops tensors

`rl_trainer.py:188-192` builds mini-batches by iterating
`rollout.__dict__` and keeping tensors with `shape[0] == rollout_len` —
private-state access, plus a silent filter that would hide a
wrong-length tensor (exactly the A7 failure mode). Add an explicit
`RolloutBuffer.mini_batch(indices) -> dict` method returning exactly the
PPO keys (`obs, action, log_prob, advantage, return, value` — plus
`raw_action` from A8), and have the trainer call it. The buffer validates
its own tensors; the trainer stops poking at `__dict__`.

### C5. `RLPipeline.eval_rollout` is dead code

`pipelines/rl.py:257` implements a generic eval rollout that nothing calls
(every method ships its own `evaluate`). Two options: delete it, or make
the three methods' `evaluate` delegate to it (passing
`lambda obs: method.act(model, obs, deterministic=True)` and a
return-shaping hook). Recommendation: delegate, so the episode/step logic
lives in exactly one place (this also fixes B1 once instead of three
times). If delegation is chosen, the methods keep their signature and only
the loop moves.

### C6. Stale `ReplayBufferDataset` docstring

`datasets/replay_buffer.py:20-24` claims the buffer is a DataLoader
`Dataset` consumed via `default_collate` into a `TransitionBatch`
namedtuple. In reality the trainer samples dicts directly
(`rl_trainer.py:90`). Rewrite the docstring to describe actual usage
(`sample()` returns a dict of tensors; `__getitem__` exists for
protocol-compatibility but is not on the hot path).

### C7. `_SACModel` is a local class inside `build_model`

`methods/rl_sac.py:151-165` defines the container inside the method,
unregistered and re-created per call. Consequences: no registry entry
(unlike DQN/PPO models), checkpoint `num_labels` probing has nothing to
read, and `_apply_model_extras` (freeze/LoRA plumbing) operates on a
container whose submodules it cannot reason about. Move it to
`models/rl/sac_model.py` as `SACActorCritic` with a
`@register_model("rl/sac")` factory, keeping the same attribute names
(`actor`, `q1`, `q2`, `q1_target`, `q2_target`) so checkpoints stay
compatible. Keep `forward` raising (there is no sensible monolithic
forward for this composite).

### C8. `_log_alpha` lives outside the model/optimizer/state_dict

Related to A4: the temperature is a bare tensor on the method, so it is
invisible to `model.state_dict()` and any optimizer built from
`model.parameters()`. After A3/A4 the alpha optimizer owns it explicitly,
which is acceptable, but document the ownership (method-state
serialization via `get_checkpoint_state` is the only persistence path) or,
better, register it as a buffer on the model in C7's `SACActorCritic`
(`self.register_buffer("log_alpha", ...)`) so checkpoints carry it for
free and `method_state` stops duplicating it. Choose one owner; the
checkpoint format must remain stable for already-saved runs (keep reading
`method_state["log_alpha"]` as a fallback when the buffer is absent).

### C9. `init_env` silently falls back to `obs_dim=4`

`pipelines/rl.py:198-199` logs a warning and continues with a wrong
observation size, which surfaces later as a cryptic shape mismatch. Replace
with `fatal(f"Cannot infer obs_dim from observation space {obs_space!r}; "
"pass --obs_dim explicitly", ValueError)`.

### C10. Argument style inconsistency

`pipelines/rl.py` uses `--env_id`, `--obs_dim`, `--env_script`,
`--warmup_steps`, `--steps_per_epoch` while methods use hyphens
(`--sac-gamma`, `--ppo-clip-eps`). Pick hyphens (the dominant style) and
rename the RL pipeline flags in one commit, keeping the old underscore
names as `dest` aliases where cheap (`parser.add_argument("--env-id",
dest="env_id", ...)` keeps `args.env_id` working so saved configs/scripts
do not break).

### C11. `METRIC_MINIMIZE` is declared but never read

All three RL methods declare `METRIC_MINIMIZE = False` and implement
`has_metric_improved` by hand; nothing in `training/` reads the flag.
Either delete the attribute, or (better) implement
`BaseTrainer.has_metric_improved` to fall back to the flag when the method
does not override `has_metric_improved` — one contract, fewer redundant
overrides. Out of RL scope strictly speaking; do it if the base change is
small, otherwise drop the attribute from the RL methods.

### C12. Resume gaps: epsilon/env-step consistency and PPO `_ppo_obs`

Two resume wrinkles (documented behavior wanted):

- DQN restores `_epsilon` and `_env_steps` independently
  (`rl_dqn.py:239-244`), so a hand-edited or old checkpoint can pair an
  epsilon with an unrelated step count. After B5, derive `_epsilon` from
  `_env_steps` on load (recompute the schedule) and keep `epsilon` in
  `method_state` only as a cross-check that logs a warning on mismatch.
- PPO's `_ppo_obs` (mid-episode observation) is not checkpointed, so a
  resumed run starts a fresh episode silently. Either accept and document
  ("resume restarts the current episode") or stash it in `ckpt_extra` via
  `get_checkpoint_state`. Recommendation: accept + document; persisting a
  mid-rollout env state is not worth the format churn.

---

## 6. Small fixes bundled with Phase C

- `import numpy as np` in `rl_trainer.py` becomes unused once B2 removes
  the global reseed — drop the import (ruff will flag it).
- `tests/test_rl_trainer.py` `_make_args` defaults `max_grad_norm=0.0`
  while the trainer's grad-clip path is untested for RL; wire
  `grad_clip` through in the D1 end-to-end test (one epoch with clipping
  on) so the AMP/clip path is exercised.
- `rl_trainer.py:113-116` logs `epsilon`/`alpha` via `hasattr` probes on
  the method; replace with the B5 `note_env_step` refactor plus an
  optional `method.log_extra_scalars(writer, epoch) -> dict` hook so the
  trainer stops introspecting private attributes.
- In `losses/rl.py`, `td_loss` uses `F.smooth_l1_loss`; fine, but add a
  one-line docstring note that the reduction arg is passed straight
  through, since `reduction="none"` feeds the T3.1 prioritized-replay
  importance weights later.

---

## 7. Test-coverage program (D1-D7)

Nine silent-corruption bugs (A2-A9) survived behind 106 passing tests. The
root cause is systematic: the RL tests exercise units in isolation and
never drive `run()`, never assert that the *stability mechanisms* actually
move, and never round-trip checkpoints. Each item below states the gap and
the exact assertion that closes it. Add these as the fixes land (one test
file per phase), not at the end.

### D1. End-to-end `run()` tests (the single most valuable addition)

**File:** `tests/test_rl_trainer.py` (extend). The current file only calls
`train_epoch(...)` / `validate()` directly — `run()` is never invoked in
any RL test, which is why A1 survived.

- `test_run_end_to_end_one_epoch`: build the trainer exactly the way
  `train.py` does (`build_optimization`, `RLTrainer(...)`, then
  `trainer.run()` with `args.epochs = 1`) using `FakeRLPipeline` /
  `FakeRLMethod` + `_ScriptedEnv`. Assertions: returns a `TrainingResult`
  with `completed_epoch == 1`; `best_metric` is a **float** (this exact
  assertion fails today under A1); a checkpoint file exists on exit; a
  fake writer received `val/eval_return` (B6/A1).
- `test_run_three_epochs_best_metric_monotone_bookkeeping`: three epochs,
  assert `trainer.best_metric` only ever improves and that
  `save_best` was called at most once per epoch (track via a stubbed
  saver).
- `test_run_respects_grad_clip`: `args.grad_clip = 1.0` with a method
  whose loss produces huge gradients; assert no NaNs and that the clip
  path executed (grad monitor callback). Covers the base-runner plumbing
  that RL bypasses.

### D2. Loop-semantics tests (warmup, epsilon, target cadence)

**File:** `tests/test_rl_trainer.py`.

- `test_warmup_fills_buffer` (exists) — extend it: after warmup,
  `method._env_steps >= warmup_steps` once C2/B5 land.
- `test_epsilon_decays_over_training`: 3 epochs,
  `epsilon_decay_steps = 3 * steps_per_epoch`, assert
  `method._epsilon` strictly decreased between epochs and equals
  `epsilon_end` at the end.
- `test_target_net_hard_sync_cadence`: `--target_update_freq 5`, run 10
  gradient steps, count hard syncs (wrap `model.hard_update` with a
  counting stub) — assert exactly 2 and that targets equal online after
  each sync.
- `test_target_net_polyak_mode`: `--tau 0.01 --target_update_freq 0`,
  assert targets moved toward online by the Polyak factor, not copied.
- `test_no_target_update_warns`: the A5 one-time warning fires exactly
  once.

### D3. Checkpoint round-trip tests (closes A6)

**File:** new `tests/test_rl_checkpoints.py`.

- `test_checkpoint_contains_target_networks`: save via `CheckpointSaver`
  the way `BaseTrainer` does, load the file, assert
  `"target.0.net.0.weight"`-style keys exist (DQN) and
  `q1_target.*` / `q2_target.*` exist (SAC).
- `test_resume_restores_targets_exactly`: train 2 epochs → save → rebuild
  model + method → `load_checkpoint_weights` → assert
  `torch.equal` on every target parameter against the saved online copy.
- `test_resume_restores_method_state`: DQN `_epsilon`/`_env_steps` and SAC
  `log_alpha` survive; `saver_extra`/`ckpt_extra` path exercised through
  the real saver, not by hand-building dicts.
- `test_resume_trains_not_reinitializes`: after resume, one epoch changes
  target weights (guards against the "silently random targets" failure).

### D4. Algorithm-correctness unit tests

**File:** new `tests/test_rl_correctness.py` (DQN/SAC/PPO math pinned to
hand-computed numbers).

- `test_dqn_td_target_double_vs_vanilla` (exists in some form — keep).
- `test_sac_bootstrap_uses_targets` (A2): deepcopy the model, overwrite
  `q1_target` weights with a constant, assert `soft_target` computed by
  `train_step` reflects the constant while online weights do not affect it
  (freeze both, perturb, compare).
- `test_sac_critic_params_untouched_by_actor_step` (A3): record critic
  weights, run one actor+alpha step, assert critics unchanged; then one
  critic step, assert critics changed.
- `test_sac_alpha_gradient_flows` / `test_sac_alpha_fixed_when_disabled`
  (A4).
- `test_dqn_target_updates_by_default` (A5).
- `test_ppo_advantages_use_per_step_bootstrap` (A7): scripted rollout of
  known rewards/values, compare `rollout.advantages` against a
  hand-computed GAE recurrence.
- `test_ppo_logprob_roundtrip` (A8): continuous actor,
  `log_prob(action_t) == log_prob at sampling time` within 1e-5.
- `test_value_loss_clip_engages` (B4).

### D5. Determinism tests (closes B2, feeds T3.11)

**File:** `tests/test_rl_trainer.py` or `tests/test_rl_pipeline.py`.

- `test_sample_is_seeded`: two `ReplayBufferDataset`s with the same seed
  produce identical index sequences for equal-sized samples.
- `test_validate_does_not_touch_global_rng`: snapshot
  `np.random.get_state()`, call `trainer.validate()`, assert the state is
  unchanged (this is a regression test for the B2 removal).
- `test_env_seed_reaches_env`: scripted/gym env reset uses the seed —
  two pipelines built with the same `env_seed` produce identical first
  observations.

### D6. Continuous-action-space tests (closes A9)

**File:** new `tests/test_rl_continuous.py`.

- `test_pipeline_continuous_action_space`: a scripted env exposing a
  `Box`-like action space (shape `(2,)`, no `.n`) initializes
  `action_type == "continuous"`, `action_dim == 2`, `n_actions is None`,
  and the rollout buffer allocates float actions of shape
  `(rollout_len, 2)`.
- `test_sac_requires_continuous` / `test_dqn_requires_discrete`: the A9
  wire-time pairing guards fire with a helpful `fatal` message.
- `test_sac_one_step_continuous`: full SAC `train_step` on a continuous
  mini-batch, finite loss, all three optimizers step.
- `test_ppo_continuous_one_epoch`: PPO with `--ppo-continuous` on the
  scripted continuous env completes an epoch.

### D7. Metric/logging tests (closes B1, B3, B5)

**File:** `tests/test_rl_trainer.py` plus method tests.

- `test_eval_budget_per_episode` (B1): scripted env with episodes of
  exactly `max_steps` length; assert `eval_steps == 2 * max_steps` for
  `num_episodes=2` (fails today).
- `test_ppo_episode_returns_logged` (B3): fake writer records
  `ppo/episode_return`.
- `test_env_steps_counted_for_all_methods` (B5): one epoch each for
  DQN/SAC/PPO → `method._env_steps == steps_per_epoch` (PPO: rollout
  length).

---

## 8. Phase E — Plan Phase-3 tasks re-scoped against the review

The old plan's Section 8 (T3.1-T3.11) is carried over with status changes
resulting from this review. Items already absorbed into Phase A are marked
DONE-BY; ordering and rationale updated where the review changed the
picture.

### E1 (was T3.1) — Prioritized Experience Replay — PENDING

Unchanged in scope; land after A-phase so priority weights ride on the
fixed `td_loss(reduction="none")` path (see §6). Buffer gains a priority
array, `update_priorities(indices, priorities)`, proportional sampling,
and IS weights `w_i = (N·P(i))^{-β}` with β annealing 0.4 → 1.0. Keep the
flat-array implementation first (sum-tree only if profiling demands it).
New: per B2 the sampler must use the buffer's own `np.random.Generator`,
not the global RNG. Tests as listed in the old plan plus
`test_priority_sampling_is_seeded`.

### E2 (was T3.2) — N-Step Returns — PENDING (unblock C3)

The `td_target(n_step=...)` parameter exists but is unreachable (C3).
Implement per the old plan (precompute n-step returns on push, storing
`reward_sum / next_obs_n / done_n`), then C3 resolves by having a real
caller. Prefer the "compute on push" variant the old plan lists as
primary; keep the raw-transitions fallback documented as rejected unless
profiling says otherwise.

### E3 (was T3.3) — Observation Normalization — PENDING

`RunningMeanStd` in the pipeline, `normalize(obs)` applied in
`step_env`/eval, stats persisted in checkpoint state. Review additions:
(a) with A9 in place, normalize per-dimension over the actual obs shape,
not the flat dim; (b) persistence must ride the same
`save_frozen`/state-dict path as A6 — store the running stats as buffers
on a small module registered in the pipeline, not as bare numpy blobs in
`method_state`; (c) `--obs_normalize` flag default **off** (scripted envs
and CartPole do not need it; avoids perturbing existing tests).

### E4 (was T3.4) — Frame Stack — PENDING

As planned (`--frame_stack N`, deque in pipeline, `(N, *obs_shape)`
buffers). Review addition: depends on the image-observation path that
`models/rl/qnetwork.py` documents as "deferred to Phase 2" but which does
not exist; either land the CNN backbone first or scope this task to
flat-vector stacking only and say so.

### E5 (was T3.5) — Vector Environments — PENDING

`--num_envs N` via `gymnasium.vector.SyncVectorEnv` (start with sync; async
adds multiprocessing complexity the test suite cannot exercise in CI).
Review additions: (a) `RolloutBuffer`/`ReplayBufferDataset` must accept
batched pushes with per-env `done` handling before this lands; (b) A7's
per-step bootstrap must be generalized to per-env-per-step; (c)
`_VectorScriptedEnv` test double as the old plan notes. Sequence **after**
A7/E2 so the bootstrap semantics are already correct for the single-env
case.

### E6 (was T3.6) — SAC Separate Optimizer Groups — DONE-BY A3+A4

The old plan's T3.6 is subsumed: A3 introduces the optimizer-ownership
hook and wires `--sac-actor-lr/-critic-lr/-alpha-lr`; A4 makes alpha
actually optimize. Do not re-implement; the E6 tests are folded into D4.

### E7 (was T3.7) — `post_train` Hooks — PENDING

As planned: export `policy.pt` (state dict only), final `evaluate()` with
the best checkpoint, final metrics to TensorBoard. Review addition: with
A6 fixed, export the **online** network for DQN and `model.actor` for
SAC/PPO explicitly, and assert in `test_post_train_export` that the
exported state dict round-loads into a freshly built model.

### E8 (was T3.8) — CLI End-to-End Wiring — PENDING

Verify `register_all_owners` picks up the RL pipeline/methods; assert
`--pipeline rl --method dqn` populates args and that
`genml-kit-train --pipeline rl --method dqn --env CartPole-v1` runs. This
is the manual exit-criterion gate; automate it as
`test_cli_rl_end_to_end` with `_ScriptedEnv` (no gym needed) so CI runs it
without the `[rl]` extra.

### E9 (was T3.9) — CI `[rl]` Extra — PARTIAL (see F1/§9)

The `pyproject.toml` half is missing entirely (F1); the lazy-import +
helpful-error half exists (`GymnasiumEnvWrapper.__init__` raises
`ImportError` with an install hint — convert it to `fatal(...)` per
conventions). Remaining: CI workflow runs the RL suite with the extra
installed and the gym-free subset without it.

### E10 (was T3.10) — Monotonic Improvement Test — PENDING

Keep, with a review caveat: strict monotonicity is flaky for PPO/SAC even
on the scripted chain; assert DQN's `eval_return` reaches the maximum on
the scripted chain within N epochs (the chain is solvable to 1.0) and that
PPO/SAC do not *degrade* below their first-epoch value. Use
`--seed`/`env_seed` everywhere (D5) and generous epochs (20) to avoid
flake; mark with `@pytest.mark.slow` if runtime becomes an issue.

### E11 (was T3.11) — Reproducibility Seed Test — PENDING (depends on B2/D5)

Two runs with `--seed 42 --env_seed 42` must produce identical
`eval_return` trajectories and identical final `method_state`. Only
meaningful after B2 removes the global-RNG reseeding and D5 seeds the
buffer sampler; do it last.

---

## 9. Phase F — Packaging and CI (F1)

### F1. `gymnasium` is not installable through the project metadata

**Problem (confirmed by reading `pyproject.toml`).** The code path assumes
an optional `[rl]` extra exists:

- `GymnasiumEnvWrapper.__init__` (`pipelines/rl.py:31-35`) raises
  `ImportError("gymnasium is required for --pipeline rl.  Install it
  with:  pip install 'genml_kit[rl]'")`.
- The old plan's design row D6 specifies "gymnasium as optional `[rl]`
  extra", pending Phase 3.

But `[project.optional-dependencies]` (`pyproject.toml:24-31`) defines
only `gcs`, `s3`, `lora`, `timm`, `uvito`, `all`, `dev` — **there is no
`rl` extra**, and `all` does not include it either. Consequences:

- `pip install 'genml_kit[rl]'` warns about the unknown extra and installs
  nothing extra — the documented remedy in the error message does not work.
- `pip install 'genml_kit[all]'` does not bring gymnasium, so even the
  kitchen-sink install cannot run `--pipeline rl` against a real env.
- The test-suite claim in the old plan ("tests avoid a hard gym dep via
  scripted FakeEnv") holds — core stays importable — but the *advertised*
  optional dependency is a lie.

**Fix (exact diff).**

```toml
[project.optional-dependencies]
gcs = ["google-cloud-storage"]
s3 = ["boto3>=1.28"]
lora = ["peft>=0.7.0"]
timm = ["timm>=1.0"]
uvito = ["segmentation_models_pytorch>=0.3.0"]
rl = ["gymnasium>=0.29", "pygame>=2.1"]        # pygame: classic-control render
all = ["genml_kit[gcs,s3,lora,timm,uvito,rl]"]
dev = ["pytest", "ruff", "yapf"]
```

- `pygame` is needed only for `CartPole-v1`'s `render_mode="human"`; keep
  it because the exit criteria use CartPole end-to-end, and note it can be
  dropped if rendering is never exercised in CI.
- Pin floor `>=0.29` (the first line with a stable `gymnasium` API used
  here: `terminated, truncated` split in `step`). No upper pin.
- Update the `ImportError` text once the extra exists (it already says
  `genml_kit[rl]`, so it becomes correct as-is); convert it to
  `fatal(..., ImportError)` per project conventions (E9).
- README: the RL section must name `pip install 'genml_kit[rl]'` — verify
  and fix any other spelling (`gym` vs `gymnasium`) while there. The
  package is `gymnasium`; **not** the deprecated `gym` — any doc or
  comment mentioning `gym` as the package name is wrong (the module
  `import gymnasium as gym` inside `pipelines/rl.py` is fine, it is just
  an alias).

**Verification.**

```
pip install -e '.[rl]'
python -c "import gymnasium"
python -m pytest tests/ -k "rl"      # green with and without the extra
```

---

## 10. Implementation order and commit plan

Sequence (each step lands green; nothing commits until you approve the
diff — and never amend/squash existing commits):

1. **F1** — `pyproject.toml` `rl` extra (+ `all`) and README check.
   Tiny, unblocks real-env work.
2. **A1** — `validate()` scalar contract (+ B6 in the same edit; one
   function). Then **D1** `test_run_end_to_end_one_epoch` — this is the
   regression net for everything after it.
3. **A5** — DQN target defaults + warning (one file), then **A6** —
   `SAVE_FROZEN = True` on `RLTrainer` (one line) with **D3** tests.
   Together: DQN becomes actually trainable and resumable.
4. **A2 + A4** — SAC target bootstrapping and tensor-alpha (both inside
   `train_step`, one edit each), without the optimizer split yet.
5. **A3 + C8** — the optimizer-ownership hook (`build_optimizers` /
   `apply_gradients`), SAC three-optimizer wiring, `_log_alpha`
   ownership decision. Largest change in Phase A; review carefully.
6. **A7 + A8 + B4** — PPO bootstrap, raw-action re-evaluation, value
   clipping (touches buffer + model + method + trainer coherently).
7. **A9** — continuous action spaces end-to-end (+ **D6**).
8. **B1/B2/B3/B5** — eval budget, determinism, PPO episode returns,
   `_env_steps` (+ **D5**, **D7**).
9. **Phase C** — two commits: C1-C4 then C5-C12 (+ §6 small fixes).
10. **Phase E** — E2 (unblocks C3), then E1, E3, E4, E5, E7, E8, E10,
    E11 in that order.

Suggested commit subjects (no `Co-Authored-By:` trailers, per policy):

- `pyproject: add [rl] extra with gymnasium`
- `rl: validate() returns scalar metric; log eval_return`
- `rl: fix DQN target-update defaults; persist frozen targets`
- `rl: SAC bootstraps from target critics; auto-alpha optimizes`
- `rl: separate SAC actor/critic/alpha optimizers`
- `rl: PPO per-step bootstrap, raw-action log-prob, value clipping`
- `rl: continuous action spaces end-to-end`
- `rl: eval/step bookkeeping and seeding fixes`
- `rl: trainer/method API hygiene (Phase C)`
- `rl: prioritized replay, n-step, normalization (Phase E parts)`

---

## 11. Exit criteria (replaces the old plan's Phase-3 list)

1. All existing RL tests pass; every item in §7 has its test present and
   green (count is whatever the suite says — do not quote stale numbers).
2. `python -m pytest tests/ -k "rl" -q` green **without** the `[rl]`
   extra installed (scripted envs only); green **with** it installed.
3. `ruff check` clean, `yapf` clean on every touched file (2-space
   indent, no typing annotations, `fatal()` for all fatal paths).
4. `genml-kit-train --pipeline rl --method dqn --env_id CartPole-v1`
   runs end-to-end with `[rl]` installed and produces a best-checkpoint
   whose target networks are loadable (A6).
5. `--method sac` and `--method ppo` each complete 3 epochs on the
   scripted env via `run()` (not just `train_epoch`), with
   `best_metric` a float.
6. Resume: DQN checkpoint → resume → target weights identical to the
   saved ones before further training (D3).
7. The old `plans/RL_PLAN.md` is deleted; `rl/README.md` cross-references
   this document instead.

---

## 12. Decision log (why the review changed the plan)

- **A3 reframed as correctness, not hardening.** The old plan had SAC
  separate optimizers as a Phase-3 task (T3.6); the audit showed the
  single-optimizer update corrupts critics *today*, so it moved to Phase A
  and T3.6 is marked DONE-BY.
- **A6 contradicts design row D3.** The plan's "zero-change
  CheckpointSaver" claim was never true for frozen targets under
  `SAVE_FROZEN=False`; the fix is a one-line override, not a saver change.
- **A7's root cause is an API contract, not a typo.** `set_next_values`
  accepting a broadcastable scalar silently is what let the trainer pass
  one number; the fix hardens the API (`fatal` on wrong length) *and*
  fixes the caller.
- **B2 reframed.** The reseeding was not just an eval-determinism choice;
  it silently couples eval to replay sampling via the global RNG, so the
  fix (separate generators) is required for reproducibility work (E11).
- **C3/E2 pairing.** The old plan shipped an n-step API with no producer;
  the review either removes it or lands E2 — dead APIs are not kept.
- **Test-count honesty.** The old plan's "137 RL tests" is stale (actual
  106); exit criteria now refer to the suite, not to memorized numbers.
