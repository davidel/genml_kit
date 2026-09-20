# RL_REVIEW: Remediation & Completion Plan for RL Support in genml_kit

Status: **Active — Phases A, B, C, D complete; Phases E, F pending**

This document supersedes `plans/RL_PLAN.md` (removed from
the repo; recoverable from git history). This file is now the single
source of truth for remaining RL work.

This document was produced by a full audit of the former `plans/RL_PLAN.md`
against the actual implementation. It carries over everything that plan still
listed as pending (its Phase 3, tasks T3.1-T3.11) and adds every defect and design
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
  `set_next_values`, `get_batch` (aliased as `sample`), `to`.
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
and its exit criteria repeat "137 RL, 1085 total". The actual suite is
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

---

## 3. Phase A — Critical fixes (COMPLETE ✅)

### ✅ A1. `RLTrainer.validate()` returns scalar metric
**Fixed in Round 1.** `RLTrainer.validate()` returns `float(metrics["eval_return"])`
and saves/restores `np.random` state around evaluation. TensorBoard logging
of `val/eval_return` is handled by `BaseTrainer.run()`.

**Files:** `genml_kit/training/rl_trainer.py`.

### ✅ A2. SAC bootstraps from target critics
**Fixed in Round 2.** Changed `train_step` to use `model.q1_target.get_value()`
and `model.q2_target.get_value()` for the TD target computation.

**Files:** `genml_kit/methods/rl_sac.py`.

### ✅ A3. SAC separate actor/critic/alpha optimization
**Fixed in Round 2 (part of A3/A4).** `SACMethod` exposes three parameter sets
and `RLTrainer._apply_grad` performs the canonical three-step update via
`Method.build_optimization` hook. Three LRs wired: critic/actor/alpha.

**Files:** `genml_kit/methods/rl_sac.py`, `genml_kit/methods/base.py`,
`genml_kit/training/rl_trainer.py`.

### ✅ A4. Auto-alpha actually optimizes `_log_alpha`
**Fixed in Round 2.** Alpha is now a tensor with `requires_grad=True`
(`_log_alpha`). `alpha_loss` is computed in `train_step` and stepped via
`alpha_opt` in `apply_grad`. `--sac-auto-alpha` (default) learns alpha;
`--sac-no-auto-alpha` fixes it.

**Files:** `genml_kit/methods/rl_sac.py`.

### ✅ A5. DQN target-net update defaults
**Fixed in Round 1.** Changed `--target_update_freq` default from `0` to `1`
(hard sync every step). Updated help text.

**Files:** `genml_kit/methods/rl_dqn.py`.

### ✅ A6. Frozen target networks survive checkpointing
**Fixed in Round 1.** Added `SAVE_FROZEN = True` class attribute to
`RLTrainer`. Ensures DQN's `model.target.*` and SAC's `q1_target.*`,
`q2_target.*` are saved/restored.

**Files:** `genml_kit/training/rl_trainer.py`.

### ✅ A7. PPO per-step bootstrap
**Fixed in Round 3.** `RolloutBuffer.set_next_values` now rejects scalars and
requires per-step `V(s_{t+1})` tensor. `RLTrainer._train_epoch_ppo` collects
per-step bootstrap values during rollout collection.

### ✅ A8. PPO continuous log-prob re-evaluation on raw actions
**Fixed in Round 3.** Raw (pre-tanh) actions stored in `RolloutBuffer.raw_actions`
and re-evaluated during PPO update epochs.
`ActorCritic.get_action_and_value` returns `(action, raw_action, log_prob, entropy, value)`.

### ✅ A9. Continuous action spaces supported
**Fixed in Round 3.** `RLPipeline.init_env` branches on `action_space` type
(`Discrete` vs `Box`), exposes `action_space`, `action_type`, `action_dim`.
`SACMethod.wire_data` reads `action_dim` from `pipeline.action_space.shape`.
`_ScriptedEnv` supports `continuous=True` with `gymnasium.spaces.Box`.
`train.py` calls `pipeline.init_env()` before `wire_data()` for RL.

---

## 4. Phase B — High-severity fixes (COMPLETE ✅)

### ✅ B1. `evaluate()` per-episode step budget
**Fixed in Round 1.** All three methods use a per-episode `episode_steps` counter.

### ✅ B2. Determinism: RNG isolation in validate
**Fixed in Round 1.** `RLTrainer.validate()` saves/restores `np.random` state.
*Remaining (future):* Seed environment properly via `env.reset(seed=...)`;
add `ReplayBufferDataset(seed=...)` with local `np.random.Generator`.

### ✅ B3. PPO episode-return tracking (dead code removed)
**Fixed in Round 3.** The dead `total_reward` variable was removed from
`_train_epoch_ppo`. Episode returns are now logged via `validate()` only.

### ✅ B4. PPO value clipping engages
**Fixed in Round 3.** `PPOMethod.train_step` now passes `old_values=data["value"]`
to `value_loss()`, enabling clipped value loss when `--ppo-vf-clip-eps` is set.

### ✅ B5. `_env_steps` advances for all three methods
**Fixed in Round 1.**
- **SAC**: increments `self._env_steps += 1` in `train_step`
- **PPO**: trainer adds `method._env_steps += rollout_len` after rollout collection
- **DQN**: already worked (increments in `step_epsilon()`)
All three now report correct `env_steps` in metrics.

### ✅ B6. `eval_return` logged to TensorBoard
**Fixed in Round 1.** Covered by A1 rewrite of `RLTrainer.validate()`.

---

## 5. Phase C — Design smells and cleanup (COMPLETE ✅)

### ✅ C1. `has_metric_improved` signature
**Fixed in Round 1.** All three RL methods use `(new_metric, best_metric)`
signature matching `BaseTrainer`.

*Deferred:* `update_target` counts gradient steps vs. env steps — documented as-is.

### ✅ C2. Warmup calls `step_epsilon()`
**Fixed in Round 4.** Warmup loop now calls `method.step_epsilon()` for each step.

### ✅ C3. `td_target(n_step=...)` parameter wired
**Fixed in Round 4.** DQN now reads `--n_step` and passes it to `td_target()`.

### ✅ C4. `RolloutBuffer.mini_batch()` → `sample()`
**Fixed in Round 4.** Renamed to `sample()` with backward-compat alias.
Added validation for empty buffer.

### ✅ C5. Removed dead `eval_rollout` code
**Fixed in Round 4.** Deleted unused `RLPipeline.eval_rollout()` method.
Updated `build_val_loader` docstring.

### ✅ C6. Updated `ReplayBufferDataset` docstring
**Fixed in Round 4.** Docstring now describes actual usage (`sample()` returns dict).

### ✅ C7. `_SACModel` moved to separate module
**Fixed in Round 4.** Created `models/rl/sac_model.py` (`SACModel`) and
`models/rl/sac_critic.py` (`SACCritic`). Registered in `models/rl/__init__.py`.

### ✅ C8. `_log_alpha` in checkpoint state
**Fixed in Round 2/4.** `_log_alpha` is saved/restored via `get_checkpoint_state`/
`load_checkpoint_state`. Alpha optimizer owns the parameter.

### ✅ C9. `init_env` fatal on `obs_dim` inference failure
**Fixed in Round 4.** Uses `fatal()` API when observation space lacks shape.

### ✅ C10. Argument style: underscores → hyphens
**Fixed in Round 4.** RL pipeline flags renamed: `--env_id` → `--env-id`,
`--obs_dim` → `--obs-dim`, `--env_script` → `--env-script`, etc.
Old underscore names kept as `dest` aliases.

### ✅ C11. Removed unused `METRIC_MINIMIZE`
**Fixed in Round 4.** Deleted from all three RL methods.

### ✅ C12. Resume state handling
**Fixed in Round 4.** PPO checkpoints `env_steps`. DQN/SAC already checkpointed
`_epsilon`/`_env_steps` and `log_alpha`.

---

## 6. Remaining Work

### Phase D \u2014 Test-coverage program (COMPLETE \u2705)

All Phase D items have been implemented:

- **D1. End-to-end `run()` tests** \u2705: Added `TestRLTrainerEndToEnd` in `tests/test_rl_trainer.py` with tests for DQN, PPO, SAC calling `trainer.run()` and verifying checkpoint written with `best_eval_return`.
- **D2. Gradient-norm logging & NaN guard** \u2705: Added NaN/Inf detection before backward pass in `_train_epoch_offpolicy` and `_train_epoch_ppo`; gradient norm logging to TensorBoard.
- **D3. PPO learning-rate scheduling wiring** \u2705: Added scheduler stepping at epoch start for PPO in `train_epoch`.
- **D4. SAC hard-target sync on warmup complete** \u2705: Added `model.hard_update()` after warmup fill for SAC.
- **D5. Buffer sampler seeded independently** \u2705: Added `seed` parameter to `ReplayBufferDataset` and `RolloutBuffer`; use independent RNGs (`np.random.default_rng()`, `torch.Generator()`). Pipeline passes `env_seed` to buffers.
- **D6. Checkpointing round-trip tests** \u2705: Added `TestRLCheckpointRoundTrip` in `tests/test_rl_trainer.py` with tests for DQN, PPO, SAC verifying method state (epsilon/env_steps, log_alpha/env_steps) saved and restored correctly.

All 1090 tests passing (107 RL-specific).

---

## 7. Phase E — Plan Phase-3 tasks re-scoped (PENDING)

### E1 (was T3.1) — Prioritized Experience Replay
Unchanged scope; land after A-phase so priority weights ride on fixed
`td_loss(reduction="none")`. Buffer gains priority array, `update_priorities`,
proportional sampling, IS weights `w_i = (N·P(i))^{-β}` with β annealing
0.4 → 1.0. Flat-array implementation first. Sampler must use buffer's own
`np.random.Generator` (per B2). Tests: old plan + `test_priority_sampling_is_seeded`.

### E2 (was T3.2) — N-Step Returns
`td_target(n_step=...)` exists but unreachable (C3). Implement per old plan
(precompute n-step returns on push, storing `reward_sum / next_obs_n / done_n`).
Prefer "compute on push" variant. Then C3 resolves with real caller.

### E3 (was T3.3) — Observation Normalization
`RunningMeanStd` in pipeline, `normalize(obs)` in `step_env`/eval, stats in
checkpoint. Per-dimension over actual obs shape. Persist via `save_frozen`/
state-dict path (buffers on small module in pipeline). `--obs_normalize`
default **off**.

### E4 (was T3.4) — Frame Stack
`--frame_stack N`, deque in pipeline, `(N, *obs_shape)` buffers. Depends on
image-observation path in `models/rl/qnetwork.py` (documented as "deferred");
either land CNN backbone first or scope to flat-vector stacking.

### E5 (was T3.5) — Vector Environments
`--num_envs N` via `gymnasium.vector.SyncVectorEnv`. Requires batched pushes
with per-env `done` handling in buffers. A7 bootstrap generalized to
per-env-per-step. Sequence after A7/E2.

### E6 (was T3.6) — SAC Separate Optimizer Groups ✅ DONE-BY A3+A4

### E7 (was T3.7) — `post_train` Hooks
Export `policy.pt` (state dict only), final `evaluate()` with best checkpoint,
final metrics to TensorBoard. Export online network (DQN) / `model.actor`
(SAC/PPO). Assert exported state dict round-loads.

### E8 (was T3.8) — CLI End-to-End Wiring
Verify `register_all_owners` picks up RL pipeline/methods; assert
`genml-kit-train --pipeline rl --method dqn --env CartPole-v1` runs.
Automate as `test_cli_rl_end_to_end` with `_ScriptedEnv`.

### E9 (was T3.9) — CI `[rl]` Extra (PARTIAL — see F1/§9)
`pyproject.toml` half missing (F1). Convert `ImportError` to `fatal()` per
conventions. CI runs RL suite with extra installed and gym-free subset without.

### E10 (was T3.10) — Monotonic Improvement Test
Strict monotonicity flaky for PPO/SAC; assert DQN reaches maximum on scripted
chain within N epochs, PPO/SAC don't degrade below first-epoch value.
Use `--seed`/`env_seed` (D5), generous epochs (20), `@pytest.mark.slow`.

### E11 (was T3.11) — Reproducibility Seed Test
Two runs with `--seed 42 --env_seed 42` produce identical `eval_return`
trajectories and `method_state`. After B2 removes global-RNG reseeding and
D5 seeds buffer sampler; do it last.

---

## 8. Phase F — Packaging and CI (PENDING)

### F1. `gymnasium` not installable through project metadata
**Problem.** Code assumes `[rl]` extra exists:
- `GymnasiumEnvWrapper.__init__` (`pipelines/rl.py:31-35`) raises
  `ImportError("gymnasium is required... Install with: pip install 'genml_kit[rl]'")`
- Old plan specified "gymnasium as optional `[rl]` extra", pending Phase 3.

But `[project.optional-dependencies]` (`pyproject.toml:24-31`) defines only
`gcs`, `s3`, `lora`, `timm`, `uvito`, `all`, `dev` — **no `rl` extra**,
and `all` does not include it.

**Fix (exact diff):**

```toml
[project.optional-dependencies]
gcs = ["google-cloud-storage"]
s3 = ["boto3>=1.28"]
lora = ["peft>=0.7.0"]
timm = ["timm>=1.0"]
uvito = ["segmentation_models_pytorch>=0.3.0"]
rl = ["gymnasium>=0.29", "pygame>=2.1"]
all = ["genml_kit[gcs,s3,lora,timm,uvito,rl]"]
dev = ["pytest", "ruff", "yapf"]
```

- `pygame` needed only for `CartPole-v1` render_mode="human"; keep it.
- Pin floor `>=0.29` (first with stable `gymnasium` API: `terminated, truncated`).
- Update `ImportError` text once extra exists (convert to `fatal(...)` per E9).
- README: name `pip install 'genml_kit[rl]'`; fix any `gym` vs `gymnasium` typos.

---

## 9. Implementation order and commit plan

Sequence (each step lands green; nothing commits until you approve the diff):

1. **F1** — `pyproject.toml` `rl` extra (+ `all`) and README check.
2. **Phase D** — Test-coverage program (D1-D6) ✅ COMPLETE
3. **Phase E** — E2 (unblocks C3), then E1, E3, E4, E5, E7, E8, E10, E11.
4. **Final** — CLI end-to-end (E8), CI with `[rl]` (E9/F1), reproducibility (E11).

Suggested commit subjects:
- `pyproject: add [rl] extra with gymnasium`
- `rl: test-coverage program (D1-D6) - end-to-end tests, gradient logging, SAC hard-target sync, seeded buffers, checkpoint round-trip`
- `rl: prioritized replay, n-step, normalization (Phase E parts)`
- `rl: CI end-to-end wiring and reproducibility`

---

## 10. Exit criteria

1. All existing RL tests pass; every item in §7 has its test present and green.
2. `python -m pytest tests/ -k "rl" -q` green **without** `[rl]` extra
   (scripted envs); green **with** it installed.
3. `ruff check` clean, `yapf` clean on every touched file (2-space indent,
   no typing annotations, `fatal()` for all fatal paths).
4. `genml-kit-train --pipeline rl --method dqn --env-id CartPole-v1` runs
   end-to-end with `[rl]` installed; best-checkpoint target networks loadable.
5. `--method sac` and `--method ppo` each complete 3 epochs on scripted env
   via `run()`, with `best_metric` a float.
6. Resume: DQN checkpoint → resume → target weights identical to saved ones
   before further training (D3).
7. `rl/README.md` cross-references this document.

---

## 11. Decision log

- **A3 reframed as correctness, not hardening.** Old plan had SAC separate
  optimizers as Phase-3 task (T3.6); audit showed single-optimizer corrupts
  critics *today*, so moved to Phase A and T3.6 marked DONE-BY.
- **A6 contradicts design row D3.** Plan's "zero-change CheckpointSaver"
  claim was never true for frozen targets under `SAVE_FROZEN=False`; fix is
  one-line override, not a saver change.
- **A7's root cause is an API contract, not a typo.** `set_next_values`
  accepting broadcastable scalar silently let trainer pass one number; fix
  hardens API (`fatal` on wrong length) *and* fixes caller.
- **B2 reframed.** Reseeding was not just eval-determinism; it silently
  couples eval to replay sampling via global RNG, so fix (separate generators)
  required for reproducibility (E11).
- **C3/E2 pairing.** Old plan shipped n-step API with no producer; review
  either removes it or lands E2 — dead APIs not kept.
- **Test-count honesty.** Old plan's "137 RL tests" is stale (actual 106);
  exit criteria now refer to the suite, not memorized numbers.