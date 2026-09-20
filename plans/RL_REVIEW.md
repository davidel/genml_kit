# RL_REVIEW: Remediation & Completion Plan for RL Support in genml_kit

Status: **Active — Phases A, B, C, D complete; Phase E pending; Phase F (F1) complete**

This document supersedes `plans/RL_PLAN.md` (removed from the repo; recoverable from git history). This file is now the single source of truth for remaining RL work.

Conventions for every code change in this document:

- Python Google style, **2-space indent**, yapf (`.style.yapf`) authoritative.
- **No typing annotations** anywhere (project style).
- All unrecoverable error paths go through `from genml_kit.utils.logging import fatal` — **never** a bare `raise`. Pattern: `fatal(f"...", ValueError)`.
- `ruff check` must stay clean; no `# noqa` suppressions without approval.
- Nothing is committed to git until the reviewer approves the diff (and never amend/squash existing commits).

---

## 1. What is done (verified)

Phases A, B, C, D and F1 are implemented:

- `RLPipeline` (`pipelines/rl.py`) with `init_env`, `reset_env`, `step_env`, `env_push`, `obs_dim`, `n_actions`, `eval_rollout`, `to_device`; scripted `_ScriptedEnv` for gym-free tests.
- `ReplayBufferDataset` (`datasets/replay_buffer.py`) — circular numpy storage, `push`, `push_batch`, `sample`, `stats`.
- `RolloutBuffer` (`datasets/rollout_buffer.py`) — `add`, `compute` (GAE), `set_next_values`, `get_batch` (aliased as `sample`), `to`.
- Models: `rl/qnet`, `rl/qnet_dueling` (`models/rl/qnetwork.py`), `rl/actor_critic` (`models/rl/actor_critic.py`) with `get_action_and_value`, `get_distribution`, `get_value`, `SACModel` (`models/rl/sac_model.py`), `SACCritic` (`models/rl/sac_critic.py`).
- Losses (`losses/rl.py`): `td_target`, `td_loss`, `gae`, `clipped_surrogate`, `value_loss`, `entropy_bonus`, `sac_q_loss`, `sac_policy_loss`, `sac_alpha_loss`.
- Methods: `DQNMethod`, `PPOMethod`, `SACMethod` implementing the `Method` contract (`act`, `train_step`, `update_target`, `evaluate`, `has_metric_improved`, checkpoint state get/load, `get_trainer_class`).
- `RLTrainer` (`training/rl_trainer.py`) with off-policy and PPO epoch flows, warmup fill, and env-based `validate`.
- `train.py` selects trainer via `Method.get_trainer_class()`, calls `pipeline.init_env(args)` unconditionally.
- Test coverage: end-to-end `run()` tests (D1), gradient-norm logging & NaN guard (D2), PPO LR scheduling (D3), SAC hard-target sync on warmup complete (D4), seeded buffer samplers (D5), checkpoint round-trip tests (D6).
- `pyproject.toml` has `[rl]` extra: `rl = ["gymnasium>=0.29", "pygame>=2.1"]` included in `all`.

Test count: **1094 total tests passing (111 RL-specific)**.

---

## 2. Remaining Work — Phase E (Plan Phase-3 tasks re-scoped)

### E1 \u2014 Prioritized Experience Replay
Buffer gains priority array, `update_priorities`, proportional sampling, IS weights `w_i = (N\u00b7P(i))^{-\u03b2}` with \u03b2 annealing 0.4 \u2192 1.0. Flat-array implementation first. Sampler must use buffer's own `np.random.Generator` (per B2). Tests: `test_priority_sampling_is_seeded` + original plan tests.

### E2 \u2014 N-Step Returns \u2713 COMPLETED
Implemented n-step returns in `ReplayBufferDataset`:
- Added `n_step` and `gamma` parameters to constructor
- Transitions buffered in `deque(maxlen=n_step)` until full or episode ends (`done=True`)
- On flush: compute discounted sum `reward + gamma*reward + ... + gamma^{n-1}*reward`
- Stores `obs` (first), `action` (first), `n_step_reward`, `next_obs` (last), `done` (last)
- Added CLI args `--n-step` (default 1) to `RLPipeline`
- DQN method passes `self._n_step` to `td_target()` for consistency
- Tests: `TestNStepReturns` (4 tests: basic, early-done, flush-on-done, n_step=1 compatibility)

### E3 \u2014 Observation Normalization
`RunningMeanStd` in pipeline, `normalize(obs)` in `step_env`/eval, stats in checkpoint. Per-dimension over actual obs shape. Persist via `save_frozen`/state-dict path (buffers on small module in pipeline). `--obs_normalize` default **off**.

### E4 — Frame Stack
`--frame_stack N`, deque in pipeline, `(N, *obs_shape)` buffers. Depends on image-observation path in `models/rl/qnetwork.py` (documented as "deferred"); either land CNN backbone first or scope to flat-vector stacking.

### E5 — Vector Environments
`--num_envs N` via `gymnasium.vector.SyncVectorEnv`. Requires batched pushes with per-env `done` handling in buffers. A7 bootstrap generalized to per-env-per-step. Sequence after A7/E2.

### E7 — `post_train` Hooks
Export `policy.pt` (state dict only), final `evaluate()` with best checkpoint, final metrics to TensorBoard. Export online network (DQN) / `model.actor` (SAC/PPO). Assert exported state dict round-loads.

### E10 — Monotonic Improvement Test
Strict monotonicity flaky for PPO/SAC; assert DQN reaches maximum on scripted chain within N epochs, PPO/SAC don't degrade below first-epoch value. Use `--seed`/`env_seed` (D5), generous epochs (20), `@pytest.mark.slow`.

### E11 — Reproducibility Seed Test
Two runs with `--seed 42 --env_seed 42` produce identical `eval_return` trajectories and `method_state`. After B2 removes global-RNG reseeding and D5 seeds buffer sampler; do it last.

---

## 3. Implementation order and commit plan

Sequence (each step lands green; nothing commits until you approve the diff):

1. **E1 \u2014 Prioritized Experience Replay**
2. **E3 \u2014 Observation Normalization**
3. **E4 \u2014 Frame Stack** (or defer until CNN backbone exists)
5. **E5 — Vector Environments**
6. **E7 — `post_train` Hooks**
7. **E10 — Monotonic Improvement Test**
8. **E11 — Reproducibility Seed Test**

Suggested commit subjects:
- `rl: n-step returns (E2)` \u2713 COMPLETED
- `rl: prioritized experience replay (E1)`
- `rl: observation normalization (E3)`
- `rl: frame stack (E4)` / `rl: vector environments (E5)`
- `rl: post_train hooks (E7)`
- `rl: monotonic improvement & reproducibility tests (E10, E11)`

---

## 4. Exit criteria

1. All existing RL tests pass; every item above has its test present and green.
2. `python -m pytest tests/ -k "rl" -q` green **without** `[rl]` extra (scripted envs); green **with** it installed.
3. `ruff check` clean, `yapf` clean on every touched file (2-space indent, no typing annotations, `fatal()` for all fatal paths).
4. `genml-kit-train --pipeline rl --method dqn --env-id CartPole-v1` runs end-to-end with `[rl]` installed; best-checkpoint target networks loadable.
5. `--method sac` and `--method ppo` each complete 3 epochs on scripted env via `run()`, with `best_metric` a float.
6. Resume: DQN checkpoint → resume → target weights identical to saved ones before further training (D3).
7. `rl/README.md` cross-references this document.