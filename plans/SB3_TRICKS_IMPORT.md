# Plan: Import SB3 "tricks" into genml_kit PPO (no SB3 dependency)

**Status:** Implemented & committed (`307d330`, `bc8436d`, `4fbdfc7`):
Phases A (reward normalization), B (orthogonal init), C (`target_kl`)
code + tests are in; §2.6 doc strings and README "Stable-Baselines3-style
tricks" section are in (this pass); Phase D empirical ent_coef sweep and
the §4 validation-matrix Pendulum runs remain open.
**Target:** Make continuous PPO (Pendulum-v1) actually *learn* toward the
reference ~−200 returns, using the same mechanisms Stable-Baselines3 bakes
into its PPO, implemented natively.
**Constraint:** No new runtime dependency on `stable-baselines3`. All tricks
are ported as small, self-contained, unit-tested modules inside `genml_kit`.

---

## 0. Executive summary

SB3's PPO converges where vanilla PPO plateaus because of a handful of
engineering details, not because of a different algorithm. The four gaps in
genml_kit today, measured against SB3's `PPO` on `Pendulum-v1`:

| # | Gap | Symptom in current logs | SB3 mechanism | Est. impact |
|---|-----|--------------------------|---------------|-------------|
| 1 | **No reward/return normalization** | `value_loss` = O(1e3–1e4), `loss` ≈ 4–5k, `log_std` pinned | `VecNormalize(norm_reward=True)` running-return stats | **huge** (losses → O(1)) |
| 2 | **Default PyTorch init** | brittle training even with normalized obs | `orthogonal_init` (gain √2 / 1.0), exposed as generic `--param_init ortho` (default `None` = PyTorch) | large |
| 3 | **No `target_kl` early-stop** | destructive minibatch updates | `target_kl` (default None; SB3 uses `k1`) | medium (safety) |
| 4 | **`ent_coef` fixed + unprincipled** | entropy term is a rounding error vs value MSE | `ent_coef` policy (SB3 default 0.0 for continuous) | medium (tuning) |

Observation normalization already exists in genml_kit (`RunningMeanStd`,
`--obs_normalize`, wired in `RLPipeline` step/reset/checkpoint) — **not** in
scope except for tests that combine it with reward normalization.

This plan implements #1–#3 fully and #4 as an optional flag, behind
explicit CLI switches so existing behavior (and all 1191 tests) stays
bit-identical by default.

---

## 1. Precise, code-grounded gap analysis

### 1.1 Reward normalization (the critical gap)

**Current flow** (`genml_kit/training/rl_trainer.py::_train_epoch_ppo`, ~line 252):
1. rollout collected → `rollout.set_next_values(...)` → `rollout.compute(gamma, lam)` (GAE) → `rollout.normalize_advantages()` (once per rollout)
2. update: `loss = pg + 0.5·value_loss + 0.01·entropy`, with raw `returns`
   = raw discounted return sums.

**Problem:** Pendulum rewards are O(−1e3) per episode. `returns`, GAE
`advantages`, and value targets are all O(1e3). `value_loss = MSE(V, returns)`
is therefore O(1e4) and the `0.5` coef still leaves it O(5e3) — dominating
the total `loss`. With `ent_coef=0.01`, the entropy term (O(0.7)) is a
rounding error, so `log_std` never gets a meaningful signal and exploration
is effectively frozen (matches the pinned `entropy=0.7258` in the logs).

**SB3 mechanism:** `VecNormalize(norm_reward=True, gamma=0.99, clip_reward=10.0)`
maintains a running mean/std of the *discounted returns* (`return_rms`) and
normalizes rewards by `return_rms.std` (not the raw reward mean). At eval
time the wrapper is disabled (`training=False`) so raw rewards are used.

### 1.2 Orthogonal initialization

**Current code** (`genml_kit/models/rl/actor_critic.py`):
- `_MLPBackbone` uses `nn.Linear` → default `kaiming_uniform_` (PyTorch).
- `GaussianActor.mean` / `ValueHead.fc` also default init; `log_std` is an
  `nn.Parameter(torch.full((action_dim,), log(0.5)))`.

**SB3 mechanism** (`sb3/common/torch_layers.py::create_mlp` +
`sb3/common/utils.py::orthogonal_init`):
- Every Linear layer `orthogonal_init(weight, gain)` with:
  - policy (actor) layers gain `√2`
  - value head gain `1.0`
  - biases initialized to 0
- Applied via `self.apply(init_weights)` after construction, with a
  per-module dispatch (Linear vs. norm vs. log_std).

This is the single most load-bearing "trick": SB3's PPO hyperparameters are
tuned around it, and it fixes the *scale* of the initial policy gradient.

### 1.3 `target_kl` early stopping

**Current code** (`genml_kit/methods/rl_ppo.py` `compute_loss`): each of
`--ppo_epochs` epochs runs all minibatches unconditionally.

**SB3 mechanism:** `PPO.train()` computes `approx_kl` (Schulman **k1** =
`0.5·mean((log_ratio)²)`) after each minibatch and, if `target_kl` is set,
**stops the whole epoch** (breaks out of the minibatch loop) when
`approx_kl > target_kl`. Note we already fixed `approx_kl` to k1 in
`bc8436d` — this is a consumer of that fix.

### 1.4 Entropy coefficient policy

**Current code:** `--ppo_entropy_coef` default `0.01`, constant across epochs.

**SB3 mechanism:** `ent_coef` is a plain constant; default `0.0` for
continuous PPO in SB3 (relying on init + normalization rather than an
entropy bonus), `0.0` also for discrete in recent versions (they found the
bonus unhelpful). genml_kit should keep `0.01` default (compat) but make it
tunable per run; reward normalization alone makes `0.01` meaningful.

---

## 2. Design

### 2.1 New module: `genml_kit/pipelines/rl_normalize.py`

Self-contained ReturnNormalizer (mirrors the already-present
`RunningMeanStd` in `pipelines/rl.py`, but for **returns** and wired into
the PPO rollout, not SAC-alpha).

```python
class ReturnNormalizer(nn.Module):
  """SB3-style discounted-return normalization (VecNormalize norm_reward).

  Maintains a running mean/std of discounted returns and normalizes each
  reward by the current std (clipped).  Disabled at eval time so raw
  rewards are returned.  Checkpointable (state_dict/load_state_dict).
  """
  def __init__(self, gamma=0.99, clip_reward=10.0, epsilon=1e-4):
    ...
    self.return_rms = RunningMeanStd(shape=())   # reuse existing class
    self.gamma = gamma
    self.clip_reward = clip_reward
    self.training = True                          # False at eval time

  def normalize_reward(self, reward, running_return) -> Tensor  # per-step
  def update(self, running_return) -> None                       # Welford update
  def denormalize(self, value) -> Tensor                         # for reporting
```

Key details (ported faithfully from SB3 `VecNormalize`):
- `running_return = gamma * running_return + reward` maintained **inside**
  the rollout loop (only during `training=True`).
- `normalized_reward = clip(reward / sqrt(return_rms.var + eps), clip_reward)`.
- Eval: `self.training=False` → `normalize_reward` returns raw reward.
- Save/load via existing checkpoint machinery (see §2.4).

### 2.2 Model init: generic `--param_init` (not RL-specific)

Instead of a PPO-only `--ortho_init`, expose a **general** parameter-init
flag `--param_init`, applied uniformly to every model via the existing
`Method._apply_model_extras` hook (`genml_kit/methods/base.py:208`).

Why generic: `_apply_model_extras` already runs `--lora`, `--freeze` and
`--grad_checkpoint` for *every* method (called by every `build_model`
after construction). Adding `--param_init` next to them gives orthogonal
init to PPO, SAC (`rl_sac.py::build_model`, line 145) and any future
method with one flag — and is usable for non-RL models too (e.g. an MLP
classifier) where orthogonal init also helps.

Design (in `genml_kit/methods/base.py` + a small `init` helper):

```python
# genml_kit/models/rl/actor_critic.py (or a shared genml_kit/models/init.py)
def init_orthogonal(module, gain_policy=math.sqrt(2.0), gain_value=1.0):
  """SB3-style orthogonal init across the ActorCritic graph."""
  for name, child in module.named_modules():
    if isinstance(child, nn.Linear):
      gain = gain_value if "critic" in name else gain_policy
      nn.init.orthogonal_(child.weight, gain=gain)
      nn.init.zeros_(child.bias)
  # log_std stays at its explicit init (log(0.5)); SB3 does not touch it.
```

CLI: `--param_init {None,ortho}` (default `None` === current behavior
= PyTorch default init, exact backward compat for all 1191 tests). In
`_apply_model_extras`: `if args.param_init == "ortho": init_orthogonal(model)`.

For non-ActorCritic models the same hook applies per-module gains
(MLP classifier: `gain=sqrt(2)` for hidden/dense layers, `1.0` for the
head) — documented as such, so the flag is genuinely reusable.

**Where `ortho` helps (document this in the flag help/docs, not assumed):**
- MLP-based RL policies/critics (PPO, SAC) on continuous control — SB3's
  tuned hyperparameters assume orthogonal init; especially together with
  reward normalization (`--reward_normalize --param_init ortho`).
- Small-to-mid MLPs (e.g. classifiers) where default PyTorch init gives a
  poorly-scaled first gradient step; orthogonal init with per-layer gains
  (√2 hidden, 1.0 head) stabilizes early training.
- **Neutral-to-mild for transformer/ViT models** (their own init is already
  tuned; `ortho` would override it) — hence the option, not the default.
- **Not appropriate** for convolutional backbones expecting
  He/Kaiming-style init — keep `default` there.

### 2.3 PPO method flags: `genml_kit/methods/rl_ppo.py`

Add to `add_args`:

```python
group.add_argument("--ppo_target_kl", type=float, default=None,
                   help="Early-stop PPO epoch when approx_kl exceeds this "
                        "(SB3-style). Default None = no early stop.")
```

And read it in `wire_data`/`build_model`: `self._target_kl = getattr(args, "ppo_target_kl", None)`.

`compute_loss` / `train`-loop integration: after each minibatch, compute
`approx_kl = 0.5 * (old_log_probs - new_log_probs).pow(2).mean()` (already
the fixed metric); if `self._target_kl is not None and approx_kl > self._target_kl` → break the minibatch loop for the current epoch.

### 2.4 Reward normalization wiring: `genml_kit/training/rl_trainer.py`

The trainer is where the rollout loop lives. Changes are minimal and
behind flags:

```python
# In __init__ (or first epoch), when args.reward_normalize:
self.ret_norm = ReturnNormalizer(gamma=self.method._gamma,
                                 clip_reward=args.reward_norm_clip)
self.ret_norm.to(self.device)
```

In `_train_epoch_ppo` rollout loop, **before** storing into the buffer:
```python
if self.ret_norm is not None and self.ret_norm.training:
  # per-step running return for the current episode
  running_return = gamma * running_return + reward
  self.ret_norm.update(running_return)          # Welford update
  reward = self.ret_norm.normalize_reward(reward, running_return)
```

Eval path (`validate`): set `ret_norm.training = False` (raw rewards);
restore to True after. This matches SB3's `VecNormalize.train(False)`.

Checkpointing: extend `ckpt_extra` / `saver_extra` to include
`ret_norm.state_dict()` and restore in `load` (mirror how `obs_rms` is
already saved/loaded in `RLPipeline`).

### 2.5 CLI / flag surface (all **opt-in**, defaults preserve current behavior)

| Flag | Default | Effect |
|------|---------|--------|
| `--reward_normalize` | `False` | Enable return-normalized rewards (SB3 `norm_reward`) |
| `--reward_norm_clip` | `10.0` | Reward clipping bound after normalization |
| `--param_init` | `None` | `--param_init ortho` = SB3-style orthogonal init (generic, all methods); `None`/default = PyTorch default init |
| `--ppo_target_kl` | `None` | Early-stop epochs on `approx_kl > target_kl` |
| `--ppo_entropy_coef` | `0.01` (unchanged) | Constant; now meaningful once rewards normalized |

All 1191 existing tests keep passing with defaults — nothing changes unless
a flag is passed.

### 2.6 Documentation & help-string requirements (from review)

Every new/affected flag must carry honest, actionable help text and
README/docs notes that state **when the knob helps and when it doesn't**:

- **`--reward_normalize`**: help/docs must say: *"Normalizes rewards by the
  running std of discounted returns (SB3 `VecNormalize(norm_reward=True)`).
  Intended for environments with large/unnormalized reward scales (e.g.
  Pendulum-v1, MuJoCo) where value loss otherwise dominates the total loss
  (O(1e3–1e4)) and swamps the entropy/policy signal. Harmless or unnecessary
  when rewards are already O(1) (e.g. small step rewards). Eval always uses
  raw rewards."* — flag is on the **generic** parser (usable by any RL
  method), documented as such.
- **`--param_init`**: help/docs must say: *"Parameter init strategy
  (default `None` = PyTorch default init). `ortho` applies orthogonal init
  (gain √2 for policy/hidden layers, 1.0 for value/head layers, zero
  biases) — the init SB3's PPO/SAC are tuned around, so it helps
  MLP-based RL policies/critics on continuous control and small-to-mid
  MLPs. Generally neutral-to-mild (or harmful) for transformer/ViT and
  conv backbones whose own init is already tuned — keep `None` there."*
- **`--ppo_target_kl`**: help/docs must say: *"Early-stop a PPO epoch when
  the k1 approx-KL exceeds this (SB3 `target_kl`). Helps avoid destructive
  updates on high-variance minibatches; a value too low (e.g. < 0.005)
  stalls learning, too high (e.g. > 0.1) disables the safety. `None`
  (default) disables."*
- **`--ppo_entropy_coef`**: help/docs must say: *"Coefficient of the
  entropy bonus. SB3 uses 0.0 for continuous PPO (relies on init +
  normalization); genml_kit keeps 0.01 as a lightly-regularizing default.
  With reward normalization in place, increasing to ~0.05–0.1 encourages
  broader exploration (may help hard exploration tasks); decreasing toward
  0.0 sharpens the policy (may help fine-tuning late in training). No single
  value is best for all tasks — sweep 0.0–0.1 on your task."*

These doc strings are part of the acceptance criteria (see §4).

**Status (2025-09-25):** all §2.6 doc strings are implemented:
- `--reward_normalize` / `--reward_norm_clip` (rl.py),
- `--param_init` (train.py),
- `--ppo_target_kl` (rl_ppo.py),
- `--ppo_entropy_coef` (rl_ppo.py, added this pass),
plus a README section "Stable-Baselines3-style tricks" documenting all
four knobs, when each helps/doesn't help, and a recommended
Pendulum-style invocation.

---

## 3. Implementation plan (ordered, with tests)

### Phase A — Reward normalization (biggest win)
1. Add `genml_kit/pipelines/rl_normalize.py::ReturnNormalizer` + unit tests
   (`tests/test_rl_normalize.py`):
   - correct running-return update + Welford stats
   - normalization clips at `clip_reward`
   - `training=False` returns raw rewards
   - state_dict round-trip via checkpoint
2. Wire into `rl_trainer.py` rollout loop + eval toggle + checkpoint.
3. Test on Pendulum-v1:
   - before: `value_loss` O(1e3–1e4), `loss` ≈ 4–5k, eval flat ~−1100.
   - after: `value_loss` O(1), `loss` O(1), eval starts climbing toward ~−200.

### Phase B — Generic `--param_init ortho`
1. Add `init_orthogonal` helper + `--param_init {None,ortho}` in
   `_apply_model_extras` (base.py) so all methods can use it.
   Default `None` = PyTorch init (bit-identical to today).
2. Tests: init respects gains (√2 policy, 1.0 value/head); biases zero;
   `log_std` untouched; `--param_init None` path identical to today;
   works on a non-RL MLP model (generic-flag check).
3. Re-run Pendulum with `--reward_normalize --param_init ortho`; expect
   faster early improvement (SB3 uses this combination by default).

### Phase C — `target_kl` early-stop
1. Add `--ppo_target_kl`, wire into `compute_loss` / minibatch loop.
2. Tests: with small `target_kl` the epoch stops early (assert fewer
   minibatches consumed / loss changes), with `None` never stops.
3. Re-run with `--ppo_target_kl 0.03` (SB3 reference) + normalization.

### Phase D — `ent_coef` policy (optional tuning; keep 0.01 default)
1. Keep `--ppo_entropy_coef` default `0.01` (review decision).
2. Sweep `--ppo_entropy_coef in {0.0, 0.01, 0.05, 0.1}` on normalized
   Pendulum to *document* (not necessarily change) how it trades off
   exploration vs. policy sharpness. Record the curve in the plan/README.
3. No default change unless the sweep shows a clearly better value;
   the documentation from §2.6 (when to raise/lower) is the deliverable if
   the default stays.

---

## 4. Validation matrix

| Test | Command / scope | Pass criterion |
|------|-----------------|----------------|
| Unit | `pytest tests/test_rl_normalize.py` | all new tests green |
| Unit | `pytest tests/test_actor_critic.py tests/test_rl_ppo.py tests/test_optim_factory.py` | no regression |
| Slow | `pytest -m slow` (3 learning-convergence tests) | still green with defaults |
| Full | `pytest -m "not slow"` | 1191+new passed |
| Lint | `ruff check` on changed files | clean |
| Format | `yapf` via `format_file` (`.style.yapf`, 2-space) | clean |
| Pendulum | your command + `--reward_normalize --param_init ortho` | `value_loss`/`loss` O(1); eval clearly improves over the −1100 plateau |
| Pendulum | + `--ppo_target_kl 0.03` | no divergence; stable improvement |

Success definition (SB3 reference on Pendulum-v1, continuous PPO):
`eval_return` meaningfully above the random baseline (~−1180), with
normalized `loss`/`value_loss` ≈ O(1). **The concrete target is empirical,
not fixed a priori**: we run the same seed/command with
`--reward_normalize --param_init ortho` (+ optionally `--ppo_target_kl 0.03`)
and record where it lands. If it reaches ≈ −200 (SB3 parity) — excellent;
if it lands meaningfully better than the −1100 plateau but not fully at
−200, that's still a success for this PR and we iterate in a follow-up
(e.g. more steps, `ent_coef` sweep from Phase D, or `target_kl` tuning).
The **hard acceptance criteria** are: (a) all flags opt-in with bit-identical
defaults; (b) `value_loss`/`loss` drop to O(1) with reward normalization;
(c) eval beats the pre-change plateau; (d) documented help strings per §2.6.

---

## 5. Risks & mitigations

- **Reward normalization changes loss scale → retuning needs.** Mitigate:
  all flags opt-in; default code path bit-identical; tuning isolated behind
  `--reward_normalize`.
- **Orthogonal init can *hurt* if the model was tuned around default init.**
  Mitigate: `--param_init` defaults to `None` (PyTorch init); test both;
  keep per-layer gains exactly as SB3; document the transformer/conv
  caveat in the flag docs (§2.6).
- **`target_kl` early-stop can stall training if set too low.**
  Mitigate: default `None` (disabled); document sensible range (0.01–0.05).
- **Checkpoint compatibility.** New flags change optimizer/model layout only
  when enabled; `ret_norm` state is additive and guarded by flag presence.
- **No SB3 import anywhere.** All tricks are re-implementations of standard
  published mechanisms (Schulman k1, Welford online stats, orthogonal init),
  MIT-style and documented inline.

---

## 6. Out of scope (deliberately)

- `VecEnv` parallelism / `SubprocVecEnv` (we have single-env stepping).
- `clip_range` linear scheduling (SB3 uses constant 0.2 by default).
- `max_grad_norm` — already present in `rl_trainer._apply_grad`.
- GAE / bootstrap masks — already correct.
- `ent_coef` scheduling (only a constant choice is needed).
- SAC/DQN paths — untouched; `ReturnNormalizer` lives under the PPO flag.

---

## 7. Decisions & remaining open questions

**Decided in review:**
1. `--reward_normalize` is **opt-in** (default `False`), with docs/help
   text explaining when it helps (§2.6).
2. `--ppo_entropy_coef` stays **0.01** by default; the deliverable is
   documentation of when to change it (§2.6), plus an empirical sweep in
   Phase D recorded in the README — not a default change.
3. `ReturnNormalizer` lives in **`genml_kit/pipelines/rl_normalize.py`**.
4. Init is a **generic `--param_init {None,ortho}`** applied via
   `Method._apply_model_extras` (all methods), not a PPO-only flag.
   Default `None` = PyTorch default init (backward compat).
5. Pendulum target is **empirical**: run with the flags, record where SB3
   changes land us; hard criteria are opt-in defaults + O(1) losses + beat
   the plateau + docs (§4).
6. **Doc deliverable shipped (2025-09-25):** §2.6 help strings on all
   four flags + README "Stable-Baselines3-style tricks" section.  The
   §2.6 requirement (acceptance criterion (d)) is now satisfied.

**Remaining open questions:**
1. `--param_init`: should `ortho` become the default for *new* continuous
   PPO/SAC runs, or stay strictly opt-in for now? (Proposal: keep `default`
   everywhere initially; flip to `ortho` for RL after the Pendulum runs.)
2. Should `--reward_normalize` also be exposed to SAC (it shares the RL
   pipeline), or stay PPO-only for this PR?
   **Research-backed recommendation: keep it PPO-only for this PR.** SAC
   with automatic temperature (`ent_coef="auto"`) already *adapts* to the
   reward scale via α, so `norm_reward` is usually unnecessary there — SB3
   practitioners explicitly avoid `norm_reward=True` for SAC
   (`normalize()` in RL Zoo uses `norm_reward=False` for SAC) because
   auto-α collapses/stabilizes reward-scale sensitivity, and normalizing
   rewards can squeeze the α adaptation range or add a moving-statistics
   bias. The one case where SAC reward normalization *can* help is
   pathological reward scales with *fixed* α — not our setup (SAC here
   uses `sac_auto_alpha=True` by default, see `rl_sac.py`). PPO has no
   such built-in scale adaptation, which is exactly why it needs the
   normalizer. So: **PPO-only** for this PR; revisit SAC only if a
   concrete SAC reward-scale problem appears (and only with fixed α).
   `ReturnNormalizer` stays reusable by design if that day comes.