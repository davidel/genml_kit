# Feature plan: record evaluation videos for RL methods (DQN / PPO / SAC)

Status: **DRAFT — awaiting approval. Do not implement until the plan is approved.**
Created: (date of writing)
Author: chatty (assistant), on behalf of the user.

This document is a complete, self-contained implementation plan. Its purpose is
that the task can be resumed tomorrow (or by anyone) with **zero information
loss**: it captures the current code, the exact changes, the CLI surface, test
strategy, and answers to the open design questions.

---

## 1. Goal

Add an opt-in CLI flag that, **only if the gym environment supports rendering**,
records **one video of an evaluation episode** during the RL training
validation phase (the `RLTrainer.validate()` path). Requirements:

- Off by default; no behavior change for existing runs.
- Works uniformly for all three RL methods: DQN, PPO, SAC.
- Must not crash when the env cannot render (missing renderer, headless box,
  custom env without `render()`): log a single **info/warning** and skip.
- Must not add overhead to the training loops (recording is validation-only).
- Must pass `ruff check .` and `pytest` (the project's CI gates; see §8).
- Must follow house style: **2-space indent, Google docstrings, yapf** with
  `.style.yapf` (`based_on_style = google`, `column_limit = 88`,
  `indent_width = 2`).

---

## 2. Current code map (verified today)

All line numbers below are **as of today** (git HEAD `90466b7`). Re-verify with
`git blame` / `get_outline` if the file has changed.

### 2.1 CLI / arg plumbing

- Entry point: `genml_kit/training/train.py` → `main()` (line ~415) →
  `normalize_args(parse_args(argv))`. The parser is filled by
  `pipeline.add_args(parser)` and `method.add_args(parser)`.
- RL-specific CLI args live in `RLPipeline.add_args` at
  `genml_kit/pipelines/rl.py:232-337` (arg group `"rl pipeline"`). Existing
  args: `--env-id`, `--obs-dim`, `--env-script`, `--warmup-steps`,
  `--replay-capacity`, `--eval-episodes`, `--env-seed`, `--steps-per-epoch`,
  `--n-step`, `--prioritized`, `--per-alpha`, `--per-beta-start`,
  `--per-beta-frames`, `--obs-normalize`, `--obs-norm-clip`.
- The relevant existing flag (for naming/co-location):
  ```python
  group.add_argument(
      "--eval-episodes",
      dest="eval_episodes",
      type=int,
      default=5,
      help="Number of episodes for policy evaluation.",
  )
  ```

### 2.2 Environment creation

`RLPipeline.init_env(self, args)` at `genml_kit/pipelines/rl.py:347-441`:

```python
def init_env(self, args):
    """Build (and cache) the environment and replay buffer.

    Called once at the start of training by ``RLTrainer``.
    """
    from genml_kit.utils.logging import fatal
    from genml_kit.utils.script import extern_call

    if self.env is not None:
      return

    if getattr(args, "env_script", None):
      self.env = extern_call(args.env_script, "make_env")
    else:
      self.env = GymnasiumEnvWrapper(args.env_id)
    ...
```

Note the two env-creation paths:

1. **Gymnasium** (default): `GymnasiumEnvWrapper(args.env_id)`.
2. **External script** (`--env-script`): `extern_call(args.env_script, "make_env")`
   — returns an arbitrary object; **we cannot assume it supports rendering**.

The gymnasium wrapper is `GymnasiumEnvWrapper` at `genml_kit/pipelines/rl.py:111-143`:

```python
class GymnasiumEnvWrapper:
  """Thin adapter around a Gymnasium ``Env`` with numpy-array observations.

  Provides the three methods that ``RLPipeline`` and ``RLTrainer`` need:
  ``reset``, ``step``, ``close``.
  """

  def __init__(self, env_id):
    try:
      import gymnasium as gym
    except ImportError as exc:
      raise ImportError("gymnasium is required for --pipeline rl.  "
                        "Install it with:  pip install 'genml_kit[rl]'") from exc
    self.env = gym.make(env_id)
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space

  def reset(self):
    obs, _info = self.env.reset()
    return obs

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    done = terminated or truncated
    return obs, reward, done, info

  def close(self):
    self.env.close()
```

**Critical gymnasium details (verified against gymnasium 1.3.0, the sandbox
version; the API contract below also holds for our min `>=0.29`):**

1. **The renderer is chosen at construction** via `gym.make(env_id,
   render_mode=...)`. If the env was created with `render_mode=None` (the
   default) and you later call `env.render()`, gymnasium raises `AttributeError`:

   > `AttributeError: Your environment must specify a valid render_mode to use the render method`

   Therefore **we must enable rendering at `gym.make()` time**, before any
   training begins — not lazily at validation time. This drives the design in §4.

2. **`metadata["render_modes"]` does NOT reflect runtime availability.** For
   CartPole, `CartPoleEnv.metadata["render_modes"]` is statically
   `['human', 'rgb_array']` **regardless of whether `pygame` is installed**.
   Verified: with `pygame` absent, `gym.make("CartPole-v1", render_mode="rgb_array")`
   *succeeds* (the renderer is only constructed lazily on the first `render()`
   call in gymnasium >= 1.0), and **that first `render()` then raises
   `gymnasium.error.DependencyNotInstalled`**. Consequence for the design: a
   probe that only checks `metadata["render_modes"]` will *falsely* report
   "renderable" on a headless / no-`pygame` box. **The probe must actually call
   `render()` (inside a try/except) to know the truth** — a metadata check alone
   is insufficient. (`pygame>=2.1` is a declared dep of the `rl` extra, so a
   properly installed environment is fine; the fragile case is CI / headless /
   partial installs — exactly the "must not crash" requirement.)

3. **`render()` requires a preceding `reset()`.** In gymnasium >= 1.0, calling
   `render()` before the first `env.reset()` raises
   `gymnasium.error.ResetNeeded` ("Cannot call `env.render()` before calling
   `env.reset()`"). In our wrapper, `reset()` is always called before the eval
   loop (via `reset_env()`), so the *recording* path is safe — **but a
   `can_render()`-style probe performed at init time (before any reset) must
   tolerate `ResetNeeded`, or be invoked only after reset.** This is settled in
   §3 D10 / §4.2 by probing *inside the eval loop, after reset*.

### 2.3 Eval loops (the three methods)

All three methods implement `evaluate(self, model, pipeline, num_episodes,
max_steps=10_000)`. They are structurally identical. Current bodies:

**PPO** — `genml_kit/methods/rl_ppo.py:223-248`:
```python
def evaluate(self, model, pipeline, num_episodes, max_steps=10_000):
    """Run evaluation episodes and return mean return."""
    total_return = 0.0
    total_steps = 0
    for _ in range(num_episodes):
      obs = pipeline.reset_env()
      episode_return = 0.0
      done = False
      episode_steps = 0
      while not done and episode_steps < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # (PPO acts via model.get_action_and_value / get_value; see file)
        ...
        obs, reward, done, _ = pipeline.step_env(act_val)
        episode_return += reward
        episode_steps += 1
        total_steps += 1
      total_return += episode_return
    return {
        "eval_return": total_return / max(num_episodes, 1),
        "eval_steps": total_steps,
    }
```

**SAC** — `genml_kit/methods/rl_sac.py:336-...`:
```python
def evaluate(self, model, pipeline, num_episodes, max_steps=10_000):
    """Run evaluation episodes and return mean return."""
    total_return = 0.0
    total_steps = 0
    for _ in range(num_episodes):
      obs = pipeline.reset_env()
      episode_return = 0.0
      done = False
      episode_steps = 0
      while not done and episode_steps < max_steps:
        action = self.act(model, obs, deterministic=True)
        ...
        obs, reward, done, _ = pipeline.step_env(act_val)
        episode_return += reward
        episode_steps += 1
        total_steps += 1
      total_return += episode_return
    return {
        "eval_return": total_return / max(num_episodes, 1),
        "eval_steps": total_steps,
    }
```

**DQN** — `genml_kit/methods/rl_dqn.py:215-...`:
```python
def evaluate(self, model, pipeline, num_episodes, max_steps=10_000):
    """Run evaluation episodes and return mean return.

    Overrides the base ``Method.evaluate`` signature because RL
    validation does not use a DataLoader.

    Args:
        model:       QNetwork with ``.online`` sub-module.
        pipeline:    ``RLPipeline`` providing ``reset_env`` / ``step_env``.
        num_episodes: Number of evaluation episodes.
        max_steps:   Hard cap on environment steps PER EPISODE to prevent
                     infinite loops with untrained policies.
    """
    total_return = 0.0
    total_steps = 0
    for _ in range(num_episodes):
      obs = pipeline.reset_env()
      episode_return = 0.0
      done = False
      episode_steps = 0
      while not done and episode_steps < max_steps:
        ...
        obs, reward, done, _ = pipeline.step_env(act_val)
        episode_return += reward
        episode_steps += 1
        total_steps += 1
      total_return += episode_return
    return {...}
```

Key observations from the three bodies:

- `step_env` returns a **4-tuple** `(obs, reward, done, info)` — the wrapper
  swallows gymnasium's `terminated`/`truncated` into `done`, but it also puts
  `info["terminated"]` and `info["truncated"]` (see `step_env` docstring,
  `rl.py:453+`). Not needed for video.
- The loop is **per-episode**; `reset_env()` is called at the top of each
  episode. This is the natural place to (re)start recording per episode.

### 2.4 The trainer's validation path

`RLTrainer.validate()` — `genml_kit/training/rl_trainer.py:354-376`:

```python
def validate(self):
    """Policy evaluation on the environment (not a DataLoader).

    Returns:
      Scalar eval_return (float) for checkpoint selection.
    """
    env_seed = getattr(self.args, "env_seed", None)
    # Isolate RNG state to avoid corrupting training sampling (B2).
    if env_seed is not None:
      old_state = np.random.get_state()
      np.random.seed(env_seed)
    else:
      old_state = None
    try:
      metrics = self.method.evaluate(
          self.model,
          self.pipeline,
          getattr(self.args, "eval_episodes", 5),
      )
      return float(metrics["eval_return"])
    finally:
      if old_state is not None:
        np.random.set_state(old_state)
```

`validate()` is called from `BaseTrainer.run()` (`genml_kit/training/trainer.py:226`)
every epoch; the return value feeds checkpoint selection. `RLTrainer.__init__`
(`rl_trainer.py:46-53`) stores `self.args` (via `BaseTrainer`), so **new args are
available as `getattr(self.args, "record_eval_video", False)`**.

Relevant pieces of `BaseTrainer.run()` (trainer.py:226 area):

```python
metrics = self.validate()
...
if self.criterion(metrics[self.best_metric_key], self.best_metric): ...
```

### 2.5 `RLPipeline` constructor + reset/step

- `RLPipeline.__init__` sets `self.env = None`, `self._obs_normalize = False`,
  `self.obs_rms = None`, etc. (`rl.py:215-230`). `init_env` is what builds the
  real env.
- `reset_env()` at `rl.py:443-451`:
  ```python
  def reset_env(self):
      """Reset the environment and return the initial observation tensor."""
      # Handle both old gym API (obs) and new gymnasium API (obs, info)
      reset_result = self.env.reset()
      obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
      if self._obs_normalize and self.obs_rms is not None:
        self.obs_rms.update(obs[None, ...])  # Add batch dim for update
        obs = self.obs_rms.normalize(obs, clip=self._obs_norm_clip)
      return obs
  ```
- `step_env(action)` at `rl.py:453+` returns `(next_obs, reward, done, info)`.

---

## 3. Design decisions (RESOLVED — from the earlier discussion)

| # | Decision | Choice |
|---|---|---|
| D1 | Flag name | `--record-eval-video` (CLI) → `args.record_eval_video` (bool, default `False`) |
| D2 | Output directory | `<checkpoint_dir>/videos/` (i.e. `os.path.join(args.checkpoint, "videos")`; `args.checkpoint` is the existing checkpoint dir used for saves/logs) |
| D3 | Recording cadence | **Every validation** when flag is on. (No separate "every N" knob in v1.) |
| D4 | Which episode(s) | **All `eval_episodes`** — each episode becomes its own numbered file (`eval_episode_00001.mp4`, …). This gives the best signal for roughly ~5 short files on classic control. |
| D5 | Format / encoder | Prefer **MP4 via `imageio[ffmpeg]`**; **fallback to animated GIF via `PIL`** (PIL is already a dependency in the environment, verified). Both written by a new shared helper (§4.3). |
| D6 | Env renderability | **Probe by actually calling `render()`** (in a try/except) once per episode *after* `reset_env()` — **not** by checking `metadata["render_modes"]` (which is statically populated and ignores missing `pygame`/headless; see §2.2). Skip recording gracefully if the probe raises or returns `None`. `--env-script` envs are probed too but assumed not renderable if they lack the API (see §4.2). |
| D7 | Render mode string | `"rgb_array"` (default gymnasium mode that returns numpy HxWx3 uint8 frames; works headless with `pygame` for classic control) |
| D8 | Extra dependency | Add `imageio[ffmpeg]` to the `rl` extra in `pyproject.toml`. GIF fallback keeps the feature working even without it. |
| D9 | Recording scope | Validation only; never inside training loops. No extra env steps are performed; frames are grabbed from the real rollout. |
| D10 | Frame source | `pipeline.render_frame()` grabs the **current rendered frame** from the env *after* each `step_env`, plus one frame *after* `reset_env` (so the video starts at the initial state). **The per-episode `can_record_video()` probe (a real `render()` attempt) must run *after* the reset**, because gymnasium >= 1.0 raises `ResetNeeded` on `render()` before the first reset (see §2.2). The probe result is cached per episode and reused by the frame-grab calls. |

---

## 4. Detailed change list

### 4.1 `pyproject.toml` — add video dependency to `rl` extra

Current (line 30):
```toml
rl = ["gymnasium>=0.29", "pygame>=2.1"]
```
Change to:
```toml
rl = ["gymnasium>=0.29", "pygame>=2.1", "imageio[ffmpeg]>=2.31"]
```
(`imageio[ffmpeg]` is the standard, lightweight way to get an ffmpeg-backed
writer without requiring a system ffmpeg binary. `PIL` is already present via
image pipeline deps, so the GIF fallback needs no new dependency.)

### 4.2 `genml_kit/pipelines/rl.py`

#### 4.2.1 `GymnasiumEnvWrapper` — render support

**Change `__init__`** to accept an optional `render_mode` (default `None`):

```python
def __init__(self, env_id, render_mode=None):
    try:
      import gymnasium as gym
    except ImportError as exc:
      raise ImportError("gymnasium is required for --pipeline rl.  "
                        "Install it with:  pip install 'genml_kit[rl]'") from exc
    self.env = gym.make(env_id, render_mode=render_mode)
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self._render_mode = render_mode
```

**Add two methods** to the wrapper:

```python
def render(self):
    """Return the current frame as a numpy RGB (H, W, 3) uint8 array.

    Returns ``None`` if the env was created without a render mode or the
    underlying env has no renderer for this mode.
    """
    if self._render_mode is None:
      return None
    try:
      return self.env.render()
    except Exception:  # noqa: BLE001 - any renderer failure => no video
      return None

def can_render(self):
    """Whether a ``render()`` call is expected to produce frames.

    **This performs a real ``render()`` attempt** (wrapped in try/except) and
    returns whether it produced a frame.  It does NOT trust
    ``metadata["render_modes"]``: that field is populated statically on the
    env class and reports e.g. ``['human', 'rgb_array']`` even when
    ``pygame`` is not installed (verified, gymnasium 1.3.0) — a metadata-only
    check would falsely report "renderable" and then crash on the first real
    ``render()`` with ``DependencyNotInstalled``.

    Precondition: called only *after* ``reset()`` (otherwise gymnasium >= 1.0
    raises ``ResetNeeded``; that, too, is caught here and maps to ``False``).
    """
    if self._render_mode is None:
      return False
    try:
      frame = self.env.render()
      return frame is not None
    except Exception:  # noqa: BLE001
      # DependencyNotInstalled (no pygame / headless), ResetNeeded, etc.
      return False
```

Design notes (revised after code inspection):

- **Real-render probe, not metadata.** The previous draft probed
  `metadata["render_modes"]` first for cheapness. That check is statically
  true on classic-control envs regardless of runtime deps, so it *cannot* be
  used to satisfy the "must not crash on headless/no-pygame" requirement. The
  probe cost (one `render()` per episode) is negligible vs. the recording
  itself, which calls `render()` per step anyway.
- **Call-site ordering matters.** `can_render()` may only be safely invoked
  *after* `reset_env()` (see §2.2 point 3). The eval loops in §4.4 therefore
  call `pipeline.can_record_video()` (which delegates here) right after
  `reset_env()` and cache the result for the episode.
- The `env.unwrapped` metadata guard is no longer needed for correctness and
  is dropped from the probe; rendering is the source of truth.

#### 4.2.2 `RLPipeline` — expose render helpers + flag

**Add** (instance) methods delegating to the wrapped env, safe on any env type:

```python
def can_record_video(self):
    """Whether the current env can produce frames for a video.

    Returns ``False`` for scripted/external envs that do not expose the
    render API, so callers can skip recording without erroring.

    **Invariant (see §2.2 / §3 D10): call only after ``reset_env()``.** The
    check is a real ``render()`` attempt, not a ``metadata["render_modes"]``
    lookup — the latter reports static class metadata and stays truthy even
    when ``pygame`` is missing (verified).  Raising is impossible here: all
    renderer failures are caught and mapped to ``False``.
    """
    env = getattr(self, "env", None)
    if env is None:
      return False
    can_render = getattr(env, "can_render", None)
    if callable(can_render):
      return bool(can_render())
    # External script envs without the helper: probe the raw API.
    # (This tolerates ResetNeeded, DependencyNotInstalled, any renderer error.)
    render = getattr(env, "render", None)
    if not callable(render):
      return False
    try:
      frame = render()
      return frame is not None
    except Exception:  # noqa: BLE001
      return False

def render_frame(self):
    """Return the latest rendered frame (numpy HxWx3 uint8) or ``None``."""
    env = getattr(self, "env", None)
    if env is None:
      return None
    render = getattr(env, "render", None)
    if callable(render):
      try:
        return render()
      except Exception:  # noqa: BLE001
        return None
    return None
```

**Change `init_env`** to thread the render mode through, and store an
`_video_enabled` flag on the pipeline so the trainer/methods know whether to
bother capturing:

```python
def init_env(self, args):
    from genml_kit.utils.logging import fatal
    from genml_kit.utils.script import extern_call

    if self.env is not None:
      return

    # D6/D7: rendering must be requested at gym.make() time.
    render_mode = ("rgb_array" if getattr(args, "record_eval_video", False)
                   else None)
    self._video_enabled = render_mode is not None

    if getattr(args, "env_script", None):
      self.env = extern_call(args.env_script, "make_env")
      # External envs are probed at validation time; we cannot force a
      # render_mode on them here.
    else:
      self.env = GymnasiumEnvWrapper(args.env_id, render_mode=render_mode)
    ...
    # (rest unchanged)
```

Also, declare `self._video_enabled = False` in `RLPipeline.__init__` (next to
`self.env = None`) so tests that bypass `init_env` have a defined value.

**Add the CLI flag** to `add_args` (after `--env-seed`, i.e. near
`rl.py:283`, grouping with the eval args):

```python
group.add_argument(
    "--record-eval-video",
    dest="record_eval_video",
    action="store_true",
    help="Record a video of each evaluation episode during validation "
         "(only if the environment supports rendering; MP4 via ffmpeg, "
         "GIF fallback).",
)
```

#### 4.2.3 `get_checkpoint_state` / `load_checkpoint_state` — no change needed

Video recording is a runtime/pure artifact; it does not affect model or buffer
state. (Verified: `get_checkpoint_state` at rl.py:495+ only saves obs_rms +
rollout buffer.)

### 4.3 New shared helper: `genml_kit/training/video_utils.py`

New module, 2-space indent, Google style. Single responsibility: frames → file.

```python
"""Small helpers to write evaluation videos for RL runs.

Kept dependency-light: MP4 via ``imageio[ffmpeg]`` when available, otherwise
animated GIF via ``PIL``.  Both writers are exercised by unit tests so the
recording path is testable in CI without a system ffmpeg binary.
"""

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def write_video(frames, path, fps=30):
  """Write *frames* (list of numpy HxWx3 uint8) to *path*.

  Format is chosen by file extension: ``.mp4`` uses imageio+ffmpeg (falling
  back to GIF if imageio is unavailable), ``.gif`` always uses PIL.

  Returns the path written, or ``None`` if no frame could be encoded
  (e.g. frames list is empty).
  """
  frames = [np.asarray(f) for f in frames if f is not None]
  if not frames:
    return None
  os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

  if path.endswith(".gif"):
    _write_gif(frames, path, fps=fps)
    return path

  try:
    _write_mp4(frames, path, fps=fps)
    return path
  except Exception as exc:  # noqa: BLE001 - degrade to GIF
    logger.warning("MP4 writer failed (%s); falling back to GIF.", exc)
    gif_path = os.path.splitext(path)[0] + ".gif"
    _write_gif(frames, gif_path, fps=fps)
    return gif_path


def _write_mp4(frames, path, fps=30):
  # Use the v3 `imageio` API: `imageio.v2` is deprecated and its shim can
  # disappear at our dependency floor (imageio==2.31). v3 exposes the same
  # `get_writer(path, fps=...)` surface.
  import imageio  # noqa: PLC0415 - lazy: optional dep
  with imageio.get_writer(path, fps=fps) as writer:
    for frame in frames:
      writer.append_data(frame)


def _write_gif(frames, path, fps=30):
  from PIL import Image  # noqa: PLC0415 - lazy import
  duration_ms = max(1, int(1000 / fps))
  images = [Image.fromarray(f) for f in frames]
  images[0].save(
      path,
      save_all=True,
      append_images=images[1:],
      duration=duration_ms,
      loop=0,
  )
```

Notes:

- `PLC0415` (import outside top-level) is **already ignored** in the repo's
  ruff config (`ignore = ["I001", "LOG015", "PLC0415", ...]`), so lazy imports
  are fine.
- The helper is placed under `genml_kit/training/` (not `utils/`) because it
  is trainer-facing and no `utils` consumer needs it; adjust if reviewer
  prefers `genml_kit/utils/`.
- Image dtype: gymnasium `rgb_array` frames are `uint8`; `PIL.Image.fromarray`
  requires it. (Do **not** cast floats to uint8 blindly — keep frames as-is
  from the env; video envs already return uint8.)

### 4.4 Methods — add `record_video` kwarg to `evaluate`

All three files change the same way. The key principle: **the method collects
frames via `pipeline.render_frame()` but does NOT write the file** — it returns
the frames in the metrics dict, and the trainer does the file IO. This keeps
methods side-effect-free and testable without touching disk.

**PPO** (`rl_ppo.py:223`):

```python
def evaluate(self, model, pipeline, num_episodes, max_steps=10_000,
             record_video=False):
    """Run evaluation episodes and return mean return.

    Args:
        model:          Policy model (actor + critic).
        pipeline:       ``RLPipeline`` providing ``reset_env``/``step_env``.
        num_episodes:   Number of evaluation episodes.
        max_steps:      Hard cap on environment steps PER EPISODE.
        record_video:   If True, capture ``render_frame()`` for each episode
                        and return them under ``metrics["episode_frames"]``
                        (list of lists; one inner list per episode).
    """
    total_return = 0.0
    total_steps = 0
    episode_frames = []
    for _ in range(num_episodes):
      obs = pipeline.reset_env()
      # D6/D10: probe *after* reset. can_record_video() performs a real
      # render() attempt (any renderer error -> False); cache the result so
      # frame grabs don't re-probe and so ResetNeeded / DependencyNotInstalled
      # are only ever hit inside the probe's try/except.
      episode_record = bool(record_video and pipeline.can_record_video())
      if record_video:
        frames = []
        if episode_record:
          frame = pipeline.render_frame()
          if frame is not None:
            frames.append(frame)
        episode_frames.append(frames)
      episode_return = 0.0
      done = False
      episode_steps = 0
      while not done and episode_steps < max_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        # ... existing act logic (unchanged) ...
        obs, reward, done, _ = pipeline.step_env(act_val)
        if episode_record:
          frame = pipeline.render_frame()
          if frame is not None:
            episode_frames[-1].append(frame)
        episode_return += reward
        episode_steps += 1
        total_steps += 1
      total_return += episode_return
    metrics = {
        "eval_return": total_return / max(num_episodes, 1),
        "eval_steps": total_steps,
    }
    if record_video:
      metrics["episode_frames"] = episode_frames
    return metrics
```

**SAC** (`rl_sac.py:336`) — identical diff except the act line:
`action = self.act(model, obs, deterministic=True)` (unchanged from current),
and same frame-capture insertions.

**DQN** (`rl_dqn.py:215`) — identical diff; the existing body already differs
only in docstring length. Preserve the existing docstring, append the
`record_video` line to it.

Contract of the kwarg:

- Default `record_video=False` → **byte-for-byte identical return dict** as
  today (no `episode_frames` key) → all existing callers/tests unaffected.
- When `True` but the env cannot render, `render_frame()` returns `None`, inner
  lists stay empty; the trainer checks for non-empty frames before writing.

### 4.5 `RLTrainer.validate()` — write the videos

`genml_kit/training/rl_trainer.py:354-376`. New version:

```python
def validate(self):
    """Policy evaluation on the environment (not a DataLoader).

    Returns:
      Scalar eval_return (float) for checkpoint selection.
    """
    env_seed = getattr(self.args, "env_seed", None)
    if env_seed is not None:
      old_state = np.random.get_state()
      np.random.seed(env_seed)
    else:
      old_state = None
    try:
      record = bool(getattr(self.args, "record_eval_video", False))
      metrics = self.method.evaluate(
          self.model,
          self.pipeline,
          getattr(self.args, "eval_episodes", 5),
          record_video=record,
      )
      if record:
        self._write_eval_videos(metrics.get("episode_frames"))
      return float(metrics["eval_return"])
    finally:
      if old_state is not None:
        np.random.set_state(old_state)

def _write_eval_videos(self, episode_frames):
    """Write one video per evaluated episode (if the env produced frames)."""
    if not episode_frames:
      return
    from genml_kit.training.video_utils import write_video  # lazy import

    video_dir = os.path.join(getattr(self.args, "checkpoint", "."), "videos")
    os.makedirs(video_dir, exist_ok=True)
    written = 0
    for idx, frames in enumerate(episode_frames, start=1):
      if not frames:
        continue
      # Prefer MP4; write_video falls back to GIF on failure.
      path = os.path.join(video_dir, f"eval_episode_{idx:06d}.mp4")
      out = write_video(frames, path, fps=30)
      if out is not None:
        written += 1
        logging.info("Wrote eval video: %s (%d frames)", out, len(frames))
    if written == 0:
      logging.warning(
          "--record-eval-video requested but the environment produced no "
          "renderable frames; skipping video recording.")
```

Notes:

- `logging` is already imported in `rl_trainer.py` (line ~9).
- `os` is **not** currently imported in `rl_trainer.py` — add `import os` at
  the top (verify import block, ruff `I` will enforce ordering).
- The `record_video` kwarg is **only** added to the three RL `evaluate`
  methods; `BaseMethod.evaluate` (`genml_kit/methods/base.py:90`) and other
  methods (classification/vo_pair) are untouched. Guard with `record_video=record`
  only in the RL trainer — which is the only caller of the RL evaluate
  signature (verified via `search_grep`).
- The RNG isolation (`np.random.get_state/set_state`) is preserved around the
  whole eval, so rendering does not perturb training sampling.

### 4.6 Docs

- `rl/README.md`: add a short subsection under evaluation describing
  `--record-eval-video`, output location (`<checkpoint>/videos/`), format
  precedence (MP4 → GIF), and the "only if env supports rendering" caveat.
- `README.md` (main): add the flag to the RL CLI reference if one exists
  (search for `--eval-episodes` in README first; mirror its formatting).
  (The main README is 80 KB — do **not** rewrite it wholesale; only patch the
  RL section.)

---

## 5. Test plan

Existing test conventions (verified):
- `tests/test_rl_pipeline.py` uses `_ScriptedEnv` (no gymnasium import needed)
  and `RLPipeline._make_scripted_env(...)`.
- `tests/test_rl_trainer.py` builds `FakeRLPipeline(RLPipeline)` subclasses
  with a scripted env; methods use `_ScriptedEnv`-backed fakes.

New/updated tests:

### 5.1 `tests/test_rl_pipeline.py` — render helpers

Add a small **renderable fake env** (no gymnasium dependency, so CI stays
green without `pygame`):

```python
class _RenderableScriptedEnv(_ScriptedEnv):
  """Scripted env that pretends to support rgb_array rendering."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._render_calls = 0

  def can_render(self):
    return True

  def render(self):
    self._render_calls += 1
    return np.zeros((64, 64, 3), dtype=np.uint8)
```

Tests:

1. `test_wrapper_render_mode_is_passed` — `GymnasiumEnvWrapper(env_id, render_mode="rgb_array")`
   sets `self._render_mode`; default `render_mode=None` → `can_render()` False.
   (This test **does not** need an actual gymnasium env; construct the wrapper
   in a way that avoids `gym.make` — or use `monkeypatch` on
   `gymnasium.make` to return the fake. Prefer monkeypatching to keep the test
   hermetic and fast.)
2. `test_pipeline_support_flag_default_off` — `RLPipeline()` without
   `init_env` → `can_record_video()` False, `render_frame()` None.
3. `test_pipeline_can_record_video_with_renderable_env` — after wiring
   `pipeline.env = _RenderableScriptedEnv()`, `can_record_video()` is True,
   `render_frame()` returns an HxWx3 uint8 array.

### 5.2 `tests/test_rl_video.py` (new)

1. `test_write_video_mp4_returns_path` — frames of `np.zeros((8,8,3), uint8)`;
   skip when the optional writer deps are absent: use
   `pytest.importorskip("imageio")` (the primary v3 API surface; the
   deprecated `imageio.v2` shim is intentionally NOT used — see §4.3/§8) —
   but still write the output under a `tmp_path` (pytest fixture), assert
   file exists and size > 0.
2. `test_write_video_gif_fallback` — monkeypatch `write_video`'s `_write_mp4`
   to raise; file extension `.mp4` input → asserts a `.gif` file is produced.
3. `test_write_video_empty_frames_returns_none` — `write_video([], ...)` → None.
4. `test_evaluate_collects_frames` — for each of DQN/PPO/SAC methods: run
   `method.evaluate(model, fake_pipeline, num_episodes=2, record_video=True)`
   with `fake_pipeline` backed by `_RenderableScriptedEnv`; assert
   `metrics["episode_frames"]` has 2 entries, each non-empty, and that
   `record_video=False` produces no `episode_frames` key.
5. `test_trainer_validate_writes_video` — `RLTrainer.validate()` with
   `args.record_eval_video=True` and a renderable fake env; assert a file
   exists under `<checkpoint>/videos/`. Reuse the `FakeRLPipeline` / fake-method
   scaffolding already in `tests/test_rl_trainer.py`.

Keep tests hermetic: **never** depend on `gymnasium`/`pygame`/`imageio`
being installed for the *logic* tests; only the video-writer tests use
`importorskip`. The GIF path (PIL) is always testable.

### 5.3 Regression safety

- Existing DQN/PPO/SAC `evaluate` tests pass unchanged because the new kwarg
  defaults to `False` and the metrics dict is unchanged.
- `RLTrainer.validate()` returns the same float; new behavior only activates
  when `--record-eval-video` is set.
- **New regression guard for the probe semantics (§2.2, §3 D6):** add a test
  that runs `evaluate(..., record_video=True)` against a fake env whose
  `render()` *raises* (simulating missing `pygame` / headless). Assert the run
  completes without raising and yields empty `episode_frames` — this locks in
  the "must not crash" guarantee that the metadata-probe draft would have
  broken.
- The `imageio` writer test must use `importorskip("imageio")` (see §4.3), so a
  partial install cannot silently skip the MP4 path coverage.

---

## 6. Ruff / yapf checklist

- Run `python -m yapf -i --style .style.yapf genml_kit/pipelines/rl.py
  genml_kit/methods/rl_ppo.py genml_kit/methods/rl_sac.py
  genml_kit/methods/rl_dqn.py genml_kit/training/rl_trainer.py
  genml_kit/training/video_utils.py tests/test_rl_pipeline.py
  tests/test_rl_video.py` (or the `format_file` tool with yapf + config).
- Run `python -m ruff check .` — must be clean. Config summary:
  `target-version = "py39"`, ignores include `I001`, `LOG015`, `PLC0415`,
  `RUF012`, `PLR0402`.
- Run `python -m pytest -q` — full suite **1118 tests currently pass**; new
  total must stay green (CI: `ruff check .` then `pytest -q`, Python 3.11;
  local sandbox runs Python 3.13.5 / pytest 9.1.1).

---

## 7. Manual verification (for whoever implements)

1. `pip install -e ".[dev,rl]"` (pulls `imageio[ffmpeg]` with the new extra).
2. Quick CartPole run showing a video is produced every validation and is
   non-trivial in size:
   ```bash
   genml-kit-train --pipeline rl --method dqn --env-id CartPole-v1 \
     --record-eval-video --epochs 2 --eval-episodes 1 --checkpoint /tmp/rlvid
   ```
   Expect: `<checkpoint>/videos/eval_episode_000001.mp4` exists; log line
   "Wrote eval video: ...".
3. Negative path — env without rendering (e.g. a custom `--env-script` whose
   `make_env` returns an object lacking `render`): run with the same flag,
   expect **no crash**, one warning, no videos dir (or empty).
4. Headless / missing-`pygame` path (the case that motivated the probe change,
   §2.2): uninstall `pygame` (or run where it is absent) and repeat step 2.
   Expect **no crash**: `can_render()` / `can_record_video()` performs a real
   `render()` attempt, catches `DependencyNotInstalled`, returns `False`, and no
   video is written. This is the scenario a `metadata["render_modes"]`-only
   check would have broken.
5. Frame-count sanity: with `--eval-episodes 1` on CartPole, the produced MP4
   should contain > 1 frame (initial frame after `reset` + one per `step_env`),
   i.e. a non-trivial video.
6. GIF fallback path: run with `imageio` absent (or `_write_mp4` forced to
   fail) and confirm the `.gif` file is produced instead — PIL is a hard
   dependency, so the fallback is always exercisable.

---

## 8. Open items / decision log (for tomorrow)

- [x] D1-D10 resolved (see §3).
- [ ] Confirm preferred home for `video_utils.py`:
      `genml_kit/training/` (plan default) vs `genml_kit/utils/`.
- [ ] Confirm whether to also expose `--eval-video-fps` (default 30) — the plan
      hardcodes 30; a flag is a trivial extension if desired.
- [ ] Confirm max video length: `max_steps` per episode already caps frames
      (10_000 default in the evaluate signature; CartPole rarely hits it).
      No extra cap planned.
- [ ] `imageio` API: prefer the v3 `import imageio; imageio.get_writer(...)`
      API, which is the supported surface for `imageio>=2.31` (our floor);
      `imageio.v2` is deprecated and its shim may be removed. Pick one at
      implementation time and make sure the writer unit test exercises it
      (see §4.3).
- [ ] README: locate the exact RL CLI section to patch (main README is big;
      rl/README.md has the primary RL docs).

---

## 9. Task-resume cheat-sheet (TL;DR for tomorrow)

1. **Approved plan** in `plans/RL_VIDEO.md` (this file).
2. Implement in this order:
   1. `pyproject.toml` rl extra += `imageio[ffmpeg]>=2.31`
   2. `video_utils.py` (new helper; use the `imageio` v3 API, not the
      deprecated `imageio.v2` shim; GIF fallback via PIL)
   3. `pipelines/rl.py` (wrapper `render_mode`/`render`/`can_render`;
      pipeline `can_record_video`/`render_frame`/`_video_enabled`;
      `init_env` render_mode wiring; `add_args` flag. Probe = **real
      `render()` attempt after reset**, NOT a `metadata` check)
   4. `methods/rl_{dqn,ppo,sac}.py` `evaluate(..., record_video=False)`
      (call `can_record_video()` after `reset_env()`, cache per episode)
   5. `training/rl_trainer.py` `validate()` + `_write_eval_videos()` + `import os`
   6. tests + docs
3. **Do NOT commit** — user reviews first; commit only after explicit approval.
4. Never amend/squash; no `Co-Authored-By`.
5. Verify: yapf (2-space), `ruff check .`, `pytest -q` (must stay all-green;
   currently 1118 passed).