"""Shared RL evaluation loop.

Extracted from ``rl_ppo.py``, ``rl_sac.py``, ``rl_dqn.py`` which each
contained a near-identical ``evaluate`` implementation.  The only
per-method hook is ``_eval_action(model, obs)`` which returns a scalar
(int) or 1-D numpy array suitable for ``pipeline.step_env``.
"""


def rl_evaluate(method,
                model,
                pipeline,
                num_episodes,
                max_steps=10_000,
                record_video=False):
  """Run evaluation episodes and return mean return.

  Args:
      method:        A ``Method`` subclass instance that implements
                     ``_eval_action(model, obs) -> action``.
      model:         The policy model.
      pipeline:      ``RLPipeline`` providing ``reset_env`` /
                     ``step_env`` / ``render_frame``.
      num_episodes:  Number of evaluation episodes.
      max_steps:     Maximum steps per episode.
      record_video:  Whether to capture frames for video.

  Returns:
      dict with ``eval_return`` (float), ``eval_steps`` (int),
      and optionally ``episode_frames`` (list of list of frames).
  """
  total_return = 0.0
  total_steps = 0
  episode_frames = []

  for _ in range(num_episodes):
    obs = pipeline.reset_env()
    # D6/D10: probe *after* reset (real render() attempt; any renderer
    # error -> False); cache so frame grabs don't re-probe.
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
      action = method._eval_action(model, obs)
      obs, reward, done, _ = pipeline.step_env(action)
      episode_return += reward
      episode_steps += 1
      total_steps += 1
      if record_video and episode_record:
        frame = pipeline.render_frame()
        if frame is not None:
          episode_frames[-1].append(frame)
    total_return += episode_return

  metrics = {
      "eval_return": total_return / max(num_episodes, 1),
      "eval_steps": total_steps,
  }
  if record_video:
    metrics["episode_frames"] = episode_frames
  return metrics
