"""Shared RL base class and evaluation loop.

``RLMethod`` holds behaviour common to ``rl_ppo.py``, ``rl_sac.py`` and
``rl_dqn.py``.  ``rl_evaluate`` was extracted from those three, which each
contained a near-identical ``evaluate`` implementation; its only per-method
hook is ``_eval_action(model, obs)``, which returns a scalar (int) or 1-D
numpy array suitable for ``pipeline.step_env``.
"""

import collections

from genml_kit.methods.base import Method
from genml_kit.utils.shared_args import DEFAULT_VISION_MODEL

# What the RL methods' ``act`` returns during rollout collection: the squashed
# action handed to the environment (numpy for continuous, int for discrete),
# its log-probability, the value estimate, and the raw pre-tanh action
# (``None`` for discrete).
ActOutput = collections.namedtuple("ActOutput",
                                   ["action", "log_prob", "value", "raw_action"])


class RLMethod(Method):
  """Shared base for RL methods (DQN, PPO, SAC).

  The only difference from :class:`Method` today is that RL models are
  deliberately **not** DDP-wrapped.

  ``DistributedDataParallel`` reduces gradients only in ``forward``.  Every RL
  method enters autograd through a *method* on the model instead --
  ``model.get_action_and_value(...)`` (PPO), ``model.online(...)`` (DQN),
  ``model.q1.get_value(...)`` (SAC).  The ``DistributedModel`` facade forwards
  those attribute accesses to the unwrapped semantic model, so no reducer hook
  is ever installed: under DDP each rank would silently train its own
  divergent model rather than erroring.

  Skipping the wrap therefore costs nothing here and avoids hiding the
  problem.  The RL trainer reduces gradients explicitly instead
  (:func:`genml_kit.utils.distributed.all_reduce_gradients`), which is
  independent of how a method reaches its parameters.
  """

  #: ``--model_arg`` keys owned by a specific RL method and therefore stripped
  #: from the kwargs forwarded to the model factory.  Subclasses override.
  MODEL_ARG_EXCLUDES = frozenset()

  def wrap_model(self, model, device, args):
    """Return *model* unwrapped; RL reduces gradients explicitly.

        See the class docstring for why a DDP wrap is wrong for these methods.
        """
    return model

  def resolve_model_name(self, args, registered):
    """Return the model factory name for an RL run.

        ``--model`` selects the registered factory.  When it is unset, or left
        at the shared vision default (:data:`DEFAULT_VISION_MODEL`) that no RL
        policy can load, fall back to *registered* (the method's own factory).
        """
    if not args.model or args.model == DEFAULT_VISION_MODEL:
      return registered
    return args.model

  def model_kwargs(self, args):
    """Return ``--model_arg`` minus this method's own (``MODEL_ARG_EXCLUDES``)."""
    return {k: v for k, v in args.model_arg.items() if k not in self.MODEL_ARG_EXCLUDES}


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
      step_out = pipeline.step_env(action)
      obs, reward, done = step_out.obs, step_out.reward, step_out.done
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
