"""Deterministic scripted environment for RL unit tests.

Test scaffolding (no gymnasium control tasks needed): lives under ``tests/``
rather than in ``genml_kit`` so the production ``RLPipeline`` module carries no
test-only code.  Imported by the RL test-suite the same way ``tests/cli_args.py``
and ``tests/iterable_streams.py`` are.
"""

import torch


class _ScriptedEnv:
  """Deterministic scripted env for unit testing (no gymnasium required).

  A simple 4-state chain: state 0 → 1 → 2 → 3 → done.
  Action 0 = advance; action 1 = stay.
  Episodes are also forcibly terminated after *max_episode_length* steps
  to prevent infinite loops with untrained policies.

  In ``continuous`` mode the env exposes a ``Box`` action space and rewards
  the policy by how close its action is to a fixed *target*
  (``reward = 1.0 - abs(action[0] - target)``).  This gives continuous
  algorithms a dense, reachable optimum without depending on gymnasium
  control tasks.  The discrete state-machine semantics are unchanged.
  """

  def __init__(self,
               obs_dim=4,
               max_episode_length=10,
               continuous=False,
               action_dim=2,
               target=0.0):
    self.obs_dim = obs_dim
    self._state = 0
    self._step_count = 0
    self._max_episode_length = max_episode_length
    self._continuous = continuous
    self._action_dim = action_dim
    self._target = target
    self.observation_space = type("S", (), {
        "shape": (obs_dim,),
    })()
    if continuous:
      import gymnasium as gym
      import numpy as np
      self.action_space = gym.spaces.Box(low=-1.0,
                                         high=1.0,
                                         shape=(action_dim,),
                                         dtype=np.float32)
    else:
      self.action_space = type("A", (), {
          "n": 2,
      })()

  def reset(self):
    self._state = 0
    self._step_count = 0
    return self._obs()

  def step(self, action):
    self._step_count += 1
    import numpy as np
    if self._continuous:
      # Continuous action: accept array, use first element for logic
      if hasattr(action, "__len__"):
        act_val = float(np.asarray(action).flat[0])
      elif hasattr(action, "item"):
        act_val = float(action.item())
      else:
        act_val = float(action)
      # Dense reward: optimal when act_val == target (max 1.0).
      reward = 1.0 - abs(act_val - self._target)
    else:
      if hasattr(action, "__len__"):
        act_val = int(np.asarray(action).flat[0])
      elif hasattr(action, "item"):
        act_val = int(action.item())
      else:
        act_val = int(action)
      if act_val == 0 and self._state < 3:
        self._state += 1
      reward = 1.0 if self._state == 3 else 0.0
    done = self._state == 3 or self._step_count >= self._max_episode_length
    return self._obs(), reward, done, {}

  def close(self):
    pass

  def _obs(self):
    obs = torch.zeros(self.obs_dim)
    obs[self._state] = 1.0
    return obs.numpy()
