"""RL space specifications: ``SpaceSpec`` + ``space_spec`` producer.

The exact same *data-side* interrogation the pipeline's ``init_env``
performed inline (obs shape from ``observation_space.shape``, action
discreteness from ``action_space.n`` vs ``action_space.shape``) is
extracted into one pure function so it is testable in isolation and has
exactly one home.  The rule "a single action head for everything"
strictly maps ``space`` -> the flat scalars that every built-in factory
already forwards (``obs_dim`` / ``n_actions`` / ``action_dim``), keeping
the ``nn.Module`` classes free of gymnasium imports.
"""

from collections import namedtuple

# Fields
# -------
# obs_shape:   (4,) for a vector env; (3, 84, 84) for an image env.
# obs_dim:     ``prod(obs_shape)`` -- the MLP input size.
# n_actions:   int for a Discrete action space, None for continuous.
# action_dim:  int for a Box/continuous action space, None for discrete.
# is_discrete: True iff the action space is Discrete.
# action_low:  ``action_space.low`` for continuous Box spaces (may be
#              None for non-Box continuous spaces such as MultiDiscrete).
# action_high: ``action_space.high`` for continuous Box spaces.
SpaceSpec = namedtuple("SpaceSpec", [
    "obs_shape",
    "obs_dim",
    "n_actions",
    "action_dim",
    "is_discrete",
    "action_low",
    "action_high",
])


def space_spec(observation_space, action_space, obs_dim=None):
  """Build a ``SpaceSpec`` from gymnasium-style spaces.

  Args:
      observation_space: A space exposing a ``shape`` attribute (typical
          gymnasium ``Box`` / ``spaces``).  ``(4,)`` is a vector env,
          ``(3, 84, 84)`` an image env.
      action_space: A space exposing either ``n`` (Discrete) or
          ``shape`` (Box/continuous).  Spaces with neither attribute are
          treated as discrete with a fallback of 2 actions, mirroring
          the pipeline's current behaviour.
      obs_dim: Optional integer override.  When set, ``obs_shape`` is
          left as-is (the caller keeps the env's own shape) but
          ``obs_dim`` takes this value -- the ``--obs_dim`` escape hatch
          for environments whose observation space cannot be inferred.

  Returns:
      SpaceSpec: The fully-populated spec.

  Raises:
      ValueError: When *observation_space* exposes no inferrable
          ``shape`` and *obs_dim* is not given (same nudge message the
          pipeline used to emit).
  """
  shape = getattr(observation_space, "shape", None)
  if obs_dim is None:
    if shape is None:
      raise ValueError(f"Cannot infer obs_dim from observation space "
                       f"{observation_space!r}; pass --obs_dim explicitly")
    obs_dim = int(_prod(shape))
  else:
    obs_dim = int(obs_dim)

  n = getattr(action_space, "n", None)
  action_shape = getattr(action_space, "shape", None)
  low = getattr(action_space, "low", None)
  high = getattr(action_space, "high", None)
  if n is not None:
    n_actions = int(n)
    action_dim = None
    is_discrete = True
    low = high = None
  elif action_shape is not None:
    n_actions = None
    action_dim = int(action_shape[0])
    is_discrete = False
  else:
    # Mirror the pipeline fallback: no n and no shape -> discrete, 2 actions.
    n_actions = 2
    action_dim = None
    is_discrete = True
    low = high = None

  return SpaceSpec(
      obs_shape=shape,
      obs_dim=obs_dim,
      n_actions=n_actions,
      action_dim=action_dim,
      is_discrete=is_discrete,
      action_low=low,
      action_high=high,
  )


def _prod(shape):
  """Product of an iterable of ints (numpy-free, cheap)."""
  total = 1
  for dim in shape:
    total *= int(dim)
  return total
