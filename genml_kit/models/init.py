"""Generic parameter-initialization helpers (SB3-style orthogonal init).

Stable-Baselines3 initializes every linear layer of its PPO/SAC policies
with orthogonal init -- gain ``sqrt(2)`` for policy/hidden layers and
``1.0`` for value/head layers, zero biases.  This is a load-bearing
"trick": SB3's hyperparameters are tuned around it, and it fixes the
*scale* of the initial policy gradient on continuous control.

The helper here is deliberately *generic* (works on any ``nn.Module``),
so it can be applied to RL and non-RL models alike via the ``--param_init``
flag wired in ``Method._apply_model_extras``.
"""

import math

from torch import nn

# SB3's default gains: sqrt(2) for policy/hidden layers, 1.0 for
# value/head layers.  Module-level constants avoid function calls in
# argument defaults (ruff B008).
_GAIN_POLICY = math.sqrt(2.0)
_GAIN_VALUE = 1.0


def init_orthogonal(module, gain_policy=_GAIN_POLICY, gain_value=_GAIN_VALUE):
  """Apply SB3-style orthogonal init across ``module``.

  Args:
      module: Any ``nn.Module`` (e.g. an ``ActorCritic`` or an MLP).
      gain_policy: Gain for policy / hidden layers.  ``sqrt(2)`` matches
          SB3's PPO/SAC actor MLP.
      gain_value: Gain for value / head layers.  ``1.0`` matches SB3's
          critic/head.

  Returns:
      The same ``module`` (for chaining).
  """
  for name, child in module.named_modules():
    if isinstance(child, nn.Linear):
      # Policy/actor and intermediate MLP layers use the larger gain;
      # critic / value / head layers use the smaller one.
      gain = gain_value if "critic" in name or "value" in name else gain_policy
      nn.init.orthogonal_(child.weight, gain=gain)
      nn.init.zeros_(child.bias)
  return module
