"""Reinforcement-learning models (Q-networks, actor-critics).

The module imports are for registry side-effects (model factories are
registered as ``rl/...``); the classes are also re-exported here so
``from genml_kit.models.rl import QNetwork`` works.
"""

from genml_kit.models.rl.actor_critic import ActorCritic
from genml_kit.models.rl.qnetwork import DuelingQHead, QHead, QNetwork
from genml_kit.models.rl.sac_critic import SACCritic
from genml_kit.models.rl.sac_model import SACModel

__all__ = [
    "ActorCritic",
    "DuelingQHead",
    "QHead",
    "QNetwork",
    "SACCritic",
    "SACModel",
]
