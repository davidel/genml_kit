"""SAC model container (actor + twin Q-critics + target networks).

This module separates the model definition from the SAC method logic,
following the same pattern as QNetwork for DQN and ActorCritic for PPO.
"""

import torch.nn as nn


class SACModel(nn.Module):
  """Container for SAC actor, twin critics, and their target networks."""

  def __init__(self, actor, q1, q2):
    super().__init__()
    self.actor = actor
    self.q1 = q1
    self.q2 = q2

  def hard_update(self):
    """Hard update target networks: target = online."""
    self.q1.hard_update()
    self.q2.hard_update()

  def soft_update(self, tau):
    """Soft Polyak update: target = (1-tau)*target + tau*online."""
    self.q1.soft_update(tau)
    self.q2.soft_update(tau)
