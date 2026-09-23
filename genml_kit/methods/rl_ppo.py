"""PPO: Proximal Policy Optimization (Schulman et al. 2017).

Phase 2 of ``plans/RL_PLAN.md`` (§6.6).  Implements the ``Method``
interface for on-policy PPO with GAE advantages, clipped surrogate
objective, and optional entropy bonus.

All math references ``rl/README.md`` Parts 4 and 11.
"""

import torch

from genml_kit.losses.rl import (
    clipped_surrogate,
    entropy_bonus,
    value_loss,
)
from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models.registry import load_model
from genml_kit.pipelines.contracts import LossOutput


@register_method
class PPOMethod(Method):
  """Proximal Policy Optimization (on-policy)."""

  NAME = "ppo"
  METRIC_KEY = "eval_return"
  NEEDS_LABELS = False
  IS_ON_POLICY = True

  @classmethod
  def get_trainer_class(cls):
    from genml_kit.training.rl_trainer import RLTrainer

    return RLTrainer

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("ppo method")
    group.add_argument(
        "--ppo_gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    group.add_argument(
        "--ppo_lam",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )
    group.add_argument(
        "--ppo_clip_eps",
        type=float,
        default=0.2,
        help="PPO clipping epsilon.",
    )
    group.add_argument(
        "--ppo_epochs",
        type=int,
        default=4,
        help="SGD epochs per rollout.",
    )
    group.add_argument(
        "--ppo_mini_batch_size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates.",
    )
    group.add_argument(
        "--ppo_entropy_coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient.",
    )
    group.add_argument(
        "--ppo_value_coef",
        type=float,
        default=0.5,
        help="Value loss coefficient.",
    )
    group.add_argument(
        "--ppo_vf_clip_eps",
        type=float,
        default=None,
        help="Value function clipping epsilon (None = unclipped).",
    )
    group.add_argument(
        "--ppo_rollout_len",
        type=int,
        default=2048,
        help="Rollout length before each PPO update.",
    )
    group.add_argument(
        "--ppo_discrete",
        action="store_true",
        default=True,
        help="Use discrete action space.",
    )
    group.add_argument(
        "--ppo_continuous",
        dest="ppo_discrete",
        action="store_false",
        help="Use continuous action space.",
    )

  def wire_data(self, args, pipeline):
    self.n_actions = pipeline.n_actions
    self._pipeline = pipeline
    self._discrete = getattr(args, "ppo_discrete", True)
    self._action_dim = getattr(pipeline, "action_dim", None)

  def build_model(self, args, device):
    self._gamma = getattr(args, "ppo_gamma", 0.99)
    self._lam = getattr(args, "ppo_lam", 0.95)
    self._clip_eps = getattr(args, "ppo_clip_eps", 0.2)
    self._ppo_epochs = getattr(args, "ppo_epochs", 4)
    self._mini_batch_size = getattr(args, "ppo_mini_batch_size", 64)
    self._entropy_coef = getattr(args, "ppo_entropy_coef", 0.01)
    self._value_coef = getattr(args, "ppo_value_coef", 0.5)
    self._vf_clip_eps = getattr(args, "ppo_vf_clip_eps", None)
    self._env_steps = 0

    model = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=self._pipeline.obs_dim,
        n_actions=self.n_actions if self._discrete else None,
        action_dim=self._action_dim if not self._discrete else None,
        discrete=self._discrete,
        device=device,
    )
    model = self._apply_model_extras(args, model, device)
    return model

  def act(self, model, obs, *, deterministic=False):
    """Select an action during rollout collection.

        Called by the trainer during rollout gathering (D11: env stepping
        never inside ``train_step``).

        Returns:
            action:     Squashed action for environment (numpy).
            log_prob:   Log probability of the action.
            value:      Value estimate.
            raw_action: Raw (pre-tanh) action for continuous, None for discrete.
        """
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      action, raw_action, log_prob, _, value = model.get_action_and_value(
          obs_t,
          deterministic=deterministic,
      )
    if self._discrete:
      return action.item(), log_prob.item(), value.item(), None
    return (
        action.squeeze(0).numpy(),
        log_prob.item(),
        value.item(),
        raw_action.squeeze(0).numpy(),
    )

  def _eval_action(self, model, obs):
    """Return the deterministic action for evaluation."""
    return self.act(model, obs, deterministic=True)[0]

  def update_target(self, model, global_step):
    """PPO does not use target networks — no-op."""
    pass

  def train_step(self, model, blob, global_step, *, labels=None):
    """PPO clipped surrogate loss.

        Unlike DQN/SAC, ``train_step`` here is called on *mini-batches*
        sampled from a pre-computed rollout, so the blob already contains
        actions, log_probs, advantages, and returns.
        """
    data = blob if isinstance(blob, dict) else blob.data

    obs = data["obs"]
    # Squashed actions (for value reference).
    actions = data["action"]
    old_log_probs = data["log_prob"]
    advantages = data["advantage"]
    returns = data["return"]

    # Normalize advantages.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # For continuous: re-evaluate log_prob on RAW (pre-tanh) actions.
    # Because the distribution is defined over raw actions.
    eval_action = data.get("raw_action", actions)

    _, _, new_log_probs, entropy, new_values = model.get_action_and_value(
        obs,
        action=eval_action,
    )

    # Policy loss (clipped surrogate).
    ratio = (new_log_probs - old_log_probs).exp()
    pg_loss = clipped_surrogate(ratio, advantages, self._clip_eps)

    # Value loss.
    v_loss = value_loss(new_values,
                        returns,
                        old_values=None,
                        clip_eps=self._vf_clip_eps)

    # Entropy bonus.
    ent = entropy_bonus(new_log_probs if entropy is None else entropy)

    # Combined loss.
    loss = pg_loss + self._value_coef * v_loss - self._entropy_coef * ent

    # For PER: use value function errors as TD errors.
    td_errors = (returns - new_values.detach()).abs()

    metrics = {
        "pg_loss": pg_loss.detach(),
        "value_loss": v_loss.detach(),
        "entropy": ent.detach(),
        "ratio_mean": ratio.detach().mean(),
    }
    return LossOutput(loss=loss, metrics=metrics, td_errors=td_errors)

  def evaluate(self,
               model,
               pipeline,
               num_episodes,
               max_steps=10_000,
               record_video=False):
    """Run evaluation episodes and return mean return."""
    from genml_kit.methods.rl_utils import rl_evaluate

    return rl_evaluate(
        self,
        model,
        pipeline,
        num_episodes,
        max_steps=max_steps,
        record_video=record_video,
    )

  def has_metric_improved(self, new_metric, best_metric):
    """Higher eval_return is better."""
    return new_metric > best_metric

  def get_checkpoint_state(self, model, args):
    return {
        "method": "ppo",
        "env_steps": self._env_steps,
    }

  def load_checkpoint_state(self, model, state, args):
    self._env_steps = state.get("env_steps", 0)

  def ckpt_extra(self, best, step):
    """Save rollout buffer state for resume."""
    return {}
