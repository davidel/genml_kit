"""PPO: Proximal Policy Optimization (Schulman et al. 2017).

Phase 2 of ``plans/RL_PLAN.md`` (§6.6).  Implements the ``Method``
interface for on-policy PPO with GAE advantages, clipped surrogate
objective, and optional entropy bonus.

All math references ``rl/README.md`` Parts 4 and 11.
"""

import torch

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

  @classmethod
  def get_trainer_class(cls):
    from genml_kit.training.rl_trainer import RLTrainer

    return RLTrainer

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("ppo method")
    group.add_argument(
        "--ppo-gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    group.add_argument(
        "--ppo-lam",
        type=float,
        default=0.95,
        help="GAE lambda (default: 0.95).",
    )
    group.add_argument(
        "--ppo-clip-eps",
        type=float,
        default=0.2,
        help="PPO clipping epsilon (default: 0.2).",
    )
    group.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="SGD epochs per rollout (default: 4).",
    )
    group.add_argument(
        "--ppo-mini-batch-size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates (default: 64).",
    )
    group.add_argument(
        "--ppo-entropy-coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient (default: 0.01).",
    )
    group.add_argument(
        "--ppo-value-coef",
        type=float,
        default=0.5,
        help="Value loss coefficient (default: 0.5).",
    )
    group.add_argument(
        "--ppo-vf-clip-eps",
        type=float,
        default=None,
        help="Value function clipping epsilon (default: None = unclipped).",
    )
    group.add_argument(
        "--ppo-rollout-len",
        type=int,
        default=2048,
        help="Rollout length before each PPO update (default: 2048).",
    )
    group.add_argument(
        "--ppo-discrete",
        action="store_true",
        default=True,
        help="Use discrete action space (default: True).",
    )
    group.add_argument(
        "--ppo-continuous",
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
    from genml_kit.losses.rl import clipped_surrogate

    pg_loss = clipped_surrogate(ratio, advantages, self._clip_eps)

    # Value loss.
    from genml_kit.losses.rl import value_loss

    v_loss = value_loss(new_values,
                        returns,
                        old_values=None,
                        clip_eps=self._vf_clip_eps)

    # Entropy bonus.
    from genml_kit.losses.rl import entropy_bonus

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
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
          action, _, _, _, _ = model.get_action_and_value(
              obs_t,
              deterministic=True,
          )
        act_val = action.item() if self._discrete else action.squeeze(0).numpy()
        obs, reward, done, _ = pipeline.step_env(act_val)
        episode_return += reward
        episode_steps += 1
        total_steps += 1
      total_return += episode_return
    return {
        "eval_return": total_return / max(num_episodes, 1),
        "eval_steps": total_steps,
    }

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
