"""SAC: Soft Actor-Critic (Haarnoja et al. 2018).

Phase 2 of ``plans/RL_PLAN.md`` (§6.6).  Implements the ``Method``
interface for maximum-entropy off-policy RL with:

- Twin Q-critics (hard-update targets).
- Reparameterization trick for continuous actions.
- Automatic temperature (alpha) tuning.

All math references ``rl/README.md`` Part 5 and §12–13.
"""

import copy
import math

import torch
import torch.nn as nn

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models.registry import load_model
from genml_kit.pipelines.contracts import LossOutput


@register_method
class SACMethod(Method):
  """Soft Actor-Critic (off-policy, continuous actions)."""

  NAME = "sac"
  METRIC_KEY = "eval_return"
  METRIC_MINIMIZE = False
  NEEDS_LABELS = False

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("sac method")
    group.add_argument(
        "--sac-gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    group.add_argument(
        "--sac-tau",
        type=float,
        default=0.005,
        help="Polyak coefficient for soft target updates (default: 0.005).",
    )
    group.add_argument(
        "--sac-alpha",
        type=float,
        default=0.2,
        help="Initial temperature alpha (default: 0.2).",
    )
    group.add_argument(
        "--sac-auto-alpha",
        action="store_true",
        default=True,
        help="Auto-tune alpha (default: True).",
    )
    group.add_argument(
        "--sac-no-auto-alpha",
        dest="sac_auto_alpha",
        action="store_false",
        help="Fix alpha (no auto-tuning).",
    )
    group.add_argument(
        "--sac-target-entropy",
        type=float,
        default=None,
        help="Target entropy for auto-alpha (default: -action_dim).",
    )
    group.add_argument(
        "--sac-critic-lr",
        type=float,
        default=3e-4,
        help="Critic learning rate (default: 3e-4).",
    )
    group.add_argument(
        "--sac-actor-lr",
        type=float,
        default=3e-4,
        help="Actor learning rate (default: 3e-4).",
    )
    group.add_argument(
        "--sac-alpha-lr",
        type=float,
        default=3e-4,
        help="Alpha learning rate (default: 3e-4).",
    )

  def wire_data(self, args, pipeline):
    self._pipeline = pipeline
    self._action_dim = pipeline.n_actions  # SAC always continuous
    self._env_steps = 0

  def build_model(self, args, device):
    self._gamma = getattr(args, "sac_gamma", 0.99)
    self._tau = getattr(args, "sac_tau", 0.005)
    self._auto_alpha = getattr(args, "sac_auto_alpha", True)

    # Temperature alpha.
    self._log_alpha = torch.tensor(
        math.log(getattr(args, "sac_alpha", 0.2)),
        device=device,
        requires_grad=True,
    )
    self._target_entropy = getattr(args, "sac_target_entropy", None)
    if self._target_entropy is None:
      self._target_entropy = -float(self._action_dim)

    # Actor (policy).
    actor = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=self._pipeline.obs_dim,
        action_dim=self._action_dim,
        discrete=False,
        device=device,
    )

    # Twin Q-critics (online + target copies).
    q1 = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=self._pipeline.obs_dim,
        action_dim=self._action_dim,
        discrete=False,
        device=device,
    )
    q2 = load_model(
        "rl/actor_critic",
        num_labels=0,
        obs_dim=self._pipeline.obs_dim,
        action_dim=self._action_dim,
        discrete=False,
        device=device,
    )

    # Targets are frozen deep-copies.
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    for p in q1_target.parameters():
      p.requires_grad = False
    for p in q2_target.parameters():
      p.requires_grad = False

    class _SACModel(nn.Module):
      def __init__(self, actor, q1, q2, q1_target, q2_target):
        super().__init__()
        self.actor = actor
        self.q1 = q1
        self.q2 = q2
        self.q1_target = q1_target
        self.q2_target = q2_target

      def forward(self, obs):
        raise NotImplementedError

    model = _SACModel(actor, q1, q2, q1_target, q2_target)
    model = self._apply_model_extras(args, model, device)

    # Build optimizers if not already built (for standalone testing)
    if not hasattr(self, "optimization") or self.optimization is None:
      self.optimization = self.build_optimization(args, model, device, {}, {})

    return model

  def _get_alpha(self):
    return self._log_alpha.exp().item()

  def build_optimization(self, args, model, device, ckpt_extra, states_to_load):
    """Build three separate optimizers for critic, actor, and alpha."""
    critic_lr = getattr(args, "sac_critic_lr", 3e-4)
    actor_lr = getattr(args, "sac_actor_lr", 3e-4)
    alpha_lr = getattr(args, "sac_alpha_lr", 3e-4)

    critic_params = list(model.q1.parameters()) + list(model.q2.parameters())
    actor_params = list(model.actor.parameters())
    alpha_params = [self._log_alpha]

    critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)
    actor_opt = torch.optim.Adam(actor_params, lr=actor_lr)
    alpha_opt = torch.optim.Adam(alpha_params, lr=alpha_lr)

    # Restore from checkpoint if available
    if ckpt_extra and "optim" in ckpt_extra:
      optim_state = ckpt_extra["optim"]
      if "critic_opt" in optim_state:
        critic_opt.load_state_dict(optim_state["critic_opt"])
      if "actor_opt" in optim_state:
        actor_opt.load_state_dict(optim_state["actor_opt"])
      if "alpha_opt" in optim_state:
        alpha_opt.load_state_dict(optim_state["alpha_opt"])

    # Return a custom optimization object
    class SACOptimization:
      def __init__(self, critic_opt, actor_opt, alpha_opt):
        self.critic_opt = critic_opt
        self.actor_opt = actor_opt
        self.alpha_opt = alpha_opt
        # For backward compatibility with logging etc.
        self.optimizer = critic_opt
        self.scheduler = None
        self.scaler = None

      def state_dict(self):
        return {
            "critic_opt": self.critic_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }

      def load_state_dict(self, state):
        self.critic_opt.load_state_dict(state["critic_opt"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.alpha_opt.load_state_dict(state["alpha_opt"])

    return SACOptimization(critic_opt, actor_opt, alpha_opt)

  def apply_grad(self, loss, scaler, amp_dtype, optimization):
    """Custom gradient application for three optimizers.

    The loss returned by train_step is the combined loss (for logging).
    Individual losses are in loss.metrics. We step each optimizer here.
    """
    # Get individual losses from metrics (they have gradients)
    critic_loss = loss.metrics["critic_loss"]
    actor_loss = loss.metrics["actor_loss"]
    alpha_loss = loss.metrics["alpha_loss"]

    # Step critic optimizer
    optimization.critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward(retain_graph=True)
    optimization.critic_opt.step()

    # Step actor optimizer
    optimization.actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward(retain_graph=True)
    optimization.actor_opt.step()

    # Step alpha optimizer (if auto_alpha)
    if self._auto_alpha:
      optimization.alpha_opt.zero_grad(set_to_none=True)
      alpha_loss.backward()
      optimization.alpha_opt.step()

  def act(self, model, obs, *, deterministic=False):
    """SAC acts by sampling from the squashed Gaussian policy."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      action, _, _, _ = model.actor.get_action_and_value(
          obs_t,
          deterministic=deterministic,
      )
    return action.squeeze(0).cpu().numpy()

  def step_epsilon(self):
    """SAC does not use epsilon — no-op."""
    pass

  def update_target(self, model, global_step):
    """Soft Polyak update for both target Q-critics."""
    with torch.no_grad():
      for p, pt in zip(model.q1.parameters(), model.q1_target.parameters()):
        pt.data.mul_(1.0 - self._tau).add_(p.data, alpha=self._tau)
      for p, pt in zip(model.q2.parameters(), model.q2_target.parameters()):
        pt.data.mul_(1.0 - self._tau).add_(p.data, alpha=self._tau)

  def train_step(self, model, blob, global_step, *, labels=None):
    """SAC update: twin critic loss + policy loss + alpha loss.

    Returns individual losses for the three optimizers. The trainer's
    apply_grad hook will step each optimizer separately.
    """
    data = blob if isinstance(blob, dict) else blob.data

    obs = data["obs"]
    rewards = data["reward"]
    next_obs = data["next_obs"]
    dones = data["done"]

    alpha = self._get_alpha()

    # --- Critic update (twin soft Q-learning) ---
    with torch.no_grad():
      next_action, next_log_prob, _, _ = model.actor.get_action_and_value(next_obs,)
      q1_next = model.q1_target.get_value(next_obs)
      q2_next = model.q2_target.get_value(next_obs)
      min_q_next = torch.min(q1_next, q2_next)
      soft_target = rewards + self._gamma * (1.0 - dones) * (min_q_next -
                                                             alpha * next_log_prob)

    # Twin Q losses.
    q1_pred = model.q1.get_value(obs)
    q2_pred = model.q2.get_value(obs)
    from genml_kit.losses.rl import sac_q_loss
    q1_loss = sac_q_loss(q1_pred, soft_target)
    q2_loss = sac_q_loss(q2_pred, soft_target)
    critic_loss = q1_loss + q2_loss

    # --- Actor update ---
    # Detach Q networks so their gradients don't flow back to critic
    for p in model.q1.parameters():
      p.requires_grad_(False)
    for p in model.q2.parameters():
      p.requires_grad_(False)

    new_action, new_log_prob, _, _ = model.actor.get_action_and_value(obs)
    q1_new = model.q1.get_value(obs)
    q2_new = model.q2.get_value(obs)
    min_q_new = torch.min(q1_new, q2_new)
    from genml_kit.losses.rl import sac_policy_loss
    actor_loss = sac_policy_loss(new_log_prob, min_q_new, alpha)

    # Re-enable critic gradients
    for p in model.q1.parameters():
      p.requires_grad_(True)
    for p in model.q2.parameters():
      p.requires_grad_(True)

    # --- Alpha update ---
    alpha_loss = torch.tensor(0.0, device=obs.device)
    if self._auto_alpha:
      from genml_kit.losses.rl import sac_alpha_loss
      alpha_loss = sac_alpha_loss(new_log_prob.detach(), self._target_entropy)

    # B5: track env steps for logging (1 env step per train_step in off-policy)
    self._env_steps += 1

    # Return combined loss (for logging) + individual losses in metrics
    total_loss = critic_loss + actor_loss + alpha_loss

    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": torch.tensor(alpha),
        "q_mean": q1_pred.detach().mean(),
    }
    return LossOutput(loss=total_loss, metrics=metrics)

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
        obs, reward, done, _ = pipeline.step_env(action)
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
        "method": "sac",
        "env_steps": self._env_steps,
        "log_alpha": self._log_alpha.item(),
    }

  def ckpt_extra(self, best, step):
    """Save three optimizers' state."""
    opt = self.optimization
    if hasattr(opt, "state_dict"):
      return {"optim": opt.state_dict()}
    return {}

  def load_checkpoint_state(self, model, state, args):
    self._env_steps = state.get("env_steps", 0)
    if "log_alpha" in state:
      self._log_alpha.data.fill_(state["log_alpha"])
