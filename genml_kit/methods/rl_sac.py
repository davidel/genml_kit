"""SAC: Soft Actor-Critic (Haarnoja et al. 2018).

Phase 2 of ``plans/RL_PLAN.md`` (§6.6).  Implements the ``Method``
interface for maximum-entropy off-policy RL with:

- Twin Q-critics (hard-update targets).
- Reparameterization trick for continuous actions.
- Automatic temperature (alpha) tuning.

All math references ``rl/README.md`` Part 5 and §12–13.
"""

import math

import torch

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.models.rl.sac_model import SACModel
from genml_kit.utils.attr import get_attribute, MISSING


@METHODS.register
class SACMethod(Method):
  """Soft Actor-Critic (off-policy, continuous actions).

  Implementation invariants
  -------------------------
  Three details are load-bearing and easy to "clean up" into a broken
  algorithm; they were each the cause of a real training failure and are
  guarded by regression tests in ``tests/test_rl_sac.py``:

  1. **The actor loss must see** ``dQ/da``.  ``train_step`` intentionally
     does *not* wrap the actor's Q evaluation in ``torch.no_grad`` — the
     whole point of the reparameterised policy gradient is to push the
     action in the direction that raises Q.  Freeze the critic
     *parameters* (``requires_grad_(False)``) instead, which stops
     gradient flowing into the critic weights while keeping the graph
     from ``action -> Q`` intact.  Wrapping (only) the Q call in
     ``no_grad`` silently reduces SAC to an entropy-only update: the
     policy never learns the task and ``q_mean`` drifts monotonically.
  2. **All backwards before any optimizer step.**  Because (1) makes the
     actor graph reference the critic networks, the critic optimizer must
     not ``step()`` until the actor ``backward()`` has run — otherwise
     autograd raises "a variable needed for gradient computation has been
     modified by an inplace operation".  See ``apply_grad``.
  3. **The temperature loss is parameterised by** ``log_alpha``, not
     ``alpha``.  ``sac_alpha_loss`` receives the optimised parameter so
     that ``dL/dlog_alpha`` does not itself depend on alpha; passing
     ``alpha = exp(log_alpha)`` makes the gradient shrink towards zero and
     drives alpha to a spurious ``~0`` fixed point (entropy term
     vanishes; SAC degenerates toward DDPG).
  """

  NAME = "sac"
  METRIC_KEY = "eval_return"
  NEEDS_LABELS = False

  @classmethod
  def get_trainer_class(cls):
    from genml_kit.training.rl_trainer import RLTrainer

    return RLTrainer

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("sac method")
    group.add_argument(
        "--sac_gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    group.add_argument(
        "--sac_tau",
        type=float,
        default=0.005,
        help="Polyak coefficient for soft target updates.",
    )
    group.add_argument(
        "--sac_alpha",
        type=float,
        default=0.2,
        help="Initial temperature alpha.",
    )
    group.add_argument(
        "--sac_auto_alpha",
        action="store_true",
        default=True,
        help="Auto-tune alpha.",
    )
    group.add_argument(
        "--sac_no_auto_alpha",
        dest="sac_auto_alpha",
        action="store_false",
        help="Fix alpha (no auto-tuning).",
    )
    group.add_argument(
        "--sac_target_entropy",
        type=float,
        default=None,
        help="Target entropy for auto-alpha.",
    )
    group.add_argument(
        "--sac_critic_lr",
        type=float,
        default=3e-4,
        help="Critic learning rate.",
    )
    group.add_argument(
        "--sac_actor_lr",
        type=float,
        default=3e-4,
        help="Actor learning rate.",
    )
    group.add_argument(
        "--sac_alpha_lr",
        type=float,
        default=3e-4,
        help="Alpha learning rate.",
    )

  def wire_data(self, args, pipeline):
    self._pipeline = pipeline
    # SAC is a continuous-action algorithm (Gaussian policy + twin
    # Q-critics over (obs, action) vectors).  Refuse discrete envs
    # loudly instead of silently building a bogus continuous policy.
    action_shape = get_attribute(pipeline, "action_space.shape")
    if action_shape is MISSING:
      action_shape = get_attribute(pipeline, "env.action_space.shape")
    is_continuous = action_shape is not MISSING and action_shape != ()
    if not is_continuous:
      from genml_kit.utils.logging import fatal

      fatal(
          "SAC supports continuous action spaces only.  Use --method dqn "
          "for discrete environments or --ppo-continuous for a continuous "
          "PPO run.",
          ValueError,
      )
    self._action_dim = int(action_shape[0])
    self._env_steps = 0

  def build_model(self, args, device):
    self._gamma = getattr(args, "sac_gamma", 0.99)
    self._tau = getattr(args, "sac_tau", 0.005)
    self._auto_alpha = getattr(args, "sac_auto_alpha", True)
    # Global grad-norm clip for the three SAC optimizers (0 disables).
    self._grad_clip = getattr(args, "grad_clip", 0.0) or 0.0

    # Temperature alpha.  The *learned* parameter is ``log_alpha`` (its
    # exponential is the temperature used everywhere else); optimising the
    # log keeps alpha strictly positive and, crucially, makes the alpha
    # loss well-conditioned (see invariant 3 in the class docstring).
    self._log_alpha = torch.tensor(
        math.log(getattr(args, "sac_alpha", 0.2)),
        device=device,
        requires_grad=True,
    )
    self._target_entropy = getattr(args, "sac_target_entropy", None)
    if self._target_entropy is None:
      self._target_entropy = -float(self._action_dim)

    # Build SAC model container (actor + twin critics + targets).
    model = SACModel.build(
        obs_dim=self._pipeline.obs_dim,
        action_dim=self._action_dim,
        hidden_dim=256,
    ).to(device)
    return model

  def _get_alpha(self):
    """Return the temperature ``alpha = exp(log_alpha)`` (> 0 always).

    Note: this value is used in the *critic target* and *actor loss*, but
    the *alpha loss* must be given ``self._log_alpha`` directly instead
    (see invariant 3 in the class docstring).
    """
    return self._log_alpha.exp()

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

    # Restore from checkpoint if available.
    # The checkpoint stores the full SACOptimization state_dict under
    # "optimizer_state_dict" (saved by CheckpointSaver via
    # SACOptimization.state_dict()).
    if ckpt_extra and "optimizer_state_dict" in ckpt_extra:
      optim_state = ckpt_extra["optimizer_state_dict"]
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

        Ordering is critical (invariant 2 in the class docstring): every
        ``backward`` must complete before the first ``step``.  The actor
        loss graph holds references into the critic networks (from
        ``get_value`` in ``train_step``), so stepping any optimizer in
        between backwards mutates parameters that a pending backward still
        needs and autograd aborts with an in-place-modified error.
        """
    # Get individual losses from metrics (they have gradients)
    critic_loss = loss.metrics["critic_loss"]
    actor_loss = loss.metrics["actor_loss"]
    alpha_loss = loss.metrics["alpha_loss"]

    # Zero all three parameter groups first.  They are distinct parameter
    # sets, so a single zeroing pass per optimiser is sufficient.
    optimization.critic_opt.zero_grad(set_to_none=True)
    optimization.actor_opt.zero_grad(set_to_none=True)
    optimization.alpha_opt.zero_grad(set_to_none=True)

    # Phase 1: ALL backward passes, no optimizer steps in between (see the
    # ordering note in this method's docstring).  Each backward accumulates
    # into the .grad buffers of its own parameter group; the alpha backward
    # only touches ``log_alpha``, so the three graphs do not interfere.
    critic_loss.backward()
    actor_loss.backward()
    if self._auto_alpha:
      alpha_loss.backward()

    # Phase 2: clipping, before stepping.  The base trainer only clips in
    # the supervised/image path; SAC steps its own optimisers, so without
    # this the ``--grad_clip`` flag would silently have no effect and
    # critic-loss spikes would hit the weights unchecked.
    if self._grad_clip > 0:
      self._clip_optimizer(optimization.critic_opt)
      self._clip_optimizer(optimization.actor_opt)
      if self._auto_alpha:
        self._clip_optimizer(optimization.alpha_opt)

    # Phase 3: now that all grads are computed and clipped, step.
    optimization.critic_opt.step()
    optimization.actor_opt.step()
    if self._auto_alpha:
      optimization.alpha_opt.step()

  def _clip_optimizer(self, optimizer):
    """Clip the grad norm of every param group owned by *optimizer*."""
    params = [p for group in optimizer.param_groups for p in group["params"]]
    torch.nn.utils.clip_grad_norm_(params, self._grad_clip)

  def act(self, model, obs, *, deterministic=False):
    """SAC acts by sampling from the squashed Gaussian policy."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
      # model.actor is ActorCritic, which has get_action_and_value
      action, _, _, _, _ = model.actor.get_action_and_value(
          obs_t,
          deterministic=deterministic,
      )
    # Return as numpy array (1D for continuous actions)
    action_np = action.squeeze(0).cpu().numpy()
    # Ensure it's a flat 1D array
    return action_np.flatten()

  def _eval_action(self, model, obs):
    """Return the deterministic action for evaluation."""
    return self.act(model, obs, deterministic=True)

  def step_epsilon(self):
    """SAC does not use epsilon — no-op."""
    pass

  def update_target(self, model, global_step):
    """Soft Polyak update for both target Q-critics."""
    model.soft_update(self._tau)

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
    # True MDP-end flag (Gymnasium 'terminated'); falls back to done
    # when the env / buffer does not distinguish truncation.
    terminated = data.get("terminated", dones)

    alpha = self._get_alpha()

    # --- Critic update (twin soft Q-learning) ---
    with torch.no_grad():
      next_action, _, next_log_prob, _, _ = model.actor.get_action_and_value(next_obs)
      # Target networks are plain Sequential modules; call forward directly.
      q1_next = model.q1.target(torch.cat([next_obs, next_action], dim=-1)).squeeze(-1)
      q2_next = model.q2.target(torch.cat([next_obs, next_action], dim=-1)).squeeze(-1)
      min_q_next = torch.min(q1_next, q2_next)
      # Use detached alpha for critic target to avoid gradient conflicts.
      alpha_detached = alpha.detach()
      # Bootstrap mask from the true-termination flag so a truncated step
      # (dones=1, terminated=0) still bootstraps gamma * V(s').
      soft_target = rewards + self._gamma * (1.0 - terminated) * (
          min_q_next - alpha_detached * next_log_prob)

    # Twin Q losses.
    action = data["action"]
    q1_pred = model.q1.get_value(obs, action)
    q2_pred = model.q2.get_value(obs, action)
    from genml_kit.losses.rl import sac_q_loss

    q1_loss = sac_q_loss(q1_pred, soft_target)
    q2_loss = sac_q_loss(q2_pred, soft_target)
    critic_loss = q1_loss + q2_loss

    # --- Actor update ---
    # Freeze the critic *parameters* (not the graph!).  ``requires_grad_
    # (False)`` on the weights stops actor-loss gradient from reaching the
    # critic optimiser's parameters, while leaving ``dQ/da`` intact below.
    # Do NOT replace this with ``torch.no_grad()``/``.detach()`` on the Q
    # values: that also severs ``dQ/da`` and the policy silently loses the
    # task gradient (see invariant 1 in the class docstring).
    for p in model.q1.net.parameters():
      p.requires_grad_(False)
    for p in model.q2.net.parameters():
      p.requires_grad_(False)

    new_action, _, new_log_prob, _, _ = model.actor.get_action_and_value(obs)
    # CRITICAL: intentionally NOT under ``torch.no_grad``.  ``new_action``
    # is a reparameterised sample (rsample -> tanh), so evaluating Q here
    # builds the chain ``action -> Q`` that the policy gradient needs in
    # order to raise Q.  The critic weights are frozen just above, so this
    # adds gradient *only* to the actor, never to the critic optimiser.
    # If this block were wrapped in no_grad, ``actor_loss`` would reduce to
    # ``alpha * log_prob``: the policy would only be regularised towards
    # entropy and never optimise return (the original bug).
    q1_new = model.q1.get_value(obs, new_action)
    q2_new = model.q2.get_value(obs, new_action)
    min_q_new = torch.min(q1_new, q2_new)
    from genml_kit.losses.rl import sac_policy_loss

    actor_loss = sac_policy_loss(new_log_prob, min_q_new, alpha)

    # Re-enable critic gradients.
    for p in model.q1.net.parameters():
      p.requires_grad_(True)
    for p in model.q2.net.parameters():
      p.requires_grad_(True)

    # --- Alpha update ---
    alpha_loss = torch.tensor(0.0, device=obs.device)
    if self._auto_alpha:
      from genml_kit.losses.rl import sac_alpha_loss

      # Pass ``log_alpha`` (the parameter the alpha optimiser actually
      # updates), NOT ``alpha = exp(log_alpha)`` (invariant 3 in the class
      # docstring).  With ``coef = log_alpha`` the gradient is simply the
      # base loss, independent of alpha; with ``coef = alpha`` the gradient
      # carries an extra factor of ``exp(log_alpha)`` that shrinks to zero,
      # creating a spurious attractor that drives alpha -> 0.
      # Detach log_prob so no gradient flows back into the policy net here
      # (the temperature update must only affect ``log_alpha``).
      alpha_loss = sac_alpha_loss(new_log_prob.detach(), self._target_entropy,
                                  self._log_alpha)

    # B5: track env steps for logging (1 env step per train_step in off-policy).
    self._env_steps += 1

    # Return combined loss (for logging) + individual losses in metrics.
    total_loss = critic_loss + actor_loss + alpha_loss

    # For PER: use critic TD errors (average of twin critics).
    td_errors = (q1_pred - soft_target.detach() + q2_pred - soft_target.detach()) * 0.5

    metrics = {
        "critic_loss": critic_loss,
        "actor_loss": actor_loss,
        "alpha_loss": alpha_loss,
        "alpha": alpha.detach(),
        "q_mean": q1_pred.detach().mean(),
    }
    return LossOutput(loss=total_loss, metrics=metrics, td_errors=td_errors)

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
        "method": "sac",
        "env_steps": self._env_steps,
        "log_alpha": self._log_alpha.item(),
    }

  def load_checkpoint_state(self, model, state, args):
    self._env_steps = state.get("env_steps", 0)
    if "log_alpha" in state:
      self._log_alpha.data.fill_(state["log_alpha"])
    if "alpha_optim" in state and hasattr(self, "_alpha_optim"):
      self._alpha_optim.load_state_dict(state["alpha_optim"])
