"""DQN method: Double-DQN with epsilon-greedy exploration.

Phase 1 of ``plans/RL_PLAN.md`` (§6.6).  Implements the ``Method``
interface for off-policy Q-learning with a configurable epsilon schedule,
hard/soft target-net updates, and Dueling support.
"""

import random

import torch

from genml_kit.methods.base import Method
from genml_kit.methods.registry import register_method
from genml_kit.models.registry import load_model
from genml_kit.pipelines.contracts import LossOutput


@register_method
class DQNMethod(Method):
  """Double-DQN with epsilon-greedy exploration."""

  NAME = "dqn"
  METRIC_KEY = "eval_return"
  NEEDS_LABELS = False

  @classmethod
  def get_trainer_class(cls):
    from genml_kit.training.rl_trainer import RLTrainer

    return RLTrainer

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("dqn method")
    group.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99).",
    )
    group.add_argument(
        "--ddqn",
        action="store_true",
        default=True,
        help="Use Double-DQN (default: True).",
    )
    group.add_argument(
        "--no_ddqn",
        dest="ddqn",
        action="store_false",
        help="Disable Double-DQN (use vanilla DQN).",
    )
    group.add_argument(
        "--dueling",
        action="store_true",
        default=False,
        help="Use dueling Q-head architecture.",
    )
    group.add_argument(
        "--epsilon_start",
        type=float,
        default=1.0,
        help="Initial exploration rate (default: 1.0).",
    )
    group.add_argument(
        "--epsilon_end",
        type=float,
        default=0.02,
        help="Final exploration rate (default: 0.02).",
    )
    group.add_argument(
        "--epsilon_decay_steps",
        type=int,
        default=50_000,
        help="Linear decay over this many env steps (default: 50000).",
    )
    group.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Polyak coefficient for soft target updates (default: 1.0 = hard).",
    )
    group.add_argument(
        "--target_update_freq",
        type=int,
        default=1,
        help=("Hard target-net sync every N env steps (0 = use soft "
              "Polyak with --tau). Default 1 = update every step for "
              "stable training."),
    )
    group.add_argument(
        "--n_step",
        type=int,
        default=1,
        help="Number of lookahead steps for n-step TD target (default: 1).",
    )

  def wire_data(self, args, pipeline):
    """Read n_actions from the pipeline (set after env init)."""
    self.n_actions = pipeline.n_actions
    self._pipeline = pipeline

  def build_model(self, args, device):
    """Build the Q-network and initialise epsilon schedule."""
    # Epsilon schedule state.
    self._eps_start = getattr(args, "epsilon_start", 1.0)
    self._eps_end = getattr(args, "epsilon_end", 0.02)
    self._decay_steps = getattr(args, "epsilon_decay_steps", 50_000)
    self._epsilon = self._eps_start
    self._env_steps = 0

    # Target-update config.
    self._target_update_freq = getattr(args, "target_update_freq", 0)
    self._tau = getattr(args, "tau", 1.0)

    # Q-learning hyper-params.
    self._gamma = getattr(args, "gamma", 0.99)
    self._ddqn = getattr(args, "ddqn", True)
    self._n_step = getattr(args, "n_step", 1)

    # Model.
    model = load_model(
        "rl/qnet_dueling" if getattr(args, "dueling", False) else "rl/qnet",
        num_labels=0,
        obs_dim=self._pipeline.obs_dim,
        n_actions=self.n_actions,
        device=device,
    )
    model = self._apply_model_extras(args, model, device)
    return model

  def act(self, model, obs, *, deterministic=False):
    """Select an action via epsilon-greedy.

        Called by the trainer, NOT inside ``train_step`` (D11).
        """
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
      # (1, n_actions)
      q = model.online(obs_t.unsqueeze(0))

    if deterministic:
      return q.argmax(dim=-1).item()

    if random.random() < self._epsilon:
      return random.randrange(self.n_actions)
    return q.argmax(dim=-1).item()

  def step_epsilon(self):
    """Decay epsilon (called by the trainer after each env step)."""
    self._env_steps += 1
    self._epsilon = max(
        self._eps_end,
        self._eps_start * (1.0 - self._env_steps / self._decay_steps),
    )

  def update_target(self, model, global_step):
    """Hard or soft target-net update (called by the trainer)."""
    if self._target_update_freq > 0:
      # Hard sync.
      if global_step % self._target_update_freq == 0:
        model.hard_update()
    elif self._tau < 1.0:
      # Soft Polyak update: target = (1-tau)*target + tau*online.
      with torch.no_grad():
        for p, pt in zip(model.online.parameters(), model.target.parameters()):
          pt.data.mul_(1.0 - self._tau).add_(p.data, alpha=self._tau)
    else:
      # Default: hard update every step.
      model.hard_update()

  def train_step(self, model, blob, global_step, *, labels=None):
    """Compute the TD loss (pure learning — no env interaction)."""
    data = blob if isinstance(blob, dict) else blob.data

    obs = data["obs"]
    action = data["action"]
    reward = data["reward"]
    next_obs = data["next_obs"]
    done = data["done"]

    # Q(s, a) for the taken actions — (B, 1).
    q = model.online(obs).gather(1, action.unsqueeze(1)).squeeze(1)

    # TD target.
    from genml_kit.losses.rl import td_target

    target = td_target(
        reward,
        next_obs,
        done,
        model.online,
        model.target,
        gamma=self._gamma,
        double_q=self._ddqn,
        n_step=self._n_step,
    )

    # TD loss.
    from genml_kit.losses.rl import td_loss

    loss = td_loss(q, target, reduction="mean")

    metrics = {
        "td_loss": loss.detach(),
        "q_mean": q.detach().mean(),
        "epsilon": self._epsilon,
        "env_steps": self._env_steps,
    }
    return LossOutput(loss=loss, metrics=metrics)

  def evaluate(self, model, pipeline, num_episodes, max_steps=10_000):
    """Run evaluation episodes and return mean return.

        Overrides the base ``Method.evaluate`` signature because RL
        validation does not use a DataLoader.

        Args:
            model:       QNetwork with ``.online`` sub-module.
            pipeline:    ``RLPipeline`` providing ``reset_env`` / ``step_env``.
            num_episodes: Number of evaluation episodes.
            max_steps:   Hard cap on environment steps PER EPISODE to prevent
                         infinite loops with untrained policies.
        """
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
        "method": "dqn",
        "epsilon": self._epsilon,
        "env_steps": self._env_steps,
        "eps_start": self._eps_start,
        "eps_end": self._eps_end,
        "decay_steps": self._decay_steps,
    }

  def load_checkpoint_state(self, model, state, args):
    self._epsilon = state.get("epsilon", self._eps_start)
    self._env_steps = state.get("env_steps", 0)
    self._eps_start = state.get("eps_start", self._eps_start)
    self._eps_end = state.get("eps_end", self._eps_end)
    self._decay_steps = state.get("decay_steps", self._decay_steps)
