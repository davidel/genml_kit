"""Learning-convergence tests on the internal ``_ScriptedEnv``.

These verify that the RL methods actually *learn* (not merely run) on a
genuinely trivial, deterministic environment:

- Discrete (``action_space.n == 2``): DQN and PPO must drive the policy to
  the optimal episode return of ``1.0`` (a random policy scores ~0.14).
- Continuous (1-D ``Box``): SAC must beat a random policy by a wide margin
  (random ~5.0, optimal ``10.0``).  This doubles as a regression guard for
  the 1-D continuous action handling (int64-scalar vs float32-vector).

They are marked ``slow`` because, although each runs in a few seconds, they
are stochastic learning checks that need the optional ``gymnasium`` extra
(for the continuous ``Box`` action space) and should not gate the fast
default test run.  Run them explicitly with::

    pytest -m slow
"""

import argparse
import random
import tempfile

import numpy as np
import pytest
import torch
from torch.optim import Adam

from genml_kit.datasets.replay_buffer import ReplayBufferDataset
from genml_kit.datasets.rollout_buffer import RolloutBuffer
from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.methods.rl_ppo import PPOMethod
from genml_kit.methods.rl_sac import SACMethod
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.training.optim_factory import Optimization
from genml_kit.training.rl_trainer import RLTrainer

pytestmark = pytest.mark.slow

# Measured references on the fixed _ScriptedEnv (see test docstrings):
#   discrete  random ~= 0.14, optimal = 1.0
#   continuous random ~= 5.05, optimal = 10.0


class _ScriptedPipeline(RLPipeline):
  """RLPipeline wired to the internal scripted env (no gymnasium control env)."""

  def __init__(self, *, continuous=False, action_dim=1, max_episode_length=3):
    super().__init__()
    self.env = _ScriptedEnv(obs_dim=4,
                            continuous=continuous,
                            action_dim=action_dim,
                            max_episode_length=max_episode_length,
                            target=0.0)
    self._obs_dim = 4
    self._n_actions = None if continuous else 2
    self._action_dim = action_dim if continuous else None
    self.action_space = self.env.action_space
    if continuous:
      self.replay_buffer = ReplayBufferDataset(obs_dim=4,
                                               capacity=10_000,
                                               action_dim=action_dim,
                                               action_dtype=np.float32,
                                               discrete=False)
    else:
      self.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=10_000)
    self.rollout_buffer = RolloutBuffer(obs_dim=4, rollout_len=64, device="cpu")


def _make_args(**overrides):
  defaults = dict(
      env_id="scripted",
      obs_dim=4,
      obs_normalize=False,
      obs_norm_clip=10.0,
      replay_capacity=10_000,
      eval_interval=1,
      record_eval_video=False,
      log_every=10**9,
      log_interval=10**9,
      max_grad_norm=0.0,
      grad_monitor=-1,
      norm_history=0,
      trend_top_n=10,
      remote_checkpoint=None,
      upload_every=0,
      state_save="opt",
      source_checkpoint=None,
      param_rename=None,
      freeze_patterns=None,
      lora_rank=None,
      lora_alpha=None,
      lora_target_modules=None,
      freeze=None,
      lora=None,
      early_stop_patience=0,
      early_stop_metric=None,
      early_stop_mode="max",
      amp_dtype=None,
      use_wandb=False,
      wandb_project=None,
      wandb_entity=None,
      wandb_tags=None,
      tb_flush_secs=120,
      grad_accum_steps=1,
      n_step=1,
      checkpoint=None,
      seed=0,
      env_seed=0,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


def _seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def _run(method, pipeline, args, epochs):
  """Run training and return the best deterministic eval return observed."""
  _seed_everything(args.seed)
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  if isinstance(method, SACMethod):
    optimization = method.build_optimization(args, model, torch.device("cpu"), {}, {})
  else:
    lr = getattr(args, "dqn_lr", None) or getattr(args, "ppo_lr", 1e-3)
    optimization = Optimization(optimizer=Adam(model.parameters(), lr=lr))
  best = float("-inf")
  with tempfile.TemporaryDirectory() as checkpoint_dir:
    args.checkpoint = checkpoint_dir
    trainer = RLTrainer(args=args,
                        model=model,
                        method=method,
                        pipeline=pipeline,
                        optimization=optimization,
                        device=torch.device("cpu"),
                        writer=None,
                        start_epoch=0,
                        best_metric=float("-inf"),
                        global_step=0)
    for epoch in range(epochs):
      trainer.train_epoch(epoch, None, epoch, None)
      best = max(best, trainer.validate())
  return best


class TestDiscreteConvergence:
  """DQN / PPO must reach the optimal return on the discrete scripted chain."""

  def test_dqn_converges(self):
    """DQN drives the greedy policy to the optimal episode return (~1.0).

    A random policy scores ~0.14; the assertion threshold (0.95) leaves a
    wide margin below the observed optimum (1.0 on every seed tested).
    """
    args = _make_args(warmup_steps=100,
                      steps_per_epoch=100,
                      eval_episodes=50,
                      dqn_lr=1e-3,
                      dqn_gamma=0.95,
                      dqn_ddqn=True,
                      dqn_dueling=False,
                      dqn_epsilon_start=1.0,
                      dqn_epsilon_end=0.05,
                      dqn_epsilon_decay_steps=500,
                      dqn_tau=1.0,
                      dqn_target_update_freq=25,
                      dqn_n_step=1)
    best = _run(DQNMethod(),
                _ScriptedPipeline(continuous=False, max_episode_length=3),
                args,
                epochs=6)
    assert best >= 0.95, f"DQN did not converge: best eval_return={best}"

  def test_ppo_converges(self):
    """PPO drives the greedy policy to the optimal episode return (~1.0).

    PPO needs a larger budget than DQN (~6k env steps); the 8k-step budget
    converged to 1.0 on every seed tested (threshold 0.9).
    """
    args = _make_args(warmup_steps=0,
                      steps_per_epoch=256,
                      eval_episodes=50,
                      ppo_discrete=True,
                      ppo_gamma=0.99,
                      ppo_lam=0.95,
                      ppo_clip_eps=0.2,
                      ppo_epochs=4,
                      ppo_batch_size=64,
                      ppo_lr=3e-4,
                      ppo_value_coef=0.5,
                      ppo_entropy_coef=0.01,
                      ppo_max_grad_norm=0.5,
                      rollout_len=64)
    best = _run(PPOMethod(),
                _ScriptedPipeline(continuous=False, max_episode_length=3),
                args,
                epochs=32)
    assert best >= 0.9, f"PPO did not converge: best eval_return={best}"


class TestContinuousConvergence:
  """SAC must learn the continuous scripted task (1-D Box action space).

  Optimal episode return is 10.0; a random policy scores ~5.05.  This also
  guards the 1-D continuous action path (the replay buffer must store
  float32 vectors, not int64 scalars).
  """

  def test_sac_learns_continuous(self):
    pytest.importorskip("gymnasium")
    args = _make_args(warmup_steps=100,
                      steps_per_epoch=100,
                      eval_episodes=50,
                      sac_lr=3e-4,
                      sac_tau=0.005,
                      sac_alpha=0.2,
                      sac_auto_alpha=True,
                      sac_target_entropy=None,
                      sac_batch_size=32)
    best = _run(SACMethod(),
                _ScriptedPipeline(continuous=True, action_dim=1, max_episode_length=10),
                args,
                epochs=8)
    # Wide margin above the random baseline (~5.05); observed optimum ~8.3.
    assert best >= 7.0, f"SAC did not learn: best eval_return={best}"
