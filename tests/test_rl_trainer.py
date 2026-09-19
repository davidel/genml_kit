"""Tests for RLTrainer (warmup + interleaved env/learn loop)."""

import argparse
import contextlib
import tempfile
from collections import namedtuple

import torch
from torch.optim import Adam

from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.models.rl.qnetwork import QNetwork
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.training.rl_trainer import RLTrainer
from genml_kit.datasets.replay_buffer import ReplayBufferDataset

Optimization = namedtuple("Optimization", ["optimizer", "scheduler", "scaler"])


@contextlib.contextmanager
def _temp_checkpoint_dir():
  """Create a temporary checkpoint directory that's auto-cleaned."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


class FakeRLMethod(DQNMethod):
  """Minimal DQN method for unit-testing RLTrainer with scripted env."""

  NAME = "fake_rl"

  @classmethod
  def add_args(cls, parser):
    pass

  def build_model(self, args, device):
    self.n_actions = 2
    self._eps_start = 1.0
    self._eps_end = 0.02
    self._decay_steps = 10_000
    self._epsilon = 1.0
    self._env_steps = 0
    self._gamma = 0.99
    self._ddqn = True
    self._tau = 1.0
    self._target_update_freq = 0
    self._pipeline = None
    return QNetwork(obs_dim=4, n_actions=2)

  def wire_data(self, args, pipeline):
    self.n_actions = 2


class FakeRLPipeline(RLPipeline):
  """Pipeline wired with a scripted env (no gymnasium needed)."""

  def __init__(self):
    super().__init__()
    self.env = _ScriptedEnv(obs_dim=4)
    self._obs_dim = 4
    self._n_actions = 2
    self.replay_buffer = ReplayBufferDataset(obs_dim=4, capacity=100)


@contextlib.contextmanager
def _temp_checkpoint_dir():
  """Create a temporary checkpoint directory that's auto-cleaned."""
  with tempfile.TemporaryDirectory() as tmpdir:
    yield tmpdir


def _make_args(**overrides):
  defaults = dict(
      batch_size=4,
      grad_accum_steps=1,
      warmup_steps=8,
      steps_per_epoch=5,
      eval_episodes=1,
      checkpoint=None,  # Set via context manager
      save_every=0,
      log_dir=None,
      log_interval=100,
      source_checkpoint=None,
      param_rename=None,
      freeze_patterns=None,
      lora_rank=None,
      lora_alpha=None,
      lora_target_modules=None,
      freeze=None,
      lora=None,
      gamma=0.99,
      ddqn=True,
      dueling=False,
      epsilon_start=1.0,
      epsilon_end=0.02,
      epsilon_decay_steps=50_000,
      tau=1.0,
      target_update_freq=0,
      obs_dim=4,
      amp_dtype=None,
      max_grad_norm=0.0,
      seed=42,
      env_seed=42,
  )
  defaults.update(overrides)
  return argparse.Namespace(**defaults)


def _build_trainer(**args_overrides):
  pipeline = FakeRLPipeline()
  method = FakeRLMethod()
  args = _make_args(**args_overrides)
  method.wire_data(args, pipeline)
  model = method.build_model(args, device=torch.device("cpu"))
  optimization = Optimization(
      optimizer=Adam(model.parameters(), lr=1e-3),
      scheduler=None,
      scaler=None,
  )
  # Trainer will be created inside _temp_checkpoint_dir context
  return model, optimization, method, pipeline, args


class TestRLTrainer:

  def test_train_epoch_runs(self):
    with _temp_checkpoint_dir() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer(
      )
      args.checkpoint = checkpoint_dir
      trainer = RLTrainer(
          args=args,
          model=model,
          method=method,
          pipeline=pipeline,
          optimization=optimization,
          device=torch.device("cpu"),
          writer=None,
          start_epoch=0,
          best_metric=float("-inf"),
          global_step=0,
      )
      avg_loss, new_step = trainer.train_epoch(0, None, step=0, monitor=None)
      assert avg_loss >= 0.0
      assert new_step > 0

  def test_validate_returns_metrics(self):
    with _temp_checkpoint_dir() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer(
      )
      args.checkpoint = checkpoint_dir
      trainer = RLTrainer(
          args=args,
          model=model,
          method=method,
          pipeline=pipeline,
          optimization=optimization,
          device=torch.device("cpu"),
          writer=None,
          start_epoch=0,
          best_metric=float("-inf"),
          global_step=0,
      )
      pipeline.init_env(trainer.args)
      eval_return = trainer.validate()
      assert isinstance(eval_return, float)
      assert eval_return >= 0.0

  def test_warmup_fills_buffer(self):
    with _temp_checkpoint_dir() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer(
          warmup_steps=8, steps_per_epoch=3)
      args.checkpoint = checkpoint_dir
      trainer = RLTrainer(
          args=args,
          model=model,
          method=method,
          pipeline=pipeline,
          optimization=optimization,
          device=torch.device("cpu"),
          writer=None,
          start_epoch=0,
          best_metric=float("-inf"),
          global_step=0,
      )
      trainer.train_epoch(0, None, step=0, monitor=None)
      assert len(pipeline.replay_buffer) >= 8
