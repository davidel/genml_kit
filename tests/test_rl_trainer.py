"""Tests for RLTrainer (warmup + interleaved env/learn loop)."""

import argparse
import tempfile
from collections import namedtuple

import numpy as np
import pytest
import torch
from torch.optim import Adam

from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.methods.rl_ppo import PPOMethod
from genml_kit.methods.rl_sac import SACMethod
from genml_kit.models.rl.qnetwork import QNetwork
from genml_kit.pipelines.rl import RLPipeline, _ScriptedEnv
from genml_kit.training.rl_trainer import RLTrainer
from genml_kit.datasets.replay_buffer import ReplayBufferDataset
from genml_kit.datasets.rollout_buffer import RolloutBuffer

Optimization = namedtuple("Optimization", ["optimizer", "scheduler", "scaler"])


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
    self._n_step = 1
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


class _TestRLPipeline(RLPipeline):
  """Test pipeline using scripted env (no gymnasium needed) for real RL methods."""

  def __init__(self,
               obs_dim=4,
               continuous=False,
               action_dim=2,
               buffer_size=100,
               rollout_len=16):
    super().__init__()
    self.env = _ScriptedEnv(obs_dim=obs_dim,
                            continuous=continuous,
                            action_dim=action_dim)
    self._obs_dim = obs_dim
    self._n_actions = action_dim if continuous else 2
    self._action_dim = action_dim if continuous else None
    # Expose action_space for continuous support
    self.action_space = self.env.action_space

    # Determine action dim and dtype for replay buffer
    if self._action_dim is not None and self._action_dim > 1:
      action_dim_rb = self._action_dim
      action_dtype = np.float32
    else:
      action_dim_rb = 1
      action_dtype = np.int64

    self.replay_buffer = ReplayBufferDataset(
        obs_dim=obs_dim,
        capacity=buffer_size,
        action_dim=action_dim_rb,
        action_dtype=action_dtype,
    )
    # On-policy rollout buffer for PPO
    self.rollout_buffer = RolloutBuffer(
        obs_dim=obs_dim,
        rollout_len=rollout_len,
        action_dim=self._action_dim,
        device="cpu",
    )


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
    with tempfile.TemporaryDirectory() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer()
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
    with tempfile.TemporaryDirectory() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer()
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
    with tempfile.TemporaryDirectory() as checkpoint_dir:
      model, optimization, method, pipeline, args = _build_trainer(warmup_steps=8,
                                                                   steps_per_epoch=3)
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


class TestRLTrainerEndToEnd:
  """End-to-end tests calling trainer.run() for each RL method."""

  def _make_args(self, **overrides):
    """Create a minimal argparse.Namespace with RL defaults."""
    defaults = dict(
        epochs=2,
        warmup_steps=4,
        steps_per_epoch=4,
        gamma=0.99,
        dqn_lr=1e-3,
        dqn_target_tau=0.1,
        dqn_dueling=False,
        ddqn=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=1000,
        env_id="CartPole-v1",
        eval_episodes=1,
        buffer_size=100,
        checkpoint=None,
        seed=None,
        grad_clip=1.0,
        log_level="ERROR",
        log_targets="stderr",
        save_every=0,
        # PPO args
        ppo_lr=3e-4,
        ppo_rollout_len=16,
        ppo_epochs=2,
        ppo_batch_size=8,
        ppo_clip=0.2,
        ppo_entropy_coef=0.01,
        ppo_value_coef=0.5,
        ppo_lam=0.95,
        ppo_max_grad_norm=0.5,
        # SAC args
        sac_lr=3e-4,
        sac_tau=0.005,
        sac_alpha=0.2,
        sac_auto_alpha=True,
        sac_target_entropy=None,
        sac_batch_size=8,
        # Trainer args
        state_save="opt,sched,amp",
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
        batch_size=4,
        grad_accum_steps=1,
        max_grad_norm=0.0,
        grad_monitor=-1,
        norm_history=0,
        trend_top_n=10,
        remote_checkpoint=None,
        upload_every=0,
        s3_region=None,
        s3_endpoint=None,
        gcs_bucket=None,
        gcs_project=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_tags=None,
        use_wandb=False,
        tb_flush_secs=120,
        amp_dtype=None,
        early_stop_patience=0,
        early_stop_metric=None,
        early_stop_mode="max",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

  def _run_e2e_dqn(self, tmp_path):
    """Helper to run DQN end-to-end for one epoch."""
    args = self._make_args(
        epochs=1,
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "dqn_ckpt"),
        seed=42,
        env_seed=42,
    )
    method = DQNMethod()
    pipeline = _TestRLPipeline(obs_dim=4, continuous=False, buffer_size=100)
    # Wire data and build model (initializes epsilon schedule)
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    optimization = Optimization(
        optimizer=Adam(model.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
    )
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
    result = trainer.run()
    # One epoch completed
    assert result.completed_epoch == 0
    # Checkpoint written
    ckpt_files = list(tmp_path.glob("dqn_ckpt*.pt"))
    assert len(ckpt_files) >= 1
    # Best metric key present
    ckpt = torch.load(ckpt_files[0], weights_only=False)
    assert "best_eval_return" in ckpt
    return result

  def _run_e2e_ppo(self, tmp_path):
    """Helper to run PPO end-to-end for one epoch."""
    args = self._make_args(
        epochs=1,
        warmup_steps=0,  # PPO doesn't use warmup
        steps_per_epoch=16,  # rollout_len
        checkpoint=str(tmp_path / "ppo_ckpt"),
        seed=42,
        env_seed=42,
        rollout_len=16,
    )
    method = PPOMethod()
    pipeline = _TestRLPipeline(obs_dim=4,
                               continuous=False,
                               buffer_size=100,
                               rollout_len=16)
    # Wire data and build model
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    optimization = Optimization(
        optimizer=Adam(model.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
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
    result = trainer.run()
    assert result.completed_epoch == 0
    ckpt_files = list(tmp_path.glob("ppo_ckpt*.pt"))
    assert len(ckpt_files) >= 1
    ckpt = torch.load(ckpt_files[0], weights_only=False)
    assert "best_eval_return" in ckpt
    return result

  def _run_e2e_sac(self, tmp_path):
    """Helper to run SAC end-to-end for one epoch."""
    args = self._make_args(
        epochs=1,
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "sac_ckpt"),
        seed=42,
        env_seed=42,
    )
    method = SACMethod()
    pipeline = _TestRLPipeline(obs_dim=4,
                               continuous=True,
                               action_dim=2,
                               buffer_size=100)
    # Wire data and build model (initializes _log_alpha)
    method.wire_data(args, pipeline)
    model = method.build_model(args, device=torch.device("cpu"))
    optimization = Optimization(
        optimizer=Adam(model.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
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
    result = trainer.run()
    assert result.completed_epoch == 0
    ckpt_files = list(tmp_path.glob("sac_ckpt*.pt"))
    assert len(ckpt_files) >= 1
    ckpt = torch.load(ckpt_files[0], weights_only=False)
    assert "best_eval_return" in ckpt
    return result

  def test_run_end_to_end_one_epoch_dqn(self, tmp_path):
    """DQN: trainer.run() completes one epoch, writes checkpoint."""
    self._run_e2e_dqn(tmp_path)

  def test_run_end_to_end_one_epoch_ppo(self, tmp_path):
    """PPO: trainer.run() completes one epoch, writes checkpoint."""
    self._run_e2e_ppo(tmp_path)

  def test_run_end_to_end_one_epoch_sac(self, tmp_path):
    """SAC: trainer.run() completes one epoch, writes checkpoint."""
    self._run_e2e_sac(tmp_path)


class TestRLCheckpointRoundTrip:
  """Checkpointing round-trip tests for RL methods."""

  def _make_args(self, **overrides):
    """Create a minimal argparse.Namespace with RL defaults."""
    defaults = dict(
        epochs=2,
        warmup_steps=4,
        steps_per_epoch=4,
        gamma=0.99,
        dqn_lr=1e-3,
        dqn_target_tau=0.1,
        dqn_dueling=False,
        ddqn=True,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=1000,
        env_id="CartPole-v1",
        eval_episodes=1,
        buffer_size=100,
        checkpoint=None,
        seed=None,
        grad_clip=1.0,
        log_level="ERROR",
        log_targets="stderr",
        save_every=0,
        # PPO args
        ppo_lr=3e-4,
        ppo_rollout_len=16,
        ppo_epochs=2,
        ppo_batch_size=8,
        ppo_clip=0.2,
        ppo_entropy_coef=0.01,
        ppo_value_coef=0.5,
        ppo_lam=0.95,
        ppo_max_grad_norm=0.5,
        # SAC args
        sac_lr=3e-4,
        sac_tau=0.005,
        sac_alpha=0.2,
        sac_auto_alpha=True,
        sac_target_entropy=None,
        sac_batch_size=8,
        # Trainer args
        state_save="opt,sched,amp",
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
        batch_size=4,
        grad_accum_steps=1,
        max_grad_norm=0.0,
        grad_monitor=-1,
        norm_history=0,
        trend_top_n=10,
        remote_checkpoint=None,
        upload_every=0,
        s3_region=None,
        s3_endpoint=None,
        gcs_bucket=None,
        gcs_project=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_tags=None,
        use_wandb=False,
        tb_flush_secs=120,
        early_stop_patience=0,
        early_stop_metric=None,
        early_stop_mode="max",
        amp_dtype=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)

  def test_dqn_checkpoint_roundtrip(self, tmp_path):
    """DQN: save checkpoint, load it, verify epsilon and env_steps preserved."""
    # First run: train for one epoch and save
    args1 = self._make_args(
        epochs=1,
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "dqn_ckpt"),
        seed=42,
        env_seed=42,
    )
    method1 = DQNMethod()
    pipeline1 = _TestRLPipeline(obs_dim=4, continuous=False, buffer_size=100)
    method1.wire_data(args1, pipeline1)
    model1 = method1.build_model(args1, device=torch.device("cpu"))
    optimization1 = Optimization(
        optimizer=Adam(model1.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
    )
    trainer1 = RLTrainer(
        args=args1,
        model=model1,
        method=method1,
        pipeline=pipeline1,
        optimization=optimization1,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=0,
        best_metric=float("-inf"),
        global_step=0,
    )
    result1 = trainer1.run()
    assert result1.completed_epoch == 0

    # Find the latest checkpoint
    ckpt_files = list(tmp_path.glob("dqn_ckpt*_latest.pt"))
    assert len(ckpt_files) == 1
    ckpt1 = torch.load(ckpt_files[0], weights_only=False)

    # Verify checkpoint has method state
    assert "method_state" in ckpt1
    assert "epsilon" in ckpt1["method_state"]
    assert "env_steps" in ckpt1["method_state"]
    epsilon1 = ckpt1["method_state"]["epsilon"]
    env_steps1 = ckpt1["method_state"]["env_steps"]

    # Second run: load checkpoint and continue
    args2 = self._make_args(
        epochs=2,  # Run one more epoch
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "dqn_ckpt"),
        seed=42,
        env_seed=42,
    )
    method2 = DQNMethod()
    pipeline2 = _TestRLPipeline(obs_dim=4, continuous=False, buffer_size=100)
    method2.wire_data(args2, pipeline2)
    model2 = method2.build_model(args2, device=torch.device("cpu"))
    optimization2 = Optimization(
        optimizer=Adam(model2.parameters(), lr=1e-3),
        scheduler=None,
        scaler=None,
    )

    # Load checkpoint state
    method2.load_checkpoint_state(model2, ckpt1["method_state"], args2)

    # Verify state was loaded correctly immediately after loading
    assert method2._epsilon == pytest.approx(epsilon1, rel=1e-3)
    assert method2._env_steps == env_steps1

    trainer2 = RLTrainer(
        args=args2,
        model=model2,
        method=method2,
        pipeline=pipeline2,
        optimization=optimization2,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=1,  # Resume from epoch 1
        best_metric=ckpt1.get("best_eval_return", float("-inf")),
        global_step=ckpt1.get("global_step", 0),
    )
    result2 = trainer2.run()
    assert result2.completed_epoch == 1

  def test_ppo_checkpoint_roundtrip(self, tmp_path):
    """PPO: save checkpoint, load it, verify state preserved."""
    args1 = self._make_args(
        epochs=1,
        warmup_steps=0,
        steps_per_epoch=16,
        checkpoint=str(tmp_path / "ppo_ckpt"),
        seed=42,
        env_seed=42,
        rollout_len=16,
    )
    method1 = PPOMethod()
    pipeline1 = _TestRLPipeline(obs_dim=4,
                                continuous=False,
                                buffer_size=100,
                                rollout_len=16)
    method1.wire_data(args1, pipeline1)
    model1 = method1.build_model(args1, device=torch.device("cpu"))
    optimization1 = Optimization(
        optimizer=Adam(model1.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
    trainer1 = RLTrainer(
        args=args1,
        model=model1,
        method=method1,
        pipeline=pipeline1,
        optimization=optimization1,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=0,
        best_metric=float("-inf"),
        global_step=0,
    )
    result1 = trainer1.run()
    assert result1.completed_epoch == 0

    ckpt_files = list(tmp_path.glob("ppo_ckpt*_latest.pt"))
    assert len(ckpt_files) == 1
    ckpt1 = torch.load(ckpt_files[0], weights_only=False)

    # PPO should have env_steps in method state
    assert "method_state" in ckpt1
    assert "env_steps" in ckpt1["method_state"]
    env_steps1 = ckpt1["method_state"]["env_steps"]

    # Second run
    args2 = self._make_args(
        epochs=2,
        warmup_steps=0,
        steps_per_epoch=16,
        checkpoint=str(tmp_path / "ppo_ckpt"),
        seed=42,
        env_seed=42,
        rollout_len=16,
    )
    method2 = PPOMethod()
    pipeline2 = _TestRLPipeline(obs_dim=4,
                                continuous=False,
                                buffer_size=100,
                                rollout_len=16)
    method2.wire_data(args2, pipeline2)
    model2 = method2.build_model(args2, device=torch.device("cpu"))
    optimization2 = Optimization(
        optimizer=Adam(model2.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
    method2.load_checkpoint_state(model2, ckpt1["method_state"], args2)

    # Verify state was loaded correctly immediately after loading
    assert method2._env_steps == env_steps1

    trainer2 = RLTrainer(
        args=args2,
        model=model2,
        method=method2,
        pipeline=pipeline2,
        optimization=optimization2,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=1,
        best_metric=ckpt1.get("best_eval_return", float("-inf")),
        global_step=ckpt1.get("global_step", 0),
    )
    result2 = trainer2.run()
    assert result2.completed_epoch == 1

  def test_sac_checkpoint_roundtrip(self, tmp_path):
    """SAC: save checkpoint, load it, verify log_alpha and env_steps preserved."""
    args1 = self._make_args(
        epochs=1,
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "sac_ckpt"),
        seed=42,
        env_seed=42,
    )
    method1 = SACMethod()
    pipeline1 = _TestRLPipeline(obs_dim=4,
                                continuous=True,
                                action_dim=2,
                                buffer_size=100)
    method1.wire_data(args1, pipeline1)
    model1 = method1.build_model(args1, device=torch.device("cpu"))
    optimization1 = Optimization(
        optimizer=Adam(model1.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
    trainer1 = RLTrainer(
        args=args1,
        model=model1,
        method=method1,
        pipeline=pipeline1,
        optimization=optimization1,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=0,
        best_metric=float("-inf"),
        global_step=0,
    )
    result1 = trainer1.run()
    assert result1.completed_epoch == 0

    ckpt_files = list(tmp_path.glob("sac_ckpt*_latest.pt"))
    assert len(ckpt_files) == 1
    ckpt1 = torch.load(ckpt_files[0], weights_only=False)

    assert "method_state" in ckpt1
    assert "log_alpha" in ckpt1["method_state"]
    assert "env_steps" in ckpt1["method_state"]
    log_alpha1 = ckpt1["method_state"]["log_alpha"]
    env_steps1 = ckpt1["method_state"]["env_steps"]

    # Second run
    args2 = self._make_args(
        epochs=2,
        warmup_steps=4,
        steps_per_epoch=4,
        checkpoint=str(tmp_path / "sac_ckpt"),
        seed=42,
        env_seed=42,
    )
    method2 = SACMethod()
    pipeline2 = _TestRLPipeline(obs_dim=4,
                                continuous=True,
                                action_dim=2,
                                buffer_size=100)
    method2.wire_data(args2, pipeline2)
    model2 = method2.build_model(args2, device=torch.device("cpu"))
    optimization2 = Optimization(
        optimizer=Adam(model2.parameters(), lr=3e-4),
        scheduler=None,
        scaler=None,
    )
    method2.load_checkpoint_state(model2, ckpt1["method_state"], args2)

    # Verify state was loaded correctly immediately after loading
    assert method2._log_alpha.item() == pytest.approx(log_alpha1, rel=1e-5)
    assert method2._env_steps == env_steps1

    trainer2 = RLTrainer(
        args=args2,
        model=model2,
        method=method2,
        pipeline=pipeline2,
        optimization=optimization2,
        device=torch.device("cpu"),
        writer=None,
        start_epoch=1,
        best_metric=ckpt1.get("best_eval_return", float("-inf")),
        global_step=ckpt1.get("global_step", 0),
    )
    result2 = trainer2.run()
    assert result2.completed_epoch == 1


class TestNStepReturns:
  """Test n-step return computation in ReplayBufferDataset."""

  def test_n_step_returns_basic(self):
    """Test that n-step returns are computed correctly."""
    from genml_kit.datasets.replay_buffer import ReplayBufferDataset

    buffer = ReplayBufferDataset(obs_dim=4, capacity=100, n_step=3, gamma=0.99, seed=42)

    # Push a sequence of transitions
    obs0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    obs1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    obs2 = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    obs3 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)

    # Step 0: reward=1.0, not done
    buffer.push(obs0, 0, 1.0, obs1, False)
    assert len(buffer) == 0  # n-step buffer not full yet

    # Step 1: reward=2.0, not done
    buffer.push(obs1, 1, 2.0, obs2, False)
    assert len(buffer) == 0  # n-step buffer not full yet

    # Step 2: reward=3.0, not done -> n-step buffer full, should push
    buffer.push(obs2, 2, 3.0, obs3, False)
    assert len(buffer) == 1

    # Check the stored transition
    # n-step reward = 1.0 + 0.99*2.0 + 0.99^2*3.0 = 1.0 + 1.98 + 2.9403 = 5.9203
    expected_reward = 1.0 + 0.99 * 2.0 + 0.99**2 * 3.0
    assert buffer.reward[0] == pytest.approx(expected_reward, rel=1e-4)
    assert np.allclose(buffer.obs[0], obs0)
    assert buffer.action[0] == 0
    assert np.allclose(buffer.next_obs[0], obs3)
    assert buffer.done[0] == 0.0

  def test_n_step_returns_early_done(self):
    """Test n-step returns terminate early when done=True."""
    from genml_kit.datasets.replay_buffer import ReplayBufferDataset

    buffer = ReplayBufferDataset(obs_dim=4, capacity=100, n_step=3, gamma=0.99, seed=42)

    obs0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    obs1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    obs2 = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)

    # Step 0: reward=1.0, not done
    buffer.push(obs0, 0, 1.0, obs1, False)
    assert len(buffer) == 0

    # Step 1: reward=2.0, DONE -> should flush n-step buffer
    buffer.push(obs1, 1, 2.0, obs2, True)
    assert len(buffer) == 1

    # n-step reward = 1.0 + 0.99*2.0 = 2.98 (stops at done)
    expected_reward = 1.0 + 0.99 * 2.0
    assert buffer.reward[0] == pytest.approx(expected_reward, rel=1e-4)
    assert buffer.done[0] == 1.0
    assert np.allclose(buffer.next_obs[0], obs2)

  def test_n_step_returns_flush_on_done(self):
    """Test n-step buffer flushes remaining transitions when episode ends."""
    from genml_kit.datasets.replay_buffer import ReplayBufferDataset

    buffer = ReplayBufferDataset(obs_dim=4, capacity=100, n_step=3, gamma=0.99, seed=42)

    obs0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    obs1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    obs2 = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    obs3 = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)

    # Episode 1: 2 steps then done
    buffer.push(obs0, 0, 1.0, obs1, False)
    buffer.push(obs1, 1, 2.0, obs2, True)
    assert len(buffer) == 1

    # Episode 2: 3 steps
    buffer.push(obs2, 2, 3.0, obs3, False)
    buffer.push(obs3, 3, 4.0, obs0, False)
    buffer.push(obs0, 0, 5.0, obs1, False)
    assert len(buffer) == 2

    # Check first transition (2-step)
    expected_1 = 1.0 + 0.99 * 2.0
    assert buffer.reward[0] == pytest.approx(expected_1, rel=1e-4)
    assert buffer.done[0] == 1.0

    # Check second transition (3-step)
    expected_2 = 3.0 + 0.99 * 4.0 + 0.99**2 * 5.0
    assert buffer.reward[1] == pytest.approx(expected_2, rel=1e-4)
    assert buffer.done[1] == 0.0

  def test_n_step_1_is_standard(self):
    """Test that n_step=1 behaves like standard replay buffer."""
    from genml_kit.datasets.replay_buffer import ReplayBufferDataset

    buffer = ReplayBufferDataset(obs_dim=4, capacity=100, n_step=1, gamma=0.99, seed=42)

    obs0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    obs1 = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    buffer.push(obs0, 0, 1.0, obs1, False)
    assert len(buffer) == 1

    assert buffer.reward[0] == 1.0
    assert buffer.done[0] == 0.0
    assert np.allclose(buffer.obs[0], obs0)
    assert np.allclose(buffer.next_obs[0], obs1)
