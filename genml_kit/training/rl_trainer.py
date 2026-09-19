"""RL trainer: supports both off-policy (DQN, SAC) and on-policy (PPO).

Extends ``BaseTrainer`` with RL-specific ``train_epoch`` and
``validate`` to bypass the DataLoader and work with environments
and replay/rollout buffers directly.
"""

import logging

import numpy as np
import torch

from genml_kit.training.trainer import BaseTrainer


class RLTrainer(BaseTrainer):
  """Training loop for RL methods.

  Inherits the full ``run()`` lifecycle from ``BaseTrainer`` (signals,
  checkpointing, AMP, grad monitor).  Overrides ``train_epoch`` and
  ``validate`` to bypass the DataLoader and work with environments
  and replay/rollout buffers directly.

  Dispatches between off-policy (DQN, SAC) and on-policy (PPO) flows
  based on ``method.NAME``.
  """

  # A6: save frozen params (target nets) so resume restores them identically
  SAVE_FROZEN = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._warmup_obs = None

  def train_epoch(self, epoch, saver, step, monitor):
    """One training epoch.

    Off-policy (DQN/SAC): warmup fill + interleaved env/learn.
    On-policy (PPO): rollout collection + GAE + SGD epochs.

    Returns:
        ``(avg_loss, new_step)``
    """
    method_name = getattr(self.method, "NAME", "dqn")
    if method_name == "ppo":
      return self._train_epoch_ppo(epoch, saver, step, monitor)
    return self._train_epoch_offpolicy(epoch, saver, step, monitor)

  # ------------------------------------------------------------------
  # Off-policy: DQN / SAC
  # ------------------------------------------------------------------

  def _train_epoch_offpolicy(self, epoch, saver, step, monitor):
    """Off-policy: warmup fill + interleaved acting + learning."""
    method = self.method
    model = self.model
    pipeline = self.pipeline
    buffer = pipeline.replay_buffer
    device = self.device
    args = self.args
    batch_size = getattr(args, "batch_size", 64)
    scaler = self.optimization.scaler
    amp_dtype = getattr(args, "amp_dtype", None)

    # Initialise environment (no-op after first epoch).
    pipeline.init_env(args)

    # Phase 1: warmup fill.
    obs = self._warmup_obs
    if obs is None:
      obs = pipeline.reset_env()
    while len(buffer) < args.warmup_steps:
      action = torch.randint(0, method.n_actions, (1,)).item()
      next_obs, reward, done, _ = pipeline.step_env(action)
      buffer.push(obs, action, reward, next_obs, float(done))
      obs = next_obs if not done else pipeline.reset_env()
    self._warmup_obs = obs

    # Phase 2: interleaved acting + learning.
    total_loss = 0.0
    batches = 0

    for _step in range(args.steps_per_epoch):
      # Act.
      action = method.act(model, obs, deterministic=False)
      next_obs, reward, done, _ = pipeline.step_env(action)
      buffer.push(obs, action, reward, next_obs, float(done))
      method.step_epsilon()

      obs = next_obs if not done else pipeline.reset_env()

      # Learn.
      batch = buffer.sample(batch_size)
      batch = pipeline.to_device(batch, device)

      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and device.type == "cuda"),
      ):
        loss_out = method.train_step(model, batch, step)

      self._apply_grad(loss_out.loss, scaler, amp_dtype)
      step += 1
      method.update_target(model, global_step=step)

      total_loss += loss_out.loss.item()
      batches += 1

      if monitor is not None:
        monitor.step(loss_out.loss, step)

    avg_loss = total_loss / max(batches, 1)
    if self.writer is not None:
      self.writer.add_scalar("train/loss", avg_loss, epoch)
      if hasattr(method, "_epsilon"):
        self.writer.add_scalar("epsilon", method._epsilon, epoch)
      if hasattr(method, "_get_alpha"):
        self.writer.add_scalar("alpha", method._get_alpha(), epoch)
      self.writer.add_scalar("env_steps", method._env_steps, epoch)

    logging.info(
        "epoch=%d  avg_loss=%.4f  env_steps=%d  buffer_size=%d",
        epoch,
        avg_loss,
        method._env_steps,
        len(buffer),
    )
    return avg_loss, step

  # ------------------------------------------------------------------
  # On-policy: PPO
  # ------------------------------------------------------------------

  def _train_epoch_ppo(self, epoch, saver, step, monitor):
    """On-policy: collect rollout → compute GAE → SGD epochs."""
    method = self.method
    model = self.model
    pipeline = self.pipeline
    device = self.device
    args = self.args
    scaler = self.optimization.scaler
    amp_dtype = getattr(args, "amp_dtype", None)

    # Initialise environment (no-op after first epoch).
    pipeline.init_env(args)

    rollout = pipeline.rollout_buffer
    rollout_len = rollout.rollout_len
    obs = getattr(self, "_ppo_obs", None)

    # Phase 1: collect rollout.
    rollout.reset()
    obs = pipeline.reset_env() if obs is None else obs
    episode_count = 0
    total_reward = 0.0

    for _ in range(rollout_len):
      action, log_prob, value = method.act(model, obs, deterministic=False)
      next_obs, reward, done, _ = pipeline.step_env(action)
      rollout.add(obs, action, log_prob, reward, value, float(done))

      total_reward += reward
      obs = next_obs

      if done:
        total_reward = 0.0
        episode_count += 1
        obs = pipeline.reset_env()

    # Bootstrap value for GAE.
    with torch.no_grad():
      obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
      next_val = model.get_value(obs_t).item()
    rollout.set_next_values(next_val)
    rollout.compute(gamma=method._gamma, lam=method._lam)
    rollout.to(device)

    self._ppo_obs = obs

    # B5: track env steps collected in this rollout
    method._env_steps += rollout_len

    # Phase 2: multiple SGD epochs over the rollout.
    total_loss = 0.0
    batches = 0
    mini_batch_size = method._mini_batch_size

    for _ppo_epoch in range(method._ppo_epochs):
      rollout_indices = torch.randperm(rollout_len)
      for start in range(0, rollout_len, mini_batch_size):
        end = min(start + mini_batch_size, rollout_len)
        mb_idx = rollout_indices[start:end]
        batch = {
            k: v[mb_idx]
            for k, v in rollout.__dict__.items()
            if isinstance(v, torch.Tensor) and v.shape[0] == rollout_len
        }
        batch = pipeline.to_device(batch, device)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and device.type == "cuda"),
        ):
          loss_out = method.train_step(model, batch, step)

        self._apply_grad(loss_out.loss, scaler, amp_dtype)
        step += 1

        total_loss += loss_out.loss.item()
        batches += 1

        if monitor is not None:
          monitor.step(loss_out.loss, step)

    avg_loss = total_loss / max(batches, 1)
    if self.writer is not None:
      self.writer.add_scalar("train/loss", avg_loss, epoch)
      self.writer.add_scalar("env_steps", method._env_steps, epoch)
      for k in ("pg_loss", "value_loss", "entropy", "ratio_mean"):
        if k in loss_out.metrics:
          self.writer.add_scalar(f"ppo/{k}", loss_out.metrics[k], epoch)

    logging.info(
        "epoch=%d  avg_loss=%.4f  env_steps=%d  episodes=%d",
        epoch,
        avg_loss,
        method._env_steps,
        episode_count,
    )
    return avg_loss, step

  # ------------------------------------------------------------------
  # Shared helpers
  # ------------------------------------------------------------------

  def _apply_grad(self, loss, scaler, amp_dtype):
    """Backward + optimizer step (shared by all RL flows)."""
    self.optimization.optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
      scaler.scale(loss).backward()
      scaler.step(self.optimization.optimizer)
      scaler.update()
    else:
      loss.backward()
      self.optimization.optimizer.step()

  def validate(self):
    """Policy evaluation on the environment (not a DataLoader).

    Returns:
      Scalar eval_return (float) for checkpoint selection.
    """
    env_seed = getattr(self.args, "env_seed", None)
    # Isolate RNG state to avoid corrupting training sampling (B2).
    if env_seed is not None:
      old_state = np.random.get_state()
      np.random.seed(env_seed)
    else:
      old_state = None
    try:
      metrics = self.method.evaluate(
          self.model,
          self.pipeline,
          getattr(self.args, "eval_episodes", 5),
      )
      return float(metrics["eval_return"])
    finally:
      if old_state is not None:
        np.random.set_state(old_state)
