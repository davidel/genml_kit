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
    # Allow method to build custom optimization (e.g., SAC three optimizers)
    custom_opt = self.method.build_optimization(self.args, self.model, self.device, {},
                                                {})
    if custom_opt is not None:
      self.optimization = custom_opt

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
    batch_size = getattr(self.args, "batch_size", 64)
    scaler = getattr(self.optimization, "scaler", None)
    amp_dtype = getattr(self.args, "amp_dtype", None)

    # Initialise environment (no-op after first epoch).
    self.pipeline.init_env(self.args)

    # Phase 1: warmup fill.
    obs = self._warmup_obs
    if obs is None:
      obs = self.pipeline.reset_env()
    while len(self.pipeline.replay_buffer) < self.args.warmup_steps:
      action = torch.randint(0, self.method.n_actions, (1,)).item()
      next_obs, reward, done, _ = self.pipeline.step_env(action)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done))
      obs = next_obs if not done else self.pipeline.reset_env()
    self._warmup_obs = obs

    # Phase 2: interleaved acting + learning.
    total_loss = 0.0
    batches = 0

    for _step in range(self.args.steps_per_epoch):
      # Act.
      action = self.method.act(self.model, obs, deterministic=False)
      next_obs, reward, done, _ = self.pipeline.step_env(action)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done))
      self.method.step_epsilon()

      obs = next_obs if not done else self.pipeline.reset_env()

      # Learn.
      batch = self.pipeline.replay_buffer.sample(batch_size)
      batch = self.pipeline.to_device(batch, self.device)

      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and self.device.type == "cuda"),
      ):
        loss_out = self.method.train_step(self.model, batch, step)

      self._apply_grad(loss_out.loss, scaler, amp_dtype)
      step += 1
      self.method.update_target(self.model, global_step=step)

      total_loss += loss_out.loss.item()
      batches += 1

      if monitor is not None:
        monitor.step(loss_out.loss, step)

    avg_loss = total_loss / max(batches, 1)
    if self.writer is not None:
      self.writer.add_scalar("train/loss", avg_loss, epoch)
      if hasattr(self.method, "_epsilon"):
        self.writer.add_scalar("epsilon", self.method._epsilon, epoch)
      if hasattr(self.method, "_get_alpha"):
        self.writer.add_scalar("alpha", self.method._get_alpha(), epoch)
      self.writer.add_scalar("env_steps", self.method._env_steps, epoch)

    logging.info(
        "epoch=%d  avg_loss=%.4f  env_steps=%d  buffer_size=%d",
        epoch,
        avg_loss,
        self.method._env_steps,
        len(self.pipeline.replay_buffer),
    )
    return avg_loss, step

  # ------------------------------------------------------------------
  # On-policy: PPO
  # ------------------------------------------------------------------

  def _train_epoch_ppo(self, epoch, saver, step, monitor):
    """On-policy: collect rollout \u2192 compute GAE \u2192 SGD epochs."""
    scaler = getattr(self.optimization, "scaler", None)
    amp_dtype = getattr(self.args, "amp_dtype", None)

    # Initialise environment (no-op after first epoch).
    self.pipeline.init_env(self.args)

    rollout = self.pipeline.rollout_buffer
    rollout_len = rollout.rollout_len
    obs = getattr(self, "_ppo_obs", None)

    # Phase 1: collect rollout.
    rollout.reset()
    obs = self.pipeline.reset_env() if obs is None else obs
    episode_count = 0
    total_reward = 0.0

    # Collect per-step bootstrap values V(s_{t+1}) during rollout.
    next_values = torch.zeros(rollout_len, dtype=torch.float32)

    for i in range(rollout_len):
      action, log_prob, value, raw_action = self.method.act(self.model,
                                                            obs,
                                                            deterministic=False)
      next_obs, reward, done, _ = self.pipeline.step_env(action)
      rollout.add(obs, action, log_prob, reward, value, float(done), raw_action)

      # Compute V(s_{t+1}) for GAE bootstrap at this step.
      with torch.no_grad():
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        next_values[i] = self.model.get_value(next_obs_t).item()

      total_reward += reward
      obs = next_obs

      if done:
        total_reward = 0.0
        episode_count += 1
        obs = self.pipeline.reset_env()

    # Bootstrap values for GAE (per-step, not a single scalar).
    rollout.set_next_values(next_values)
    rollout.compute(gamma=self.method._gamma, lam=self.method._lam)
    rollout.to(self.device)

    self._ppo_obs = obs

    # B5: track env steps collected in this rollout
    self.method._env_steps += rollout_len

    # Phase 2: multiple SGD epochs over the rollout.
    total_loss = 0.0
    batches = 0
    mini_batch_size = self.method._mini_batch_size

    for _ppo_epoch in range(self.method._ppo_epochs):
      rollout_indices = torch.randperm(rollout_len)
      for start in range(0, rollout_len, mini_batch_size):
        end = min(start + mini_batch_size, rollout_len)
        mb_idx = rollout_indices[start:end]
        batch = {
            k: v[mb_idx]
            for k, v in rollout.__dict__.items()
            if isinstance(v, torch.Tensor) and v.shape[0] == rollout_len
        }
        batch = self.pipeline.to_device(batch, self.device)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and self.device.type == "cuda"),
        ):
          loss_out = self.method.train_step(self.model, batch, step)

        self._apply_grad(loss_out.loss, scaler, amp_dtype)
        step += 1

        total_loss += loss_out.loss.item()
        batches += 1

        if monitor is not None:
          monitor.step(loss_out.loss, step)

    avg_loss = total_loss / max(batches, 1)
    if self.writer is not None:
      self.writer.add_scalar("train/loss", avg_loss, epoch)
      self.writer.add_scalar("env_steps", self.method._env_steps, epoch)
      for k in ("pg_loss", "value_loss", "entropy", "ratio_mean"):
        if k in loss_out.metrics:
          self.writer.add_scalar(f"ppo/{k}", loss_out.metrics[k], epoch)

    logging.info(
        "epoch=%d  avg_loss=%.4f  env_steps=%d  episodes=%d",
        epoch,
        avg_loss,
        self.method._env_steps,
        episode_count,
    )
    return avg_loss, step

  # ------------------------------------------------------------------
  # Shared helpers
  # ------------------------------------------------------------------

  def _apply_grad(self, loss, scaler, amp_dtype):
    """Backward + optimizer step (shared by all RL flows).

    If the method provides a custom ``apply_grad``, delegate to it.
    This enables multi-optimizer patterns (e.g., SAC).
    """
    if hasattr(self.method, "apply_grad"):
      self.method.apply_grad(loss, scaler, amp_dtype, self.optimization)
    else:
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
