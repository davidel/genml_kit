"""RL trainer: supports both off-policy (DQN, SAC) and on-policy (PPO).

Extends ``BaseTrainer`` with RL-specific ``train_epoch`` and
``validate`` to bypass the DataLoader and work with environments
and replay/rollout buffers directly.
"""

import logging
import os

import numpy as np
import torch

from genml_kit.training.train_reporting import TrainReporting
from genml_kit.training.trainer import BaseTrainer
from genml_kit.training.video_utils import write_video
from genml_kit.utils.attr import get_attribute, MISSING


def _terminated_from(info, done):
  """Extract the true MDP-end flag from a step_env ``info`` dict.

  ``RLPipeline.step_env`` surfaces ``info["terminated"]`` (Gymnasium
  5-tuple API).  For old-gym 4-tuple envs (or scripted envs) it is
  absent, in which case we fall back to ``done`` (terminated-or-
  truncated), preserving the historical behaviour.
  """
  if isinstance(info, dict):
    term = info.get("terminated")
    if term is not None:
      return float(term)
  return float(done)


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
    # NOTE: method.build_optimization() is now called by optim_factory
    # with the proper ckpt_extra, so we no longer need to override here.

  def train_epoch(self, epoch, saver, step, monitor):
    """One training epoch.

    Off-policy (DQN/SAC): warmup fill + interleaved env/learn.
    On-policy (PPO): rollout collection + GAE + SGD epochs.

    Returns:
        ``(avg_loss, new_step)``
    """
    if getattr(self.method, "IS_ON_POLICY", False):
      # D3: Step PPO scheduler if present
      if self.optimization.scheduler is not None:
        self.optimization.scheduler.step()
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
    # Determine if continuous action space: Discrete has .n, Box has .shape
    action_space = getattr(self.pipeline.env, 'action_space', None)
    is_continuous = action_space is not None and hasattr(
        action_space, 'shape') and getattr(action_space, 'shape', ()) != ()
    action_dim = getattr(action_space, 'shape', (1,))[0] if is_continuous else 1
    while len(self.pipeline.replay_buffer) < self.args.warmup_steps:
      if is_continuous:
        # Sample random continuous action from [-1, 1]
        action = np.random.uniform(-1, 1, size=action_dim).astype(np.float32)
      else:
        action = int(torch.randint(0, self.method.n_actions, (1,)).item())
      next_obs, reward, done, info = self.pipeline.step_env(action)
      terminated = _terminated_from(info, done)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done),
                                       terminated)
      obs = next_obs if not done else self.pipeline.reset_env()
      if hasattr(self.method, 'step_epsilon'):
        self.method.step_epsilon()
    self._warmup_obs = obs

    # D4: SAC hard-target sync on warmup complete
    # For SAC, after warmup, sync target networks with online networks
    if hasattr(self.method, '_get_alpha') and hasattr(self.model, 'hard_update'):
      self.model.hard_update()
      logging.info("SAC: Hard target update after warmup complete")

    # Phase 2: interleaved acting + learning.
    total_loss = 0.0
    batches = 0

    reporter = TrainReporting(
        total_batches=self.args.steps_per_epoch,
        log_every=getattr(self.args, "log_every", 50),
        writer=self.writer,
        device=self.device,
        optimizer=self.optimization.optimizer,
        throughput_unit="step",
    )

    # Determine if continuous action space (needed for step_env)
    action_space = getattr(self.pipeline.env, 'action_space', None)
    is_continuous = action_space is not None and hasattr(
        action_space, 'shape') and getattr(action_space, 'shape', ()) != ()

    for _step in range(self.args.steps_per_epoch):
      # Act.
      action = self.method.act(self.model, obs, deterministic=False)
      # For continuous, step_env expects the full action array; for discrete, a scalar.
      if is_continuous:
        # action is already a 1D numpy array from SAC.act()
        action_for_env = action
      else:
        action_for_env = int(action) if hasattr(action, 'item') else int(action)
      next_obs, reward, done, info = self.pipeline.step_env(action_for_env)
      terminated = _terminated_from(info, done)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done),
                                       terminated)
      if hasattr(self.method, 'step_epsilon'):
        self.method.step_epsilon()

      obs = next_obs if not done else self.pipeline.reset_env()

      # Learn.
      batch = self.pipeline.replay_buffer.sample(batch_size)
      # Anneal beta for PER
      if hasattr(self.pipeline.replay_buffer, 'anneal_beta'):
        self.pipeline.replay_buffer.anneal_beta(self.method._env_steps)
      batch = self.pipeline.to_device(batch, self.device)

      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and self.device.type == "cuda"),
      ):
        loss_out = self.method.train_step(self.model, batch, step)

      # D2: NaN guard - check for NaN loss before backward
      if torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any():
        logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
        continue

      # PER: Update priorities if supported
      if hasattr(self.pipeline.replay_buffer, 'update_priorities') and \
         hasattr(loss_out, 'td_errors') and loss_out.td_errors is not None:
        indices = batch.get('indices')
        if indices is not None:
          td_errors = loss_out.td_errors.detach().cpu().numpy()
          new_priorities = np.abs(td_errors) + 1e-6
          self.pipeline.replay_buffer.update_priorities(indices.numpy(), new_priorities)

      self._apply_grad(loss_out, scaler, amp_dtype)

      # D2: Gradient norm logging
      if self.writer is not None and hasattr(self.method, 'apply_grad'):
        # For SAC with custom apply_grad, gradients are already applied
        pass
      elif self.writer is not None and self.optimization.optimizer is not None:
        # Log gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
          if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        self.writer.add_scalar("train/grad_norm", total_norm, step)

      step += 1
      self.method.update_target(self.model, global_step=step)

      total_loss += loss_out.loss.item()
      batches += 1

      if monitor is not None:
        monitor.step(loss_out.loss, step)

      # Build extra metrics for the reporter.
      extra = {}
      if hasattr(self.method, "_epsilon"):
        extra["epsilon"] = self.method._epsilon
      if hasattr(self.method, "_get_alpha"):
        extra["alpha"] = self.method._get_alpha()
      extra["env_steps"] = self.method._env_steps
      extra["buffer_size"] = len(self.pipeline.replay_buffer)
      for k, v in loss_out.metrics.items():
        extra[k] = v.item() if hasattr(v, "item") else v

      reporter.step(
          batch_idx=batches,
          batch_size=batch_size,
          loss_value=loss_out.loss.item(),
          global_step=step,
          extra_metrics=extra,
          report_now=(batches + 1 == self.args.steps_per_epoch),
      )

    reporter.summary()
    return reporter.epoch_avg_loss(), step

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
      next_obs, reward, done, info = self.pipeline.step_env(action)
      terminated = _terminated_from(info, done)
      rollout.add(obs, action, log_prob, reward, value, float(done), raw_action,
                  terminated)

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
    # Total number of mini-batches across all PPO epochs.
    total_pico_batches = self.method._ppo_epochs * (
        rollout_len // mini_batch_size)

    reporter = TrainReporting(
        total_batches=total_pico_batches,
        log_every=getattr(self.args, "log_every", 50),
        writer=self.writer,
        device=self.device,
        optimizer=self.optimization.optimizer,
        throughput_unit="step",
    )

    for _ppo_epoch in range(self.method._ppo_epochs):
      # Use rollout's sample method to get properly formatted mini-batches
      for _ in range(0, rollout_len, mini_batch_size):
        batch = rollout.sample(mini_batch_size)
        batch = self.pipeline.to_device(batch, self.device)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and self.device.type == "cuda"),
        ):
          loss_out = self.method.train_step(self.model, batch, step)

        # D2: NaN guard - check for NaN loss before backward
        if torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any():
          logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
          continue

        self._apply_grad(loss_out, scaler, amp_dtype)

        # D2: Gradient norm logging
        if self.writer is not None and self.optimization.optimizer is not None:
          # Log gradient norm
          total_norm = 0.0
          for p in self.model.parameters():
            if p.grad is not None:
              param_norm = p.grad.data.norm(2)
              total_norm += param_norm.item()**2
          total_norm = total_norm**0.5
          self.writer.add_scalar("train/grad_norm", total_norm, step)

        step += 1

        total_loss += loss_out.loss.item()
        batches += 1

        if monitor is not None:
          monitor.step(loss_out.loss, step)

        # Build extra metrics for the reporter.
        extra = {"episodes": episode_count}
        extra["env_steps"] = self.method._env_steps
        for k, v in loss_out.metrics.items():
          extra[k] = v.item() if hasattr(v, "item") else v

        reporter.step(
            batch_idx=batches - 1,
            batch_size=mini_batch_size,
            loss_value=loss_out.loss.item(),
            global_step=step,
            extra_metrics=extra,
            report_now=(batches == total_pico_batches),
        )

    reporter.summary()
    return reporter.epoch_avg_loss(), step

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
      # loss is a LossOutput namedtuple; extract the scalar loss tensor
      loss_tensor = loss.loss if hasattr(loss, 'loss') else loss
      self.optimization.optimizer.zero_grad(set_to_none=True)
      if scaler is not None:
        scaler.scale(loss_tensor).backward()
        scaler.step(self.optimization.optimizer)
        scaler.update()
      else:
        loss_tensor.backward()
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
      record = bool(getattr(self.args, "record_eval_video", False))
      metrics = self.method.evaluate(
          self.model,
          self.pipeline,
          getattr(self.args, "eval_episodes", 5),
          record_video=record,
      )
      if record:
        self._write_eval_videos(metrics.get("episode_frames"))
      return float(metrics["eval_return"])
    finally:
      if old_state is not None:
        np.random.set_state(old_state)

  def _write_eval_videos(self, episode_frames):
    """Write one video per evaluation episode under ``<checkpoint>/videos/``.

    Args:
      episode_frames: Optional list of per-episode frame lists (from the
        ``episode_frames`` metric).  Episodes with no captured frames are
        skipped; ``None``/missing is a no-op.
    """
    if not episode_frames:
      return
    out_dir = os.path.join(getattr(self.args, "checkpoint", "."), "videos")
    for i, frames in enumerate(episode_frames):
      frames = [f for f in (frames or []) if f is not None]
      if not frames:
        continue
      path = os.path.join(out_dir, f"eval_episode_{i:03d}.mp4")
      write_video(frames, path)

  def saver_extra(self):
    """Extra state attached to every checkpoint write (method state)."""
    extra = {"method_state": self.method.get_checkpoint_state(self.model, self.args)}
    # Include pipeline state (e.g., obs normalization RMS)
    fn = get_attribute(self.pipeline, "get_checkpoint_state")
    if fn is not MISSING:
      extra.update(fn())
    # Include rollout buffer state for PPO resume
    fn = get_attribute(self.pipeline, "rollout_buffer.state_dict")
    if fn is not MISSING:
      extra["rollout_buffer"] = fn()
    return extra

  def ckpt_extra(self, best, step):
    """Extra state attached to best-checkpoint and exit saves."""
    return {}
