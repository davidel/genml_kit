"""RL trainer: warmup buffer + interleaved env/learn loop.

Phase 1 of ``plans/RL_PLAN.md`` (§6.10).  Extends ``BaseTrainer`` with
an RL-specific ``train_epoch`` (warmup + env stepping + learning) and
``validate`` (policy evaluation on the environment).
"""

import logging

import numpy as np
import torch

from genml_kit.training.trainer import BaseTrainer


class RLTrainer(BaseTrainer):
  """Training loop for off-policy RL methods (DQN, …).

  Inherits the full ``run()`` lifecycle from ``BaseTrainer`` (signals,
  checkpointing, AMP, grad monitor).  Overrides ``train_epoch`` and
  ``validate`` to bypass the DataLoader and work with environments
  and replay buffers directly.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._warmup_obs = None

  def train_epoch(self, epoch, saver, step, monitor):
    """One training epoch: warmup (if needed) + interleaved env/learn.

    Returns:
        ``(avg_loss, new_step)``
    """
    method = self.method
    model = self.model
    pipeline = self.pipeline
    buffer = pipeline.replay_buffer
    device = self.device
    args = self.args
    batch_size = getattr(args, "batch_size", 64)
    scaler = self.optimization.scaler
    amp_dtype = getattr(args, "amp_dtype", None)

    # ---- Initialise environment (no-op after first epoch) ----
    pipeline.init_env(args)

    # ---- Phase 1: warmup fill ----
    obs = self._warmup_obs
    if obs is None:
      obs = pipeline.reset_env()
    while len(buffer) < args.warmup_steps:
      action = torch.randint(0, method.n_actions, (1,)).item()
      next_obs, reward, done, _ = pipeline.step_env(action)
      buffer.push(obs, action, reward, next_obs, float(done))
      obs = next_obs if not done else pipeline.reset_env()
    self._warmup_obs = obs

    # ---- Phase 2: interleaved acting + learning ----
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

      loss = loss_out.loss
      self.optimization.optimizer.zero_grad(set_to_none=True)

      if scaler is not None:
        scaler.scale(loss).backward()
        if args.grad_accum_steps > 1:
          pass  # grad accum via no_sync context (handled by parent).
        scaler.step(self.optimization.optimizer)
        scaler.update()
      else:
        loss.backward()
        if args.grad_accum_steps > 1:
          pass  # simplified: no grad accum context for now.
        self.optimization.optimizer.step()

      step += 1
      method.update_target(model, global_step=step)

      total_loss += loss_out.loss.item()
      batches += 1

      if monitor is not None:
        monitor.step(loss_out.loss, step)

    avg_loss = total_loss / max(batches, 1)
    if self.writer is not None:
      self.writer.add_scalar("train/loss", avg_loss, epoch)
      self.writer.add_scalar("epsilon", method._epsilon, epoch)
      self.writer.add_scalar("env_steps", method._env_steps, epoch)

    logging.info(
        "epoch=%d  avg_loss=%.4f  epsilon=%.4f  env_steps=%d  "
        "buffer_size=%d",
        epoch,
        avg_loss,
        method._epsilon,
        method._env_steps,
        len(buffer),
    )
    return avg_loss, step

  def validate(self):
    """Policy evaluation on the environment (not a DataLoader)."""
    env_seed = getattr(self.args, "env_seed", None)
    if env_seed is not None:
      np.random.seed(env_seed)
    return self.method.evaluate(
        self.model,
        self.pipeline,
        getattr(self.args, "eval_episodes", 5),
    )
