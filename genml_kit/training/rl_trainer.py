"""RL trainer: supports both off-policy (DQN, SAC) and on-policy (PPO).

Extends ``BaseTrainer`` with RL-specific ``train_epoch`` and
``validate`` to bypass the DataLoader and work with environments
and replay/rollout buffers directly.
"""

import logging
import os

import numpy as np
import torch

from genml_kit.datasets.rollout_buffer import RolloutBuffer
from genml_kit.training.train_reporting import (MetricKind, metric, TrainReporting)
from genml_kit.training.trainer import BaseTrainer
from genml_kit.training.video_utils import write_video
from genml_kit.utils.args import amp_dtype_from_args
from genml_kit.utils.attr import get_attribute, maybe_call, MISSING
from genml_kit.utils.distributed import (
    all_gather_tensor,
    all_reduce_gradients,
    any_rank_true,
    is_distributed,
    is_main_process,
    rank,
    world_size,
)
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import resolve_seed


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


def _reduce_rl_gradients(model):
  """All-reduce *model*'s gradients across ranks, when distributed.

  RL models are **not** DDP-wrapped (see ``RLMethod.wrap_model``): their
  autograd path runs through methods like ``model.get_action_and_value``,
  which never enter DDP's reducer, so gradients would otherwise stay
  rank-local and each rank would train a different model.

  The ``is_distributed()`` guard lives here, at the call site, so the
  single-device path -- the common one -- never materialises a parameter or
  gradient list.  No-op outside ``torchrun``.
  """
  if is_distributed():
    all_reduce_gradients(model.parameters())


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
    # §2.15: ``--grad_accum_steps`` is a supervised-only knob; RL bypasses
    # BaseTrainer.train_epoch, so it would be silently ignored here and
    # quietly break the documented "effective batch = local x accum x
    # world_size" relationship.  Reject it and point at the per-algorithm
    # knobs that actually change the effective batch: PPO ->
    # --ppo_mini_batch_size; off-policy (DQN/SAC) -> --batch_size.
    if self.args.grad_accum_steps > 1:
      fatal(f"--grad_accum_steps={self.args.grad_accum_steps} is not supported "
            f"for RL methods. Use --ppo_mini_batch_size (PPO) or --batch_size "
            f"(off-policy) to enlarge the effective batch.")
    self._warmup_obs = None
    # One-shot guard: sync target nets to the online nets exactly once,
    # after the very first warmup fill (not on every epoch).
    self._targets_synced = False
    # NOTE: method.build_optimization() is now called by optim_factory
    # with the proper ckpt_extra, so we no longer need to override here.

  def train_epoch(self, epoch, saver, step, monitor):
    """One training epoch.

    Off-policy (DQN/SAC): warmup fill + interleaved env/learn.
    On-policy (PPO): rollout collection + GAE + SGD epochs.

    Returns:
        ``(avg_loss, new_step)``
    """
    if self.method.IS_ON_POLICY:
      if self.optimization.scheduler is not None:
        self.optimization.scheduler.step()
      return self._train_epoch_ppo(epoch, saver, step, monitor)
    return self._train_epoch_offpolicy(epoch, saver, step, monitor)

  def _train_epoch_offpolicy(self, epoch, saver, step, monitor):
    """Off-policy: warmup fill + interleaved acting + learning."""
    batch_size = self.args.batch_size
    scaler = getattr(self.optimization, "scaler", None)
    amp_dtype = amp_dtype_from_args(self.args)

    # Initialise environment (no-op after first epoch).
    self.pipeline.init_env(self.args)

    # Phase 1: warmup fill.
    obs = self._warmup_obs
    if obs is None:
      obs = self.pipeline.reset_env()
    action_shape = get_attribute(self.pipeline, "env.action_space.shape")
    is_continuous = action_shape is not MISSING and action_shape != ()
    while len(self.pipeline.replay_buffer) < self.args.warmup_steps:
      if is_continuous:
        action = np.random.uniform(-1, 1, size=action_shape[0]).astype(np.float32)
      else:
        action = int(torch.randint(0, self.method.n_actions, (1,)).item())
      next_obs, reward, done, info = self.pipeline.step_env(action)
      terminated = _terminated_from(info, done)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done),
                                       terminated)
      obs = next_obs if not done else self.pipeline.reset_env()
      maybe_call(self.method, "step_epsilon")
    self._warmup_obs = obs

    # D4: SAC hard-target sync, exactly ONCE after the first warmup fill.
    #
    # This whole method runs once *per epoch*, and the warmup ``while``
    # above is already satisfied after epoch 0 (``_warmup_obs`` is cached),
    # so without the ``_targets_synced`` guard this block would fire every
    # epoch -- re-copying ``target <- online`` and destroying the target
    # lag that SAC's soft Polyak updates rely on.  Symptom of the missing
    # guard: "SAC: Hard target update after warmup complete" logged once
    # per epoch, with the critic bootstrapping off its own drifting weights.
    if (not self._targets_synced and hasattr(self.method, '_get_alpha') and
        maybe_call(self.model, "hard_update") is not MISSING):
      self._targets_synced = True
      logging.info("SAC: Hard target update after warmup complete")

    total_loss = 0.0
    batches = 0

    reporter = TrainReporting(
        total_batches=self.args.steps_per_epoch,
        log_every=self.args.log_every,
        logger=self.logger,
        device=self.device,
        optimizer=self.optimization.optimizer,
        throughput_unit="step",
        epoch=epoch,
    )

    # Determine if continuous action space (needed for step_env)
    action_shape = get_attribute(self.pipeline, "env.action_space.shape")
    is_continuous = action_shape is not MISSING and action_shape != ()

    for _step in range(self.args.steps_per_epoch):
      # Act.
      action = self.method.act(self.model, obs, deterministic=False)
      # For continuous, step_env expects the full action array; for discrete, a scalar.
      # action is already a 1D numpy array from SAC.act()
      action_for_env = action if is_continuous else int(action)
      next_obs, reward, done, info = self.pipeline.step_env(action_for_env)
      terminated = _terminated_from(info, done)
      self.pipeline.replay_buffer.push(obs, action, reward, next_obs, float(done),
                                       terminated)
      maybe_call(self.method, "step_epsilon")

      obs = next_obs if not done else self.pipeline.reset_env()

      # Learn.
      batch = self.pipeline.replay_buffer.sample(batch_size)
      # Anneal beta for PER
      maybe_call(self.pipeline.replay_buffer, "anneal_beta", self.method._env_steps)
      batch = self.pipeline.to_device(batch, self.device)

      with torch.amp.autocast(
          "cuda",
          dtype=amp_dtype,
          enabled=(amp_dtype is not None and self.device.type == "cuda"),
      ):
        loss_out = self.method.train_step(self.model, batch, step)

      # Reduce the skip decision across ranks: ``continue`` below also skips
      # ``_apply_grad`` -> ``_reduce_rl_gradients``, so a rank-local skip would
      # enter the all-reduce on a subset of ranks (hang, or the
      # ``all_reduce_gradients`` count guard firing).  One rank's NaN discards
      # the step on *all* ranks, which is the only choice that keeps replicas
      # in lockstep.  Matches the PPO path's ``any_rank_true`` below.
      is_bad = bool(
          torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any())
      if any_rank_true(is_bad):
        if is_bad:
          logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
        continue

      update_priorities = get_attribute(self.pipeline.replay_buffer,
                                        "update_priorities")
      td_errors = get_attribute(loss_out, "td_errors")
      if update_priorities is not MISSING and td_errors is not MISSING and \
         td_errors is not None:
        indices = batch.get('indices')
        if indices is not None:
          new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
          update_priorities(indices.numpy(), new_priorities)

      self._apply_grad(loss_out, scaler, amp_dtype)

      if hasattr(self.method, 'apply_grad'):
        pass
      elif self.optimization.optimizer is not None:
        total_norm = 0.0
        for p in self.model.parameters():
          if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        self.logger.log_step("train/grad_norm",
                             total_norm,
                             step=step,
                             resolution="window",
                             kind=MetricKind.LAST)

      step += 1
      self.method.update_target(self.model, global_step=step)

      total_loss += loss_out.loss.item()
      batches += 1

      if monitor is not None:
        monitor.step(loss_out.loss, step)

      # Build extra metrics for the reporter.  ``env_steps`` / ``buffer_size``
      # are monotonic counters, so they report their last value in the epoch
      # summary rather than a meaningless mean over the ramp.
      extra = {}
      epsilon = get_attribute(self.method, "_epsilon")
      if epsilon is not MISSING:
        extra["epsilon"] = metric("epsilon", epsilon)
      alpha = maybe_call(self.method, "_get_alpha")
      if alpha is not MISSING:
        extra["alpha"] = metric("alpha", alpha)
      extra["env_steps"] = metric("env_steps", self.method._env_steps, MetricKind.LAST,
                                  ".0f")
      extra["buffer_size"] = metric("buffer_size", len(self.pipeline.replay_buffer),
                                    MetricKind.LAST, ".0f")
      for k, v in loss_out.metrics.items():
        extra[k] = metric(k, v)

      reporter.step(
          batch_idx=batches,
          batch_size=batch_size,
          # ``TrainReporting`` computes ``loss = sum(loss_value) / sum(batch_size)``
          # (per-sample semantics), while RL losses are reported per-*batch*
          # means.  Scale by the mini-batch size so the printed ``loss`` is the
          # true mean objective (historical bug: the reported loss was
          # ``raw_loss / batch_size``, e.g. ~65 instead of ~5300 for PPO).
          loss_value=loss_out.loss.item() * batch_size,
          global_step=step,
          extra_metrics=extra,
          report_now=(batches + 1 == self.args.steps_per_epoch),
      )

    reporter.summary()
    return reporter.epoch_avg_loss(), step

  def _train_epoch_ppo(self, epoch, saver, step, monitor):
    """On-policy: collect rollout \u2192 compute GAE \u2192 SGD epochs."""
    scaler = getattr(self.optimization, "scaler", None)
    amp_dtype = amp_dtype_from_args(self.args)

    # Initialise environment (no-op after first epoch).
    self.pipeline.init_env(self.args)

    rollout = self.pipeline.rollout_buffer
    rollout_len = rollout.rollout_len
    obs = getattr(self, "_ppo_obs", None)

    # Sharded rollouts (default under DDP): each rank collects
    # ``rollout_len / world`` steps from its *own* env, then the per-rank
    # segments are all-gathered into one global rollout that is identical on
    # every rank.  ``--dist_arg rollout_shard=false`` reverts to one full
    # rollout per rank (today's behaviour).
    world = world_size()
    sharded = bool(self.args.dist_arg.get("rollout_shard", True)) and world > 1
    if sharded and rollout_len % world != 0:
      fatal(f"--ppo_rollout_len ({rollout_len}) must be divisible by the world "
            f"size ({world}) when rollout sharding is enabled; pass "
            f"--dist_arg rollout_shard=false to collect a full rollout per rank.")
    local_len = rollout_len // world if sharded else rollout_len

    # Collect into a rank-local buffer in sharded mode; reuse the pipeline's
    # buffer (already sized rollout_len) otherwise.
    collect = (RolloutBuffer(obs_dim=rollout.obs_dim,
                             rollout_len=local_len,
                             action_dim=rollout.action_dim,
                             device="cpu") if sharded else rollout)

    collect.reset()
    obs = self.pipeline.reset_env() if obs is None else obs
    episode_count = 0
    total_reward = 0.0
    # Reward/return normalization (SB3 VecNormalize) - PPO-only.
    # getattr: fake/test pipelines may not define ret_norm.
    ret_norm = getattr(self.pipeline, "ret_norm", None)
    if ret_norm is not None:
      ret_norm.train(True)
      ret_norm.reset_per_episode()

    # Collect per-step bootstrap values V(s_{t+1}) during rollout.
    next_values = torch.zeros(local_len, dtype=torch.float32)

    for i in range(local_len):
      action, log_prob, value, raw_action = self.method.act(self.model,
                                                            obs,
                                                            deterministic=False)
      next_obs, reward, done, info = self.pipeline.step_env(action)
      terminated = _terminated_from(info, done)
      # Update the discounted-return running stats with the RAW reward,
      # then normalize what is stored in the buffer (SB3 sequence).
      if ret_norm is not None:
        ret_norm.update(reward)
        reward = ret_norm.normalize_reward(reward)
      collect.add(obs, action, log_prob, reward, value, float(done), raw_action,
                  terminated)

      with torch.no_grad():
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
        next_values[i] = self.model.get_value(next_obs_t).item()

      total_reward += reward
      obs = next_obs

      if done:
        if ret_norm is not None:
          ret_norm.reset_per_episode()
        total_reward = 0.0
        episode_count += 1
        obs = self.pipeline.reset_env()

    # Bootstrap values for GAE (per-step, not a single scalar).  GAE is
    # computed *per rank* over that rank's own contiguous segment (each
    # segment is a self-contained trajectory with its own done/terminated
    # markers and bootstrap value), so no cross-rank boundary is bridged.
    collect.set_next_values(next_values)
    collect.compute(gamma=self.method._gamma, lam=self.method._lam)

    if sharded:
      # Gather the rank segments into one global rollout on every rank, then
      # standardise advantages ONCE over the *global* rollout (matching the
      # single-process statistics).
      self._gather_rollout(collect, rollout)
    # Standardise advantages ONCE over the full rollout (not per
    # mini-batch) -- see ``RolloutBuffer.normalize_advantages``.
    rollout.normalize_advantages()
    rollout.to(self.device)

    self._ppo_obs = obs

    # B5: track env steps collected in this rollout.  Count the steps this
    # rank actually collected: in sharded mode each rank takes
    # ``rollout_len / world`` from its own env, so adding the global
    # ``rollout_len`` here would over-count the fleet's real env interaction
    # by a factor of ``world``.
    self.method._env_steps += local_len

    total_loss = 0.0
    batches = 0
    mini_batch_size = self.method._mini_batch_size
    # Total number of mini-batches across all PPO epochs.
    total_pico_batches = self.method._ppo_epochs * (rollout_len // mini_batch_size)

    # Sharded mode: every rank draws the *same* global minibatch indices
    # (shared generator, no rank offset) and then trains on a disjoint
    # 1/world slice.  DDP averages the per-rank gradients, so this is
    # numerically identical to single-process training on the global rollout.
    if sharded and mini_batch_size % world != 0:
      fatal(f"--ppo_mini_batch_size ({mini_batch_size}) must be divisible by "
            f"the world size ({world}) when rollout sharding is enabled; pass "
            f"--dist_arg rollout_shard=false to train a full minibatch per rank.")
    sample_rng = None
    if sharded:
      sample_rng = torch.Generator()
      sample_rng.manual_seed(resolve_seed(self.args.seed) + epoch)

    reporter = TrainReporting(
        total_batches=total_pico_batches,
        log_every=self.args.log_every,
        logger=self.logger,
        device=self.device,
        optimizer=self.optimization.optimizer,
        throughput_unit="step",
        epoch=epoch,
    )

    target_kl = getattr(self.method, "_target_kl", None)
    for _ppo_epoch in range(self.method._ppo_epochs):
      # Use rollout's sample method to get properly formatted mini-batches
      stop_epoch = False
      for _ in range(0, rollout_len, mini_batch_size):
        if sharded:
          # Identical global indices on every rank -> the rank slices tile
          # exactly one global minibatch.
          idx = rollout.sample_indices(mini_batch_size, generator=sample_rng)
          batch = rollout.batch_at(idx[rank()::world])
        else:
          batch = rollout.sample(mini_batch_size)
        batch = self.pipeline.to_device(batch, self.device)

        with torch.amp.autocast(
            "cuda",
            dtype=amp_dtype,
            enabled=(amp_dtype is not None and self.device.type == "cuda"),
        ):
          loss_out = self.method.train_step(self.model, batch, step)

        # SB3-style target_kl early-stop: if the k1 approx-KL from this
        # minibatch update exceeds the threshold, stop the whole epoch
        # (remaining minibatches are skipped) -- prevents destructive
        # updates when the policy drifts too far in one epoch.  The decision
        # is reduced across ranks: a rank-local early stop would desync the
        # DDP gradient all-reduce and hang the job.
        if target_kl is not None:
          kl = loss_out.metrics.get("approx_kl")
          stop_epoch = any_rank_true(kl is not None and kl > target_kl)

        # Likewise, the NaN/Inf skip must be agreed on by every rank (else
        # some ranks call backward and others do not -> collective hang).
        is_bad = bool(
            torch.isnan(loss_out.loss).any() or torch.isinf(loss_out.loss).any())
        if any_rank_true(is_bad):
          if is_bad:
            logging.warning("NaN/Inf loss detected at step %d, skipping update", step)
          continue

        self._apply_grad(loss_out, scaler, amp_dtype)

        if self.optimization.optimizer is not None:
          total_norm = 0.0
          for p in self.model.parameters():
            if p.grad is not None:
              param_norm = p.grad.data.norm(2)
              total_norm += param_norm.item()**2
          total_norm = total_norm**0.5
          self.logger.log_step("train/grad_norm",
                               total_norm,
                               step=step,
                               resolution="window",
                               kind=MetricKind.LAST)

        step += 1

        total_loss += loss_out.loss.item()
        batches += 1

        if monitor is not None:
          monitor.step(loss_out.loss, step)

        # Build extra metrics for the reporter.  ``episodes`` / ``env_steps``
        # are monotonic counters -> last value in the epoch summary.
        extra = {"episodes": metric("episodes", episode_count, MetricKind.LAST, ".0f")}
        extra["env_steps"] = metric("env_steps", self.method._env_steps,
                                    MetricKind.LAST, ".0f")
        for k, v in loss_out.metrics.items():
          extra[k] = metric(k, v)

        reporter.step(
            batch_idx=batches - 1,
            batch_size=mini_batch_size,
            # Per-batch mean loss * batch size -> TrainReporting divides by the
            # summed batch sizes to recover the true mean (see off-policy note).
            loss_value=loss_out.loss.item() * mini_batch_size,
            global_step=step,
            extra_metrics=extra,
            report_now=(batches == total_pico_batches),
        )

        # SB3-style target_kl: stop the *whole* epoch once exceeded.
        if stop_epoch:
          break

      if stop_epoch:
        break

    reporter.summary()
    return reporter.epoch_avg_loss(), step

  def _gather_rollout(self, local, global_rollout):
    """Gather per-rank rollout *local* into *global_rollout* on every rank.

    Each rank contributed ``global_rollout.rollout_len / world`` contiguous
    steps; they are concatenated in rank order so every rank holds the same
    global rollout.  GAE was already computed per rank (each segment is a
    self-contained trajectory), so only the resulting arrays are exchanged;
    ``normalize_advantages`` then runs on the assembled global rollout.
    """
    arrays = local.array_state()
    gathered = {}
    for key, value in arrays.items():
      gathered[key] = None if value is None else all_gather_tensor(value)
    global_rollout.load_arrays(gathered)

  def _apply_grad(self, loss, scaler, amp_dtype):
    """Backward + optimizer step (shared by all RL flows).

    If the method provides a custom ``apply_grad``, delegate to it.
    This enables multi-optimizer patterns (e.g., SAC); in that case the
    method is responsible for its own ``backward``/``step`` ordering and
    for honouring ``--grad_clip`` (see ``SACMethod.apply_grad``).

    NOTE: unlike the supervised/image path (``BaseTrainer``), the generic
    RL branch below historically ignored ``--grad_clip``, so off-policy
    methods without a custom ``apply_grad`` trained unclipped.  Clipping is
    applied here now, *before* the optimizer step.
    """
    if maybe_call(self.method, "apply_grad", loss, scaler, amp_dtype,
                  self.optimization) is MISSING:
      # loss is a LossOutput namedtuple; extract the scalar loss tensor
      loss_tensor = get_attribute(loss, "loss")
      if loss_tensor is MISSING:
        loss_tensor = loss
      self.optimization.optimizer.zero_grad(set_to_none=True)
      if scaler is not None:
        scaler.scale(loss_tensor).backward()
        # Unscale before clipping so the clip sees true (unscaled) norms.
        if self.args.grad_clip > 0:
          scaler.unscale_(self.optimization.optimizer)
        # Reduce before clipping: the clip threshold must act on the global
        # gradient norm, not each rank's local one.
        _reduce_rl_gradients(self.model)
        if self.args.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.args.grad_clip)
        scaler.step(self.optimization.optimizer)
        scaler.update()
      else:
        loss_tensor.backward()
        # Reduce before clipping (see above).
        _reduce_rl_gradients(self.model)
        if self.args.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                         max_norm=self.args.grad_clip)
        self.optimization.optimizer.step()

  def validate(self):
    """Policy evaluation on the environment (not a DataLoader).

    Distributed policy mirrors ``BaseTrainer.validate``: only the owning rank
    evaluates, the others return ``None`` and wait at the caller's
    ``barrier()``.  This override previously ran on *every* rank, and because
    each rank's env is seeded differently (``per_rank_seed``) the ranks
    evaluated different policies -- N-way redundant work plus an inconsistent
    "New best" metric logged by every rank.

    Returns:
      Scalar eval_return (float) for checkpoint selection, or ``None`` on a
      rank that does not own validation (see ``_should_validate``).
    """
    if not self._should_validate():
      return None
    env_seed = self.args.env_seed
    # Isolate RNG state to avoid corrupting training sampling (B2).
    if env_seed is not None:
      old_state = np.random.get_state()
      np.random.seed(env_seed)
    else:
      old_state = None
    try:
      record = bool(self.args.record_eval_video)
      # Eval must see RAW rewards (SB3 VecNormalize train(False)):
      # the running-return stats are not updated, and rewards are not
      # normalized, so eval_return stays comparable across runs.
      ret_norm = getattr(self.pipeline, "ret_norm", None)
      if ret_norm is not None:
        ret_norm.train(False)
      try:
        metrics = self.method.evaluate(
            self.model,
            self.pipeline,
            self.args.eval_episodes,
            record_video=record,
        )
      finally:
        if ret_norm is not None:
          ret_norm.train(True)
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
    # Rank 0 owns every file sink (checkpoints, logs): without this guard each
    # rank wrote the same eval_episode_*.mp4 paths and clobbered the others.
    if not episode_frames or not is_main_process():
      return
    out_dir = os.path.join(self.args.checkpoint, "videos")
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
