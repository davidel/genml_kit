"""Tests for RL under DDP (P4b) and the explicit-gradient-reduction seam.

Two classes of test live here:

* **Single-process** tests for the pure functions the distributed path is
  built from -- ``all_reduce_gradients`` argument handling,
  ``RolloutBuffer.sample_indices``/``batch_at``, ``_gather_rollout`` ordering,
  GAE-per-segment equivalence, and ``per_rank_seed``.
* **Multi-process** tests that spawn two gloo ranks on CPU.  These are the
  only ones that can prove the collective behaviour actually works, and they
  cover the two bugs found while writing this file:

  1. DDP reduces gradients only in ``forward``; every RL method enters
     autograd through a *method* on the model (``model.get_action_and_value``
     etc.), so a DDP wrap never fired and ranks silently diverged.  Fixed by
     not wrapping RL models and calling ``all_reduce_gradients`` explicitly.
  2. The driver seeds each rank differently (``seed + rank()``), so
     ``build_model`` drew different initial weights per rank.  The supervised
     path was masked by ``DistributedDataParallel.__init__`` broadcasting rank
     0's parameters; RL had no such rescue.  Fixed by an explicit
     ``broadcast_model_state``.

They are skipped when ``torch.distributed`` is unavailable.
"""

import os

import pytest
import torch

from genml_kit.datasets.rollout_buffer import RolloutBuffer
from genml_kit.methods.rl_dqn import DQNMethod
from genml_kit.methods.rl_ppo import PPOMethod
from genml_kit.methods.rl_sac import SACMethod
from genml_kit.methods.rl_utils import RLMethod
from genml_kit.utils import distributed
from tests.cli_args import train_args


def _gloo_available():
  return distributed.dist.is_available()


requires_gloo = pytest.mark.skipif(not _gloo_available(),
                                   reason="torch.distributed unavailable")

# Single-process: the pieces the distributed path is assembled from.


class TestRLMethodDoesNotWrap:
  """RL methods must not be DDP-wrapped (see RLMethod docstring)."""

  @pytest.mark.parametrize("method_cls", [DQNMethod, PPOMethod, SACMethod])
  def test_rl_methods_derive_from_rl_method(self, method_cls):
    assert issubclass(method_cls, RLMethod)

  @pytest.mark.parametrize("method_cls", [DQNMethod, PPOMethod, SACMethod])
  def test_wrap_model_returns_the_model_unwrapped(self, method_cls):
    method = method_cls()
    model = torch.nn.Linear(2, 2)
    args = train_args(["--pipeline", "rl"], dist_arg={})
    assert method.wrap_model(model, torch.device("cpu"), args) is model

  def test_base_method_wraps(self):
    """The default ``Method.wrap_model`` delegates to the DDP helper."""
    from genml_kit.methods.classification import ClassificationMethod

    method = ClassificationMethod()
    model = torch.nn.Linear(2, 2)
    args = train_args(["--pipeline", "rl"], dist_arg={})
    # Not distributed -> identity, but routed through the helper.
    assert method.wrap_model(model, torch.device("cpu"), args) is model


class TestAllReduceGradientsSingleProcess:
  """Outside a process group the helper must be a free no-op."""

  def test_noop_when_not_distributed(self):
    param = torch.nn.Parameter(torch.zeros(3))
    param.grad = torch.tensor([1.0, 2.0, 3.0])
    before = param.grad.clone()
    distributed.all_reduce_gradients([param])
    assert torch.equal(param.grad, before)

  def test_tolerates_missing_grads(self):
    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = None
    distributed.all_reduce_gradients([param])  # must not raise

  def test_broadcast_is_noop_when_not_distributed(self):
    model = torch.nn.Linear(2, 2)
    before = [p.detach().clone() for p in model.parameters()]
    distributed.broadcast_model_state(model)
    for old, new in zip(before, model.parameters()):
      assert torch.equal(old, new.detach())


class TestPerRankSeed:
  """``per_rank_seed`` must offset by rank so env streams diverge."""

  def _args(self, **kw):
    base = dict(seed=42, env_seed=None)
    base.update(kw)
    return train_args(["--pipeline", "rl"], **base)

  def test_falls_back_to_global_seed(self, monkeypatch):
    monkeypatch.setattr("genml_kit.pipelines.rl.rank", lambda: 0)
    from genml_kit.pipelines.rl import per_rank_seed

    assert per_rank_seed(self._args(seed=42)) == 42

  def test_env_seed_wins_over_global_seed(self, monkeypatch):
    monkeypatch.setattr("genml_kit.pipelines.rl.rank", lambda: 0)
    from genml_kit.pipelines.rl import per_rank_seed

    assert per_rank_seed(self._args(seed=42, env_seed=7)) == 7

  def test_ranks_get_distinct_seeds(self, monkeypatch):
    from genml_kit.pipelines.rl import per_rank_seed

    args = self._args(seed=42)
    monkeypatch.setattr("genml_kit.pipelines.rl.rank", lambda: 0)
    seed0 = per_rank_seed(args)
    monkeypatch.setattr("genml_kit.pipelines.rl.rank", lambda: 1)
    seed1 = per_rank_seed(args)
    assert seed1 == seed0 + 1


class TestRolloutBufferSlicing:
  """``sample_indices`` + ``batch_at`` are the sharded-minibatch primitive."""

  def _filled(self, rollout_len=8, obs_dim=2):
    buf = RolloutBuffer(obs_dim=obs_dim, rollout_len=rollout_len, seed=0)
    for i in range(rollout_len):
      buf.add(obs=[float(i), 0.0],
              action=0,
              log_prob=-0.1,
              reward=float(i),
              value=0.0,
              done=False)
    buf.set_next_values(torch.zeros(rollout_len))
    buf.compute(gamma=0.99, lam=0.95)
    return buf

  def test_sample_indices_shape_and_range(self):
    buf = self._filled()
    idx = buf.sample_indices(4)
    assert idx.shape == (4,)
    assert int(idx.min()) >= 0 and int(idx.max()) < buf.rollout_len

  def test_sample_indices_reproducible_with_generator(self):
    buf = self._filled()
    gen_a = torch.Generator().manual_seed(7)
    gen_b = torch.Generator().manual_seed(7)
    assert torch.equal(buf.sample_indices(5, gen_a), buf.sample_indices(5, gen_b))

  def test_batch_at_matches_sample(self):
    """``sample`` must be exactly ``batch_at(sample_indices(...))``."""
    buf = self._filled()
    gen = torch.Generator().manual_seed(11)
    idx = buf.sample_indices(4, gen)
    via_batch_at = buf.batch_at(idx)
    gen_replay = torch.Generator().manual_seed(11)
    via_sample = buf.sample(4, generator=gen_replay)
    for key in via_batch_at:
      assert torch.equal(via_batch_at[key], via_sample[key]), key

  def test_rank_slices_are_disjoint_and_cover_the_batch(self):
    """The (A) compute mode: same global indices, rank-strided slices."""
    buf = self._filled()
    batch_size = 8
    world = 2
    idx = buf.sample_indices(batch_size, torch.Generator().manual_seed(3))
    slices = [idx[r::world] for r in range(world)]
    assert sum(len(s) for s in slices) == batch_size
    # Positional (not value) disjointness: the two slices must reassemble the
    # original batch exactly when interleaved.
    rebuilt = torch.stack(slices, dim=1).reshape(-1)
    assert torch.equal(rebuilt, idx)


class TestGatherRollout:
  """``_gather_rollout`` must concatenate rank segments in rank order."""

  def _local(self, rollout_len, offset):
    buf = RolloutBuffer(obs_dim=1, rollout_len=rollout_len)
    for i in range(rollout_len):
      buf.add(obs=[float(offset + i)],
              action=0,
              log_prob=0.0,
              reward=float(offset + i),
              value=0.0,
              done=False)
    buf.set_next_values(torch.zeros(rollout_len))
    buf.compute(gamma=0.99, lam=0.95)
    return buf

  def test_single_rank_gather_is_identity(self):
    """With no process group, gather returns the local arrays unchanged."""
    from genml_kit.training.rl_trainer import RLTrainer

    local = self._local(4, offset=0)
    global_buf = RolloutBuffer(obs_dim=1, rollout_len=4)
    RLTrainer._gather_rollout(None, local, global_buf)
    assert torch.equal(global_buf.obs, local.obs)
    assert torch.equal(global_buf.rewards, local.rewards)
    assert global_buf.full

  def test_gaper_segment_equals_whole_when_boundary_is_done(self):
    """Per-segment GAE must equal whole-rollout GAE if the split is a boundary.

    This is the correctness guard for sharded rollouts: each rank runs GAE
    over its own segment, which is only valid when no episode crosses the
    segment boundary.
    """
    from genml_kit.losses.rl import gae

    torch.manual_seed(0)
    total, split = 8, 4
    rewards = torch.randn(total)
    values = torch.randn(total)
    next_values = torch.randn(total)
    dones = torch.zeros(total)
    dones[split - 1] = 1.0  # episode ends exactly at the boundary

    whole, _ = gae(rewards, values, next_values, dones, 0.99, 0.95)
    seg_a, _ = gae(rewards[:split], values[:split], next_values[:split], dones[:split],
                   0.99, 0.95)
    seg_b, _ = gae(rewards[split:], values[split:], next_values[split:], dones[split:],
                   0.99, 0.95)
    assert torch.allclose(whole, torch.cat([seg_a, seg_b]), atol=1e-6)

  def test_segment_gae_differs_without_boundary(self):
    """Discriminating counterpart: no boundary -> per-segment GAE is wrong.

    Without this, the test above could pass trivially.
    """
    from genml_kit.losses.rl import gae

    torch.manual_seed(0)
    total, split = 8, 4
    rewards = torch.randn(total)
    values = torch.randn(total)
    next_values = torch.randn(total)
    dones = torch.zeros(total)

    whole, _ = gae(rewards, values, next_values, dones, 0.99, 0.95)
    seg_a, _ = gae(rewards[:split], values[:split], next_values[:split], dones[:split],
                   0.99, 0.95)
    seg_b, _ = gae(rewards[split:], values[split:], next_values[split:], dones[split:],
                   0.99, 0.95)
    assert not torch.allclose(whole, torch.cat([seg_a, seg_b]), atol=1e-6)


# Multi-process: real collectives, 2 gloo ranks on CPU.


def _init(rank, world, port):
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world)
  os.environ["LOCAL_RANK"] = str(rank)
  distributed.dist.init_process_group("gloo", rank=rank, world_size=world)


def _grad_reduce_worker(rank, world, port, tmpdir):
  """Two ranks with different grads; both must end on the mean."""
  _init(rank, world, port)
  try:
    param = torch.nn.Parameter(torch.zeros(3))
    param.grad = torch.full((3,), float(rank + 1))
    distributed.all_reduce_gradients([param])
    torch.save(param.grad.clone(), os.path.join(tmpdir, f"reduce{rank}.pt"))
  finally:
    distributed.destroy_distributed()


def _mixed_dtype_worker(rank, world, port, tmpdir):
  """Gradients of several dtypes are reduced without silent upcasting."""
  _init(rank, world, port)
  try:
    lo = torch.nn.Parameter(torch.zeros(2))
    lo.grad = torch.full((2,), float(rank + 1))
    hi = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    hi.grad = torch.full((2,), 2.0 * (rank + 1), dtype=torch.float64)
    half = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    half.grad = torch.full((2,), 4.0 * (rank + 1), dtype=torch.bfloat16)
    distributed.all_reduce_gradients([lo, hi, half])
    torch.save(
        {
            "lo": lo.grad.clone(),
            "hi": hi.grad.clone(),
            "half": half.grad.clone(),
        }, os.path.join(tmpdir, f"mixed{rank}.pt"))
  finally:
    distributed.destroy_distributed()


def _broadcast_worker(rank, world, port, tmpdir):
  """Ranks build different weights; broadcast must unify them on rank 0's."""
  _init(rank, world, port)
  try:
    torch.manual_seed(1000 + rank)  # different init per rank, like seed+rank
    model = torch.nn.Linear(4, 4)
    before = model.weight.detach().clone()
    distributed.broadcast_model_state(model)
    torch.save({
        "before": before,
        "after": model.weight.detach().clone()
    }, os.path.join(tmpdir, f"bcast{rank}.pt"))
  finally:
    distributed.destroy_distributed()


def _spawn(fn, tmp_path, port, world=2, extra=()):
  # ``spawn`` pickles the target, so ``fn`` must be module level.
  torch.multiprocessing.spawn(fn,
                              args=(world, port, str(tmp_path), *extra),
                              nprocs=world,
                              join=True)


@requires_gloo
def test_gradients_are_averaged_across_ranks(tmp_path):
  """rank0 grad=1, rank1 grad=2 -> both see 1.5 (mean, like DDP)."""
  _spawn(_grad_reduce_worker, tmp_path, port=29801)
  g0 = torch.load(tmp_path / "reduce0.pt", weights_only=False)
  g1 = torch.load(tmp_path / "reduce1.pt", weights_only=False)
  assert torch.allclose(g0, torch.full((3,), 1.5))
  assert torch.equal(g0, g1)


@requires_gloo
def test_mixed_dtype_gradients_do_not_upcast(tmp_path):
  """bf16/fp64 grads keep their dtype and get the correct mean."""
  _spawn(_mixed_dtype_worker, tmp_path, port=29803)
  for rank in (0, 1):
    saved = torch.load(tmp_path / f"mixed{rank}.pt", weights_only=False)
    assert saved["lo"].dtype == torch.float32
    assert saved["hi"].dtype == torch.float64
    assert saved["half"].dtype == torch.bfloat16
    assert torch.allclose(saved["lo"], torch.full((2,), 1.5))
    assert torch.allclose(saved["hi"], torch.full((2,), 3.0, dtype=torch.float64))
    assert torch.allclose(saved["half"].float(), torch.full((2,), 6.0))


@requires_gloo
def test_broadcast_model_state_unifies_ranks(tmp_path):
  """Differing per-rank init is overwritten with rank 0's weights."""
  _spawn(_broadcast_worker, tmp_path, port=29805)
  r0 = torch.load(tmp_path / "bcast0.pt", weights_only=False)
  r1 = torch.load(tmp_path / "bcast1.pt", weights_only=False)
  # Precondition: the two ranks really did start from different weights.
  assert not torch.equal(r0["before"], r1["before"])
  # After the broadcast both hold rank 0's values.
  assert torch.equal(r0["after"], r1["after"])
  assert torch.equal(r0["after"], r0["before"])


def _ppo_epoch_worker(rank, world, port, tmpdir, rollout_len, mini_batch):
  """Run one sharded PPO epoch; dump the post-step parameters."""
  from torch.optim import Adam

  from genml_kit.training.optim_factory import Optimization
  from genml_kit.training.rl_trainer import RLTrainer

  from tests.test_rl_trainer import _make_args, _TestRLPipeline

  _init(rank, world, port)
  try:
    from genml_kit.utils.seed import resolve_seed, seed_everything
    # Exactly what cli/train.py does: a different seed per rank.
    seed_everything(resolve_seed(42) + distributed.rank())

    args = _make_args(
        ppo_rollout_len=rollout_len,
        ppo_mini_batch_size=mini_batch,
        ppo_epochs=1,
        env_steps_per_epoch=rollout_len,
        checkpoint=None,
        seed=42,
        env_seed=42,
        dist_arg={},
    )
    pipeline = _TestRLPipeline(obs_dim=4,
                               continuous=False,
                               buffer_size=100,
                               rollout_len=rollout_len)
    method = PPOMethod()
    method.wire_data(args, pipeline)
    raw = method.build_model(args, device=torch.device("cpu"))
    model = method.wrap_model(raw, torch.device("cpu"), args)
    distributed.broadcast_model_state(model)

    optimization = Optimization(optimizer=Adam(model.parameters(), lr=3e-4),
                                scheduler=None,
                                scaler=None)
    trainer = RLTrainer(args=args,
                        model=model,
                        method=method,
                        pipeline=pipeline,
                        optimization=optimization,
                        device=torch.device("cpu"),
                        logger=None,
                        start_epoch=0,
                        best_metric=float("-inf"),
                        global_step=0)
    trainer.train_epoch(0, None, step=0, monitor=None)
    params = torch.cat([p.detach().flatten() for p in model.parameters()])
    torch.save({
        "params": params,
        "rollout_len": pipeline.rollout_buffer.rollout_len,
    }, os.path.join(tmpdir, f"ppo{rank}.pt"))
  finally:
    distributed.destroy_distributed()


@requires_gloo
def test_sharded_ppo_epoch_keeps_ranks_identical(tmp_path):
  """The regression test for both bugs found while writing this file.

  A sharded PPO epoch must leave *identical* parameters on every rank.  Before
  the fix this failed: no DDP reducer fired (RL enters autograd via
  ``model.get_action_and_value``) and each rank had started from different
  initial weights.
  """
  _spawn(_ppo_epoch_worker, tmp_path, port=29807,
         extra=(8, 4))  # rollout_len, mini_batch
  r0 = torch.load(tmp_path / "ppo0.pt", weights_only=False)
  r1 = torch.load(tmp_path / "ppo1.pt", weights_only=False)

  # The sharded buffer is the GLOBAL rollout on every rank (each rank collects
  # rollout_len / world, then they are gathered into one buffer).
  assert r0["rollout_len"] == 8
  assert r1["rollout_len"] == 8
  assert torch.equal(r0["params"],
                     r1["params"]), ("post-step parameters diverged across ranks")


def _sac_step_worker(rank, world, port, tmpdir, steps):
  """Run SAC updates; dump critic/actor/alpha params post-step.

  SAC is the one RL method with a *custom* ``apply_grad``: it owns its own
  gradient reduction (three optimizers plus the bare ``log_alpha`` leaf
  tensor), so the trainer's ``_reduce_rl_gradients`` path never runs for it.
  This worker proves that per-method reduction directly.

  Each rank seeds the replay buffer with rank-distinct transitions, so the
  local gradients genuinely differ.  Without the reduction the post-step
  parameters would diverge; with it they are bit-identical.
  """
  import numpy as np

  from genml_kit.methods.rl_sac import SACMethod

  from tests.test_rl_trainer import _make_args, _TestRLPipeline

  _init(rank, world, port)
  try:
    from genml_kit.utils.seed import resolve_seed, seed_everything

    # Exactly what cli/train.py does: a different seed per rank.
    seed_everything(resolve_seed(42) + distributed.rank())

    args = _make_args(
        sac_gamma=0.99,
        sac_tau=0.005,
        sac_alpha=0.2,
        sac_auto_alpha=True,
        sac_critic_lr=1e-3,
        sac_actor_lr=1e-3,
        sac_alpha_lr=1e-3,
        grad_clip=0.0,  # isolate the reduction from any clipping effect
        checkpoint=None,
        seed=42,
    )
    # ``action_dim=2`` (not 1): the test pipeline picks its replay-buffer
    # dtype with an ``action_dim > 1`` heuristic, so a 1-D continuous action
    # would be stored as int64 scalars.
    pipeline = _TestRLPipeline(obs_dim=4, continuous=True, action_dim=2, buffer_size=64)
    method = SACMethod()
    method.wire_data(args, pipeline)
    raw = method.build_model(args, device=torch.device("cpu"))
    model = method.wrap_model(raw, torch.device("cpu"), args)
    distributed.broadcast_model_state(model)
    # ``log_alpha`` is derived deterministically from ``--sac_alpha``
    # (``log(0.2)``), not from the RNG, so it already starts identical on
    # every rank and needs no synchronisation.

    # Rank-distinct transitions so local gradients differ before reduction.
    rng = np.random.default_rng(1000 + rank)
    buffer = pipeline.replay_buffer
    for _ in range(32):
      obs = rng.normal(size=4).astype(np.float32)
      nxt = rng.normal(size=4).astype(np.float32)
      buffer.push(obs,
                  rng.uniform(-1, 1, size=2).astype(np.float32), float(rng.normal()),
                  nxt, 0.0, False)

    optimization = method.build_optimization(args,
                                             model,
                                             torch.device("cpu"),
                                             ckpt_extra={},
                                             states_to_load={})

    for _step in range(steps):
      batch = pipeline.to_device(buffer.sample(16), torch.device("cpu"))
      loss_out = method.train_step(model, batch, 0)
      method.apply_grad(loss_out, None, None, optimization)

    critic_params = method._optimizer_params(optimization.critic_opt)
    actor_params = method._optimizer_params(optimization.actor_opt)
    torch.save(
        {
            "critic": torch.cat([p.detach().flatten() for p in critic_params]),
            "actor": torch.cat([p.detach().flatten() for p in actor_params]),
            "log_alpha": method._log_alpha.detach().clone(),
        }, os.path.join(tmpdir, f"sac{rank}.pt"))
  finally:
    distributed.destroy_distributed()


@requires_gloo
def test_sac_two_rank_step_keeps_optimizers_and_alpha_identical(tmp_path):
  """SAC's per-method reduction must unify all three optimizers + log_alpha.

  ``SACMethod.apply_grad`` reduces gradients itself (it steps three separate
  optimizers), so this exercises that custom path -- the trainer's shared
  ``_reduce_rl_gradients`` never runs for SAC.  Post-step, every rank must
  hold bit-identical critic/actor weights *and* the temperature.
  """
  _spawn(_sac_step_worker, tmp_path, port=29809, extra=(3,))
  r0 = torch.load(tmp_path / "sac0.pt", weights_only=False)
  r1 = torch.load(tmp_path / "sac1.pt", weights_only=False)

  assert torch.equal(r0["critic"], r1["critic"]), "critic weights diverged across ranks"
  assert torch.equal(r0["actor"], r1["actor"]), "actor weights diverged across ranks"
  assert torch.equal(r0["log_alpha"],
                     r1["log_alpha"]), "log_alpha diverged across ranks"
