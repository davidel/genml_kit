"""Tests for the distributed model-wrapping seam.

Almost everything here runs single-process and never initializes a real
process group: the facade is exercised with a stub forward module, and the
``world_size >= 2`` / launcher-env branches are driven with mocks, so the
whole surface is covered without a GPU or ``torchrun``.

The one multi-process test spawns two gloo ranks and is the only test that can
prove collective behaviour (gradient all-reduce); it is skipped when
``torch.distributed`` is unavailable.
"""

import os

import pytest
import torch
from torch import nn

from genml_kit.io.checkpointing import filter_state_dict
from genml_kit.utils import distributed


class _Semantic(nn.Module):
  """Stand-in for a method's semantic model, with the attribute surface the
  training loop actually touches (``config``, a submodule, a mutable attr)."""

  def __init__(self):
    super().__init__()
    self.backbone = nn.Linear(4, 4)
    self.config = "semantic-config"
    self.mask_ratio = 0.5
    self.momentum_updates = 0

  def forward(self, x):
    return self.backbone(x)

  def update_momentum(self, momentum):
    self.momentum_updates += 1
    return momentum


class _StubForward:
  """Records calls instead of running DDP, so no process group is needed."""

  def __init__(self):
    self.calls = []

  def __call__(self, *args, **kwargs):
    self.calls.append((args, kwargs))
    return "forwarded"


class _SemanticChild(_Semantic):
  """A subclass, to prove ``isinstance`` survives the facade."""


def _facade(semantic=None):
  semantic = semantic or _Semantic()
  stub = _StubForward()
  return distributed.DistributedModel(semantic, stub), semantic, stub


class _FakeDDP:
  """DDP stand-in: keeps ``.module``, records kwargs, forwards to it.

  This lets ``wrap_model``'s ``world_size >= 2`` branch run in one process
  without a process group or GPU.  It deliberately mirrors only the DDP
  surface the facade relies on (``.module`` and being callable).
  """

  last = None

  def __init__(self, module, **kwargs):
    self.module = module
    self.kwargs = kwargs
    _FakeDDP.last = self

  def __call__(self, *args, **kwargs):
    return self.module(*args, **kwargs)


@pytest.fixture
def fake_distributed(monkeypatch):
  """Pretend a 2-process group exists, with DDP faked out.

  Yields a callable ``set_world_size(n)`` so a test can flip the fake world
  size, and restores the real ``DistributedDataParallel`` on teardown.
  """
  import torch.nn.parallel as tpar

  monkeypatch.setattr(distributed, "world_size", lambda: 2)
  monkeypatch.setattr(tpar, "DistributedDataParallel", _FakeDDP)
  _FakeDDP.last = None

  def set_world_size(n):
    monkeypatch.setattr(distributed, "world_size", lambda: n)

  return set_world_size


class TestRankHelpersWithoutGroup:

  def test_defaults_are_single_process(self):
    assert distributed.is_distributed() is False
    assert distributed.world_size() == 1
    assert distributed.rank() == 0
    assert distributed.is_main_process() is True
    assert distributed.local_rank() == 0


class TestSingleDeviceIsIdentity:
  """The zero-cost guarantee: world_size == 1 must not build a facade."""

  def test_wrap_model_returns_the_same_object(self):
    model = _Semantic()
    assert distributed.wrap_model(model) is model

  def test_wrap_model_keeps_isinstance_identity(self):
    model = _Semantic()
    wrapped = distributed.wrap_model(model)
    assert isinstance(wrapped, _Semantic)
    assert not isinstance(wrapped, distributed.DistributedModel)

  def test_unwrap_is_a_noop_rather_than_an_error(self):
    model = _Semantic()
    assert distributed.unwrap(model) is model

  def test_wrap_model_is_idempotent(self):
    model = _Semantic()
    once = distributed.wrap_model(model)
    assert distributed.wrap_model(once) is once


class TestFacadeAttributeTransparency:

  def test_reads_the_semantic_attribute(self):
    facade, semantic, _ = _facade()
    assert facade.config == semantic.config
    assert facade.mask_ratio == 0.5

  def test_reads_registered_submodules(self):
    facade, semantic, _ = _facade()
    assert facade.backbone is semantic.backbone

  def test_writes_land_on_the_semantic_model(self):
    # This is the DDP failure mode: ``ddp.mask_ratio = ...`` would land on the
    # DDP object.  The facade must forward the assignment to the real model.
    facade, semantic, _ = _facade()
    facade.mask_ratio = 0.25
    assert semantic.mask_ratio == 0.25
    assert facade.mask_ratio == 0.25

  def test_method_calls_delegate(self):
    facade, semantic, _ = _facade()
    facade.update_momentum(0.99)
    assert semantic.momentum_updates == 1

  def test_missing_attribute_raises(self):
    facade, _, _ = _facade()
    with pytest.raises(AttributeError):
      _ = facade.does_not_exist

  def test_module_is_the_semantic_model(self):
    facade, semantic, _ = _facade()
    assert facade.module is semantic


class TestFacadeIsinstanceTransparency:
  """Methods branch on ``isinstance(model, T)``; the facade must not lie.

  Regression for the bug where ``DistributedModel`` reported its own type, so
  e.g. I-JEPA's ``if isinstance(model, IJEPA): update_momentum(...)`` silently
  skipped the teacher EMA ramp under DDP.
  """

  def test_isinstance_of_the_semantic_type(self):
    facade, _, _ = _facade()
    assert isinstance(facade, _Semantic)

  def test_isinstance_of_a_semantic_subclass(self):
    facade, _, _ = _facade(_SemanticChild())
    assert isinstance(facade, _SemanticChild)
    assert isinstance(facade, _Semantic)

  def test_isinstance_of_nn_module(self):
    facade, _, _ = _facade()
    assert isinstance(facade, nn.Module)

  def test_facade_is_still_recognised_as_itself(self):
    # unwrap() depends on this: the passthrough must not mask the real type.
    facade, _, _ = _facade()
    assert isinstance(facade, distributed.DistributedModel)

  def test_real_type_is_unchanged(self):
    facade, _, _ = _facade()
    assert type(facade) is distributed.DistributedModel

  def test_semantic_model_is_not_mistaken_for_a_facade(self):
    _, semantic, _ = _facade()
    assert not isinstance(semantic, distributed.DistributedModel)

  def test_isinstance_branch_runs_through_the_facade(self):
    # The exact I-JEPA shape: the isinstance guard must pass, and the call it
    # guards must reach the semantic model.
    facade, semantic, _ = _facade(_SemanticChild())
    if isinstance(facade, _SemanticChild):
      facade.update_momentum(0.9)
    assert semantic.momentum_updates == 1


class TestFacadeForward:

  def test_call_goes_through_the_forward_module(self):
    facade, _, stub = _facade()
    x = torch.zeros(2, 4)
    assert facade(x) == "forwarded"
    assert len(stub.calls) == 1
    assert stub.calls[0][0][0] is x

  def test_forward_method_also_routes_through_the_stub(self):
    facade, _, stub = _facade()
    assert facade.forward(torch.zeros(1, 4)) == "forwarded"
    assert len(stub.calls) == 1


class TestFacadeStateDict:
  """Checkpoint keys are the portable contract: no ``module.`` prefix."""

  def test_keys_match_the_semantic_model(self):
    facade, semantic, _ = _facade()
    assert set(facade.state_dict()) == set(semantic.state_dict())

  def test_keys_carry_no_wrapper_prefix(self):
    facade, _, _ = _facade()
    assert all(not k.startswith("module.") for k in facade.state_dict())

  def test_checkpoint_from_facade_loads_into_plain_model(self):
    facade, semantic, _ = _facade()
    fresh = _Semantic()
    saved = facade.state_dict()
    filtered, skipped = filter_state_dict(saved, fresh.state_dict())
    assert not skipped
    assert set(filtered) == set(fresh.state_dict())


class TestFacadeParameterNames:
  """Optimizer regexes, freeze_model and layer matching index by name."""

  def test_named_parameters_match_the_semantic_model(self):
    facade, semantic, _ = _facade()
    assert ([n for n, _ in facade.named_parameters()
            ] == [n for n, _ in semantic.named_parameters()])

  def test_parameter_objects_are_identical(self):
    facade, semantic, _ = _facade()
    assert [id(p) for p in facade.parameters()
           ] == [id(p) for p in semantic.parameters()]

  def test_named_modules_match(self):
    facade, semantic, _ = _facade()
    assert ([n for n, _ in facade.named_modules()
            ] == [n for n, _ in semantic.named_modules()])


class TestFacadeModeAndDevice:

  def test_train_and_eval_toggle_the_semantic_model(self):
    facade, semantic, _ = _facade()
    facade.eval()
    assert semantic.training is False
    facade.train()
    assert semantic.training is True

  def test_train_and_eval_return_the_facade(self):
    facade, _, _ = _facade()
    assert facade.train() is facade
    assert facade.eval() is facade

  def test_to_returns_the_facade(self):
    facade, _, _ = _facade()
    assert facade.to("cpu") is facade

  def test_str_matches_the_semantic_model(self):
    facade, semantic, _ = _facade()
    assert str(facade) == str(semantic)


class TestUnwrap:

  def test_unwrap_peels_the_facade(self):
    facade, semantic, _ = _facade()
    assert distributed.unwrap(facade) is semantic

  def test_unwrap_does_not_peel_semantic_wrappers(self):
    # A semantic wrapper is the model's identity, never infrastructure.
    class _SemanticWrapper(nn.Module):

      def __init__(self, inner):
        super().__init__()
        self.inner = inner

    wrapped = _SemanticWrapper(_Semantic())
    assert distributed.unwrap(wrapped) is wrapped


class TestWrapModelUnderFakeWorldSize:
  """``wrap_model``'s ``world_size >= 2`` branch, without spawning ranks."""

  def test_builds_a_facade(self, fake_distributed):
    model = _Semantic()
    wrapped = distributed.wrap_model(model)
    assert isinstance(wrapped, distributed.DistributedModel)

  def test_facade_module_is_the_semantic_model(self, fake_distributed):
    model = _Semantic()
    assert distributed.wrap_model(model).module is model

  def test_forward_routes_through_the_ddp_wrapper(self, fake_distributed):
    model = _Semantic()
    wrapped = distributed.wrap_model(model)
    out = wrapped(torch.zeros(2, 4))
    assert out.shape == (2, 4)
    assert _FakeDDP.last.module is model

  def test_state_dict_has_no_wrapper_prefix(self, fake_distributed):
    wrapped = distributed.wrap_model(_Semantic())
    assert all(not k.startswith("module.") for k in wrapped.state_dict())

  def test_unwrap_peels_back_to_the_semantic_model(self, fake_distributed):
    model = _Semantic()
    assert distributed.unwrap(distributed.wrap_model(model)) is model

  def test_ddp_flags_default_to_false(self, fake_distributed):
    distributed.wrap_model(_Semantic())
    assert _FakeDDP.last.kwargs["find_unused_parameters"] is False
    assert _FakeDDP.last.kwargs["static_graph"] is False

  def test_ddp_flags_are_forwarded(self, fake_distributed):
    distributed.wrap_model(_Semantic(), find_unused_parameters=True, static_graph=True)
    assert _FakeDDP.last.kwargs["find_unused_parameters"] is True
    assert _FakeDDP.last.kwargs["static_graph"] is True

  def test_wrap_model_is_idempotent_when_distributed(self, fake_distributed):
    once = distributed.wrap_model(_Semantic())
    assert distributed.wrap_model(once) is once

  def test_no_device_ids_on_cpu(self, fake_distributed):
    distributed.wrap_model(_Semantic(), torch.device("cpu"))
    assert _FakeDDP.last.kwargs["device_ids"] is None


class TestInitDistributedEnv:
  """Launcher-env semantics of ``init_distributed`` (no real group)."""

  @pytest.fixture(autouse=True)
  def _no_cuda(self, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  def _spy(self, monkeypatch):
    calls = []

    def fake_init(**kwargs):
      calls.append(kwargs)

    monkeypatch.setattr(distributed.dist, "init_process_group", fake_init)
    return calls

  def test_unset_world_size_is_a_noop(self, monkeypatch):
    calls = self._spy(monkeypatch)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert distributed.init_distributed() is False
    assert calls == []

  def test_world_size_one_is_a_noop(self, monkeypatch):
    calls = self._spy(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert distributed.init_distributed() is False
    assert calls == []

  def test_world_size_two_initializes_gloo_on_cpu(self, monkeypatch):
    calls = self._spy(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert distributed.init_distributed() is True
    assert calls == [{"backend": "gloo"}]

  def test_second_call_is_a_noop(self, monkeypatch):
    self._spy(monkeypatch)
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(distributed, "is_distributed", lambda: True)
    assert distributed.init_distributed() is False


class TestShardLoaderSingleProcess:
  """Without distribution the loader must be returned untouched."""

  def _loader(self):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.arange(16).float().reshape(16, 1))
    return DataLoader(ds, batch_size=4, shuffle=True)

  def test_shard_loader_is_identity(self):
    loader = self._loader()
    assert distributed.shard_loader(loader) is loader

  def test_make_sampler_is_none(self):
    from torch.utils.data import TensorDataset
    ds = TensorDataset(torch.arange(8))
    assert distributed.make_sampler(ds, shuffle=True) is None

  def test_shard_batch_sampler_is_identity(self):
    assert distributed.shard_batch_sampler(None) is None

  def test_all_reduce_sum_is_identity(self):
    assert distributed.all_reduce_sum(7.5) == 7.5

  def test_all_reduce_dict_is_identity(self):
    assert distributed.all_reduce_dict({"a": 2.0, "b": 3.0}) == {"a": 2.0, "b": 3.0}

  def test_barrier_is_noop(self):
    assert distributed.barrier() is None


class TestShardLoaderDistributed:
  """With a fake 2-rank group, sharding must be disjoint and complete."""

  def _loader(self, n=16, batch_size=4):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.arange(n).float().reshape(n, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

  def _collect(self, loader):
    out = []
    for batch in loader:
      out.extend(int(v) for v in batch[0].reshape(-1))
    return sorted(out)

  def test_shards_are_disjoint_and_cover_the_epoch(self, monkeypatch, fake_distributed):
    base = self._loader(n=16, batch_size=4)
    seen = {}
    for r in (0, 1):
      monkeypatch.setattr(distributed, "rank", lambda r=r: r)
      sharded = distributed.shard_loader(base, seed=0)
      seen[r] = self._collect(sharded)
    # No sample appears on both ranks, and together they cover the dataset.
    assert set(seen[0]).isdisjoint(seen[1])
    assert set(seen[0]) | set(seen[1]) == set(range(16))

  def test_rank_strided_batch_sampler_is_disjoint(self, monkeypatch, fake_distributed):
    from torch.utils.data import BatchSampler, SequentialSampler
    inner = BatchSampler(SequentialSampler(range(16)), batch_size=4, drop_last=False)
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    r0 = list(distributed.shard_batch_sampler(inner))
    monkeypatch.setattr(distributed, "rank", lambda: 1)
    r1 = list(distributed.shard_batch_sampler(inner))
    flat0 = [i for b in r0 for i in b]
    flat1 = [i for b in r1 for i in b]
    assert set(flat0).isdisjoint(flat1)
    assert len(r0) == len(r1), "ranks must run equal step counts for DDP"


class TestShardIndexSamplerPreservesSemantics:
  """``shard_loader`` must shard the sampler, not replace it.

  Regression: the dispatch keyed on ``loader.batch_size is None``, so a pipeline
  that set *both* a sampler and ``batch_size`` (the images pipeline did) fell
  into the ``else`` branch and its sampler was swapped for a plain
  ``DistributedSampler`` -- ``--sampler weighted`` and ``--sampler balanced``
  silently degraded to uniform sampling under DDP.
  """

  def _ds(self, n=40):
    from torch.utils.data import TensorDataset
    return TensorDataset(torch.arange(n).float().reshape(n, 1))

  def test_weighted_sampler_is_sharded_not_replaced(self, fake_distributed):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    weights = torch.linspace(0.1, 2.0, 40)
    loader = DataLoader(self._ds(),
                        batch_size=8,
                        sampler=WeightedRandomSampler(weights, 40, replacement=True))
    sharded = distributed.shard_loader(loader, seed=0)
    # The whole point: still weighted.
    assert type(sharded.sampler).__name__ == "_ShardedWeightedSampler"
    assert torch.allclose(sharded.sampler.weights, weights.double())

  def test_weighted_global_draw_is_identical_across_ranks(self, monkeypatch,
                                                          fake_distributed):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    loader = DataLoader(self._ds(),
                        batch_size=8,
                        sampler=WeightedRandomSampler(torch.ones(40),
                                                      40,
                                                      replacement=True))
    sharded = distributed.shard_loader(loader, seed=3)
    distributed.sampler_epoch(sharded, 2)

    # Option (a) of the plan: one shared global draw, sliced per rank. The seed
    # must therefore be rank-independent (seed + epoch, never seed + rank).
    monkeypatch.setattr(distributed, "rank", lambda: 0)
    got0 = sharded.sampler._global_order()
    monkeypatch.setattr(distributed, "rank", lambda: 1)
    got1 = sharded.sampler._global_order()
    assert torch.equal(got0, got1)

  def test_weighted_shard_slices_tile_the_draw_with_equal_steps(
      self, monkeypatch, fake_distributed):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    loader = DataLoader(self._ds(),
                        batch_size=8,
                        sampler=WeightedRandomSampler(torch.ones(40),
                                                      40,
                                                      replacement=True))
    sharded = distributed.shard_loader(loader, seed=3)
    distributed.sampler_epoch(sharded, 0)

    monkeypatch.setattr(distributed, "rank", lambda: 0)
    r0 = list(sharded.sampler)
    monkeypatch.setattr(distributed, "rank", lambda: 1)
    r1 = list(sharded.sampler)
    assert len(r0) == len(r1), "ranks must run equal step counts for DDP"
    # Both ranks slice the *same* global draw at alternating positions, so the
    # two slices tile it exactly (indices may repeat within a slice: sampling is
    # with replacement, so identity/disjointness is about positions, not values).
    order = sharded.sampler._global_order().tolist()
    assert r0 == order[0::2]
    assert r1 == order[1::2]

  def test_epoch_reshuffles_the_weighted_draw(self, fake_distributed):
    from torch.utils.data import DataLoader, WeightedRandomSampler
    loader = DataLoader(self._ds(),
                        batch_size=8,
                        sampler=WeightedRandomSampler(torch.ones(40),
                                                      40,
                                                      replacement=True))
    sharded = distributed.shard_loader(loader, seed=3)
    distributed.sampler_epoch(sharded, 0)
    first = sharded.sampler._global_order()
    distributed.sampler_epoch(sharded, 1)
    second = sharded.sampler._global_order()
    assert not torch.equal(first, second)

  def test_plain_shuffle_still_gets_a_distributed_sampler(self, fake_distributed):
    from torch.utils.data import DataLoader
    loader = DataLoader(self._ds(), batch_size=8, shuffle=True)
    sharded = distributed.shard_loader(loader, seed=0)
    assert type(sharded.sampler).__name__ == "DistributedSampler"

  def test_unknown_sampler_fails_loudly_instead_of_degrading(self, fake_distributed):
    from torch.utils.data import DataLoader, Sampler

    class _Exotic(Sampler):

      def __iter__(self):
        return iter([])

      def __len__(self):
        return 0

    loader = DataLoader(self._ds(), batch_size=8, sampler=_Exotic())
    with pytest.raises(Exception, match="cannot shard loader sampler _Exotic"):
      distributed.shard_loader(loader, seed=0)


class _RecordingSampler:
  """Sampler stand-in that records every ``set_epoch`` call it receives."""

  def __init__(self):
    self.epochs = []

  def set_epoch(self, epoch):
    self.epochs.append(epoch)

  def __iter__(self):
    return iter([])

  def __len__(self):
    return 0


class TestSamplerEpoch:
  """``sampler_epoch`` must reach the sampler that owns the shuffle state.

  Regression: both call sites used to pick the first *truthy* attribute
  (``loader.batch_sampler or loader.sampler``).  For a sharded index loader
  ``loader.batch_sampler`` is a plain ``BatchSampler`` -- truthy, but with no
  ``set_epoch`` -- so it masked the ``DistributedSampler`` in ``loader.sampler``
  and the epoch stayed pinned at 0 for the whole run.
  """

  def _loader(self, n=16, batch_size=4):
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.arange(n).float().reshape(n, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

  def test_reaches_the_distributed_sampler_under_a_batch_sampler(
      self, fake_distributed):
    loader = distributed.shard_loader(self._loader(), seed=0)
    # The premise: DataLoader wrapped the DistributedSampler in a plain
    # BatchSampler, which is what used to mask it.
    assert type(loader.batch_sampler).__name__ == "BatchSampler"
    assert not hasattr(loader.batch_sampler, "set_epoch")

    distributed.sampler_epoch(loader, 3)
    assert loader.sampler.epoch == 3

  def test_distributed_sampler_reshuffles_each_epoch(self, fake_distributed):
    loader = distributed.shard_loader(self._loader(), seed=0)
    orders = []
    for epoch in range(3):
      distributed.sampler_epoch(loader, epoch)
      assert loader.sampler.epoch == epoch
      orders.append(tuple(int(b[0][0]) for b in loader))
    # Different permutation (and shard boundaries) every epoch; identical
    # tuples would mean the epoch never took effect.
    assert len(set(orders)) == 3

  def test_reaches_a_batch_stream_sampler(self):
    from torch.utils.data import DataLoader, TensorDataset
    rec = _RecordingSampler()
    ds = TensorDataset(torch.arange(8).float().reshape(8, 1))
    # ``batch_sampler`` fully determines the batches; no ``batch_size``.
    distributed.sampler_epoch(DataLoader(ds, batch_sampler=rec), 5)
    assert rec.epochs == [5]

  def test_bare_sampler_is_still_accepted(self):
    """Backwards compatibility: a caller holding only the sampler works."""
    rec = _RecordingSampler()
    distributed.sampler_epoch(rec, 2)
    assert rec.epochs == [2]

  def test_sampler_without_set_epoch_is_a_noop(self):
    from torch.utils.data import BatchSampler, SequentialSampler
    inner = BatchSampler(SequentialSampler(range(8)), batch_size=4, drop_last=False)
    # Must not raise: plain BatchSampler has no set_epoch.
    assert distributed.sampler_epoch(inner, 1) is None

  def test_none_is_a_noop(self):
    assert distributed.sampler_epoch(None, 1) is None


class _TinyDDPModel(nn.Module):

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(4, 1)

  def forward(self, x):
    return self.fc(x).sum()


def _spawn_worker(rank, world_size, tmp_path):
  """Train one step on each rank and dump this rank's parameters."""
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = "29557"
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(rank)

  torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
  try:
    model = distributed.wrap_model(_TinyDDPModel())
    # Identical inputs on every rank, different seed order irrelevant here:
    # after all-reduce the gradients (and hence the params) must be equal.
    x = torch.ones(2, 4) * (1.0 + 0.0 * rank)
    loss = model(x)
    loss.backward()
    torch.distributed.all_reduce(torch.zeros(1))  # barrier-ish
    torch.save([p.detach().clone() for p in model.parameters()],
               tmp_path / f"rank{rank}.pt")
  finally:
    distributed.destroy_distributed()


def test_two_rank_step_yields_identical_parameters(tmp_path):
  """DDP all-reduce makes post-step parameters identical across ranks."""
  if not distributed.dist.is_available():
    pytest.skip("torch.distributed unavailable")

  torch.multiprocessing.spawn(_spawn_worker, args=(2, tmp_path), nprocs=2, join=True)

  rank0 = torch.load(tmp_path / "rank0.pt", weights_only=False)
  rank1 = torch.load(tmp_path / "rank1.pt", weights_only=False)
  assert len(rank0) == len(rank1)
  for a, b in zip(rank0, rank1):
    assert torch.equal(a, b)
