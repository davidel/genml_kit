"""Distributed-training plumbing and the model-wrapping seam.

This module owns the *infrastructure* side of multi-GPU training: the process
group, plus the one place where a model is wrapped for data-parallel execution.

Why a facade (see ``docs/MULTI_GPU_PLAN.md`` s 7, Decision C)
-------------------------------------------------------------
``torch.nn.parallel.DistributedDataParallel`` wraps a model as
``DDP(inner)`` and keeps ``inner`` under ``DDP.module``.  It adds no
attribute forwarding (``DDP`` defines no ``__getattr__``), so ``ddp.config``
or ``ddp.backbone`` raise ``AttributeError``; ``ddp.state_dict()`` returns
``module.*`` keys; and ``ddp.some_attr = value`` lands on the DDP object
instead of on the model.

That collides with how this harness uses a model.  Methods read and mutate the
model *inside* ``train_step`` (``model.status``, ``model.q1``,
``model.update_momentum()``, ``model.mask_ratio = ...``), introspection reads
``model.config``, and checkpointing relies on the model's own state-dict keys
(``backbone.*`` / ``classifier.*``) being the portable contract.

:class:`DistributedModel` is therefore a transparent facade: it forwards the
semantic model's surface (attributes, ``state_dict``, ``parameters``,
``train``, ``eval``, ``to``, ...) to the semantic model, while only
``forward`` is routed through DDP -- which is where the gradient all-reduce
happens.  Consequences:

1. checkpoint keys remain the semantic model's own keys, so single-GPU and
   multi-GPU checkpoints are interchangeable;
2. parameter names stay clean, so optimizer regexes, ``freeze_model``,
   layer matching and the gradient monitor keep working unchanged;
3. introspection (``config``, the model report) reaches the semantic model;
4. ``isinstance(model, T)`` still holds, because the facade reports the
   semantic model's ``__class__`` (methods branch on the model's type).

The single-device path is untouched: :func:`wrap_model` returns the model
*unchanged* when the world size is one, so no facade is ever built and no
indirection is added.  :func:`unwrap` peels only infrastructure wrappers and
never PEFT or semantic ones.
"""

import logging
import os
import random

import torch.distributed as dist

from genml_kit.utils.logging import fatal


def barrier():
  """Synchronise all ranks; no-op when not distributed."""
  if is_distributed():
    dist.barrier()


def _reduce_device():
  """Pick the device for a collective tensor (CUDA if available, else CPU)."""
  import torch

  if torch.cuda.is_available() and is_distributed():
    return torch.device("cuda", local_rank())
  return torch.device("cpu")


def all_reduce_sum(value):
  """Sum *value* across ranks; returns the same type it was given.

  Args:
      value: A ``float``/``int`` scalar or a ``torch.Tensor``.

  Returns:
      The reduced value: a Python scalar for a scalar input, else a tensor.
  """
  import torch

  if not is_distributed():
    return value
  is_scalar = not torch.is_tensor(value)
  tensor = torch.tensor(float(value)) if is_scalar else value.clone().detach()
  tensor = tensor.to(_reduce_device())
  dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
  return tensor.item() if is_scalar else tensor


def all_reduce_mean(value, world=None):
  """Average *value* across ranks (same type contract as :func:`all_reduce_sum`)."""
  total = all_reduce_sum(value)
  denom = world if world is not None else world_size()
  return total / denom


def all_reduce_dict(mapping, *, average=False):
  """Reduce every value of *mapping* across ranks, returning a new dict.

  Args:
      mapping: ``{key: float}`` (e.g. per-key validation metrics).
      average: When True divide by the world size, else sum.

  Returns:
      A new dict with the same keys and reduced values.
  """
  if not is_distributed():
    return dict(mapping)
  out = {}
  for key, value in mapping.items():
    out[key] = all_reduce_mean(value) if average else all_reduce_sum(value)
  return out


def any_rank_true(flag):
  """Return ``True`` when *flag* is truthy on **any** rank.

  A collective: every rank must call it in lockstep.  Used to keep
  per-rank control flow (early-stop, NaN skip) identical across ranks so a
  divergent branch cannot desynchronise the DDP gradient all-reduce.

  Args:
      flag: Any value interpretable as a boolean.

  Returns:
      ``True`` if any rank passed a truthy *flag*, else ``False``.
  """
  if not is_distributed():
    return bool(flag)
  return all_reduce_sum(1 if flag else 0) > 0


def all_gather_tensor(tensor):
  """Gather *tensor* from every rank, concatenated in rank order.

  Every rank must pass a tensor of identical shape, so the gather is a plain
  concatenation along dim 0.  Returns *tensor* unchanged when not distributed.

  Args:
      tensor: A ``torch.Tensor`` with the same shape on every rank.

  Returns:
      A tensor of shape ``(world_size * tensor.shape[0], ...)`` on the same
      device as *tensor*.
  """
  import torch

  if not is_distributed():
    return tensor
  t = tensor.to(_reduce_device())
  if not t.is_contiguous():
    t = t.contiguous()
  gathered = [torch.empty_like(t) for _ in range(world_size())]
  dist.all_gather(gathered, t)
  return torch.cat(gathered, dim=0).to(tensor.device)


def broadcast_model_state(model):
  """Broadcast *model*'s parameters and buffers from rank 0 to every rank.

  Needed because the driver seeds each rank differently (``seed + rank()``) so
  that per-rank data streams diverge -- but that also makes ``build_model``
  draw different initial weights on each rank.  Training must start from one
  shared model or the ranks are not replicas of anything.

  The supervised path happens to be masked today because
  ``DistributedDataParallel.__init__`` broadcasts rank 0's parameters when it
  wraps the model.  RL is not wrapped (see ``RLMethod.wrap_model``), so it had
  no such rescue and every rank trained a different network.  Doing the
  broadcast explicitly makes the invariant the driver's, rather than an
  incidental side effect of wrapping.

  Buffers are included so replicated state (normalisation statistics, running
  averages) starts identical too.  No-op when not distributed.

  Args:
      model: The model to synchronise, in place.
  """
  import torch

  if not is_distributed():
    return
  device = _reduce_device()
  with torch.no_grad():
    for tensor in list(model.parameters()) + list(model.buffers()):
      # A broadcast requires a contiguous staging copy on the collective
      # device; the result is copied back into the original storage.
      staged = tensor.data.to(device)
      if not staged.is_contiguous():
        staged = staged.contiguous()
      dist.broadcast(staged, src=0)
      tensor.data.copy_(staged.to(tensor.device))


def all_reduce_gradients(tensors):
  """All-reduce (mean) the ``.grad`` of every tensor in *tensors*.

  Averages rather than sums, matching DDP's default gradient semantics, so an
  explicit reduction is numerically interchangeable with a DDP-wrapped
  ``forward``.

  Needed by flows whose autograd path does not go through a model ``forward``.
  RL is the case that matters: every RL method enters autograd through a
  *method* on the model (``model.get_action_and_value``, ``model.online``,
  ``model.q1.get_value``), never via ``model(x)``, so a DDP reducer would
  never fire and ranks would diverge silently.

  Gradients are bucketed by ``(device, dtype)`` before the collective.  This is
  not optional: ``torch.cat`` type-promotes across dtypes, so a single flat
  buffer would silently upcast ``bf16``/``fp16`` gradients to ``fp32``.  One
  collective per bucket (typically one or two for an RL model) instead of one
  per parameter.

  Tensors with ``.grad is None`` are skipped, but the *number* of non-``None``
  gradients must agree across ranks -- a rank that ran a different set of
  backwards passes would otherwise deadlock the collective, so that is checked
  (cheaply, via a scalar all-reduce) before any gather starts.

  No-op (and free) when not distributed, so callers may invoke it
  unconditionally; hot paths still guard at the call site to avoid building
  the tensor list at all.

  Args:
      tensors: Iterable of ``torch.Tensor`` -- parameters, or any leaf tensor
          such as SAC's standalone ``log_alpha``.
  """
  import torch

  if not is_distributed():
    return
  grads = [t.grad for t in tensors if t.grad is not None]
  if not grads:
    return

  # Deadlock guard: the collective below requires the same tensor count on
  # every rank.  Runs first, so a mismatch raises instead of hanging.
  expected = all_reduce_sum(len(grads)) // world_size()
  if len(grads) != expected:
    fatal(f"inconsistent gradient sets across ranks ({len(grads)} vs "
          f"{expected}): a rank skipped a backward pass")

  buckets = {}
  for grad in grads:
    if not grad.is_contiguous():
      grad.data = grad.data.contiguous()
    buckets.setdefault((grad.device, grad.dtype), []).append(grad)

  for group in buckets.values():
    flat = torch.cat([g.reshape(-1) for g in group])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size())
    offset = 0
    for grad in group:
      numel = grad.numel()
      # copy_ casts back to the gradient's original dtype.
      grad.copy_(flat[offset:offset + numel].view_as(grad))
      offset += numel


def is_distributed():
  """Return ``True`` when a torch process group is initialized."""
  return dist.is_available() and dist.is_initialized()


def rank():
  """Return this process' global rank (``0`` when not distributed)."""
  return dist.get_rank() if is_distributed() else 0


def world_size():
  """Return the number of processes (``1`` when not distributed)."""
  return dist.get_world_size() if is_distributed() else 1


def is_main_process():
  """Return ``True`` on the process that owns logging and checkpoint writes."""
  return rank() == 0


def local_rank():
  """Return this process' node-local rank (``0`` when not distributed)."""
  if not is_distributed():
    return 0
  return int(getattr(dist, "get_node_local_rank", lambda: 0)() or 0)


def init_distributed(backend=None):
  """Join the process group when launched under ``torchrun``.

  Does nothing when ``WORLD_SIZE`` is absent or one, so an ordinary
  ``python -m genml_kit.cli.train`` invocation is unaffected.

  Args:
      backend: Process-group backend; defaults to ``nccl`` on CUDA and
          ``gloo`` otherwise.

  Returns:
      ``True`` if this call created the process group, ``False`` if it was
      a no-op (not launched distributed, or already initialized).
  """
  if is_distributed():
    return False
  _rank, size, local = _env_ranks()
  if size <= 1:
    return False
  if backend is None:
    backend = "nccl" if _cuda_available() else "gloo"
  if backend == "nccl" and _cuda_available():
    import torch

    torch.cuda.set_device(local)
  dist.init_process_group(backend=backend)
  logging.info("Distributed init: backend=%s rank=%d/%d local_rank=%d", backend, rank(),
               size, local)
  return True


def destroy_distributed():
  """Tear the process group down; safe to call unconditionally."""
  if is_distributed():
    dist.destroy_process_group()


def _env_ranks():
  """Return ``(rank, world_size, local_rank)`` from the launcher env vars."""

  def _int(name, default):
    raw = os.environ.get(name)
    try:
      return int(raw) if raw not in (None, "") else default
    except ValueError:
      return default

  return _int("RANK", 0), _int("WORLD_SIZE", 1), _int("LOCAL_RANK", 0)


def _cuda_available():
  import torch

  return torch.cuda.is_available()


def wrap_model(model, device=None, *, find_unused_parameters=False, static_graph=False):
  """Wrap *model* for data-parallel training, or return it unchanged.

  This is the single wrap site of the whole harness.  Call it *after* the
  checkpoint has been restored and the optimizer built, so both see the
  semantic model rather than the infrastructure wrapper.

  Args:
      model: The semantic (or PEFT) model returned by ``build_model``.
      device: Optional device; used to derive ``device_ids`` on CUDA.
      find_unused_parameters: Forwarded to DDP; for methods whose forward
          does not touch every parameter.
      static_graph: Forwarded to DDP; lets DDP optimize for a fixed graph.

  Returns:
      *model* itself when the world size is one (identity -- no facade, no
      indirection), else a :class:`DistributedModel` around it.
  """
  if world_size() < 2:
    return model
  if isinstance(model, DistributedModel):
    return model

  import torch
  from torch.nn.parallel import DistributedDataParallel

  device_ids = None
  if device is not None and torch.device(device).type == "cuda":
    device_ids = [torch.device(device).index or 0]
  ddp = DistributedDataParallel(
      model,
      device_ids=device_ids,
      find_unused_parameters=find_unused_parameters,
      static_graph=static_graph,
  )
  return DistributedModel(model, ddp)


def _is_infra_wrapper(model):
  """Return ``True`` for infrastructure wrappers only (DDP/DataParallel)."""
  from torch.nn.parallel import DataParallel, DistributedDataParallel

  return isinstance(model, (DistributedDataParallel, DataParallel))


def unwrap(model):
  """Peel infrastructure wrappers, returning the semantic model.

  Only :class:`DistributedModel` and ``torch``'s own parallel wrappers are
  removed; PEFT and semantic wrappers are left in place because they *are*
  the model's identity and define the checkpoint key layout.

  Args:
      model: Any model handle.

  Returns:
      The innermost non-infrastructure module.
  """
  while True:
    if isinstance(model, DistributedModel) or _is_infra_wrapper(model):
      model = model.module
    else:
      return model


class DistributedModel:
  """Transparent facade over a DDP-wrapped model.

  ``forward`` goes through DDP (gradient all-reduce); every other attribute
  access, assignment and state-dict operation is delegated to the semantic
  model so that introspection, checkpointing and ``train_step`` keep seeing
  the model they expect.  :attr:`module` is the one explicit way to reach the
  semantic model.

  Attributes:
      module: The semantic model held by the facade.
  """

  def __init__(self, semantic, forward_module):
    """Bind the semantic model and the module that runs the forward pass.

    Args:
        semantic: The model that owns the parameters, the state-dict layout
            and the semantic API.
        forward_module: The object that executes ``forward`` (the DDP
            wrapper, or any callable).
    """
    # Bypass ``__setattr__`` so the handles cannot be forwarded to a model
    # that is not bound yet (and so they never appear in the model's
    # ``state_dict``).
    object.__setattr__(self, "_semantic", semantic)
    object.__setattr__(self, "_forward_module", forward_module)

  # -- the one explicit peel ------------------------------------------------

  @property
  def module(self):
    """The semantic model (this is the only supported unwrap point)."""
    return object.__getattribute__(self, "_semantic")

  @property
  def __class__(self):
    """Report the semantic model's type so ``isinstance`` stays transparent.

    Methods branch on ``isinstance(model, SomeType)`` (e.g. I-JEPA ramps the
    teacher EMA only ``if isinstance(model, IJEPA)``, and the causal-LM path
    infers its attention backend from ``isinstance(model, PreTrainedModel)``).
    Without this, those branches would silently take the wrong path under DDP.
    ``isinstance`` consults ``__class__`` only when the real type does not
    already match, so ``isinstance(facade, DistributedModel)`` (used by
    :func:`unwrap`) keeps working via ``type(self)``.
    """
    return object.__getattribute__(self, "_semantic").__class__

  # -- attribute transparency ----------------------------------------------

  def __getattr__(self, name):
    # Reached only when normal lookup failed, so our own handles (which live
    # in ``__dict__``) never come through here.  The underscore guard stops
    # us forwarding ``_parameters``/``_buffers``/... to the inner model.
    if name.startswith("_"):
      raise AttributeError(name)
    return getattr(object.__getattribute__(self, "_semantic"), name)

  def __setattr__(self, name, value):
    if name.startswith("_"):
      object.__setattr__(self, name, value)
    else:
      setattr(object.__getattribute__(self, "_semantic"), name, value)

  # -- the training entry point --------------------------------------------

  def __call__(self, *args, **kwargs):
    """Run the forward pass through the DDP wrapper (grad all-reduce)."""
    return object.__getattribute__(self, "_forward_module")(*args, **kwargs)

  def forward(self, *args, **kwargs):
    """Alias of :meth:`__call__` for callers that prefer an explicit name."""
    return self(*args, **kwargs)

  # -- delegation of the surface the harness relies on ---------------------

  def state_dict(self, *args, **kwargs):
    """Return the semantic model's state dict (no ``module.*`` prefix)."""
    return self.module.state_dict(*args, **kwargs)

  def load_state_dict(self, *args, **kwargs):
    """Load into the semantic model, keeping checkpoint keys unwrapped."""
    return self.module.load_state_dict(*args, **kwargs)

  def parameters(self, *args, **kwargs):
    """Iterate the semantic model's parameters (clean, unprefixed names)."""
    return self.module.parameters(*args, **kwargs)

  def named_parameters(self, *args, **kwargs):
    """Iterate the semantic model's named parameters."""
    return self.module.named_parameters(*args, **kwargs)

  def buffers(self, *args, **kwargs):
    """Iterate the semantic model's buffers."""
    return self.module.buffers(*args, **kwargs)

  def modules(self, *args, **kwargs):
    """Iterate the semantic model's modules."""
    return self.module.modules(*args, **kwargs)

  def named_modules(self, *args, **kwargs):
    """Iterate the semantic model's named modules."""
    return self.module.named_modules(*args, **kwargs)

  def children(self, *args, **kwargs):
    """Iterate the semantic model's immediate children."""
    return self.module.children(*args, **kwargs)

  def train(self, mode=True):
    """Set the semantic model to train mode; returns the facade."""
    self.module.train(mode)
    return self

  def eval(self):
    """Set the semantic model to eval mode; returns the facade."""
    self.module.eval()
    return self

  def to(self, *args, **kwargs):
    """Move the semantic model; returns the facade."""
    self.module.to(*args, **kwargs)
    return self

  def __repr__(self):
    return repr(self.module)

  def __str__(self):
    return str(self.module)


def make_sampler(dataset,
                 *,
                 shuffle,
                 seed=0,
                 batch_size=None,
                 drop_last=False,
                 epoch=0):
  """Return a sampler that shards *dataset* across ranks, or ``None``.

  ``None`` means "no explicit sampler": the caller should use the loader's own
  ``shuffle`` flag.  That is the single-process behaviour, preserved exactly.

  When distributed, this returns a ``DistributedSampler`` (over *dataset*,
  possibly wrapped in a ``BatchSampler`` when *batch_size* is given) so each
  rank sees a disjoint, non-overlapping shard of the epoch.

  Args:
      dataset: The dataset to shard.
      shuffle: Whether the epoch order should be shuffled (train: True,
          validation: False).
      seed: Base seed; each rank offsets it so shards differ but are
          reproducible.
      batch_size: When given, the sampler yields *batches* (a batch sampler),
          used by pipelines that drive a ``batch_sampler``.
      drop_last: Forwarded to the wrapping ``BatchSampler``.
      epoch: Initial epoch, forwarded to ``set_epoch`` for shuffling.

  Returns:
      A ``DistributedSampler``, a ``BatchSampler`` around it, or ``None``.
  """
  if world_size() < 2:
    return None
  from torch.utils.data import BatchSampler, DistributedSampler

  sampler = DistributedSampler(
      dataset,
      num_replicas=world_size(),
      rank=rank(),
      shuffle=shuffle,
      seed=seed,
      drop_last=drop_last and batch_size is not None,
  )
  sampler.set_epoch(epoch)
  if batch_size is None:
    return sampler
  return BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)


def shard_batch_sampler(batch_sampler):
  """Shard a pipeline's own batch sampler across ranks, or return it unchanged.

  For pipelines that already produce batches (balanced / length-bucketed
  sampling), ``DistributedSampler`` cannot be used directly because it indexes
  a dataset, not a batch stream.  This wraps such a sampler so each rank runs a
  disjoint, equal-length slice of the epoch.  A no-op when not distributed.

  Args:
      batch_sampler: A sampler yielding lists of indices.

  Returns:
      The original sampler when single-process, else a sharding wrapper.
  """
  if world_size() < 2 or batch_sampler is None:
    return batch_sampler
  return _ShardedBatchSampler(batch_sampler)


def shard_loader(loader, *, seed=0):
  """Rebuild *loader* so each rank sees a disjoint shard of the epoch.

  ``DataLoader`` forbids swapping its sampler after construction (the index
  sampler is built in ``__init__``), so a distributed run must construct the
  loader with the sharded sampler.  This rebuilds an equivalent loader from the
  existing one's attributes, replacing only the sampling strategy:

  * a loader with a ``batch_sampler`` (balanced / length-bucketed) gets a
    rank-strided slice of that batch stream;
  * a loader with a plain ``sampler`` or ``shuffle`` gets a
    ``DistributedSampler`` over its dataset.

  Returns the loader unchanged when not distributed, so the single-device path
  is unaffected.

  Args:
      loader: An already-built ``DataLoader``.
      seed: Base seed for the ``DistributedSampler`` shuffle.

  Returns:
      A new ``DataLoader`` (distributed) or *loader* itself (single-process).
  """
  if world_size() < 2:
    return loader
  from torch.utils.data import DataLoader

  kwargs = {
      "num_workers": loader.num_workers,
      "collate_fn": loader.collate_fn,
      "pin_memory": loader.pin_memory,
      "timeout": loader.timeout,
      "worker_init_fn": loader.worker_init_fn,
      "persistent_workers": loader.persistent_workers,
      "prefetch_factor": loader.prefetch_factor if loader.num_workers else None,
      "generator": getattr(loader, "generator", None),
  }
  # A loader constructed with ``batch_size`` owns an auto-generated
  # ``batch_sampler``; one constructed with an explicit ``batch_sampler`` has
  # ``batch_size is None``.  Only the latter needs batch-stream sharding.
  if loader.batch_size is None and loader.batch_sampler is not None:
    kwargs["batch_sampler"] = shard_batch_sampler(loader.batch_sampler)
  else:
    kwargs["sampler"] = make_sampler(loader.dataset,
                                     shuffle=_is_shuffling(loader),
                                     seed=seed)
    kwargs["batch_size"] = loader.batch_size
    kwargs["drop_last"] = loader.drop_last
  return DataLoader(loader.dataset, **kwargs)


def _is_shuffling(loader):
  """Infer whether *loader* shuffles, from its sampler type.

  ``DataLoader`` exposes no ``shuffle`` attribute, so the intent is recovered
  from the sampler it built (``RandomSampler`` means ``shuffle=True``).
  """
  from torch.utils.data import RandomSampler

  return isinstance(loader.sampler, RandomSampler)


def sampler_epoch(sampler, epoch):
  """Advance a sampler's epoch so each epoch reshuffles differently.

  Safe on ``None`` and on samplers without ``set_epoch`` (e.g. the balanced
  sampler), so callers can call it unconditionally each epoch.
  """
  set_epoch = getattr(sampler, "set_epoch", None)
  if set_epoch is not None:
    set_epoch(epoch)


class _ShardedBatchSampler:
  """Give each rank a disjoint, equal-sized slice of another batch sampler.

  Used for pipelines that already yield *batches* (balanced / length-bucketed
  samplers) and therefore cannot be wrapped in ``DistributedSampler`` directly.
  Batches are assigned by stride (``rank, rank + world, ...``) and the count is
  truncated to a multiple of the world size, so every rank runs the same number
  of steps -- a DDP requirement -- and the per-rank union covers the epoch.

  Args:
      batch_sampler: Any iterable yielding lists of indices (one per batch).
      epoch: Current epoch, used to reshuffle the stride offset.
  """

  def __init__(self, batch_sampler, epoch=0):
    self.batch_sampler = batch_sampler
    self.epoch = epoch

  def set_epoch(self, epoch):
    """Record the epoch (reshuffles the offset on the next iteration)."""
    self.epoch = epoch

  def __iter__(self):
    batches = list(self.batch_sampler)
    world = world_size()
    # Truncate to a whole multiple of the world size: equal step counts per
    # rank, which DDP requires to avoid a hang in the all-reduce.
    usable = (len(batches) // world) * world
    batches = batches[:usable]
    # Deterministic per-epoch rotation so each rank sees a different subset
    # that still tiles the whole epoch.
    rng = random.Random(self.epoch)
    offset = rng.randrange(world) if world else 0
    mine = (rank() - offset) % world
    yield from batches[mine::world]

  def __len__(self):
    world = world_size()
    return len(self.batch_sampler) // world if world else len(self.batch_sampler)
