"""Distributed-training plumbing and the model-wrapping seam.

This module owns the *infrastructure* side of multi-GPU training: the process
group, plus the one place where a model is wrapped for data-parallel execution.

Why a facade
------------
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
from itertools import islice

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import (BatchSampler, DataLoader, DistributedSampler,
                              RandomSampler, SequentialSampler, WeightedRandomSampler)

from genml_kit.datasets.base import is_iterable, is_random_access
from genml_kit.utils.logging import fatal


def barrier():
  """Synchronise all ranks; no-op when not distributed."""
  if is_distributed():
    dist.barrier()


def _reduce_device():
  """Pick the device for a collective tensor (CUDA if available, else CPU)."""
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
  if not is_distributed():
    return tensor
  t = tensor.to(_reduce_device())
  if not t.is_contiguous():
    t = t.contiguous()
  gathered = [torch.empty_like(t) for _ in range(world_size())]
  dist.all_gather(gathered, t)
  return torch.cat(gathered, dim=0).to(tensor.device)


def all_gather_object(obj):
  """Gather a picklable *obj* from every rank, in rank order.

  Unlike :func:`all_gather_tensor` the per-rank objects may differ in shape or
  type, so this pickles them (``all_gather_object``).  Used to collect per-rank
  validation results for a cross-rank merge.  Returns ``[obj]`` when not
  distributed.

  Args:
      obj: Any picklable object, e.g. a ``ValidationResult``.

  Returns:
      A list of length ``world_size()`` in rank order.
  """
  if not is_distributed():
    return [obj]
  out = [None] * world_size()
  dist.all_gather_object(out, obj)
  return out


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

  def no_sync(self):
    """Expose DDP's ``no_sync`` context manager through the facade.

    Only the forward module is DDP-wrapped; the semantic model returned by
    :attr:`module` / :func:`unwrap` has no ``no_sync``.  The trainer uses this
    to defer the gradient all-reduce on non-final accumulation micro-steps.
    """
    return object.__getattribute__(self, "_forward_module").no_sync()

  def to(self, *args, **kwargs):
    """Move the semantic model; returns the facade."""
    self.module.to(*args, **kwargs)
    return self

  def __repr__(self):
    return repr(self.module)

  def __str__(self):
    return str(self.module)


class _SkippedSampler:
  """Yield an index sampler's indices after skipping a fixed prefix.

    Used by ``DataPipeline.seek_batch`` to resume a random-access epoch in
    ``O(1)``: the loader keeps building and collating batches exactly as before,
    but the index stream starts where the checkpoint left off, so no sample is
    decoded only to be thrown away.

    The skip is expressed in *samples* (``batch_idx * batch_size``) because the
    wrapping ``BatchSampler`` groups whatever the inner sampler yields.

    ``__len__`` reports the remaining count so ``len(loader)`` stays truthful
    after a seek.
    """

  def __init__(self, inner, skip):
    self._inner = inner
    self._skip = max(0, int(skip))

  def __iter__(self):
    return islice(iter(self._inner), self._skip, None)

  def __len__(self):
    return max(0, len(self._inner) - self._skip)

  def __getattr__(self, name):
    # Anything else (``set_epoch``, ``total_size``, ...) belongs to the wrapped
    # sampler; epoch control must keep reaching it through us.
    return getattr(self._inner, name)


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


class _ShardedWeightedSampler:
  """Rank-sharded ``WeightedRandomSampler``: one global draw, disjoint slices.

  The single-process sampler draws ``num_samples`` indices *with replacement*
  from per-sample weights.  Each rank must therefore agree on that **global**
  draw and then keep only its ``rank::world`` slice, so the slices tile the draw
  exactly like a ``DistributedSampler`` shard and every rank runs the same
  number of steps.

  Note that tiling is by *position*, not by value: because sampling is with
  replacement, the same index can legitimately land on both ranks.  That is the
  same guarantee the single-process sampler gives (each draw is an independent
  sample), merely split across ranks instead of repeated within one.

  Agreement requires a rank-independent RNG: the generator is seeded with
  ``seed + epoch`` and never ``seed + rank()``.  This is the one place where the
  per-rank streams must *not* diverge: the sampling decision has to be global,
  only the data streams need to differ.

  The epoch is republished by :func:`sampler_epoch` each epoch (this class
  exposes ``set_epoch`` for exactly that reason), so the draw changes per epoch
  while staying identical across ranks.

  Args:
      weights: Per-sample weights, as accepted by ``WeightedRandomSampler``.
      seed: Base seed (already the rank-independent one).
      epoch: Initial epoch.
  """

  def __init__(self, weights, *, seed=0, epoch=0):
    self.weights = torch.as_tensor(weights, dtype=torch.double)
    self.num_samples = len(self.weights)
    self.seed = seed
    self.epoch = epoch

  def set_epoch(self, epoch):
    """Record the epoch (reshuffles the global draw on the next iteration)."""
    self.epoch = epoch

  def _global_order(self):
    """The full, rank-independent index draw for the current epoch."""
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    return torch.multinomial(self.weights,
                             self.num_samples,
                             replacement=True,
                             generator=generator)

  def __iter__(self):
    world = world_size() or 1
    order = self._global_order()
    # Truncate to a whole multiple of the world size so every rank runs the
    # same number of steps -- a DDP requirement, as in _ShardedBatchSampler.
    usable = (len(order) // world) * world
    yield from order[:usable][rank()::world].tolist()

  def __len__(self):
    world = world_size() or 1
    return self.num_samples // world


class _ShardedBatchLoader:
  """Give each rank a disjoint share of a streaming loader's batches.

  A streaming (``IterableDataset``) loader has no index space, so there is no
  sampler whose type could describe how to split it -- ``DataLoader`` rejects
  both ``sampler`` and ``batch_sampler`` for such datasets and installs an
  internal ``_InfiniteConstantSampler`` placeholder that yields ``None``
  forever.  The split therefore cannot be expressed as a sampler, and this
  wrapper is the only available seam: it consumes the loader and hands each
  rank its own share of the batch stream.

  The same mechanism does the shuffling.  A stream cannot be shuffled by index,
  so batches are permuted inside a buffer of ``world_size * k`` batches (a
  "round"):

  * every rank builds the **same** round from the **same** stream and permutes
    it with the **same** rank-independent seed, so the permutation agrees;
  * rank *r* then keeps positions ``j % world == r``, which is disjoint and
    exhaustive across ranks by construction;
  * each rank yields exactly ``k`` batches per full round, so all ranks run the
    same number of steps and none leaves a collective early.

  ``k=0`` degenerates to the minimal round-robin (a round of ``world_size``
  batches, kept in stream order); larger ``k`` widens the window and improves
  mixing at a cost of ``batch_size * world_size * k`` items resident per rank.

  The wrapped loader is used *as built*: nothing is reconstructed, so
  ``num_workers``, ``prefetch_factor``, ``worker_init_fn`` and the rest keep
  their configured values.  Every other attribute (``dataset``, ``batch_size``,
  ``sampler``, ...) is forwarded to it, so introspection and
  ``DataPipeline.seek_batch`` see through the wrapper.

  Args:
      loader: An already-built streaming ``DataLoader``.
      rank: This process' rank.
      world: Total number of ranks.
      k: Batches per rank per round (``--shuffle_buffer_k``).
      seed: Base seed for the shared permutation.
  """

  def __init__(self, loader, rank, world, *, k=10, seed=0):
    self._loader = loader
    self._rank = rank
    self._world = max(1, world)
    # A round needs at least one batch per rank, so k is clamped to >= 1; k=0
    # is the documented "round-robin without permuting" case and is handled in
    # __iter__ rather than by disabling the stride.
    self._k = max(0, int(k))
    self._seed = seed
    self.epoch = 0

  def set_epoch(self, epoch):
    """Record the epoch: the next iteration permutes rounds differently."""
    self.epoch = epoch

  def _round_size(self):
    """Batches pulled per round by every rank together."""
    return self._world * max(1, self._k)

  def __iter__(self):
    it = iter(self._loader)
    round_no = 0
    while True:
      window = list(islice(it, self._round_size()))
      if not window:
        return
      # Drop the ragged tail of the epoch so the round splits evenly: a
      # partial round would give some ranks an extra step and desynchronise
      # the DDP all-reduce.
      usable = (len(window) // self._world) * self._world
      if not usable:
        return
      window = window[:usable]
      if self._k:
        # Rank-independent: every rank must land on the same permutation for
        # the stride to be disjoint.  The round number is part of the seed so
        # successive rounds do not permute identically.
        random.Random(f"{self._seed}-{self.epoch}-{round_no}").shuffle(window)
      for position, batch in enumerate(window):
        if position % self._world == self._rank:
          yield batch
      round_no += 1

  def __len__(self):
    """Estimated batches this rank will yield (informational only)."""
    try:
      return len(self._loader) // self._world
    except TypeError:
      return 0

  def __getattr__(self, name):
    # Everything else belongs to the wrapped loader.  DataPipeline.seek_batch
    # and the trainer's length accounting reach dataset/num_workers/sampler
    # through here, so the wrapper stays transparent.
    if name.startswith("_"):
      raise AttributeError(name)
    return getattr(self._loader, name)


def shard_index_sampler(loader, *, seed=0):
  """Return a rank-sharded equivalent of *loader*'s index sampler.

  ``shard_loader`` must not blanket-replace ``loader.sampler``: the pipeline
  chose it for a reason (class weights, class balance), and substituting a plain
  ``DistributedSampler`` silently changes the training objective.  This
  dispatches on the sampler's *type* instead:

  * ``WeightedRandomSampler`` -> :class:`_ShardedWeightedSampler`, which keeps
    the weights and shards one global draw;
  * ``RandomSampler`` / ``SequentialSampler`` / absent -> a
    ``DistributedSampler``, i.e. the plain dataset shard;
  * anything else -> ``fatal``, because silently degrading it is exactly the
    bug this function exists to prevent.

  Args:
      loader: An already-built ``DataLoader``.
      seed: Base seed for the shard.

  Returns:
      A sampler to pass as ``sampler=`` when rebuilding the loader.
  """
  sampler = getattr(loader, "sampler", None)
  if isinstance(sampler, WeightedRandomSampler):
    return _ShardedWeightedSampler(sampler.weights, seed=seed)
  if sampler is None or isinstance(sampler, (RandomSampler, SequentialSampler)):
    return make_sampler(loader.dataset, shuffle=_is_shuffling(loader), seed=seed)
  fatal(
      f"cannot shard loader sampler {type(sampler).__name__} across ranks: "
      f"it would be silently replaced by an unweighted DistributedSampler. "
      f"Pass --dist_arg shard=false to keep it per-rank, or teach "
      f"shard_index_sampler about it.", NotImplementedError)


def shard_loader(loader, *, seed=0, shuffle_buffer_k=10):
  """Shard *loader* so each rank sees a disjoint share of the epoch.

  Two strategies, dispatched on the dataset:

  * **Streaming** (``IterableDataset``) -- the loader is *wrapped* in
    :class:`_ShardedBatchLoader`, which splits the batch stream by rank and
    permutes it in a shared round-robin buffer.  There is no index space, so
    neither of the sampler strategies below applies and the loader is kept
    exactly as built.
  * **Random access** -- ``DataLoader`` forbids swapping its sampler after
    construction (the index sampler is built in ``__init__``), so the loader is
    *rebuilt* with the sharded sampler:
    a loader driven by an explicit ``batch_sampler`` (balanced /
    length-bucketed) gets a rank-strided slice of that batch stream; a loader
    driven by an index ``sampler`` or ``shuffle`` gets a shard of whatever that
    sampler was (:func:`shard_index_sampler`).

  The dispatch keys on *how the loader samples*, not on ``batch_size``: a
  pipeline may set both a sampler and ``batch_size`` (the images pipeline does),
  and keying on ``batch_size`` alone discarded the sampler.

  Returns the loader unchanged when not distributed, so the single-device path
  is unaffected.

  Args:
      loader: An already-built ``DataLoader``.
      seed: Base seed for the sharded sampler.
      shuffle_buffer_k: Batches per rank per shuffle round for a streaming
        loader (``--shuffle_buffer_k``); ignored for random access, which
        shuffles by sampler.

  Returns:
      A new ``DataLoader`` or :class:`_ShardedBatchLoader` (distributed), or
      *loader* itself (single-process).
  """
  if world_size() < 2:
    return loader

  dataset = getattr(loader, "dataset", None)
  # Dispatch on the dataset's *capabilities*, not on its class.  A wrapper such
  # as FieldSelectorDataset exposes whatever the dataset it wraps exposes, so over
  # a stream it is iterable without being indexable -- and isinstance(..,
  # IterableDataset) would be False for it, sending it down the sampler branch
  # where DistributedSampler calls len() and fails.
  if is_iterable(dataset) and not is_random_access(dataset):
    # A stream has no index space, so its "sampler" is DataLoader's internal
    # _InfiniteConstantSampler placeholder and says nothing about how to split
    # it.  Wrap the loader as built: both sampler branches below are
    # meaningless here.
    if isinstance(loader, _ShardedBatchLoader):
      return loader
    return _ShardedBatchLoader(loader,
                               rank(),
                               world_size(),
                               k=shuffle_buffer_k,
                               seed=seed)

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
  # A loader constructed with an explicit ``batch_sampler`` has
  # ``batch_size is None``; only that one needs batch-stream sharding.  Anything
  # else samples by index, whatever its ``batch_size`` says.
  if loader.batch_size is None and loader.batch_sampler is not None:
    kwargs["batch_sampler"] = shard_batch_sampler(loader.batch_sampler)
  else:
    kwargs["sampler"] = shard_index_sampler(loader, seed=seed)
    kwargs["batch_size"] = loader.batch_size
    kwargs["drop_last"] = loader.drop_last
  return DataLoader(loader.dataset, **kwargs)


def _is_shuffling(loader):
  """Infer whether *loader* shuffles, from its sampler type.

  ``DataLoader`` exposes no ``shuffle`` attribute, so the intent is recovered
  from the sampler it built (``RandomSampler`` means ``shuffle=True``).
  """
  return isinstance(loader.sampler, RandomSampler)


def _epoch_targets(root):
  """Collect the distinct objects under *root* that may own per-epoch state.

  Three places can hold the epoch, and which one does depends on how the
  ``DataLoader`` was built:

  * ``loader.sampler`` -- a plain index sampler, e.g. the ``DistributedSampler``
    that :func:`shard_loader` installs;
  * ``loader.batch_sampler`` -- a pipeline's own batch-stream sampler
    (length-bucketed), or a :class:`_ShardedBatchSampler`;
  * ``loader.batch_sampler.sampler`` -- when ``batch_size`` was passed,
    ``DataLoader`` wraps the index sampler in a *plain* ``BatchSampler``, so the
    real sampler is one level down.

  ``root`` itself comes first, so a bare sampler needs no special case.
  Callers then filter on capability (:func:`sampler_epoch`); collecting the
  whole chain and filtering is what makes the epoch reach the
  ``DistributedSampler``.  Picking the first *truthy* attribute instead is a
  silent no-op: ``loader.batch_sampler`` is a ``BatchSampler``, which has no
  ``set_epoch``, and being truthy it masks ``loader.sampler``.

  The loader, a plain ``BatchSampler`` and the sampler it wraps can alias each
  other, so identity duplicates are dropped here rather than by the caller.

  Args:
      root: A ``DataLoader``, a sampler, or any stand-in exposing these
        attributes.

  Returns:
      A list of non-``None`` candidate objects, outermost first, without
      identity duplicates.
  """
  batch_sampler = getattr(root, "batch_sampler", None)
  targets = []
  for candidate in (
      root,
      batch_sampler,
      getattr(root, "sampler", None),
      getattr(batch_sampler, "sampler", None),
  ):
    if candidate is None:
      continue
    # Deduplicate by identity, not equality: a sampler may define __eq__, and
    # two distinct samplers that compare equal must each receive the epoch.
    if not any(candidate is target for target in targets):
      targets.append(candidate)
  return targets


def sampler_epoch(root, epoch):
  """Advance every per-epoch sampler reachable from *root*.

  Callers invoke this unconditionally each epoch; it is safe on ``None``, on a
  loader with no samplers, and on samplers without ``set_epoch`` (the balanced
  sampler has none).

  Args:
      root: A loader (preferred) or a bare sampler; both are accepted, so a
        caller holding only the sampler object keeps working.
      epoch: The epoch index to publish.
  """
  for target in _epoch_targets(root):
    set_epoch = getattr(target, "set_epoch", None)
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
