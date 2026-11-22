"""Factory functions for creating optimizers and schedulers.

Used by ``train.py`` to replace hardcoded optimizer /
scheduler creation with configurable versions driven by CLI arguments.
"""

import collections
import logging
import math
import re

import torch
import torch.optim as optim

from genml_kit.io.checkpointing import restore_training_state
from genml_kit.methods.base import Method
from genml_kit.utils.args import amp_dtype_from_args
from genml_kit.utils.logging import fatal
from genml_kit.utils.script import extern_call, is_script_spec


# Everything needed to drive the training loop: the param groups (as
# logged/reported), the optimizer, the LR scheduler (may be None) and the
# AMP GradScaler (only set for float16 AMP on CUDA, else None).
class Optimization:
  """Thin wrapper binding optimizer + scheduler + scaler.

  Implements ``state_dict()`` / ``load_state_dict()`` that delegate to
  the inner optimizer, so ``CheckpointSaver`` can treat the full
  optimization object uniformly (both for standard single-optimizer
  methods and for custom wrappers like ``SACOptimization``).
  """

  def __init__(self, param_groups=None, optimizer=None, scheduler=None, scaler=None):
    self.param_groups = param_groups
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.scaler = scaler

  def state_dict(self):
    return self.optimizer.state_dict()

  def load_state_dict(self, state):
    self.optimizer.load_state_dict(state)


def build_optimization(args, model, device, ckpt_extra, states_to_load, method=None):
  """Build param groups, optimizer, scheduler and AMP scaler for a run.

  Wraps the optimizer/scheduler/scaler setup that ``train.py`` and
  ``pretrain.py`` used to carry as near-verbatim copies, then restores
  any optimizer/scheduler/scaler state requested via ``--state_load``.

  If *method* provides a custom ``build_optimization`` override (e.g.
  SAC with three separate optimizers), it is called instead of the
  standard single-optimizer path.

  Args:
      args: Parsed CLI args (lr, weight_decay, llrd_decay, lr_group,
          optimizer, opt_arg, scheduler, sched_arg, epochs, amp_dtype).
      model: The model whose ``named_parameters()`` feed the groups.
      device: The run's ``torch.device`` (selects the GradScaler device).
      ckpt_extra: Extra dict from the resumed checkpoint (may carry
          optimizer/scheduler/scaler state); may be empty.
      states_to_load: State sources to restore, from ``--state_load``.
      method: Optional method instance; if it overrides
          ``build_optimization``, that override is called directly.

  Returns:
      An ``Optimization`` namedtuple (or compatible custom object).
  """
  # Delegate to method-specific optimization if available.
  if (method is not None and
      type(method).build_optimization is not Method.build_optimization):
    logging.info("Using method-specific build_optimization (%s)", type(method).__name__)
    return method.build_optimization(args, model, device, ckpt_extra, states_to_load)

  if args.llrd_decay is not None:
    param_groups = build_param_groups_llrd(
        dict(model.named_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        decay_factor=args.llrd_decay,
    )
  else:
    param_groups = build_param_groups(
        dict(model.named_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_groups=args.lr_group,
    )
  optimizer = create_optimizer(
      param_groups,
      name=args.optimizer,
      **args.opt_arg,
  )

  # ``--sched_arg`` is the explicit per-scheduler channel and therefore wins
  # over the generic ``--warmup_epochs`` convenience flag.  The generic flag
  # is only forwarded to the warmup_* schedules: plain torch schedulers
  # would reject the extra keyword.
  sched_kwargs = dict(args.sched_arg)
  if (args.scheduler or "").startswith("warmup_"):
    sched_kwargs.setdefault("warmup_epochs", args.warmup_epochs)
  scheduler = create_scheduler(
      optimizer,
      name=args.scheduler,
      epochs=args.epochs,
      base_lr=args.lr,
      **sched_kwargs,
  )

  # Only use GradScaler with float16 AMP (not bfloat16 which has native
  # wider dynamic range and doesn't need loss scaling).
  amp_dtype = amp_dtype_from_args(args)
  scaler = (torch.amp.GradScaler(device)
            if amp_dtype == torch.float16 and device.type == "cuda" else None)
  if scaler is not None:
    logging.info("GradScaler enabled for float16 AMP stability.")

  restore_training_state(
      ckpt_extra,
      optimizer,
      scheduler,
      scaler,
      states_to_load,
  )
  return Optimization(param_groups, optimizer, scheduler, scaler)


def build_param_groups(named_params, lr, weight_decay, lr_groups=None):
  """Build optimizer param groups, optionally with per-group LR by regex.

  Args:
      named_params: Dict mapping parameter names to Parameters (e.g.
          ``dict(model.named_parameters())``).
      lr: Default / fallback learning rate for unmatched params.
      weight_decay: Weight decay applied to every group.
      lr_groups: Optional list of ``"regex=lr"`` strings.  Regexes are
          matched against parameter names; first match wins.  Unmatched
          trainable params fall back to *lr*.

  Returns:
      List of optimizer param-group dicts.
  """
  # Parse lr_groups into (compiled regex, lr_value) pairs.
  parsed = []
  if lr_groups:
    for spec in lr_groups:
      if "=" not in spec:
        fatal(f"Invalid --lr_group spec {spec!r}: expected 'regex=lr'", ValueError)
      pattern, lr_str = spec.rsplit("=", 1)
      try:
        compiled = re.compile(pattern)
      except re.error as exc:
        fatal(f"Invalid regex in --lr_group {spec!r}: {exc}", ValueError)
      try:
        val = float(lr_str)
      except ValueError:
        fatal(f"Invalid LR value in --lr_group {spec!r}: {lr_str!r}", ValueError)
      parsed.append((compiled, val))

  # If no groups, return a single flat group.
  if not parsed:
    trainable = [p for p in named_params.values() if p.requires_grad]
    if not trainable:
      return []
    return [{"params": trainable, "lr": lr, "weight_decay": weight_decay}]

  # Bucket parameters by first-matching regex.
  # group_buckets maps regex index -> list of parameters.
  group_buckets = {i: [] for i in range(len(parsed))}
  fallback = []

  for name, param in named_params.items():
    if not param.requires_grad:
      continue
    matched = False
    for i, (pat, _) in enumerate(parsed):
      if pat.search(name):
        group_buckets[i].append(param)
        matched = True
        break
    if not matched:
      fallback.append(param)

  # Check every regex matched at least one param.
  for i, (pat, _) in enumerate(parsed):
    if not group_buckets[i]:
      fatal(
          f"--lr_group regex {pat.pattern!r} matched no parameters. "
          f"Available params: {list(named_params.keys())}", ValueError)

  # Build the final list of param-group dicts.
  groups = []
  for i, (_, group_lr) in enumerate(parsed):
    params = group_buckets[i]
    logging.info("  LR group %d: lr=%.2e, weight_decay=%.2e, params=%d", i, group_lr,
                 weight_decay, len(params))
    groups.append({"params": params, "lr": group_lr, "weight_decay": weight_decay})

  if fallback:
    logging.info("  LR group default: lr=%.2e, weight_decay=%.2e, params=%d", lr,
                 weight_decay, len(fallback))
    groups.append({"params": fallback, "lr": lr, "weight_decay": weight_decay})

  return groups


def _infer_depth(names):
  """Build a trie from dotted parameter names.

  Args:
      names: Iterable of parameter names (e.g. from
          ``model.named_parameters()``).

  Returns:
      Nested dict (trie) representing the parameter namespace.
  """

  def insert(root, name):
    parent = root
    for p in name.split("."):
      cnode = parent.get(p)
      if cnode is None:
        parent[p] = cnode = {}
      parent = cnode

  root = {}
  for name in names:
    insert(root, name)
  return root


def _depthize(root, dest=None, names=None, depth=0):
  """Walk the trie and assign a numeric depth to each leaf path.

  Numeric keys (e.g. ``"0"``, ``"11"``) represent repeated blocks and
  receive increasing depths.  Non-numeric keys inherit the current depth.

  Args:
      root: Current trie node.
      dest: Accumulator dict mapping ``["dotted", "path"]`` → int depth.
      names: Path prefix accumulated so far.
      depth: Current depth counter.

  Returns:
      The *dest* dict (populated in-place).
  """
  cdest = {} if dest is None else dest
  cnames = [] if names is None else names

  layers = []
  for cname, cnode in root.items():
    name = ".".join(cnames + [cname])
    if cname.isdigit():
      layers.append((cname, cnode))
    else:
      cdest[name] = depth
      _depthize(cnode, dest=cdest, names=cnames + [cname], depth=depth)

  n = 1
  for lid, lnode in sorted(layers, key=lambda x: int(x[0])):
    _depthize(lnode, dest=cdest, names=cnames + [lid], depth=depth + n)
    n += 1

  return cdest


def compute_params_depths(names):
  """Infer a depth for every parameter name via the trie structure.

  Args:
      names: Iterable of dotted parameter names.

  Returns:
      Dict mapping each parameter name to an integer depth.  Deeper
      blocks (e.g. later transformer layers) receive higher values.
  """
  return _depthize(_infer_depth(names))


def build_param_groups_llrd(named_params, lr, weight_decay, decay_factor=0.85):
  """Build optimizer param groups with layer-wise learning-rate decay.

  Shallow blocks (e.g. early transformer layers) receive a smaller
  learning rate, while deeper blocks and non-block parameters use a
  rate closer to *lr*.

  A ``decay_factor`` of ``None`` or ``<= 0`` disables the layer-wise
  decay and returns a single flat-LR group (identical to
  :func:`build_param_groups` with no regex groups).  This guards
  against the degenerate ``decay_factor=0.0`` configuration in which
  every level deeper than the first receives ``lr * 0`` --- silently
  freezing (zero learning rate) nearly the whole network while the
  logged spread ``lr=[0.0, lr]`` makes it look healthy.

  Args:
      named_params: Dict mapping parameter names to Parameters (e.g.
          ``dict(model.named_parameters())``).
      lr: Base learning rate.
      weight_decay: Weight decay for parameters with ``ndim > 1``.
      decay_factor: Multiplicative decay per depth level.  A value of
          0.85 means each shallower level has 85% of the next level's
          learning rate.

  Returns:
      List of param-group dicts suitable for ``torch.optim``.
  """
  depth_map = compute_params_depths(named_params.keys())
  max_depth = max(depth_map.values()) if depth_map else 0

  dparams = collections.defaultdict(list)
  for name, param in named_params.items():
    if param.requires_grad:
      depth = depth_map.get(name, 0)
      dparams[depth].append((name, param))

  # Flat fallback: a decay factor of None or <= 0 means "no layer-wise
  # decay" -- a single group with the base LR (and per-param weight
  # decay).  This prevents decay_factor=0.0 from silently zeroing the
  # learning rate of every layer deeper than the first.
  if decay_factor is None or decay_factor <= 0:
    return build_param_groups(named_params, lr, weight_decay, lr_groups=None)

  param_groups = []
  for depth, params in sorted(dparams.items()):
    lr_value = lr * (decay_factor**(max_depth - depth))

    wdecay_params = []
    ndecay_params = []
    for _, param in params:
      if param.ndim > 1:
        wdecay_params.append(param)
      else:
        ndecay_params.append(param)

    if wdecay_params:
      param_groups.append({
          "params": wdecay_params,
          "lr": lr_value,
          "depth": depth,
          "weight_decay": weight_decay,
      })
    if ndecay_params:
      param_groups.append({
          "params": ndecay_params,
          "lr": lr_value,
          "depth": depth,
          "weight_decay": 0.0,
      })

  return param_groups


def create_optimizer(params, *, name="AdamW", lr=1e-4, weight_decay=0.01, **kwargs):
  """Create an optimizer by name, forwarding extra keyword arguments.

  Three dispatch modes:

  1. **name ends with ``.py``** — loads the script and calls its
     ``create_optimizer(params, **kwargs)``.
  2. **name is a ``torch.optim`` class name** — instantiates it directly.
  3. **name is ``None``** — falls back to the default ``AdamW``.

  *params* can be either a plain iterable of parameters (the legacy
  interface) or a list of param-group dicts as returned by
  :func:`build_param_groups`.  In the latter case the ``lr`` and
  ``weight_decay`` keyword arguments are ignored because each group
  already carries its own values.
  """
  if name is not None and is_script_spec(name):
    return _load_optimizer_script(params,
                                  name,
                                  lr=lr,
                                  weight_decay=weight_decay,
                                  **kwargs)

  cls = getattr(optim, name, None)
  if cls is None or not isinstance(cls, type):
    available = [
        x for x in dir(optim)
        if not x.startswith("_") and isinstance(getattr(optim, x), type)
    ]
    fatal(f"Unknown optimizer {name!r}. "
          f"Available: {', '.join(sorted(available))}", ValueError)

  param_groups_input = (isinstance(params, (list, tuple)) and params and
                        isinstance(params[0], dict))
  if param_groups_input:
    logging.info("Creating %s optimizer with %d param groups", name, len(params))
    for i, g in enumerate(params):
      logging.info("  group %d: lr=%.2e, weight_decay=%.2e, params=%d", i, g["lr"],
                   g["weight_decay"], len(g["params"]))
    if kwargs:
      logging.info("  Extra optimizer kwargs: %s", kwargs)
    optimizer = cls(params, **kwargs)
  else:
    logging.info("Creating %s optimizer (lr=%.2e, weight_decay=%.2e)", name, lr,
                 weight_decay)
    if kwargs:
      logging.info("  Extra optimizer kwargs: %s", kwargs)
    optimizer = cls(params, lr=lr, weight_decay=weight_decay, **kwargs)

  # Seed initial_lr so schedulers see the correct per-group peak LR.
  for g in optimizer.param_groups:
    g.setdefault("initial_lr", g["lr"])

  return optimizer


def report_lr(optimizer, logger=None, step=0):
  """Return a string representation of the current learning rate(s).

  For a single param group returns ``"lr=1.23e-04"``.  For multiple
  groups returns ``"lr=[1.23e-04, 5.67e-05]"`` (min, max).

  If *logger* is provided, also emits ``Train/lr`` (single group) or
  ``Train/lr_group{i}`` (multiple groups) per-step scalars to the sinks.

  Args:
      optimizer: An optimizer with one or more param groups.
      logger: Optional ``MetricsLogger``.
      step: The global training step for the metric sinks.
  """
  from genml_kit.training.metrics_sinks import ensure_logger

  logger = ensure_logger(logger)
  groups = optimizer.param_groups
  if not groups:
    return "lr=?"
  if len(groups) == 1:
    lr = groups[0]["lr"]
    logger.log_step("Train/lr", lr, step=step, resolution="running")
    return f"lr={lr:.2e}"
  lrs = []
  for i, g in enumerate(groups):
    lr = g["lr"]
    lrs.append(lr)
    logger.log_step(f"Train/lr_group{i}", lr, step=step, resolution="running")
  lr_min, lr_max = min(lrs), max(lrs)
  if lr_min == lr_max:
    return f"lr={lr_min:.2e}"
  return f"lr=[{lr_min:.2e}, {lr_max:.2e}]"


class WarmupScheduler(optim.lr_scheduler.LRScheduler):
  """LR schedule with a linear warmup ramp, stepped once per epoch.

  The base LR in the optimizer param groups is the *peak* LR.  During warmup
  (the first ``warmup_epochs`` epochs, fractionally allowed) the multiplier
  ramps linearly 0 -> 1; afterwards the *shape* branch takes over and returns
  the multiplier of the wrapped shape function.

  The trainer steps its scheduler once per epoch
  (``BaseTrainer.run``), so the unit of "steps" here is epochs.  The class
  deliberately has no external warmup wrapper: warmup and shape live in one
  ``get_lr`` so the ramp and the decay can never disagree about which epoch
  is which.

  Args:
      optimizer: Bound optimizer; its ``param_groups`` LRs are the peaks.
      shape_fn: Callable ``epoch_multiplier(int epoch) -> float`` giving the
          post-warmup decay shape as a multiplier of the peak LR.  Built by
          :func:`_build_shape_fn` from the ``warmup_shape`` name.
      warmup_epochs: Length of the linear ramp in epochs (fractions allowed).
      warmup_shape: Shape name for logging/error messages only.
  """

  def __init__(self, optimizer, shape_fn, warmup_epochs=0.0, warmup_shape=""):
    self._shape_fn = shape_fn
    self._warmup_epochs = max(0.0, float(warmup_epochs))
    self._warmup_shape = warmup_shape
    super().__init__(optimizer)

  def get_lr(self):
    if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
      # Linear ramp over the first ``warmup_epochs`` epochs.  ``(t + 1) / w``
      # (not ``t / w``) so epoch 0 trains at a fraction of the peak instead
      # of exactly zero, which would waste the epoch.
      return [
          base_lr * min(1.0, (self.last_epoch + 1.0) / self._warmup_epochs)
          for base_lr in self.base_lrs
      ]
    return [base_lr * self._shape_fn(self.last_epoch) for base_lr in self.base_lrs]


class _WarmupShape:
  """Post-warmup decay branch of :class:`WarmupScheduler`.

  A picklable callable instead of a closure/lambda: the scheduler is
  saved inside every checkpoint (``optimizer`` state includes the
  scheduler in torch's serialization path), and ``torch.save`` pickles
  its attributes.  A local lambda raised ``AttributeError: Can't pickle
  local object`` and made every ``warmup_*`` run crash at the first
  best/latest save.

  Args:
      shape: Decay branch name, ``"cosine"`` | ``"linear"`` | ``"rsqrt"``.
      epochs: Total training epochs (linear/rsqrt denominators).
      eta_min: Cosine floor as a fraction of the peak LR.
      total: Cosine ``T_max``: the epoch of the first minimum.
      rsqrt_offset: ``rsqrt`` warm-squared offset.
  """

  def __init__(self, shape, epochs, eta_min=0.0, total=0, rsqrt_offset=0.0):
    self.shape = shape
    self.epochs = epochs
    self.eta_min = eta_min
    self.total = total
    self.rsqrt_offset = rsqrt_offset

  def __call__(self, ep):
    """Return the LR multiplier for post-warmup epoch *ep*."""
    if self.shape == "cosine":
      # ``T_max`` keeps its torch meaning: the epoch of the first minimum.
      total = max(1, self.total)
      return (self.eta_min + (1.0 - self.eta_min) * 0.5 *
              (1.0 + math.cos(math.pi * min(1.0, ep / total))))
    if self.shape == "linear":
      total = max(1, self.epochs)
      return max(0.0, 1.0 - ep / total)
    # rsqrt
    warmup_squared = self.rsqrt_offset**2
    return math.sqrt(max(1, self.epochs) / max(1.0, ep + warmup_squared + 1.0))


def _build_shape_fn(name, epochs, kwargs):
  """Build the post-warmup decay branch of :class:`WarmupScheduler`.

  Returns a callable ``epoch -> multiplier of the peak LR``.
  """
  kwargs = dict(kwargs)
  if name == "cosine":
    eta_min = float(kwargs.pop("eta_min", 0.0))
    total = max(1, int(kwargs.pop("T_max", epochs)))
    return _WarmupShape("cosine",
                        epochs=max(1, int(epochs)),
                        eta_min=eta_min,
                        total=total)

  if name == "linear":
    return _WarmupShape("linear", epochs=max(1, int(epochs)))

  if name == "rsqrt":
    return _WarmupShape("rsqrt",
                        epochs=max(1, int(epochs)),
                        rsqrt_offset=float(kwargs.pop("rsqrt_offset", 0.0)))

  fatal(f"Unknown warmup shape {name!r}. Available: cosine, linear, rsqrt.", ValueError)


_WARMUP_SHAPES = ("warmup_cosine", "warmup_linear", "warmup_rsqrt")


def create_scheduler(optimizer, *, name=None, epochs=100, base_lr=1e-4, **kwargs):
  """Create a LR scheduler by name, forwarding extra keyword arguments.

  Three dispatch modes:

  1. **name is None / empty** — returns ``None`` (no scheduler).
  2. **name ends with ``.py``** — loads the script and calls its
     ``create_scheduler(optimizer, **kwargs)`` function (similar to
     custom image processors).
  3. **Otherwise** — resolves *name* via ``torch.optim.lr_scheduler``
     and forwards ``epochs``, ``base_lr``, and remaining *kwargs*.

  Args:
      optimizer: A ``torch.optim.Optimizer`` instance.
      name: A ``torch.optim.lr_scheduler`` class name (e.g.
          ``"CosineAnnealingLR"``, ``"StepLR"``), a path / URL to a
          Python script, or ``None`` for no scheduling.
      epochs: Total number of training epochs.
      base_lr: The target / peak learning rate.
      **kwargs: Forwarded to the scheduler constructor (e.g.
          ``T_max``, ``eta_min``, ``step_size``, ``gamma``).

  Returns:
      A ``torch.optim.lr_scheduler._LRScheduler`` instance, or
      ``None`` if *name* is ``None``.
  """
  if not name:
    logging.info("No scheduler requested.")
    return None

  if name in _WARMUP_SHAPES:
    shape_name = name[len("warmup_"):]
    warmup_epochs = float(kwargs.pop("warmup_epochs", 0.0))
    logging.info("Creating %s scheduler (warmup_epochs=%.3g, kwargs=%s)", name,
                 warmup_epochs, kwargs)
    return WarmupScheduler(optimizer,
                           shape_fn=_build_shape_fn(shape_name, epochs, kwargs),
                           warmup_epochs=warmup_epochs,
                           warmup_shape=shape_name)

  if is_script_spec(name):
    return _load_scheduler_script(optimizer,
                                  name,
                                  epochs=epochs,
                                  base_lr=base_lr,
                                  **kwargs)

  cls = getattr(optim.lr_scheduler, name, None)
  if cls is None or not isinstance(cls, type):
    available = [
        x for x in dir(optim.lr_scheduler)
        if not x.startswith("_") and isinstance(getattr(optim.lr_scheduler, x), type)
    ]
    fatal(f"Unknown scheduler {name!r}. "
          f"Available: {', '.join(sorted(available))}", ValueError)

  logging.info("Creating %s scheduler (kwargs=%s)", name, kwargs)
  return cls(optimizer, **kwargs)


def _load_scheduler_script(optimizer, path_or_url, **extra):
  """Load a custom scheduler from a Python file or URL.

  The script must define a callable
  ``create_scheduler(optimizer, **kwargs)``.
  """
  return extern_call(path_or_url, "create_scheduler", optimizer, **extra)


def _load_optimizer_script(params, path_or_url, lr=1e-4, weight_decay=0.01, **extra):
  """Load a custom optimizer from a Python file or URL.

  The script must define a callable
  ``create_optimizer(params, **kwargs)``.

  ``lr`` and ``weight_decay`` are forwarded so that scripts receive the
  same values the user passed via ``--lr`` / ``--weight_decay``.
  """
  extra.setdefault("lr", lr)
  extra.setdefault("weight_decay", weight_decay)
  return extern_call(path_or_url, "create_optimizer", params, **extra)
