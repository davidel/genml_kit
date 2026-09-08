"""Shared checkpoint save/load utilities.

Extracted from ``train.py`` to avoid code duplication with ``pretrain.py``.
Both scripts import these functions rather than maintaining separate copies.
"""

import io
import logging
import os
import re
import tarfile
import tempfile

import torch

from genml_kit.io.storage_utils import save_checkpoint, storage_download
from genml_kit.training.param_align import AlignConfig, align_state_dicts, report_to_str
from genml_kit.utils.attr import MISSING, get_attribute
from genml_kit.utils.logging import fatal


def rename_keys(state_dict, patterns):
  """Apply regex-based key renaming to a state dict.

    Each pattern is a string ``SEARCH;REPLACE`` where *SEARCH* is a
    Python regex and *REPLACE* is a replacement string that may use
    ``$1``, ``$2``, … for capture groups (``$N`` is automatically
    converted to ``\\g<N>`` for Python's ``re.sub``).

    Patterns are applied in order.  The last pattern wins for any key
    that matches multiple patterns.

    Args:
        state_dict: The state dictionary to rename keys on.
        patterns: List of ``"search;replace"`` strings.

    Returns:
        A new dictionary with renamed keys.
  """
  compiled = []
  for pat_str in patterns:
    if ";" not in pat_str:
      fatal(
          f"Invalid --param_rename pattern {pat_str!r}: "
          "expected 'SEARCH;REPLACE'.", ValueError)
    search, replace = pat_str.split(";", 1)
    # Python re.sub only supports \N backreferences, not $N.
    # Convert user-friendly $N to \g<N> for clarity and safety.
    replace = re.sub(r"\$(\d+)", r"\\g<\1>", replace)
    compiled.append((re.compile(search), replace))

  new_state = {}
  for key, value in state_dict.items():
    new_key = key
    for regex, replacement in compiled:
      new_key = regex.sub(replacement, new_key)
    new_state[new_key] = value
  return new_state


def serialize_lora_state(model):
  """Serialize PEFT adapter state to bytes.

  The adapter files are saved via ``save_pretrained``, packed into a
  tar archive (contents only, no top-level directory), and returned
  as ``bytes``.

  Requires ``peft`` to be installed.  *model* must be a ``PeftModel``.
  """
  from peft import PeftModel

  if not isinstance(model, PeftModel):
    fatal(
        f"Expected a PeftModel instance, got {type(model).__name__}",
        TypeError,
    )

  with tempfile.TemporaryDirectory() as tmpdir:
    model.save_pretrained(tmpdir)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
      for entry in os.listdir(tmpdir):
        tar.add(os.path.join(tmpdir, entry), arcname=entry)
    blob = buf.getvalue()
    logging.info("  LoRA adapter blob: %d bytes", len(blob))
    return blob


def deserialize_lora_state(model, blob):
  """Restore PEFT adapter state from a tar blob.

  *blob* must have been produced by :func:`serialize_lora_state`.
  Works whether *model* is already a ``PeftModel`` or a plain model.

  Returns:
      Tuple of ``(model, loaded_keys)`` where *loaded_keys* is the set of
      state-dict keys that were restored from the blob.
  """
  from peft import PeftModel

  with tempfile.TemporaryDirectory() as tmpdir:
    buf = io.BytesIO(blob)
    with tarfile.open(fileobj=buf, mode="r") as tar:
      tar.extractall(tmpdir, filter="data")
    # When the model is already a PeftModel (e.g. resume_checkpoint called
    # after apply_lora), PeftModel.from_pretrained would double-wrap it,
    # producing mangled keys like "base_model.model.base_model.model.*".
    # Instead, swap the existing adapter weights in place.
    if isinstance(model, PeftModel):
      model.delete_adapter("default")
      before = set(model.state_dict().keys())
      model.load_adapter(tmpdir, adapter_name="default")
      model.set_adapter("default")
    else:
      before = set(model.state_dict().keys())
      model = PeftModel.from_pretrained(model, tmpdir)

  loaded = set(model.state_dict().keys()) - before
  return model, loaded


def select_best_checkpoint(root_path):
  """Return the best checkpoint path, falling back to latest.

    Checkpoint files are assumed to follow the ``<root>_best.pt`` /
    ``<root>_latest.pt`` naming convention.

    Args:
        root_path: The checkpoint root path (without ``_best.pt`` or
            ``_latest.pt`` suffix).

    Returns:
        The path to the available checkpoint, or ``None`` if neither
        exists.
  """
  best = root_path + "_best.pt"
  latest = root_path + "_latest.pt"
  if os.path.isfile(best):
    return best
  if os.path.isfile(latest):
    logging.warning(f"Best checkpoint not found ({best}). "
                    f"Falling back to latest: {latest}")
    return latest
  logging.error(f"No checkpoint found at {root_path} "
                f"(tried {best}, {latest}).")


_VALID_STATE_FLAGS = {"opt", "sched", "amp", "none"}


def format_count(num, suffixes=("K", "M", "G", "T")):
  """Format an integer count into a human-readable string with a suffix.

    *suffixes* should have entries for 1024¹, 1024², 1024³, and 1024⁴
    respectively.  Values below 1 024 are returned as-is with no suffix.
    """
  if num < 1024:
    return f"{num}"
  for power, sfx in enumerate(suffixes, start=1):
    if num < 1024**(power + 1):
      return f"{num / 1024**power:.2f}{sfx}"
  # Exceeds the largest suffix — use the last one.
  return f"{num / 1024**len(suffixes):.2f}{suffixes[-1]}"


def _format_bytes(num_bytes):
  """Convert a byte count into a human-readable string (B, KB, MB, GB)."""
  if num_bytes < 1024:
    return f"{num_bytes} B"
  return format_count(num_bytes, suffixes=(" KB", " MB", " GB", " TB"))


def create_model_report(model):
  """Return a full model report string: network structure and parameter details.

    The report includes the ``str(model)`` representation (layer tree), a
    summary line with total/trainable parameter counts, and a detailed
    per-parameter table showing name, shape, element count, and memory
    size.
    """
  parts = []

  parts.append(f"Model structure:\n{model}")

  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  parts.append(f"Model params: {format_count(total)} total, "
               f"{format_count(trainable)} trainable")

  parts.append(_format_model_param_table(model))
  return "\n".join(parts)


def _format_model_param_table(model):
  """Return a formatted table of every parameter's name, shape, count, and size.

    Finishes with a summary line showing total parameter count and total
    memory consumed by the model's parameters.  All columns are
    dynamically aligned into a tabular layout.
    """
  from genml_kit.utils.table import format_table

  params = list(model.named_parameters())
  if not params:
    return "Model has no parameters."

  # Pre-compute display values.
  total_params = 0
  total_bytes = 0
  trainable_params = 0
  data_rows = []
  for name, param in params:
    numel = param.numel()
    param_bytes = numel * param.element_size()
    total_params += numel
    total_bytes += param_bytes
    if param.requires_grad:
      trainable_params += numel
    data_rows.append([
        name,
        str(tuple(param.shape)),
        format_count(numel),
        _format_bytes(param_bytes),
        "yes" if param.requires_grad else "no",
    ])

  headers = ["Parameter", "Shape", "Params", "Size", "Trainable"]
  aligns = ["left", "right", "right", "right", "right"]
  footer = [
      "TOTAL",
      "",
      format_count(total_params),
      _format_bytes(total_bytes),
      format_count(trainable_params),
  ]
  lines = ["Model parameter details:"]
  lines.extend(format_table(headers, data_rows, align=aligns, footer=footer))
  return "\n".join(lines)


def parse_state_flags(flag_value):
  """Parse a comma-separated state flag string into a set of tokens.

    Returns a set like ``{"opt", "sched", "amp"}``.
    If the string contains ``"none"``, returns an empty set.
    Raises ValueError on invalid tokens or empty input.
    """
  tokens = {t.strip().lower() for t in flag_value.split(",")}
  if not tokens:
    fatal("state flag string must not be empty", ValueError)
  invalid = tokens - _VALID_STATE_FLAGS
  if invalid:
    fatal(f"Invalid state flag(s): {invalid}. Allowed: {_VALID_STATE_FLAGS}",
          ValueError)
  if "none" in tokens:
    return set()
  return tokens


def checkpoint_dict(model,
                    optimizer,
                    scheduler,
                    epoch,
                    states_to_save=None,
                    scaler=None,
                    save_frozen=True,
                    **extra):
  """Build a standard checkpoint dict.

    ``states_to_save`` is a set like ``{"opt", "sched", "amp"}``.
    If ``None``, everything is saved (backward compat).
    Any additional keyword arguments are merged into the dict as-is.

    When *save_frozen* is ``False``, only trainable parameters are
    included in the model state dict (via :func:`trainable_state_dict`).
    This drastically reduces checkpoint size for fine-tuning runs with
    a frozen backbone.
    """
  from genml_kit.training.model_utils import trainable_state_dict

  # When a LoRA blob is present the adapter weights live in the blob;
  # only persist the non-adapter part of the state dict here.
  # trainable_state_dict() correctly respects requires_grad, yielding
  # the unfrozen classifier weights (and nothing else).  The lora_
  # keys are then stripped so the same blob can fully restore them.
  if extra.get("lora_state_blob") is not None:
    base = (model.state_dict() if save_frozen else trainable_state_dict(model))
    sd = {k: v for k, v in base.items() if "lora_" not in k}
  else:
    sd = (model.state_dict() if save_frozen else trainable_state_dict(model))
  d = {
      "model_state_dict": sd,
      "epoch": epoch,
  }
  # Persist num_labels so downstream loaders never need to guess.
  num_labels = get_attribute(model, "config.num_labels")
  if num_labels is not MISSING:
    d["num_labels"] = num_labels
  id2label = get_attribute(model, "config.id2label")
  if id2label is not MISSING:
    d["id2label"] = id2label
  if states_to_save is None or "opt" in states_to_save:
    d["optimizer_state_dict"] = optimizer.state_dict()
  if states_to_save is None or "sched" in states_to_save:
    d["scheduler_state_dict"] = scheduler.state_dict(
    ) if scheduler is not None else None
  if states_to_save is None or "amp" in states_to_save:
    d["scaler_state_dict"] = scaler.state_dict() if scaler is not None else None
  d.update(extra)
  return d


def filter_state_dict(ckpt_state, model_state):
  """Filter a checkpoint state dict to only include keys compatible with the model.

    Skips keys whose tensor shape differs between checkpoint and model.
    Returns ``(filtered_state, skipped)`` where *skipped* is a list of
    ``(key, reason)`` tuples describing why each key was dropped.
    """
  filtered = {}
  skipped = []
  for k, v in ckpt_state.items():
    if k not in model_state:
      skipped.append((k, "missing in model"))
    elif v.shape != model_state[k].shape:
      skipped.append((
          k,
          (f"shape mismatch: checkpoint {list(v.shape)} " \
           f"vs model {list(model_state[k].shape)}"),
      ))
    else:
      filtered[k] = v
  return filtered, skipped


def should_save_periodic(global_step, save_every):
  """Return True when *global_step* hits a positive *save_every* multiple.

    Both training scripts increment *global_step* once per optimizer step
    (gradient accumulation already folded in), so the trigger is
    accumulation-agnostic. ``save_every <= 0`` disables periodic saving.
  """
  return save_every > 0 and global_step > 0 and global_step % save_every == 0


class CheckpointSaver:
  """Binds checkpoint state sources once; per-save data passed per call.

    The state-dict things (model, optimizer, scheduler, ``_states_to_save``,
    scaler, ``_save_frozen``) and the destination (``_root`` path plus
    remote sync URI) are fixed at construction, as is the optional
    ``_extra_fn`` hook, which is called on every save to compute
    script-specific KV extras (e.g. ``method_state``) at save time.
    Per-save key/value data -- ``epoch``, ``global_step`` and one-off
    extras -- are passed to the ``save*`` methods, which delegate to
    :func:`checkpoint_dict` and :func:`save_checkpoint`.  Keys from
    ``_extra_fn`` are overridden by same-named call-site keys.

    All constructor-injected fields are internal (underscore-prefixed);
    the public surface is exactly ``should_save`` / ``save`` /
    ``save_best`` / ``save_latest``.
  """

  def __init__(self,
               model,
               optimizer,
               scheduler,
               root,
               states_to_save=None,
               scaler=None,
               save_frozen=True,
               remote_uri=None,
               save_every=0,
               extra_fn=None):
    self._model = model
    self._optimizer = optimizer
    self._scheduler = scheduler
    self._root = root
    self._states_to_save = states_to_save
    self._scaler = scaler
    self._save_frozen = save_frozen
    self._remote_uri = remote_uri
    self._save_every = save_every
    self._extra_fn = extra_fn

  def should_save(self, global_step):
    """Return True when *global_step* hits a ``_save_every`` multiple.

    ``_save_every <= 0`` disables periodic saving; step 0 never triggers.
    Callers pass ``epoch - 1`` (last fully completed epoch) when saving
    mid-epoch so a resume restarts the interrupted epoch.
    """
    return should_save_periodic(global_step, self._save_every)

  def save(self, suffix, epoch, **extra):
    """Write ``_root + suffix`` (e.g. ``"_latest.pt"``) with *extra* KV pairs."""
    path = self._root + suffix
    if self._extra_fn is not None:
      extra = {**self._extra_fn(), **extra}
    save_checkpoint(
        checkpoint_dict(
            self._model,
            self._optimizer,
            self._scheduler,
            epoch,
            states_to_save=self._states_to_save,
            scaler=self._scaler,
            save_frozen=self._save_frozen,
            **extra,
        ),
        path,
        remote_uri=self._remote_uri,
    )
    return path

  def save_best(self, epoch, **extra):
    """Save to ``<root>_best.pt``; *extra* forwarded to checkpoint_dict."""
    return self.save("_best.pt", epoch, **extra)

  def save_latest(self, epoch, **extra):
    """Save to ``<root>_latest.pt``; *extra* forwarded to checkpoint_dict."""
    return self.save("_latest.pt", epoch, **extra)


def resume_checkpoint(ckpt_latest, ckpt_best, model, device):
  """Resume model state from an existing checkpoint.

    Looks for *ckpt_latest* first, then *ckpt_best*.  Restores model weights
    (filtering out shape-mismatched keys) and LoRA adapter state. Training
    states are restored separately by :func:`restore_training_state` after
    the final trainable parameter set has been established.

    Returns ``(start_epoch, best_metric, extra)`` where *extra* contains
    auxiliary checkpoint state and other non-model metadata.
    """

  resume_path = None
  if os.path.exists(ckpt_latest):
    resume_path = ckpt_latest
  elif os.path.exists(ckpt_best):
    resume_path = ckpt_best

  if not resume_path:
    return model, 0, 0.0, {}

  logging.info(f"Resuming from checkpoint: {resume_path}")
  # weights_only=False is intentional: resume checkpoints contain
  # optimizer/scheduler/AMP state objects that the restricted unpickler
  # cannot reconstruct. These files are produced by this toolkit itself,
  # not fetched from untrusted sources.
  ckpt = torch.load(resume_path, map_location=device, weights_only=False)
  logging.info(f"  Checkpoint keys: {list(ckpt.keys())}")

  ckpt_sd = ckpt.get("model_state_dict", {})
  logging.info(f"  model_state_dict: {len(ckpt_sd)} keys")
  if ckpt_sd:
    for k, v in ckpt_sd.items():
      logging.info(f"    {k}  {tuple(v.shape)}")

  # Restore LoRA adapters first so their keys are present before we load
  # the non-LoRA model weights (this lets us cleanly separate truly
  # missing keys from expected LoRA keys in the load_state_dict report).
  lora_blob = ckpt.get("lora_state_blob")
  lora_keys = set()
  if lora_blob is not None:
    logging.info(f"  lora_state_blob: {len(lora_blob)} bytes")
    model, lora_keys = deserialize_lora_state(model, lora_blob)
    logging.info(f"  Restored {len(lora_keys)} LoRA keys from blob")

  filtered, skipped = filter_state_dict(
      ckpt["model_state_dict"],
      model.state_dict(),
  )
  if skipped:
    for k, reason in skipped:
      logging.warning(f"  Skipped key '{k}': {reason}")

  result = model.load_state_dict(filtered, strict=False)
  matched = len(filtered) - len(result.unexpected_keys)
  logging.info(f"  Restored model weights ({matched}/{len(filtered)} keys)")

  truly_missing = set(result.missing_keys) - lora_keys
  if truly_missing:
    logging.warning(f"  Missing keys ({len(truly_missing)} not loaded"
                    f" from checkpoint):")
    cur_sd = model.state_dict()
    for mk in sorted(truly_missing):
      shape = tuple(cur_sd[mk].shape) if mk in cur_sd else "N/A"
      logging.warning(f"    {mk}  {shape}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys (ignored): "
                    f"{result.unexpected_keys}")

  start_epoch = ckpt.get("epoch", -1) + 1
  best_metric = ckpt.get("best_macro_f1", 0.0)
  extra = {
      k: v
      for k, v in ckpt.items()
      if k not in {"model_state_dict", "lora_state_blob", "epoch", "best_macro_f1"}
  }
  extra["_model_state_skipped"] = bool(skipped)

  logging.info(f"  Resumed at epoch {start_epoch}, best_metric={best_metric:.4f}")
  return model, start_epoch, best_metric, extra


def fetch_remote_checkpoint(remote_uri, ckpt_latest, ckpt_best):
  """Pull missing local checkpoints from *remote_uri* before auto-resume.

    :func:`resume_checkpoint` only reads local files, so a run restarted
    on a fresh machine would otherwise silently train from scratch.  For
    each candidate, in the same precedence order as
    :func:`resume_checkpoint` (latest, then best), the remote copy is
    downloaded when — and only when — the local file is absent.  Existing
    local files are never overwritten: for a single-writer run the local
    copy is authoritative.

    Args:
        remote_uri: Remote prefix URI (the ``--remote_checkpoint``
            value); ``None`` disables the fallback entirely.
        ckpt_latest: Local path of the ``_latest.pt`` checkpoint.
        ckpt_best: Local path of the ``_best.pt`` checkpoint.

    Returns:
        List of local paths restored from remote storage.
    """
  if not remote_uri:
    return []
  restored = []
  for path in (ckpt_latest, ckpt_best):
    if os.path.isfile(path):
      logging.info(f"  Local checkpoint present, skipping remote fetch: {path}")
      continue
    if storage_download(remote_uri, path):
      restored.append(path)
  return restored


def open_resume_context(args, model, device):
  """Fetch remote checkpoints and auto-resume into ``model``.

  Wraps the resume boilerplate shared by ``train.py`` and ``pretrain.py``:
  derive the ``_latest.pt`` / ``_best.pt`` paths from ``--checkpoint``,
  pull missing local copies from ``--remote_checkpoint``, then resume.

  Args:
      args: Parsed CLI args (``checkpoint``, ``remote_checkpoint``).
      model: The freshly built model to load weights into.
      device: Device used for tensor remapping during the load.

  Returns:
      Tuple ``(model, start_epoch, best_macro_f1, ckpt_extra)``.
  """
  ckpt_latest = args.checkpoint + "_latest.pt"
  ckpt_best = args.checkpoint + "_best.pt"
  fetch_remote_checkpoint(args.remote_checkpoint, ckpt_latest, ckpt_best)
  model, start_epoch, best_macro_f1, ckpt_extra = resume_checkpoint(
      ckpt_latest,
      ckpt_best,
      model,
      device,
  )
  return model, start_epoch, best_macro_f1, ckpt_extra


def restore_training_state(extra, optimizer, scheduler, scaler, states_to_load):
  """Restore optimizer, scheduler, and AMP state from checkpoint extras.

  This must be called after the final model trainability policy is applied and
  the optimizer has been created from the resulting trainable parameters.
  """
  skipped = extra.get("_model_state_skipped", False)

  opt_state = extra.get("optimizer_state_dict")
  if "opt" in states_to_load and opt_state is not None:
    if skipped:
      logging.warning("  Skipped optimizer restore (model architecture changed)")
    else:
      optimizer.load_state_dict(opt_state)
      logging.info("  Restored optimizer state")
  else:
    logging.info("  Skipped optimizer state")

  sched_state = extra.get("scheduler_state_dict")
  if "sched" in states_to_load and sched_state is not None:
    if scheduler is None:
      logging.info("  Skipped scheduler restore (no scheduler in current run)")
    elif skipped:
      logging.warning("  Skipped scheduler restore (model architecture changed)")
    else:
      scheduler.load_state_dict(sched_state)
      logging.info("  Restored scheduler state")
  else:
    logging.info("  Skipped scheduler state")

  if "amp" in states_to_load:
    scaler_dict = extra.get("scaler_state_dict")
    if scaler_dict is not None and scaler is not None:
      scaler.load_state_dict(scaler_dict)
      logging.info("  Restored GradScaler state")
    else:
      logging.info("  Skipped GradScaler state")


def _format_loaded_params(report, state, unexpected_keys):
  """Build the "actually loaded" summary for :func:`load_checkpoint_weights`.

    The alignment report already lists what *could* be matched
    (unmatched/unused/divergent); this summarises what was really
    transferred into the model, grouped by common dotted prefix on both
    sides so large checkpoints collapse to a handful of lines.

    Args:
        report: ``AlignReport`` whose *mapping* is ``{new_key: old_key}``.
        state: The (already renamed) source state dict the mapping
            points into; used to count transferred parameters.
        unexpected_keys: Keys rejected by ``load_state_dict``; excluded
            from the summary and reported separately by the caller.

    Returns:
        A ready-to-log string.  Empty when nothing was loaded.
    """
  unexpected = set(unexpected_keys)
  loaded = {k: v for k, v in report.mapping.items() if k not in unexpected}
  if not loaded:
    return ""

  groups = {}
  for new_key, old_key in loaded.items():
    new_prefix = new_key.rpartition(".")[0]
    old_prefix = old_key.rpartition(".")[0]
    groups.setdefault((new_prefix, old_prefix), []).append(new_key)

  param_total = sum(state[old_key].numel() for old_key in loaded.values())
  lines = [
      f"param_align: {len(loaded)} keys actually loaded "
      f"({param_total:,} parameters)"
  ]
  for (new_prefix, old_prefix), keys in sorted(groups.items()):
    if len(keys) == 1:
      # A lone key carries full information — print it verbatim.
      lines.append(f"  {keys[0]} <- {loaded[keys[0]]}")
    else:
      # An empty prefix means the keys live at the state-dict root.
      new_label = f"{new_prefix}.*" if new_prefix else "(root)"
      old_label = f"{old_prefix}.*" if old_prefix else "(root)"
      lines.append(f"  {new_label} <- {old_label}  ({len(keys)} keys)")
  return "\n".join(lines)


def load_checkpoint_weights(path,
                            model,
                            device="cpu",
                            strict=False,
                            param_rename=None,
                            max_distance=None):
  """Load model weights from a source checkpoint, aligned by shape.

    ``align_state_dicts`` matches source keys to model keys by tensor
    shape and weighted token distance.  ``param_rename`` patterns are
    applied *before* alignment so that manual renames take priority.

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        device: Device to map tensors to.
        strict: If True, raise on missing/unexpected keys.
        param_rename: Optional list of ``"SEARCH;REPLACE"`` patterns
            for renaming checkpoint keys via regex before alignment.
        max_distance: Override for ``AlignConfig.max_distance``.
            ``None`` uses the default (0.25).

    Returns:
        The ``AlignReport`` produced by :func:`align_state_dicts`.
  """
  if not os.path.isfile(path):
    fatal(
        f"Checkpoint file not found: {path}. "
        "Ensure the checkpoint exists before calling load_checkpoint_weights().",
        FileNotFoundError)

  # weights_only=False is intentional: checkpoints written by older
  # versions of this toolkit may carry non-tensor payload objects
  # (e.g. numpy scalars in custom keys). Files are user-produced, not
  # fetched from untrusted sources.
  ckpt = torch.load(path, map_location=device, weights_only=False)

  state = ckpt.get("model_state_dict")
  if state is None:
    logging.warning(
        "Loading raw state dictionary from '%s'; checkpoint metadata is "
        "unavailable.", path)
    state = ckpt

  if param_rename:
    state = rename_keys(state, param_rename)

  config_kwargs = {}
  if max_distance is not None:
    config_kwargs["max_distance"] = max_distance
  config = AlignConfig(**config_kwargs) if config_kwargs else None

  report = align_state_dicts(state, model.state_dict(), config=config)

  logging.info(report_to_str(report))

  aligned = {}
  for new_key, old_key in report.mapping.items():
    aligned[new_key] = state[old_key]

  result = model.load_state_dict(aligned, strict=strict)
  matched = len(aligned) - len(result.unexpected_keys)
  logging.info(f"  Loaded weights from {path} ({matched}/{len(aligned)} keys)")
  loaded_block = _format_loaded_params(report, state, result.unexpected_keys)
  if loaded_block:
    logging.info(loaded_block)
  if result.missing_keys:
    logging.warning(f"  Missing keys: {result.missing_keys}")
  if result.unexpected_keys:
    logging.warning(f"  Unexpected keys: {result.unexpected_keys}")
  return report
