"""Shared model loading and preprocessing utilities."""

import logging
import re
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from genml_kit.io.checkpointing import format_count
from genml_kit.models import load_model, load_processor
from genml_kit.utils.attr import MISSING, maybe_call, maybe_setattr
from genml_kit.utils.logging import fatal


def set_train_mode(net, mode='train'):
  """Set a model to train or eval, respecting frozen sub-trees.

  When *mode* is ``'train'``, frozen modules whose parameters were all
  frozen via :func:`freeze_model` are forced back to eval mode after
  the (recursive) ``model.train(True)`` call.  This prevents
  BatchNorm running-statistics updates and Dropout activation inside
  frozen backbones.

  Parameters
  ----------
  net : nn.Module
  mode : str, optional
      ``'train'`` or ``'eval'``.  Defaults to ``'train'``.
  """
  if mode == 'eval':
    net.train(False)
  else:
    frozen = _find_frozen_modules(net)
    # Normal recursive train — fires any custom train() overrides.
    net.train(True)
    # Re-freeze: calls each module's own train(False), respecting
    # any custom override rather than setting .training directly.
    for m in frozen:
      m.train(False)


def _find_frozen_modules(net):
  """Return the set of modules where the entire sub-tree is frozen.

  A module is considered *fully frozen* when **all** of its own
  parameters (non-recursive) have ``requires_grad=False`` **and**
  every child sub-tree is also fully frozen.
  """
  frozen = set()

  def net_process(mod):
    children_frozen = all(net_process(c) for c in mod.children())
    own_frozen = all(not p.requires_grad for p in mod.parameters(recurse=False))
    fully_frozen = children_frozen and own_frozen
    if fully_frozen:
      frozen.add(mod)
    return fully_frozen

  net_process(net)
  return frozen


@contextmanager
def model_mode(model, mode):
  """Context manager to temporarily set a model's training mode.

    Args:
        model: A ``torch.nn.Module``.
        mode: Either ``"train"`` or ``"eval"``.

    Yields the model itself so callers can write::

        with model_mode(model, "eval"):
            out = model(x)
  """
  was_training = model.training
  try:
    model.train(mode == "train")
    yield model
  finally:
    if was_training:
      model.train()
    else:
      model.eval()


def _find_head_module(model):
  """Locate the classification head on a model.

    Searches for common head attribute names in the order most HF and
    custom models use them.  Returns ``(attr_name, module)`` or ``None``.
    """
  for attr in ("classifier", "head", "fc"):
    mod = getattr(model, attr, None)
    if mod is not None and isinstance(mod, torch.nn.Module):
      return attr, mod


def extract_lora_params(model):
  """Return the set of parameter names belonging to PEFT adapter layers.

  Uses ``get_peft_model_state_dict`` to identify adapter modules, then
  resolves the full parameter names from ``model.named_parameters()``
  (the state dict strips the adapter name, e.g. ``.default.``).
  """
  from peft import get_peft_model_state_dict
  sd = get_peft_model_state_dict(model)
  prefixes = {key.rsplit(".", 1)[0] for key in sd}
  return {
      n for n, _ in model.named_parameters() if any(
          n.startswith(p + ".") for p in prefixes)
  }


def freeze_model(model, trainable_prefixes):
  """Freeze all parameters except those matching *trainable_prefixes*.

  Each element in *trainable_prefixes* is a regex tested with
  ``re.search(pattern, name)`` (substring match).  Prefix with ``^``
  to anchor at the start.  A parameter is thawed when **any** pattern
  matches.
  """
  compiled = [re.compile(p) for p in trainable_prefixes]
  for p in model.parameters():
    p.requires_grad = False
  thawed = 0
  for name, p in model.named_parameters():
    if any(pat.search(name) for pat in compiled):
      p.requires_grad = True
      thawed += 1
  total = sum(1 for _ in model.parameters())
  logging.info(f"Froze {total - thawed}/{total} parameters "
               f"(thawed patterns: {', '.join(trainable_prefixes)})")
  return total, thawed


def trainable_state_dict(model):
  """Return a state dict containing only parameters with ``requires_grad=True``.

  Useful when saving checkpoints for fine-tuning with a frozen backbone:
  frozen weights can be reloaded from the original source, so only the
  trainable parameters need to be persisted.
  """
  trainable = {name for name, p in model.named_parameters() if p.requires_grad}
  return {k: v for k, v in model.state_dict().items() if k in trainable}


def apply_lora(model, *, r=8, alpha=16, dropout=0.0, target_modules=None):
  """Wrap *model* with LoRA adapters via the PEFT library.

  Returns the wrapped ``PeftModel``.  Requires the ``peft`` package
  (``pip install genml_kit[lora]``).
  """
  from peft import LoraConfig, get_peft_model

  lora_config = LoraConfig(
      r=r,
      lora_alpha=alpha,
      lora_dropout=dropout,
      target_modules=target_modules,
      bias="none",
  )
  model = get_peft_model(model, lora_config)
  trainable, total = model.get_nb_trainable_parameters()
  pct = 100.0 * trainable / total if total else 0.0
  logging.info(
      "LoRA trainable params: %s || all params: %s || trainable%%: %.4f",
      format_count(trainable),
      format_count(total),
      pct,
  )
  return model


def build_val_transform(processor, image_size):
  """Build validation transform: Resize → CenterCrop → Processor."""
  return v2.Compose([
      v2.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
      v2.CenterCrop(image_size),
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
      v2.Normalize(
          mean=processor.image_mean,
          std=processor.image_std,
      ),
  ])


def load_model_for_inference(
    model_name,
    checkpoint_path,
    *,
    device="cuda",
    cache_dir=None,
    model_kwargs=None,
    proc_kwargs=None,
    num_labels=None,
    image_size,
):
  """Load a fine-tuned model ready for inference.

    Args:
        model_name: HuggingFace model name or local path defining architecture,
                    or a registered custom model name (e.g. "convvit").
        checkpoint_path: Path to a raw state dictionary or a wrapped checkpoint
            containing ``model_state_dict``. Only model weights are loaded here.
        device: Target device.
        cache_dir: Optional HF cache directory.
        model_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_model` (e.g. ``{"depth": 6}``).
        proc_kwargs: Optional dict of extra kwargs forwarded to
            :func:`load_processor`.
        num_labels: Number of output classes.  When ``None`` the function
            recovers it from the checkpoint metadata (``num_labels`` or
            ``id2label``).
        image_size: Input resolution (pixels).

    Returns:
        (model, processor) tuple.
    """
  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

  # Resolve num_labels (priority: explicit > checkpoint metadata)
  if num_labels is None:
    num_labels = ckpt.get("num_labels")
  if num_labels is None and "id2label" in ckpt:
    num_labels = len(ckpt["id2label"])
  if num_labels is None:
    fatal(
        "Cannot infer num_labels from checkpoint. "
        "Expected 'num_labels' or 'id2label' in checkpoint metadata.", ValueError)

  # Load id2label mapping from checkpoint if available.
  id2label = ckpt.get("id2label")
  label2id = None
  if id2label is not None:
    label2id = {v: k for k, v in id2label.items()}

  # Use the unified registry — it dispatches to HF or custom transparently.
  model = load_model(
      model_name,
      num_labels=num_labels,
      id2label=id2label,
      label2id=label2id,
      image_size=image_size,
      device=torch.device(device),
      checkpoint_path=checkpoint_path,
      cache_dir=cache_dir,
      **(model_kwargs or {}),
  )

  # Restore label mapping if not provided via checkpoint.
  if id2label is None:
    n = num_labels
    model.config.id2label = {str(i): f"LABEL_{i}" for i in range(n)}
    model.config.label2id = {f"LABEL_{i}": str(i) for i in range(n)}
    logging.warning(
        f"Checkpoint has no id2label — using generic LABEL_0..LABEL_{n - 1}. "
        "Re-train with the updated code to get proper label names.")
  else:
    model.config.id2label = id2label
    model.config.label2id = label2id

  processor = load_processor(
      model_name,
      image_size=image_size,
      cache_dir=cache_dir,
      **(proc_kwargs or {}),
  )

  model.to(device).eval()
  return model, processor


def extract_features(outputs):
  """Extract features from a HuggingFace model output.

    If pooler_output is available (already pooled to [N, hidden_size]),
    use it directly.  Otherwise, apply global average pooling to
    last_hidden_state.

    Args:
        outputs: Model output object (e.g. BaseModelOutputWithPoolingAndNoAttention).

    Returns:
        torch.Tensor of shape [N, hidden_size].
    """
  if outputs.pooler_output is not None:
    return outputs.pooler_output
  return outputs.last_hidden_state.mean([-2, -1])


def extract_backbone_features(model, pixel_values):
  """Extract features from the penultimate layer via a forward hook.

    Supports any model that has a ``.classifier`` attribute (common for HF
    image classifiers and custom wrappers that follow the same convention).

    1. If the model provides an ``extract_backbone_features(pixel_values)``
       method (the genml_kit protocol), delegate to it.
    2. Otherwise, attach a one-shot forward-hook to ``model.classifier``
       to capture the input features just before the classification head.

    The hook is removed after extraction to avoid side-effects.

    The model is temporarily placed in eval mode during feature extraction
    so that BatchNorm/Dropout behave deterministically, and restored to
    its original mode afterwards.

    Raises:
        ValueError: if no ``classifier`` attribute is found and the model
                    does not implement the protocol method.
    """
  with model_mode(model, "eval"):
    return _extract_backbone_features_impl(model, pixel_values)


def _extract_backbone_features_impl(model, pixel_values):
  """Internal implementation of backbone feature extraction.

    .. note::

       The caller (``extract_backbone_features``) is responsible for
       setting the model to eval mode.  This function assumes the
       model is already in eval mode.
    """
  # 1. Image-model protocol
  out = maybe_call(model, "extract_backbone_features", pixel_values)
  if out is not MISSING:
    if isinstance(out, torch.Tensor):
      return out.mean(dim=(2, 3)) if out.ndim == 4 else out
    if isinstance(out, list):
      last = out[-1]
      return last.mean(dim=(2, 3)) if last.ndim == 4 else last
    return extract_features(out)

  # 2. HF hidden states
  try:
    out = model(pixel_values=pixel_values, output_hidden_states=True)
    if hasattr(out, "hidden_states") and out.hidden_states:
      last = out.hidden_states[-1]  # (B, N, D) or (B, D, H, W)
      if last.ndim == 4:
        return last.mean(dim=(2, 3))  # spatial average-pool
      if last.ndim == 3:
        return last[:, 0]  # CLS token
      return last
  except TypeError:
    pass  # model doesn't accept output_hidden_states

  # 3. Hook-based fallback via the classification head
  head = _find_head_module(model)
  if head is None:
    fatal(
        "Cannot extract backbone features — no classification head found "
        "and the model does not implement extract_backbone_features().", ValueError)
  _, head_module = head
  captured = []

  def hook(module, inp, out):
    captured.append(inp[0])

  handle = head_module.register_forward_hook(hook)
  try:
    model(pixel_values=pixel_values)
  finally:
    handle.remove()

  if not captured:
    fatal(
        "Hook did not capture any features. "
        "Ensure the model's head module is called during forward().", ValueError)

  return captured[0]


def collect_features(model, dataset, device, batch_size=128):
  """Extract backbone features for all samples in a dataset.

    Args:
        model: Any model (HF or custom) in eval mode, on device.
        dataset: HFDatasetProxy (already has transform applied).
        device: torch device.
        batch_size: batch size for feature extraction (default 128).

    Returns:
        features: np.ndarray of shape [N, hidden_size], dtype float32.
        labels: np.ndarray of shape [N], dtype int64.
    """
  with model_mode(model, "eval"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features_list = []
    labels_list = []

    with torch.no_grad():
      for pixel_values, batch_labels in loader:
        pixel_values = pixel_values.to(device)
        features = extract_backbone_features(model, pixel_values)
        features_list.append(features.cpu().numpy())
        labels_list.append(batch_labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


def enable_grad_checkpointing(model):
  """Enable gradient checkpointing on *model* (in-place).

  Dispatches to the backend's native API when available, otherwise sets
  a flag that custom models (ConvViT / UVito) pick up in their
  transformer loops.

  Detection order (first match wins):

  1. timm unwrapped — ``model.set_grad_checkpointing(enable=True)``
  2. timm via wrapper — ``model.model.set_grad_checkpointing(enable=True)``
  3. HuggingFace via wrapper — ``model.backbone.gradient_checkpointing_enable()``
  4. HuggingFace unwrapped — ``model.gradient_checkpointing_enable()``
  5. ConvViT / UVito unwrapped — ``model.use_grad_checkpoint = True``
  6. ConvViT / UVito via wrapper — ``model.model.use_grad_checkpoint = True``

  Logs a warning if no known checkpointing mechanism is found.
  """
  # timm native API (unwrapped or via wrapper)
  if maybe_call(model, 'set_grad_checkpointing', enable=True) is not MISSING:
    logging.info("Gradient checkpointing enabled (timm native API).")
    return
  if maybe_call(model, 'model.set_grad_checkpointing', enable=True) is not MISSING:
    logging.info("Gradient checkpointing enabled (timm native API via wrapper).")
    return

  # HuggingFace native API (via wrapper or unwrapped)
  if maybe_call(model, 'backbone.gradient_checkpointing_enable') is not MISSING:
    logging.info("Gradient checkpointing enabled (HuggingFace API via wrapper).")
    return
  if maybe_call(model, 'gradient_checkpointing_enable') is not MISSING:
    logging.info("Gradient checkpointing enabled (HuggingFace native API).")
    return

  # Custom model per-block flag (unwrapped or via wrapper)
  if maybe_setattr(model, 'use_grad_checkpoint', True) is not MISSING:
    logging.info("Gradient checkpointing enabled (per-block, transformer loop).")
    return
  if maybe_setattr(model, 'model.use_grad_checkpoint', True) is not MISSING:
    logging.info(
        "Gradient checkpointing enabled (per-block, transformer loop via wrapper).")
    return

  logging.warning(
      "Gradient checkpointing requested but no known mechanism found "
      "for model type %s. Checkpointing will NOT be active.",
      type(model).__name__,
  )
