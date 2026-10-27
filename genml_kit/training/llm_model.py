"""Model loading helpers shared by the LLM generation and eval CLIs.

Both ``genml-kit-llm generate`` and ``genml-kit-llm eval`` need the same
"load a causal LM ready for inference" step, covering three weight sources:

- the **base model** -- a HuggingFace repo id or a registered custom model,
  loaded through the ``MODELS`` registry (``load_model(task="causal_lm")``);
- an optional **raw state-dict checkpoint** -- a ``.pt`` file, either a bare
  state dict or a ``CheckpointSaver`` dict (``model_state_dict`` key), or a
  checkpoint *root* resolving to ``<root>_best.pt`` / ``<root>_latest.pt``;
- an optional **PEFT adapter** -- either a ``save_pretrained`` adapter
  directory, or the ``lora_state_blob`` embedded in a ``CheckpointSaver``
  checkpoint.

Nothing here is HuggingFace-specific: the base model goes through the same
registry the trainer uses, so custom models work unchanged.
"""

import logging
import os

import torch

from genml_kit.io.checkpointing import (deserialize_lora_state, filter_state_dict,
                                        select_best_checkpoint)
from genml_kit.models import load_model, load_tokenizer
from genml_kit.utils.logging import fatal


def resolve_checkpoint_file(path):
  """Return a concrete checkpoint file for *path*.

    *path* may be a ``.pt`` file directly, or a checkpoint *root* (the value
    ``--checkpoint`` takes during training), in which case the ``_best.pt``
    file is preferred and ``_latest.pt`` is the fallback.
    """
  if os.path.isfile(path):
    return path
  selected = select_best_checkpoint(path)
  if selected is None:
    fatal(
        f"No checkpoint found at {path!r} "
        f"(expected a file or a <root>_best.pt / <root>_latest.pt pair).",
        FileNotFoundError)
  return selected


def load_checkpoint_into_model(model, checkpoint, device):
  """Load model weights (and any embedded LoRA blob) from *checkpoint*.

    Same-architecture load: shape-mismatched keys are skipped with a warning
    (via :func:`filter_state_dict`), mirroring :func:`resume_checkpoint`.
    When the checkpoint carries a ``lora_state_blob`` it is applied first, so
    the adapter keys are present before the non-adapter weights land.
    """
  path = resolve_checkpoint_file(checkpoint)
  logging.info("Loading checkpoint: %s", path)
  ckpt = torch.load(path, map_location=device, weights_only=False)
  blob = ckpt.get("lora_state_blob") if isinstance(ckpt, dict) else None
  state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

  if blob is not None:
    logging.info("  Restoring LoRA adapters from embedded blob (%d bytes).", len(blob))
    model, lora_keys = deserialize_lora_state(model, blob)
    logging.info("  Restored %d LoRA keys.", len(lora_keys))

  filtered, skipped = filter_state_dict(state, model.state_dict())
  for key, reason in skipped:
    logging.warning("  Skipped key '%s': %s", key, reason)
  result = model.load_state_dict(filtered, strict=False)
  matched = len(filtered) - len(result.unexpected_keys)
  logging.info("  Restored model weights (%d/%d keys).", matched, len(filtered))
  if result.missing_keys:
    logging.warning("  Missing keys: %s", result.missing_keys)
  return model


def load_lora_adapter(model, adapter_dir):
  """Attach a PEFT adapter from *adapter_dir* (a ``save_pretrained`` dir)."""
  try:
    from peft import PeftModel
  except ImportError:
    fatal(
        "Loading a LoRA adapter requires the 'peft' package. "
        "Install it with: pip install genml_kit[lora]", ImportError)
  if not os.path.isdir(adapter_dir):
    fatal(f"LoRA adapter directory not found: {adapter_dir!r}", FileNotFoundError)
  logging.info("Loading LoRA adapter: %s", adapter_dir)
  if isinstance(model, PeftModel):
    model.load_adapter(adapter_dir, adapter_name="default")
    model.set_adapter("default")
  else:
    model = PeftModel.from_pretrained(model, adapter_dir)
  return model


def load_lm_for_inference(model_name,
                          tokenizer_name=None,
                          checkpoint=None,
                          lora_adapter=None,
                          device="cpu",
                          cache_dir=None,
                          trust_remote_code=False,
                          attn_implementation=None):
  """Load a causal LM and its tokenizer, ready to generate.

    Args:
        model_name: HuggingFace repo id or registered custom model name.
        tokenizer_name: tokenizer override; defaults to *model_name*.
        checkpoint: optional raw state-dict / ``CheckpointSaver`` path.
        lora_adapter: optional PEFT adapter directory.
        device: torch device (string or object).
        cache_dir: HuggingFace cache directory.
        trust_remote_code: allow custom code from the Hub.
        attn_implementation: attention backend (e.g. ``"sdpa"``).

    Returns:
        ``(model, tokenizer)`` with the model in ``eval`` mode on *device*.
    """
  device = torch.device(device)
  tokenizer = load_tokenizer(tokenizer_name or model_name, cache_dir=cache_dir)
  model = load_model(
      model_name,
      task="causal_lm",
      num_labels=0,
      cache_dir=cache_dir,
      device=device,
      trust_remote_code=trust_remote_code,
      attn_implementation=attn_implementation,
  )
  if checkpoint:
    model = load_checkpoint_into_model(model, checkpoint, device)
  if lora_adapter:
    model = load_lora_adapter(model, lora_adapter)
  model.to(device)
  model.eval()
  return model, tokenizer
